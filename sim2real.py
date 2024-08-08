import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
from visualization_msgs.msg import Marker
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        
        self.declare_parameter('goal_x', 1.0)
        self.declare_parameter('goal_y', 1.0)
        self.declare_parameter('angle', 0.0)
        self.declare_parameter('GOAL_REACHED_DIST', 0.1)

        self.goal_x = self.get_parameter('goal_x').get_parameter_value().double_value
        self.goal_y = self.get_parameter('goal_y').get_parameter_value().double_value
        self.angle = self.get_parameter('angle').get_parameter_value().double_value
        self.GOAL_REACHED_DIST = self.get_parameter('GOAL_REACHED_DIST').get_parameter_value().double_value

        self.previous_action = [0.0, 0.0]
        self.distance = 0.0
        self.theta = 0.0
        self.odom_x = 0.0
        self.odom_y = 0.0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor = Actor(24, 2).to(self.device)
        self.actor.load_state_dict(torch.load('actor.pth', map_location=self.device))
        self.actor.eval()
        
        self.laser_subscriber = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.odom_subscriber = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_publisher = self.create_publisher(Marker, '/laser_markers', 10)

    def laser_callback(self, msg):
        min_laser = [min(msg.ranges[i:i+18]) for i in range(-90, 90, 9)]
        min_laser = self.clean_laser_data(min_laser)
        state = min_laser + [self.distance, self.theta] + self.previous_action
        if len(state) != 24:
            self.get_logger().error(f"Unexpected state vector length: {len(state)}")
            return
        
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(state_tensor).squeeze().cpu().numpy()

        twist = Twist()
        twist.linear.x = float(action[0])
        twist.angular.z = float(action[1])
        self.cmd_vel_publisher.publish(twist)

        self.previous_action = [float(action[0]), float(action[1])]
        self.publish_laser_markers(min_laser)

    def clean_laser_data(self, laser_data):
        max_range = 10.0  # Replace this with the maximum range of your laser scanner
        cleaned_data = [x if np.isfinite(x) else max_range for x in laser_data]
        return cleaned_data

    def publish_laser_markers(self, min_laser):
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'laser_scan'
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1  # Size
        marker.scale.y = 0.1
        marker.color.a = 1.0  # Alpha
        marker.color.r = 1.0  # Red
        marker.color.g = 0.0  # Green
        marker.color.b = 0.0  # Blue

        for i, distance in enumerate(min_laser):
            angle = (i * np.pi / 10)  
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            marker.points.append(Point(x=x, y=y, z=0.0))

        self.marker_publisher.publish(marker)

    def odom_callback(self, msg):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        self.angle = msg.pose.pose.orientation.z

        self.distance = np.linalg.norm([self.goal_x - self.odom_x, self.goal_y - self.odom_y])
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(skew_x ** 2 + skew_y ** 2)
        mag2 = math.sqrt(1 ** 2 + 0 ** 2)
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            beta = -beta if skew_x < 0 else -beta
        theta = beta - self.angle
        self.theta = (theta + np.pi) % (2 * np.pi) - np.pi

def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
