import rclpy
import os
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Actor neural network model
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

class TD3(object):
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)

    def get_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def load(self, filename, directory):
        directory = os.path.expanduser(directory)
        actor_path = os.path.join(directory, "%s_actor.pth" % filename)
        print(f"Loading the actor model from: {actor_path}")
        self.actor.load_state_dict(torch.load(actor_path, map_location = torch.device('cpu')))

# Define the RobotController class that extends the ROS2 Node class
class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        
        # Declare parameters for goal position and control thresholds
        self.declare_parameter('goal_x', 1.0)
        self.declare_parameter('goal_y', 1.0)
        self.declare_parameter('angle', 0.0)
        self.declare_parameter('GOAL_REACHED_DIST', 0.1)

        # Initialize parameters
        self.goal_x = self.get_parameter('goal_x').get_parameter_value().double_value
        self.goal_y = self.get_parameter('goal_y').get_parameter_value().double_value
        self.angle = self.get_parameter('angle').get_parameter_value().double_value
        self.GOAL_REACHED_DIST = self.get_parameter('GOAL_REACHED_DIST').get_parameter_value().double_value

        # Initialize variables for previous actions, distance to goal, and robot's orientation
        self.previous_action = [0.0, 0.0]
        self.distance = 0.0
        self.theta = 0.0
        self.odom_x = 0.0
        self.odom_y = 0.0

        # Set up the device and load the pre-trained actor model
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.actor = Actor(24, 2).to(self.device)
        # self.actor.load_state_dict(torch.load('actor.pth', map_location=self.device, weights_only=True))
        # self.actor.eval()

        # Set up the device and load the pre-trained actor model
        self.actor = TD3(24, 2)
        try:
            self.actor.load("TD3_velodyne", "~/Downloads/pytorch_models")
        except:
            raise ValueError("Could not load the stored model parameters")
        
        # Set up ROS2 subscriptions and publishers
        self.laser_subscriber = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.odom_subscriber = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_publisher = self.create_publisher(Marker, '/laser_markers', 10)

    # Callback function for laser scan data
    def laser_callback(self, msg):
        # Process laser scan data into 20 segments
        min_laser = []
        for i in range(-90, 90, 9):
            segment = msg.ranges[i:i+18]
            if segment:
                valid_distances = [dist for dist in segment if not math.isnan(dist) and not math.isinf(dist)]
                if valid_distances:
                    min_laser.append(min(valid_distances))
                else:
                    min_laser.append(float('inf'))  # Use a large value to indicate no valid range data
            else:
                min_laser.append(float('inf'))  # Use a large value to indicate no valid range data

        # Create state vector including laser data, distance to goal, angle to goal, and previous actions
        state = min_laser + [self.distance, self.theta] + self.previous_action
        if len(state) != 24:
            self.get_logger().error(f"Unexpected state vector length: {len(state)}")
            return
        
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        # Predict action using the actor model
        with torch.no_grad():
            action = self.actor.get_action(state_tensor)

        # Scale actions to desired ranges
        action[0] = ((action[0] + 1) / 2) * 0.4  # Scale action[0] to [0, 0.4]
        action[1] = action[1] * 0.5  # Scale action[1] to [-0.5, 0.5]

        # Log the state and action
        self.get_logger().info(f"State: {state}")
        self.get_logger().info(f"Action: {action}")

        # Publish the predicted action as a velocity command
        twist = Twist()
        twist.linear.x = float(action[0])
        twist.angular.z = float(action[1])
        self.cmd_vel_publisher.publish(twist)

        # Update the previous action and publish laser markers for visualization
        self.previous_action = [float(action[0]), float(action[1])]
        self.publish_laser_markers(min_laser)

    # Function to publish laser markers for visualization in RViz
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

        # Convert laser scan data into points
        for i, distance in enumerate(min_laser):
            if math.isinf(distance) or math.isnan(distance):
                continue  # Skip invalid distances
            angle = (i * np.pi / 10)
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            marker.points.append(Point(x=x, y=y, z=0.0))

        self.marker_publisher.publish(marker)

    # Callback function for odometry data
    def odom_callback(self, msg):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        self.angle = msg.pose.pose.orientation.z

        # Calculate distance and angle to the goal
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

        # Log the odometry data
        self.get_logger().info(f"Odometry - x: {self.odom_x}, y: {self.odom_y}, angle: {self.angle}, distance: {self.distance}, theta: {self.theta}")

# Main function to initialize the ROS2 node and start spinning
def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

