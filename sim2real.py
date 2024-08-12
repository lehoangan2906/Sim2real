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
from std_msgs.msg import Bool
import math

# Define the actor network
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
        
        # Parameters for the robot's goal and control behavior
        self.declare_parameter('goal_x', -1.0)
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('GOAL_REACHED_DIST', 0.3)
        self.declare_parameter('COLLISION_DIST_THRESHOLD', 0.2)  # Add the collision distance threshold

        self.goal_x = self.get_parameter('goal_x').get_parameter_value().double_value
        self.goal_y = self.get_parameter('goal_y').get_parameter_value().double_value
        self.GOAL_REACHED_DIST = self.get_parameter('GOAL_REACHED_DIST').get_parameter_value().double_value
        self.COLLISION_DIST_THRESHOLD = self.get_parameter('COLLISION_DIST_THRESHOLD').get_parameter_value().double_value

        # Initial values
        self.previous_action = [0.0, 0.0]
        self.distance = 0.0
        self.theta = 0.0
        self.odom_x = 0.0
        self.odom_y = 0.0

        # Load the neural network model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor = Actor(24, 2).to(self.device)
        self.actor.load_state_dict(torch.load('TD3_velodyne_actor.pth', map_location=self.device))
        self.actor.eval()
        
        # ROS2 subscriptions and publishers
        self.laser_subscriber = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.odom_subscriber = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_publisher = self.create_publisher(Marker, '/laser_markers', 10)
        self.emergency_stop_sub = self.create_subscription(Bool, '/emergency_stop', self.emergency_stop_callback, 10)

        # EMA parameters
        self.ema_alpha = 0.2  # Smoothing factor for EMA
        self.previous_ema = [12.0] * 20  # Initial EMA values (fallback to max range)
        self.emergency_stop = False  # Track emergency stop status

    def laser_callback(self, msg):
        # Process laser scan data using EMA and combined strategy
        min_laser = self.process_lidar_data(msg.ranges)
        
        # Check for collision risk
        if any(dist < self.COLLISION_DIST_THRESHOLD for dist in min_laser):
            self.get_logger().warn("Obstacle detected within collision distance threshold!")
            self.safe_stop()  # Stop the robot if an obstacle is too close
            return
        
        # Update state vector
        state = min_laser + [self.distance, self.theta] + self.previous_action
        
        # State validation and error handling
        if len(state) != 24:
            self.get_logger().error(f"Unexpected state vector length: {len(state)}")
            self.safe_stop()  # Stop the robot if state vector is incorrect
            return
        
        # Convert state to tensor and predict action
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_tensor).squeeze().cpu().numpy()

        # Action clipping to ensure safe operation
        linear_vel = np.clip(action[0], 0.0, 0.4)  # Example range for linear velocity
        angular_vel = np.clip(action[1], -0.6, 0.6)  # Example range for angular velocity
        
        if not self.emergency_stop:
            twist = Twist()
            twist.linear.x = float(linear_vel)
            twist.angular.z = float(angular_vel)
            self.cmd_vel_publisher.publish(twist)

        # Update previous action and publish laser markers for visualization
        self.previous_action = [float(linear_vel), float(angular_vel)]
        self.publish_laser_markers(min_laser)

    def process_lidar_data(self, ranges):
        """Process LiDAR data by combining Min Value and EMA to improve obstacle detection."""
        num_gaps = 20  # Number of gaps we want to split the FOV into
        gap_size = len(ranges) // num_gaps  # Calculate the size of each gap

        processed_data = []
        for i in range(num_gaps):
            gap_start = i * gap_size
            gap_end = (i + 1) * gap_size
            gap_data = ranges[gap_start:gap_end]

            if len(gap_data) > 0:
                current_min = min(gap_data)
            else:
                current_min = 12.0  # Fallback value if gap_data is empty

            # Calculate the EMA for this gap
            ema_value = self.ema_alpha * current_min + (1 - self.ema_alpha) * self.previous_ema[i]
            
            # Combine the Min Value and EMA Value
            combined_value = min(current_min, ema_value)  # Use the minimum of both as the combined value

            processed_data.append(combined_value)
            self.previous_ema[i] = ema_value  # Update the previous EMA for the next iteration

        return processed_data

    def safe_stop(self):
        """Stop the robot by publishing zero velocities."""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_publisher.publish(twist)

    def emergency_stop_callback(self, msg):
        """Handle emergency stop signal."""
        self.emergency_stop = msg.data  # msg.data will be True or False
        if self.emergency_stop:
            self.safe_stop()

    def publish_laser_markers(self, min_laser):
        """Publish laser markers for visualization in RViz."""
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'laser_scan'
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        for i, distance in enumerate(min_laser):
            angle = (i * np.pi / 10)  
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            marker.points.append(Point(x=x, y=y, z=0.0))

        self.marker_publisher.publish(marker)

    def odom_callback(self, msg):
        """Update odometry data and calculate distance to goal."""
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

    def update_model(self, new_model_path):
        """Update the model with a new set of parameters."""
        self.actor.load_state_dict(torch.load(new_model_path, map_location=self.device))
        self.actor.eval()

def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
