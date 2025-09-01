import gtsam
import numpy as np

class MotionModel:
    def __init__(self):
        self.vehicle_param_H = 0.76  # Distance between encoder back left wheel and center back axis
        self.vehicle_param_L = 2.83  # Distance between back and front wheels
        self.vehicle_param_a = 3.78  # Distance from rear wheel to sensor
        self.vehicle_param_b = 0.5   # Distance from rear center to sensor

        self.max_steering = np.pi/6  # Maximum steering angle (30 degrees)
    
    def normalize_angle(self, angle: float) -> float:
        """Your current angle normalization is too restrictive"""
        # Should use proper angle normalization:
        while angle > np.pi:
            angle -= 2*np.pi
        while angle < -np.pi:
            angle += 2*np.pi
        return angle

    # def predict(self, initial_pose: gtsam.Pose2, speed: float, steering: float, dt: float) -> gtsam.Pose2:
    #     """
    #     Predict next pose using Ackerman (bicycle) model in global frame.
    #     """
        
    #     theta = initial_pose.theta()
    #     # theta = self.normalize_angle(theta)
    #     speed = self.transform_speed(speed, steering)
    #     steering = np.clip(steering, -self.max_steering, self.max_steering)

    #     if abs(steering) < 1e-4:
    #         dx = speed * np.cos(theta) * dt
    #         dy = speed * np.sin(theta) * dt
    #         dtheta = 0
    #     else:
    #         dx = speed * np.cos(theta) * dt
    #         dy = speed * np.sin(theta) * dt
    #         dtheta = (speed / self.vehicle_param_L) * np.tan(steering) * dt  # Ensure (H/L) is correct

        
    #     # Update global pose directly
    #     x_new = initial_pose.x() + dx
    #     y_new = initial_pose.y() + dy
    #     theta_new = initial_pose.theta() + dtheta

    #     # Normalize theta to be within [-pi, pi]
    #     # theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi

    #     # Create new Pose2
    #     new_pose = gtsam.Pose2(x_new, y_new, theta_new)
        
    #     return new_pose
    
    def predict(self, initial_pose: gtsam.Pose2, speed: float, steering: float, dt: float):
        theta = initial_pose.theta()
        speed = self.transform_speed(speed, steering)
        steering = np.clip(steering, -self.max_steering, self.max_steering)

        if abs(steering) < 1e-4:
            dx = speed * np.cos(theta) * dt
            dy = speed * np.sin(theta) * dt
            dtheta = 0
        else:
            # Use proper bicycle model equations for curved motion
            turning_radius = self.vehicle_param_L / np.tan(steering)
            dtheta = (speed / turning_radius) * dt
            dx = turning_radius * (np.sin(theta + dtheta) - np.sin(theta))
            dy = turning_radius * (np.cos(theta) - np.cos(theta + dtheta))
        
        return gtsam.Pose2(
            initial_pose.x() + dx,
            initial_pose.y() + dy,
            initial_pose.theta() + dtheta
        )
    
    def transform_speed(self, speed: float, steering: float) -> float:
        """Transform speed and steering angle into a new speed"""
        return speed / (1 - (self.vehicle_param_H/self.vehicle_param_L) * np.tan(steering))

