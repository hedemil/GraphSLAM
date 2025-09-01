import gtsam
import numpy as np
from data_loader import DataLoader, DeadReckoning, GPSData, LaserScan
from typing import List
from motion_model import MotionModel 
from landmark_manager import LandmarkManager
from tree_detector import TreeDetector, TreeDetection
from gtsam.symbol_shorthand import X, L
import matplotlib.pyplot as plt
import logging
from gtsam.utils.plot import plot_pose2, plot_point2, plot_trajectory, plot_covariance_ellipse_2d

class SLAM:
    def __init__(self):
        # Initialize classes
        self.data_loader = DataLoader()
        self.motion_model = MotionModel()
        self.landmark_manager = LandmarkManager()
        self.tree_detector = TreeDetector()

        # Initialize graph and initial estimate
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.result = None

        # Initialize noise models
        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.7]))
        self.gps_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([2, 2, 0.5]))
        
        # Define initial and minimum noise sigmas for bearing and range
        self.initial_sigma_bearing = 5.0     # radians (adjust as needed)
        self.min_sigma_bearing = 0.05        # radians

        self.initial_sigma_range = 10.0       # meters (adjust as needed)
        self.min_sigma_range = 0.1           # meters

        # Define high noise for inactive landmarks
        self.inactive_sigma_bearing = 5.0    # radians (example value)
        self.inactive_sigma_range = 10.0     # meters (example value)

        # Initialize graph with prior
        self.graph.add(gtsam.PriorFactorPose2(X(0), gtsam.Pose2(0, 0, 0), self.prior_noise))
        self.initial_estimate.insert(X(0), gtsam.Pose2(0, 0, 0))

        # Initialize logger
        self.logger = logging.getLogger('SLAM')

    def get_dynamic_noise(self, landmark: LandmarkManager):
        """
        Compute a dynamic noise model for a landmark based on the number of observations and active status.
        
        Args:
            landmark (Landmark): The landmark for which to compute the noise.
        
        Returns:
            gtsam.NoiseModel: The dynamic noise model.
        """
        if not landmark.active:
            # Set high noise for inactive landmarks
            sigma_bearing = self.inactive_sigma_bearing
            sigma_range = self.inactive_sigma_range
        else:
            # Prevent division by zero
            times_observed = max(1, landmark.times_observed)

            # Compute dynamic sigmas
            sigma_bearing = max(self.min_sigma_bearing, self.initial_sigma_bearing / np.sqrt(times_observed))
            sigma_range = max(self.min_sigma_range, self.initial_sigma_range / np.sqrt(times_observed))

        # Create a diagonal noise model with the computed sigmas
        dynamic_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([sigma_bearing, sigma_range]))

        return dynamic_noise

    def add_landmark_measurement(self, pose_idx: int, pose: gtsam.Pose2, laser: LaserScan):
        try:
            detections = self.tree_detector.detect_trees(laser.ranges, laser.angles)
            trees = self.landmark_manager.process_scan(detections, pose, laser.timestamp)

            for lm, tree in trees:
                # Convert tree.angle from clockwise +y to counterclockwise +x
                gtsam_bearing = (np.pi / 2) - tree.angle

                # relative_bearing = gtsam_bearing - pose.theta()

                # Retrieve dynamic noise based on the landmark's observation count and active status
                dynamic_noise = self.get_dynamic_noise(lm)

                landmark_pose = gtsam.Pose2(lm.x, lm.y, 0)

                relative_pose = pose.between(landmark_pose)
                
                relative_bearing = relative_pose.theta()


                # Log whether the landmark is active or inactive
                if not lm.active:
                    self.logger.debug(f"Landmark ID {lm.id} is inactive. Using increased noise.")

                # Add the measurement factor with dynamic noise
                self.graph.add(gtsam.BearingRangeFactor2D(
                    X(pose_idx), 
                    L(lm.id), 
                    gtsam.Rot2(relative_bearing), 
                    tree.distance, 
                    dynamic_noise
                ))

                # Only add initial estimate if it's a new landmark
                if not self.initial_estimate.exists(L(lm.id)):
                    # Initialize landmark position based on robot's pose and measurement
                    initial_x = pose.x() + tree.distance * np.cos(gtsam_bearing)
                    initial_y = pose.y() + tree.distance * np.sin(gtsam_bearing)
                    self.initial_estimate.insert_point2(L(lm.id), gtsam.Point2(initial_x, initial_y))

        except Exception as e:
            self.logger.error(f"Error adding landmark measurement: {e}")


    
    
    def odometry_update(self, pose_idx: int, odom_data: DeadReckoning, dt: float):

        if self.result and self.result.exists(X(pose_idx)):
            previous_pose = self.result.atPose2(X(pose_idx))
        else:
            previous_pose = self.initial_estimate.atPose2(X(pose_idx))
    
        new_pose = self.motion_model.predict(previous_pose, odom_data.speed, odom_data.steering, dt)
        relative_pose = previous_pose.between(new_pose)
        # Best result is 0.15, 0.15, 0.1 + 0.05*abs(odom_data.steering)
        steering_noise = 0.1 + 0.05*abs(odom_data.steering)
        speed_noise = 0.15 #+ 0.1*abs(odom_data.speed)
        odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([speed_noise, speed_noise, steering_noise]))
        self.graph.add(gtsam.BetweenFactorPose2(X(pose_idx), X(pose_idx + 1), relative_pose, odom_noise))
        if not self.initial_estimate.exists(X(pose_idx + 1)):
            self.initial_estimate.insert(X(pose_idx + 1), new_pose)


    def add_gps_measurement(self, pose_idx: int, gps_data: GPSData):
        try:
            gps_x = gps_data.longitude
            gps_y = gps_data.latitude


            if self.result and self.result.exists(X(pose_idx)):
                current_pose = self.result.atPose2(X(pose_idx))
            else:
                current_pose = self.initial_estimate.atPose2(X(pose_idx))

            gps_factor = gtsam.PriorFactorPose2(X(pose_idx), 
                                            gtsam.Pose2(gps_x, gps_y, current_pose.theta()), 
                                            self.gps_noise)
            self.graph.add(gps_factor)
        except Exception as e:
            self.logger.error(f"Error adding GPS measurement: {e}")
        
    def optimize(self):
        # Optimize
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimate)
        self.result = optimizer.optimize()
        return self.result
    
    
    def plot_results(self, gps_data: List[GPSData]):

        optimized_poses = []
        optimized_landmarks = []
        for key in self.result.keys():
            sym = gtsam.Symbol(key)         # Convert raw Key -> Symbol
            if sym.chr() == ord('x'):
                pose = self.result.atPose2(key)
                optimized_poses.append(pose)
            elif sym.chr() == ord('l'):
                landmark = self.result.atPoint2(key)
                optimized_landmarks.append(landmark)

        # initial_poses = []
        # initial_landmarks = []
        # for key in self.initial_estimate.keys():
        #     sym = gtsam.Symbol(key)         # Convert raw Key -> Symbol
        #     if sym.chr() == ord('x'):
        #         pose = self.initial_estimate.atPose2(key)
        #         initial_poses.append(pose)
        #     elif sym.chr() == ord('l'):
        #         landmark = self.initial_estimate.atPoint2(key)
        #         initial_landmarks.append(landmark)
        
        # initial_pose_x = [pose.x() for pose in initial_poses]
        # initial_pose_y = [pose.y() for pose in initial_poses]
        # plt.plot(initial_pose_x, initial_pose_y, 'r-', label='Initial')

        # initial_landmark_x = [landmark[0] for landmark in initial_landmarks]
        # initial_landmark_y = [landmark[1] for landmark in initial_landmarks]

        # if initial_landmark_x and initial_landmark_y:
        #     plt.scatter(initial_landmark_x, initial_landmark_y, c='red', s=10, label='Landmarks')
                
        print(f'Number of Landmarks: {len(optimized_landmarks)}')
        # Plot the optimized poses
        optimized_x = [pose.x() for pose in optimized_poses]
        optimized_y = [pose.y() for pose in optimized_poses]

        plt.plot(optimized_x, optimized_y, 'b-', label='Optimized')

        if optimized_x and optimized_y:
                # Mark start and end
                plt.scatter([optimized_x[0]], [optimized_y[0]], c='green', s=100, label='Start')
                plt.scatter([optimized_x[-1]], [optimized_y[-1]], c='red', s=100, label='End')

        landmarks_x = [landmark[0] for landmark in optimized_landmarks]
        landmarks_y = [landmark[1] for landmark in optimized_landmarks]
        landmarks_active = [self.landmark_manager.get_landmark(lm.id).active for lm in self.landmark_manager.landmarks.values()]

        if landmarks_x and landmarks_y:
            # Differentiate active and inactive landmarks
            active_x = [x for x, active in zip(landmarks_x, landmarks_active) if active]
            active_y = [y for y, active in zip(landmarks_y, landmarks_active) if active]
            inactive_x = [x for x, active in zip(landmarks_x, landmarks_active) if not active]
            inactive_y = [y for y, active in zip(landmarks_y, landmarks_active) if not active]

            if active_x and active_y:
                plt.scatter(active_x, active_y, c='orange', s=10, label='Active Landmarks')
            # if inactive_x and inactive_y:
            #     plt.scatter(inactive_x, inactive_y, c='red', marker='x', s=50, label='Inactive Landmarks')


        gps_x = [gps.longitude for gps in gps_data]
        gps_y = [gps.latitude for gps in gps_data]
        
        if gps_x and gps_y:
            plt.plot(gps_x, gps_y, 'g--', label='GPS')
        
        # # Add covariance ellipses for landmarks
        # marginals = gtsam.Marginals(self.graph, self.result)
        # for i, landmark in enumerate(optimized_landmarks):
        #     try:
        #         covariance = marginals.marginalCovariance(L(i))
        #         plot_covariance_ellipse_2d(covariance, landmark, 0.95)
        #     except RuntimeError:
        #         continue
                
        # # Plot factor graph connections
        # for i in range(self.graph.size()):
        #     factor = self.graph.at(i)
        #     if isinstance(factor, gtsam.BearingRangeFactor2D):
        #         pose_key = factor.keys()[0]
        #         lm_key = factor.keys()[1]
        #         pose = self.result.atPose2(pose_key)
        #         landmark = self.result.atPoint2(lm_key)
        #         plt.plot([pose.x(), landmark[0]], [pose.y(), landmark[1]], 'k:', alpha=0.1)
        

        plt.title("SLAM Results with Loop Closure (Tree Landmarks)", fontsize=14)
        plt.xlabel("X (m)", fontsize=12)
        plt.ylabel("Y (m)", fontsize=12)
        plt.grid(True)
        plt.axis('equal')
        plt.legend(fontsize=10)

        plt.tight_layout()
        plt.show()

        self.plot_error_metrics(gps_data, optimized_poses)
    
    # Add to slam_main.py:
    def plot_error_metrics(self, gps_data, optimized_poses):
        errors = []
        times = []
        for i, pose in enumerate(optimized_poses):
            if i < len(gps_data):
                error = np.sqrt((pose.x() - gps_data[i].longitude)**2 + 
                            (pose.y() - gps_data[i].latitude)**2)
                errors.append(error)
                times.append(gps_data[i].timestamp)
        
        plt.figure(figsize=(10,6))
        plt.plot(times, errors)
        plt.xlabel('Time (s)')
        plt.ylabel('Position Error (m)')
        plt.title('SLAM Position Error vs GPS Ground Truth')
        plt.grid(True)
        plt.show()
    
    import matplotlib.pyplot as plt
    from gtsam.utils.plot import plot_covariance_ellipse_2d

    def plot_landmark_covariances(self, confidence_level=0.95):
        """
        Plot landmark covariances as ellipses.

        Args:
            slam: SLAM object containing result and graph.
            confidence_level: Confidence level for the covariance ellipses (default 95%).
        """
        if self.result is None:
            print("SLAM optimization result is not available.")
            return
        
        # Compute chi-square value for confidence level
        from scipy.stats import chi2
        chi2_val = chi2.ppf(confidence_level, 2)  # For 2 degrees of freedom
        
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.title(f"Landmark Covariances (Confidence: {int(confidence_level * 100)}%)")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis('equal')
        plt.grid(True)

        # Plot each landmark
        for key in self.result.keys():
            sym = gtsam.Symbol(key)
            if sym.chr() == ord('l'):  # Check if it's a landmark
                landmark = self.result.atPoint2(key)
                try:
                    marginals = gtsam.Marginals(self.graph, self.result)
                    covariance = marginals.marginalCovariance(key)
                    plot_covariance_ellipse_2d(covariance, landmark, chi2_val, ax=ax)
                    plt.plot(landmark[0], landmark[1], 'ro')  # Plot landmark mean
                except RuntimeError as e:
                    print(f"Error computing covariance for landmark {sym.index()}: {e}")
                    continue
        
        plt.legend(['Landmark Mean', 'Covariance Ellipse'], loc='best')
        plt.tight_layout()
        plt.show()



def main():
    slam = SLAM()
    OPTIMIZE_INTERVAL = 10000

    # Load data
    laser_data = slam.data_loader.load_laser_data('LASER.txt')
    dr_data = slam.data_loader.load_dead_reckoning('DRS.txt')
    GT_data = slam.data_loader.load_gps_data_mat('aa3_gpsx.mat')
    gps_data = slam.data_loader.load_gps_data('GPS.txt')
    sensor_events = slam.data_loader.load_sensor_manager('Sensors_manager.txt')

    # slam.first_iteration(slam.initial_estimate.atPose2(X(0)), dr_data)

    current_pose_idx = 0
    optimized_idx = 1
    # sensor_events = sensor_events[:20000]

    trees_total = []
    accepted_trees = []

    for event in sensor_events:
        if event.sensor_id == 1:  # GPS
            slam.add_gps_measurement(current_pose_idx, gps_data[event.index - 1])
            # continue
        elif event.sensor_id == 2:  # Dead Reckoning
            if event.index == 1:
                slam.odometry_update(current_pose_idx, dr_data[event.index - 1], 0.025)
                current_pose_idx += 1
                continue
            dt = dr_data[event.index -1].timestamp - dr_data[event.index - 2].timestamp
            slam.odometry_update(current_pose_idx, dr_data[event.index - 1], dt)
            if current_pose_idx % OPTIMIZE_INTERVAL == 0:
                slam.optimize()
                print(f'Optimized nr: {optimized_idx}')
                trees = slam.landmark_manager.landmarks
                print(f'Number of Landmarks: {len(trees)}')
                print(f'Number of Active Landmarks: {sum([t.active for t in trees.values()])}')
                optimized_idx += 1
            current_pose_idx += 1
        elif event.sensor_id == 3:  # Laser
            if slam.result and slam.result.exists(X(current_pose_idx)):
                current_pose = slam.result.atPose2(X(current_pose_idx))
            else:
                current_pose = slam.initial_estimate.atPose2(X(current_pose_idx))
            slam.add_landmark_measurement(current_pose_idx, current_pose, laser_data[event.index - 1])
            trees = slam.landmark_manager.landmarks
            trees_total.append(len(trees))
            accepted_trees.append(sum([t.active for t in trees.values()]))
            # continue
        
            
            
    trees = slam.landmark_manager.landmarks
    print(f'Number of Landmarks: {len(trees)}')
    print(f'Number of Active Landmarks: {sum([t.active for t in trees.values()])}')
    # Print maximum pos of landmarks
        
    slam.optimize()

    

    slam.plot_results(GT_data)
    #slam.plot_landmark_covariances()

    # Plotting the Number of Trees and Active Trees Over Time
    plot_landmarks_over_time(trees_total, accepted_trees)

def plot_landmarks_over_time(trees_total: List[int], accepted_trees: List[int]):
    """
    Plots the total number of trees and active trees over time.

    Parameters:
    - trees_total (List[int]): List containing the total number of trees after each laser measurement.
    - accepted_trees (List[int]): List containing the number of active trees after each laser measurement.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(trees_total, label='Total Trees', color='blue', linewidth=2)
    plt.plot(accepted_trees, label='Active Trees', color='green', linewidth=2)
    plt.xlabel('Laser Measurement Index', fontsize=14)
    plt.ylabel('Number of Trees', fontsize=14)
    plt.title('Number of Trees and Active Trees Over Time', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
if __name__ == "__main__":
    main()
