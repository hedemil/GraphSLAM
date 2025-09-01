import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from tree_detector import TreeDetection
import gtsam
from scipy.spatial import KDTree
import logging

@dataclass
class Landmark:
    id: int
    x: float
    y: float
    diameter: float
    first_observed: int
    times_observed: int
    covariance: np.ndarray
    last_observed: int
    observation_quality: float
    # History for stationary validation
    positions: List[Tuple[float, float]]  # Recent position measurements
    diameters: List[float]                # Recent diameter measurements
    position_variance: float = 0.0        # Variance in position measurements
    active: bool = field(default=True)    # Flag to indicate if the landmark is active

 

class LandmarkManager:
    def __init__(self):
        self.landmarks: Dict[int, Landmark] = {}
        self.next_id = 0
        self.landmark_positions = []
        self.kdtree = None

        # Core thresholds (adjusted for stationary trees)
        self.position_threshold = 5.0     # Increased from 2.0m
        self.diameter_threshold = 0.5     # Increased to be more tolerant

        # Quality parameters
        self.min_observations = 3         
        self.max_age = 1000
        self.quality_threshold = 0.5
        self.max_history = 5              # Reduced history since we only need it for validation
        self.max_position_variance = 1    # Maximum allowed variance for stable trees

        # Add Mahalanobis distance parameters
        self.mahalanobis_threshold = 5.99 # Chi-square 5.99 - 95% confidence for 2 DOF 7.81 for 90%

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("LandmarkManager")

        self.landmark_list: List[Landmark] = []
        self.removed_landmarks = set()  # Optionally keep if needed

    def build_kdtree(self):
        if self.landmarks:
            self.landmark_list = list(self.landmarks.values())
            self.landmark_positions = [(lm.x, lm.y) for lm in self.landmark_list]
            self.kdtree = KDTree(self.landmark_positions)
        else:
            self.kdtree = None
            self.landmark_list = []

    def find_closest_landmark(self, x: float, y: float, diameter: float) -> Tuple[Optional[Landmark], float]:
        if not self.landmarks or self.kdtree is None:
            return None, float('inf')

        indices = self.kdtree.query_ball_point([x, y], self.position_threshold)
        matches = []
        pos = np.array([x, y])

        for idx in indices:
            lm = self.landmark_list[idx]
            diff = pos - np.array([lm.x, lm.y])
            try:
                inv_cov = np.linalg.inv(lm.covariance)
                mahalanobis_dist = np.sqrt(diff.T @ inv_cov @ diff)
            except np.linalg.LinAlgError:
                self.logger.warning(f"Covariance matrix for landmark ID {lm.id} is singular. Regularizing.")
                lm.covariance += np.eye(2) * 1e-6
                try:
                    inv_cov = np.linalg.inv(lm.covariance)
                    mahalanobis_dist = np.sqrt(diff.T @ inv_cov @ diff)
                except np.linalg.LinAlgError:
                    self.logger.error(f"Failed to invert regularized covariance for landmark ID {lm.id}. Skipping.")
                    continue

            euclidean_dist = np.linalg.norm(diff)
            diameter_diff = abs(lm.diameter - diameter)

            if (mahalanobis_dist < self.mahalanobis_threshold and 
                euclidean_dist < self.position_threshold and 
                diameter_diff < self.diameter_threshold):
                
                # Enhanced scoring system
                position_score = 1.0 - (euclidean_dist / self.position_threshold)
                mahalanobis_score = 1.0 - (mahalanobis_dist / self.mahalanobis_threshold)
                diameter_score = 1.0 - (diameter_diff / self.diameter_threshold)
                stability_score = 1.0 - min(1.0, lm.position_variance / self.max_position_variance)
                observation_score = min(1.0, lm.times_observed / self.min_observations)

                total_score = (0.4 * position_score +
                              0.4 * mahalanobis_score +
                              0.1 * diameter_score +
                              0.05 * stability_score +
                              0.05 * observation_score)


                matches.append((lm, euclidean_dist, total_score))

        if not matches:
            return None, float('inf')

        best_match = max(matches, key=lambda x: x[2])
        return best_match[0], best_match[1]

    def update_landmark(self, landmark: Landmark, x: float, y: float,
                       diameter: float, timestamp: int) -> Landmark:
        # Update position history
        landmark.positions.append((x, y))
        landmark.diameters.append(diameter)

        # Maintain fixed history size
        if len(landmark.positions) > self.max_history:
            landmark.positions.pop(0)
            landmark.diameters.pop(0)

        # Calculate position variance for stability assessment
        if len(landmark.positions) >= 2:
            positions = np.array(landmark.positions)
            landmark.position_variance = np.var(positions[:, 0]) + np.var(positions[:, 1])

        # Adaptive update rate based on observation history
        if landmark.times_observed < self.min_observations:
            alpha = 0.3
        else:
            # More conservative updates for established landmarks
            alpha = 1.0 / (landmark.times_observed + 1)
            alpha = max(0.05, min(0.3, alpha))

        # Kalman-like covariance update
        # Assume observation_noise is the measurement covariance
        observation_noise = np.eye(2) * (0.1 + 0.02 * np.sqrt(x*x + y*y))
        try:
            kalman_gain = landmark.covariance @ np.linalg.inv(landmark.covariance + observation_noise)
        except np.linalg.LinAlgError:
            kalman_gain = np.eye(2)  # Fallback to no gain

        # Calculate the measurement residual
        residual = np.array([x, y]) - np.array([landmark.x, landmark.y])

        # Update the landmark's position using the Kalman gain
        landmark.x += kalman_gain[0, 0] * residual[0] + kalman_gain[0, 1] * residual[1]
        landmark.y += kalman_gain[1, 0] * residual[0] + kalman_gain[1, 1] * residual[1]

        landmark.diameter += alpha * (diameter - landmark.diameter)
        landmark.covariance = (np.eye(2) - kalman_gain) @ landmark.covariance

        # Update metadata
        landmark.times_observed += 1
        landmark.last_observed = timestamp

        # Update quality score with emphasis on position stability
        stability_score = 1.0 / (1.0 + landmark.position_variance)
        diameter_consistency = 1.0 / (1.0 + np.var(landmark.diameters))

        landmark.observation_quality = (0.7 * stability_score +
                                       0.3 * diameter_consistency)
        
        # Determine if the landmark should be reactivated
        if not landmark.active and landmark.observation_quality >= self.quality_threshold and \
        landmark.position_variance < self.max_position_variance:
            landmark.active = True
            self.logger.info(f"Reactivated landmark ID {landmark.id} based on improved quality.")
    

        return landmark

    def add_landmark(self, x: float, y: float, diameter: float, timestamp: int) -> Landmark:
        initial_covariance = np.eye(2) * (0.1 + 0.02 * np.sqrt(x**2 + y**2))

        lm = Landmark(
            id=self.next_id,  # Start from 0
            x=x,
            y=y,
            diameter=diameter,
            first_observed=timestamp,
            last_observed=timestamp,
            times_observed=1,
            covariance=initial_covariance,
            observation_quality=1.0,
            positions=[(x, y)],
            diameters=[diameter],
            position_variance=0.0,
        )
        self.landmarks[self.next_id] = lm
        self.landmark_list.append(lm)  # Keep the list in sync
        self.logger.debug(f"Assigning ID {self.next_id} to new landmark at ({x}, {y})")
        self.next_id += 1

        # Rebuild KD-Tree after adding new landmark
        self.build_kdtree()

        # self.logger.info(f"Added new landmark: ID={lm.id}, Position=({lm.x}, {lm.y}), Diameter={lm.diameter}")
        return lm

    def process_scan(self, detections: List[TreeDetection],
                vehicle_pose: gtsam.Pose2, timestamp: float) -> List[Tuple[Landmark, TreeDetection]]:
        observed_landmarks = []
        current_landmarks = set()
        
        for d in detections:
            gx, gy = self.convert_to_global(d, vehicle_pose)
            lm, dist = self.find_closest_landmark(gx, gy, d.diameter)
        
            if lm is not None: # and lm.position_variance < self.max_position_variance:
                if lm.id not in current_landmarks:
                    lm = self.update_landmark(lm, gx, gy, d.diameter, timestamp)
                    observed_landmarks.append((lm, d))
                    current_landmarks.add(lm.id)
                    
                else:
                    self.logger.debug(f"Landmark {lm.id} already processed in this scan.")
            else:
                new_lm = self.add_landmark(gx, gy, d.diameter, timestamp)
                observed_landmarks.append((new_lm, d))
                current_landmarks.add(new_lm.id)
        
        self.cleanup_landmarks(timestamp)
        self.build_kdtree()
        
        return observed_landmarks
    

    def cleanup_landmarks(self, timestamp: int):
        to_deactivate = []
        for lm in self.landmarks.values():
            if (lm.observation_quality < self.quality_threshold or
                lm.position_variance >= self.max_position_variance):
                to_deactivate.append(lm.id)
        
        for lm_id in to_deactivate:
            self.landmarks[lm_id].active = False
            self.removed_landmarks.add(lm_id)  # Optionally keep track if needed
            # self.logger.info(f"Deactivated landmark ID {lm_id} due to low quality or high variance.")

    # Angle from x-axis counter-clockwise
    def convert_to_global(self, detection: TreeDetection, vehicle_pose: gtsam.Pose2) -> Tuple[float, float]:
        
        global_angle = vehicle_pose.theta() + (np.pi/2 - detection.angle)
    
        dx = detection.distance * np.cos(global_angle)
        dy = detection.distance * np.sin(global_angle)

        return vehicle_pose.x() + dx, vehicle_pose.y() + dy
    

    def convert_to_global_homogeneous(self, detection, vehicle_pose: gtsam.Pose2) -> Tuple[float, float]:
        # vehicle_pose: (x_robot, y_robot, theta_robot) in global
        # detection: local bearing, range, etc.
        
        # Build transform T from robot frame -> global
        
        T = np.array([
            [ np.cos(vehicle_pose.theta()), -np.sin(vehicle_pose.theta()), vehicle_pose.x()],
            [ np.sin(vehicle_pose.theta()),  np.cos(vehicle_pose.theta()), vehicle_pose.y()],
            [ 0,                0,               1  ]
        ])

        # Local point (assuming detection.angle is measured CW from +Y in robot frame)


        dx = detection.distance * np.sin(detection.angle)
        dy = detection.distance * np.cos(detection.angle)
        local_point = np.array([dx, dy, 1.0])
        
        global_point = T @ local_point
        return (global_point[0], global_point[1])


    def get_landmark(self, landmark_id: int) -> Optional[Landmark]:
        return self.landmarks.get(landmark_id, None)
