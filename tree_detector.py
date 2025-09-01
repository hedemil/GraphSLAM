import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist

@dataclass
class TreeDetection:
    """Class for storing detected tree information"""
    distance: float    # Distance to tree center
    angle: float       # Angle to tree center (radians)
    diameter: float    # Diameter of tree

class TreeDetector:
    def __init__(self):
        self.M11 = 30.0         # Max range
        self.M10 = 3.0          # Min range
        self.M2 = 1.5           # Range difference threshold
        self.M3 = 2.0           # Reduced max tree diameter from 2.0
        self.M5 = 0.5           # Min tree diameter
        self.min_points = 5     # Increased from 4
        self.max_points = 15    # Reduced from 15
        self.max_range_variance = 0.5  # Maximum allowed range variance
        self.min_circularity = 0.8     # Minimum circularity score

        self.dbscan_eps = 0.5  # Maximum distance between points to be in the same cluster
        self.dbscan_min_samples = 4  # Minimum number of points to form a cluster

    def validate_cluster(self, cluster_x: np.ndarray, cluster_y: np.ndarray, 
                        cluster_ranges: np.ndarray) -> bool:
        # Fit circle to points
        x_mean = np.mean(cluster_x)
        y_mean = np.mean(cluster_y)
        r = np.mean(np.sqrt((cluster_x - x_mean)**2 + (cluster_y - y_mean)**2))
        
        # Calculate circularity score
        distances = np.abs(np.sqrt((cluster_x - x_mean)**2 + 
                                 (cluster_y - y_mean)**2) - r)
        circularity = 1.0 - np.std(distances) / r
        
        # Calculate range consistency
        range_variance = np.var(cluster_ranges) / np.mean(cluster_ranges)**2
        
        return (circularity > self.min_circularity and 
                range_variance < self.max_range_variance)


    def detect_trees(self, ranges: np.ndarray, angles: np.ndarray) -> List[TreeDetection]:
        # Convert to Cartesian coordinates
        x = ranges * np.sin(angles)
        y = ranges * np.cos(angles)

        # Filter valid ranges
        valid_idx = (ranges < self.M11) & (ranges > self.M10)
        X = x[valid_idx]
        Y = y[valid_idx]

        if len(X) == 0:
            return []

        points = np.vstack((X, Y)).T

        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit(points)
        labels = clustering.labels_

        trees = []
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label

        for label in unique_labels:
            cluster_points = points[labels == label]
            cluster_ranges = ranges[valid_idx][labels == label]
            cluster_angles = angles[valid_idx][labels == label]

            if len(cluster_points) < self.min_points:
                continue

            # Estimate diameter
            angle_span = cluster_angles.max() - cluster_angles.min()
            mean_range = cluster_ranges.mean()
            diameter_test = 2.0 * mean_range * np.sin(angle_span / 2.0)
            if len(cluster_points) > 1:
                diameter = np.max(pdist(cluster_points))
            else:
                diameter = 0.0

            if (self.M5 < diameter < self.M3 and
                self.validate_cluster(cluster_points[:,0], cluster_points[:,1], cluster_ranges)):
                # Compute average angle
                # avg_angle = np.arctan2(cluster_points[:,1].mean(), cluster_points[:,0].mean())
                avg_angle = cluster_angles.mean()
                trees.append(TreeDetection(
                    distance=mean_range,
                    angle=avg_angle,
                    diameter=diameter
                ))

        # Remove duplicates if necessary
        return self.remove_duplicates(trees)

    def remove_duplicates(self, trees: List[TreeDetection]) -> List[TreeDetection]:
        if not trees:
            return []

        # Sort trees based on distance and angle
        trees_sorted = sorted(trees, key=lambda t: (t.distance, t.angle))
        filtered_trees = []

        for tree in trees_sorted:
            duplicate = False
            for accepted in filtered_trees:
                dx = tree.distance * np.cos(tree.angle) - accepted.distance * np.cos(accepted.angle)
                dy = tree.distance * np.sin(tree.angle) - accepted.distance * np.sin(accepted.angle)
                dist = np.sqrt(dx*dx + dy*dy)

                if dist < max(tree.diameter, accepted.diameter):
                    duplicate = True
                    break

            if not duplicate:
                filtered_trees.append(tree)

        return filtered_trees
