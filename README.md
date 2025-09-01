# Graph-SLAM for Natural Landmark Mapping

A robust implementation of Graph-SLAM (Simultaneous Localization and Mapping) using GTSAM factor graphs for mobile robot navigation in natural environments with tree landmarks.

## Project Overview

This project implements a complete SLAM pipeline that processes real-world sensor data to simultaneously build maps and localize a mobile robot. The system is specifically designed to handle natural landmarks (trees) detected from LASER scans, making it suitable for outdoor navigation scenarios.

**Key Features:**
- Factor graph-based SLAM using GTSAM optimization
- Real-time tree detection from LASER scans using DBSCAN clustering
- Adaptive landmark management with quality scoring
- Dynamic noise modeling for robust performance
- Comprehensive evaluation on the Victoria Park dataset

## Architecture

### Core Components

- **`slam_main.py`** - Main SLAM orchestrator with GTSAM factor graph optimization
- **`data_loader.py`** - Multi-sensor data parsing and synchronization (GPS, odometry, LASER)
- **`motion_model.py`** - Bicycle/Ackerman motion model for vehicle kinematics  
- **`tree_detector.py`** - LASER-based tree detection using DBSCAN clustering and validation
- **`landmark_manager.py`** - Landmark lifecycle management with Kalman-style updates

### Technical Stack

- **Optimization**: GTSAM (Georgia Tech Smoothing and Mapping) library
- **Clustering**: DBSCAN for tree detection from point clouds
- **Motion Model**: Ackerman steering geometry for wheeled vehicles
- **State Estimation**: Kalman-like filtering for landmark position refinement

## Getting Started

### Prerequisites

```bash
# Required Python packages
pip install gtsam numpy matplotlib scipy scikit-learn
```

### Required Data Files

Place the following files in the project directory:
- `LASER.txt` - Raw LASER scan data (362 values per scan)
- `DRS.txt` - Dead reckoning measurements (timestamp, speed, steering)  
- `GPS.txt` - GPS coordinates (timestamp, latitude, longitude)
- `aa3_gpsx.mat` - Ground truth GPS data (MATLAB format)
- `Sensors_manager.txt` - Sensor event timing and synchronization

### Running the System

```bash
# Run complete SLAM pipeline
python3 slam_main.py

# Test data loading only
python3 data_loader.py
```

## Algorithm Details

### Graph-SLAM Framework

The system constructs a factor graph where:
- **Nodes** represent robot poses and landmark positions
- **Factors** encode constraints from sensor measurements:
  - Prior factors for GPS measurements
  - Between factors for odometry constraints  
  - Bearing-range factors for landmark observations

### Tree Detection Pipeline

1. **Preprocessing**: Filter LASER points by range (3-30m) and convert to Cartesian coordinates
2. **Clustering**: Apply DBSCAN to group points into potential tree candidates
3. **Validation**: Assess cluster circularity and range consistency
4. **Feature Extraction**: Estimate tree position, diameter, and bearing

### Landmark Management

- **Data Association**: Mahalanobis distance-based matching with position and diameter constraints
- **State Updates**: Kalman-like filtering with adaptive learning rates
- **Quality Assessment**: Multi-factor scoring based on observation count and position stability
- **Lifecycle Management**: Dynamic activation/deactivation based on reliability metrics

## Experimental Results

### Performance Evaluation

The system was evaluated on the Victoria Park dataset with the following key findings:

- **Odometry-only**: Significant trajectory drift over time
- **SLAM with landmarks**: ~70% reduction in position error compared to odometry-only
- **Noise sensitivity**: Careful parameter tuning critical for optimal performance
- **Computational efficiency**: Periodic optimization every 10,000 poses for real-time capability

### Key Metrics

- **Landmark Detection**: Successfully identifies 200+ tree landmarks
- **Position Accuracy**: Mean error < 5m with optimized noise models
- **Robustness**: Handles sensor degradation through adaptive noise modeling

## Configuration

### Key Parameters

```python
# SLAM Optimization
OPTIMIZE_INTERVAL = 10000  # Poses between optimizations

# Noise Models
GPS_NOISE = [2.0, 2.0, 0.5]  # [x, y, theta] standard deviations
ODOM_NOISE = [0.15, 0.15, 0.1]  # Base odometry noise

# Tree Detection
RANGE_LIMITS = [3.0, 30.0]  # Min/max detection range (meters)
DIAMETER_LIMITS = [0.5, 2.0]  # Valid tree diameter range
DBSCAN_EPS = 0.5  # Clustering distance threshold
MIN_CLUSTER_SIZE = 4  # Minimum points per cluster

# Landmark Management  
POSITION_THRESHOLD = 5.0  # Association distance threshold
MAHALANOBIS_THRESHOLD = 5.99  # 95% confidence interval
```

## Future Improvements

- **Loop Closure Detection**: Implement robust place recognition for large-scale mapping
- **Online Optimization**: Integrate iSAM2 for real-time incremental optimization
- **Multi-Session SLAM**: Support for map persistence and reuse across sessions
- **Semantic Understanding**: Extend beyond geometric features to semantic landmarks

## Technical Report

For detailed methodology, experimental analysis, and theoretical background, see [`Final_project.pdf`](Final_project.pdf).

## Contributing

This project was developed as part of advanced coursework in Applied Estimation at KTH Royal Institute of Technology. Contributions and improvements are welcome!

## License

This project is available under the MIT License - see the LICENSE file for details.

---

*Developed as part of Applied Estimation coursework, demonstrating practical implementation of state-of-the-art SLAM algorithms for real-world robotic navigation.*