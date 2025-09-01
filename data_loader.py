import numpy as np
from dataclasses import dataclass
from typing import List, Dict
import scipy.io
import logging

@dataclass
class LaserScan:
    """Class for storing laser scan data"""
    timestamp: float
    ranges: np.ndarray  # 361 measurements (0 to 180 degrees)
    angles: np.ndarray  # Corresponding angles in radians

@dataclass
class DeadReckoning:
    """Class for storing dead reckoning data"""
    timestamp: float
    speed: float      # m/s
    steering: float   # radians

@dataclass
class GPSData:
    """Class for storing GPS data"""
    timestamp: float
    latitude: float  # meters
    longitude: float  # meters

@dataclass
class SensorEvent:
    """Class for storing sensor event data"""
    timestamp: float
    sensor_id: int  # 1 = GPS, 2 = Dead Reckoning, 3 = Laser
    index: int

class DataLoader:
    def __init__(self):
        # Configure logging at the beginning of your script
        logging.basicConfig(
            level=logging.INFO,  # Set the logging level
            format='%(asctime)s - %(levelname)s - %(message)s',  # Define the log message format
            handlers=[
                logging.StreamHandler()  # Log to console; you can add FileHandler for file logging
            ]
        )

        logger = logging.getLogger(__name__)  # Create a logger for the current module

    def load_laser_data(self, laser_data_path: str) -> List[LaserScan]:
        """Load laser scan data from .txt file"""
        laser_scans = []
        try:
            # Load all data and reshape
            data = np.loadtxt(laser_data_path)
            print(f"Raw data shape: {data.shape}")
            
            # Reshape data into a matrix with the correct number of values per scan (362)
            # First value in each row is timestamp, followed by 361 measurements
            num_scans = len(data) // 362
            reshaped_data = data[:num_scans * 362].reshape(num_scans, 362)
            
            # Create angle array (0 to 180 degrees in radians)
            angles = np.linspace(0, np.pi, 361)
            
            # Process each scan
            for scan_data in reshaped_data:
                timestamp = scan_data[0]  # First value is timestamp
                ranges = scan_data[1:]    # Remaining values are ranges
                
                laser_scans.append(LaserScan(
                    timestamp=timestamp,
                    ranges=ranges,
                    angles=angles
                ))
                
            print(f"Successfully loaded {len(laser_scans)} laser scans")
            # if laser_scans:
            #     print(f"Second scan timestamp: {laser_scans[1].timestamp}")
            #     print(f"Second scan ranges shape: {laser_scans[1].ranges.shape}")
            #     print(f"Sample ranges from first scan: {laser_scans[1].ranges[:5]}")
                
        except Exception as e:
            print(f"Error loading laser data: {e}")
            import traceback
            print(traceback.format_exc())
            
        return laser_scans



    def load_dead_reckoning(self, dr_data_path: str) -> List[DeadReckoning]:
        """Load dead reckoning data from .txt file"""
        dr_measurements = []
        try:
            with open(dr_data_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    timestamp, speed, steering = map(float, parts)
                    dr_measurements.append(DeadReckoning(timestamp=timestamp, speed=speed, steering=steering))
            print(f"Loaded {len(dr_measurements)} dead reckoning measurements")
        except Exception as e:
            print(f"Error loading dead reckoning data: {e}")
        return dr_measurements

    def load_gps_data(self, gps_data_path: str) -> List[GPSData]:
        """Load GPS data from .txt file"""
        gps_measurements = []
        try:
            with open(gps_data_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    timestamp, latitude, longitude = map(float, parts)
                    gps_measurements.append(GPSData(timestamp=timestamp, latitude=latitude, longitude=longitude))
            print(f"Loaded {len(gps_measurements)} GPS measurements")
            
        except Exception as e:
            print(f"Error loading GPS data: {e}")
        return gps_measurements

    def load_gps_data_mat(self, gps_data_path: str) -> List[GPSData]:
        """Load GPS data from a .mat file and translate coordinates to start at origin."""
        gps_measurements = []
        try:
            # Load the .mat file
            mat = scipy.io.loadmat(gps_data_path)
            
            # Extract variables based on the provided structure
            time_gps = mat.get('timeGps')
            la_m = mat.get('La_m')
            lo_m = mat.get('Lo_m')
            
            if time_gps is None or la_m is None or lo_m is None:
                raise KeyError("One or more required variables ('timeGps', 'La_m', 'Lo_m') not found in the .mat file.")
            
            # Flatten the arrays in case they are loaded as 2D arrays
            time_gps = np.array(time_gps).flatten()
            la_m = np.array(la_m).flatten()
            lo_m = np.array(lo_m).flatten()
            
            # Check if all arrays have the same length
            if not (len(time_gps) == len(la_m) == len(lo_m)):
                raise ValueError("Mismatch in lengths of 'timeGps', 'La_m', and 'Lo_m' arrays.")
            
            # Convert time from milliseconds to seconds
            time_seconds = time_gps / 1000.0
            
            # Initialize variables for origin adjustment
            origin_lat = None
            origin_lon = None
            origin_set = False
            
            previous_ts = None
            for ts, lat_m, lon_m in zip(time_seconds, la_m, lo_m):
                if np.isnan(lat_m) or np.isnan(lon_m) or np.isnan(ts):
                    print(f"Skipping entry with NaN values at timestamp {ts}.")
                    continue  # Skip this entry
                
                
                # Set origin based on the first valid GPS data point
                if not origin_set:
                    origin_lat = lat_m
                    origin_lon = lon_m
                    origin_set = True
                    print(f"Origin set to Latitude: {origin_lat}, Longitude: {origin_lon}")
                
                # Translate coordinates to start at origin
                translated_lat = lat_m - origin_lat
                translated_lon = lon_m - origin_lon
                
                gps_measurements.append(GPSData(timestamp=float(ts), latitude=float(translated_lat), longitude=float(translated_lon)))
                previous_ts = ts
            
            print(f"Loaded {len(gps_measurements)} GPS measurements from '{gps_data_path}'.")
        
        except FileNotFoundError:
            print(f"File not found: {gps_data_path}")
        except KeyError as e:
            print(f"Missing data in .mat file: {e}")
        except ValueError as ve:
            print(f"Data format error: {ve}")
        except Exception as e:
            print(f"An unexpected error occurred while loading GPS data: {e}")
        
        return gps_measurements


    def load_sensor_manager(self, sensor_manager_path: str) -> List[SensorEvent]:
        """Load sensor manager data from .txt file"""
        events = []
        try:
            with open(sensor_manager_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    timestamp, sensor_id, index = float(parts[0]), int(parts[1]), int(parts[2])
                    events.append(SensorEvent(
                        timestamp=timestamp,
                        sensor_id=sensor_id,
                        index=index
                    ))
            print(f"Loaded {len(events)} sensor events from Sensor_manager.txt")
        except Exception as e:
            print(f"Error loading sensor manager data: {e}")
        return events

    # def synchronize_data(self, sensor_events: List[SensorEvent], laser_data: List[LaserScan], 
    #                      dr_data: List[DeadReckoning], gps_data: List[GPSData]) -> Dict[str, List]:
    #     """Synchronize data from different sensors based on sensor events"""
    #     synchronized_data = {"laser": [], "dead_reckoning": [], "gps": []}
        
    #     for event in sensor_events:
    #         if event.sensor_id == 1:  # GPS
    #             synchronized_data["gps"].append(gps_data[event.index - 1])
    #         elif event.sensor_id == 2:  # Dead Reckoning
    #             synchronized_data["dead_reckoning"].append(dr_data[event.index - 1])
    #         elif event.sensor_id == 3:  # Laser
    #             synchronized_data["laser"].append(laser_data[event.index - 1])

    #     print(f"Synchronized {len(synchronized_data['laser'])} laser scans, \
    #           {len(synchronized_data['dead_reckoning'])} dead reckoning entries, \
    #           and {len(synchronized_data['gps'])} GPS entries.")
    #     return synchronized_data

if __name__ == "__main__":
    # Instantiate the loader
    loader = DataLoader()

    # Paths to the data files
    laser_data_path = "LASER.txt"
    dr_data_path = "DRS.txt"
    gps_data_path = "GPS.txt"
    sensor_manager_path = "Sensors_manager.txt"

    # Load data
    sensor_events = loader.load_sensor_manager(sensor_manager_path)
    laser_data = loader.load_laser_data(laser_data_path)
    dr_data = loader.load_dead_reckoning(dr_data_path)
    gps_data = loader.load_gps_data(gps_data_path)
    
    i = 0

    for event in sensor_events:
        if event.sensor_id == 1: # GPS
            print(f'GPS DATA: {gps_data[event.index - 1]}')
            print(f'EVENT: {event}')
        elif event.sensor_id == 2: # Dead Reckoning
            print(f'DR DATA: {dr_data[event.index - 1]}')
            print(f'EVENT: {event}')
        elif event.sensor_id == 3: # Laser
            print(f'Laser DATA: {laser_data[event.index - 1].timestamp}')
            print(f'EVENT: {event}')
        if i == 5:
            break
        i += 1