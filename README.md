# Enhanced Smart Wheelchair Navigation System

## Overview

This is an enhanced smart wheelchair navigation system that uses **POV (Point of View) camera analysis** for real-time obstacle detection in the navigation path. The system analyzes the forward-looking video feed from a wheelchair-mounted camera to detect obstacles and dynamically adjust the navigation path.

## Key Features

### 1. Enhanced POV Camera Analysis
- **Path-Focused Obstacle Detection**: Analyzes obstacles specifically in the navigation path, not just general obstacles
- **Region of Interest (ROI)**: Focuses on the forward path area of the camera feed for efficient analysis
- **Multi-Method Detection**: Uses four different detection methods:
  - Depth estimation (focus/blur analysis)
  - Structural analysis (adaptive thresholding)
  - Motion detection (background subtraction)
  - Edge analysis (Canny edge detection)

### 2. Real-Time Path Replanning
- Automatically replans path when obstacles are detected in the navigation path
- Dynamic obstacle tracking and filtering
- Temporal consistency checking (obstacles must be detected consistently across frames)

### 3. Navigation System
- A* pathfinding algorithm with obstacle avoidance
- Destination-oriented navigation
- Smooth path planning and execution
- Emergency stop and recovery mechanisms

### 4. User Interface
- Dual-panel GUI with POV camera feed and navigation map
- Real-time obstacle visualization
- Path overlay on camera feed
- System status and performance monitoring

## System Architecture

### Components

1. **POVCameraProcessor** (`camera_processor.py`)
   - Handles POV camera initialization and frame capture
   - Performs path-focused obstacle detection
   - Multi-method detection algorithms
   - Temporal filtering for consistent results

2. **SmartWheelchairApp** (`main.py`)
   - Main application controller
   - Coordinates navigation, camera analysis, and path planning
   - Handles system state and error recovery
   - Manages threading for real-time processing

3. **PathPlanner** (`path_planner.py`)
   - A* pathfinding implementation
   - Dynamic path replanning
   - Path smoothing and optimization

4. **MapManager** (`map_manager.py`)
   - Map loading and management
   - Landmark tracking
   - Obstacle management

5. **WheelchairGUI** (`gui_interface.py`)
   - User interface with camera feed and map display
   - Real-time visualization
   - Control panel and status display

## Installation

### Requirements

```bash
pip install opencv-python numpy pillow tkinter
```

### Dependencies
- Python 3.7+
- OpenCV (cv2)
- NumPy
- PIL (Pillow)
- Tkinter (usually included with Python)

## Usage

### Running the System

```bash
python main.py
```

### Setting Up the Camera

1. Connect a webcam to your computer (typically index 0)
2. Mount the camera on the wheelchair facing forward (POV position)
3. The system will automatically detect and initialize the camera

### Navigation Workflow

1. **Load a Map**: Select a map from the dropdown (e.g., `hospital_floor1.json`)
2. **Set Destination**: Click on the map to set a destination
3. **Start Navigation**: Click "Start Navigation" button
4. **Monitor Progress**: Watch the POV camera feed and map for real-time updates

### Camera Calibration

The system includes automatic calibration:
- Click "Calibrate Camera" to calibrate obstacle detection
- Calibration captures background frames for motion detection
- Recommended: Calibrate in the starting environment before navigation

## Configuration

Edit `config.json` or use `config.py` to customize:

- **Camera Settings**: Resolution, FPS, index
- **Grid Settings**: Grid size, cell size
- **Detection Parameters**: Threshold, min/max obstacle area
- **Navigation Settings**: Movement cost, diagonal movement
- **GUI Settings**: Window size, refresh rate

## POV Camera Analysis Details

### Path ROI (Region of Interest)
- Focuses on the lower-center region of the camera frame
- This is where obstacles in the forward path typically appear
- Configurable based on camera mounting position

### Obstacle Detection Methods

1. **Depth Estimation**
   - Uses Laplacian variance to measure focus/sharpness
   - Closer objects appear sharper
   - Identifies potential obstacles in the path

2. **Structural Analysis**
   - Adaptive thresholding to identify dark regions
   - Morphological operations to clean up detections
   - Validates obstacles by checking darkness

3. **Motion Detection**
   - Background subtraction using MOG2
   - Detects moving objects in the path
   - Useful for dynamic obstacles (people, moving objects)

4. **Edge Analysis**
   - Canny edge detection
   - Contour analysis
   - Shape-based filtering

### Temporal Filtering
- Tracks obstacles across multiple frames
- Only considers obstacles detected consistently
- Reduces false positives from noise

### Path-Focused Analysis
- Filters obstacles to only those in or near the navigation path
- Reduces computational load
- Focuses on obstacles that matter for navigation

## Performance

- **Processing Speed**: ~30 FPS camera analysis
- **Detection Latency**: < 50ms per frame
- **Path Replanning**: < 100ms for typical grids
- **Memory Usage**: Optimized for real-time operation

## Troubleshooting

### Camera Not Detected
- Check camera connection
- Verify camera index in config (default: 0)
- Try different camera indices if multiple cameras are connected

### Poor Obstacle Detection
- Calibrate camera in the environment
- Adjust detection threshold in config
- Ensure adequate lighting
- Check camera focus and positioning

### Path Planning Issues
- Verify map is loaded correctly
- Check destination is not blocked
- Ensure start position is valid
- Review obstacle detection results

## Future Enhancements

- Machine learning-based obstacle classification
- 3D depth sensing integration
- Multi-camera fusion
- Advanced path smoothing
- Predictive obstacle avoidance
- Cloud-based map sharing

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions are welcome! Please ensure code follows the existing style and includes appropriate tests.

## Acknowledgments

- OpenCV for computer vision capabilities
- A* algorithm implementation
- Tkinter for GUI framework

---

**Note**: This system is designed for simulation and testing. For real-world deployment, additional safety mechanisms and hardware integration would be required.


