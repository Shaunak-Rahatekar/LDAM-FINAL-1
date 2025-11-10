
"""
Enhanced Smart Wheelchair Navigation System with POV Camera Analysis

This system provides autonomous navigation for wheelchairs using:
- Enhanced POV (Point of View) camera feed analysis from wheelchair-mounted camera
- Real-time path-focused obstacle detection using multiple computer vision methods
- Destination-oriented path planning with dynamic replanning
- A* pathfinding algorithm with obstacle avoidance
- Multi-method obstacle detection (depth estimation, structural analysis, motion detection, edge analysis)

The wheelchair-mounted POV camera continuously analyzes the forward-looking video feed
to detect obstacles specifically in the navigation path, enabling real-time path
adjustment for safe navigation to the destination.

Key Features:
- Path-focused obstacle detection (analyzes obstacles in navigation path)
- Region of Interest (ROI) focusing on forward path area
- Temporal filtering for consistent obstacle detection
- Multi-method detection for robustness
- Real-time path replanning based on camera analysis
"""

import time
import threading
import queue
import numpy as np
import logging
import traceback
import json
from typing import List, Tuple, Optional, Dict, Any
from collections import deque
from datetime import datetime

import tkinter as tk

from map_manager import MapManager
from path_planner import PathPlanner
from camera_processor import CameraProcessor
from gui_interface import WheelchairGUI
from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('smart_wheelchair.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('SmartWheelchair')


class SmartWheelchairApp:
    """Enhanced main application controller with comprehensive error handling"""

    def __init__(self):
        # Initialize logging
        self.logger = logger
        self.logger.info("Initializing Smart Wheelchair System...")

        # System status tracking
        self.system_status = {
            'camera_available': False,
            'map_loaded': False,
            'navigation_active': False,
            'emergency_stop': False,
            'degraded_mode': False,
            'last_error': None
        }

        # Initialize modules with error handling
        self.map_manager = self._initialize_module(MapManager, "MapManager")
        self.path_planner = self._initialize_module(PathPlanner, "PathPlanner")
        self.camera_processor = self._initialize_module(CameraProcessor, "CameraProcessor")

        # System state with validation
        self.current_position: Optional[Tuple[int, int]] = None
        self.destination: Optional[Tuple[int, int]] = None
        self.path: List[Tuple[int, int]] = []
        self.dynamic_obstacles: List[Tuple[int, int]] = []
        self.temporary_obstacles: List[Tuple[int, int]] = []
        self.current_direction: str = "STOP"

        # Enhanced threading and synchronization
        self.is_running = False
        self.is_paused = False
        self.navigation_thread: Optional[threading.Thread] = None
        self.camera_thread: Optional[threading.Thread] = None
        self.command_thread: Optional[threading.Thread] = None
        self.gui_update_queue = queue.Queue(maxsize=10)  # Prevent queue overload
        self.command_queue = queue.Queue(maxsize=50)     # Command buffering

        # Locks to prevent race conditions
        self.state_lock = threading.Lock()
        self.path_lock = threading.Lock()
        self.obstacle_lock = threading.Lock()

        # Enhanced performance monitoring
        self.performance_stats = {
            'total_frames_processed': 0,
            'average_processing_time': 0.0,
            'total_replans': 0,
            'total_distance_traveled': 0.0,
            'system_uptime': 0.0,
            'error_count': 0,
            'recovery_count': 0,
            'camera_failures': 0,
            'path_failures': 0
        }

        # Path smoothing and optimization
        self.path_history = deque(maxlen=10)    # Store recent paths for smoothing
        self.command_history = deque(maxlen=20) # Command history for analysis

        # Simulation mode
        self.simulation_mode = False
        self.simulated_obstacles: List[Tuple[int, int]] = []

        # GUI root
        self.root: Optional[tk.Tk] = None
        self.gui: Optional[WheelchairGUI] = None

        # Initialize system components
        self._initialize_system()

        logger.info("Smart Wheelchair System initialized successfully")

    def set_destination(self, destination: Tuple[int, int]):
        """Set destination for wheelchair navigation - system will navigate using live camera feed"""
        try:
            # Validate destination coordinates
            if not (0 <= destination[0] < config['GRID_WIDTH'] and
                    0 <= destination[1] < config['GRID_HEIGHT']):
                self._show_user_message("Invalid Destination",
                                        f"Destination {destination} is out of bounds!")
                return False

            # Check if destination is not blocked
            if not self._is_position_valid(destination):
                self._show_user_message("Invalid Destination",
                                        f"Destination {destination} is blocked by obstacles!")
                return False

            # Set destination for wheelchair navigation
            with self.state_lock:
                self.destination = destination
            
            self._update_navigation_status(f"Destination set: {destination}")
            logger.info(f"✓ Destination set for wheelchair navigation: {destination}")
            logger.info("  System will use live webcam feed to navigate to destination")

            # Plan path to destination using current camera-obstacle data
            if self.is_running:
                self.plan_path()
            else:
                # Plan path now even if not running - will be used when navigation starts
                self.plan_path()

            logger.info(f"✓ Path planned to destination using live camera analysis")
            return True

        except Exception as e:
            logger.error(f"Error setting destination: {e}")
            self._show_user_message("Error", f"Failed to set destination: {e}")
            return False

    def get_available_destinations(self) -> Dict[str, Tuple[int, int]]:
        """Get available destinations from current map"""
        try:
            if self.map_manager and self.map_manager.current_map:
                landmarks = self.map_manager.get_landmarks()
                return landmarks
            return {}
        except Exception as e:
            logger.error(f"Error getting destinations: {e}")
            return {}

    def set_destination_to_landmark(self, landmark_name: str):
        """Set destination to a named landmark"""
        try:
            landmarks = self.get_available_destinations()
            if landmark_name in landmarks:
                return self.set_destination(landmarks[landmark_name])
            else:
                self._show_user_message("Landmark Not Found",
                                        f"Landmark '{landmark_name}' not found in current map!")
                return False
        except Exception as e:
            logger.error(f"Error setting landmark destination: {e}")
            return False

    def _initialize_module(self, module_class, module_name):
        """Safely initialize a module with error handling"""
        try:
            instance = module_class()
            logger.info(f"{module_name} initialized successfully")
            return instance
        except Exception as e:
            logger.error(f"Failed to initialize {module_name}: {e}")
            self.system_status['degraded_mode'] = True
            self.performance_stats['error_count'] += 1
            return None

    def _initialize_system(self):
        """Initialize the complete system with error recovery"""
        try:
            # Create default maps if they don't exist
            if self.map_manager:
                self.map_manager.create_default_maps()

            # Initialize GUI in main thread
            self.setup_gui()

            # Setup system components
            self.setup_system()

            # Start background monitoring
            self._start_system_monitor()

        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            self.system_status['last_error'] = str(e)
            self._enter_degraded_mode()

    def setup_gui(self):
        """Initialize the GUI with error handling"""
        try:
            self.root = tk.Tk()
            self.gui = WheelchairGUI(self.root, self)
            # Setup GUI update handling
            self.root.after(100, self.process_gui_updates)
            # Handle window close event
            self.root.protocol("WM_DELETE_WINDOW", self.shutdown)
            logger.info("GUI initialized successfully")
        except Exception as e:
            logger.error(f"GUI initialization failed: {e}")
            self.system_status['degraded_mode'] = True

    def setup_system(self):
        """Setup the complete system with comprehensive error handling"""
        logger.info("Setting up system components...")

        # Load default map with error handling
        map_loaded = self._safe_load_map("hospital_floor1.json")
        if not map_loaded:
            logger.warning("Failed to load default map, using fallback")
            self._setup_fallback_map()

        # Initialize POV camera with fallback to simulation
        self._initialize_pov_camera_with_fallback()

        # Update GUI with initial state
        self.update_gui_display()

        logger.info("System setup completed")

    def _safe_load_map(self, map_file: str) -> bool:
        """Safely load map with error handling"""
        try:
            if not self.map_manager:
                logger.error("Map manager not available")
                return False

            success = self.map_manager.load_map(map_file)
            if success:
                start_pos = self.map_manager.get_start_position()
                with self.state_lock:
                    self.current_position = tuple(start_pos) if start_pos else (0, 0)
                self.reset_navigation()
                if hasattr(self, 'gui') and self.gui:
                    self.gui.update_landmark_display()
                self.system_status['map_loaded'] = True
                logger.info(f"Map loaded successfully: {map_file}")
                return True
            else:
                logger.warning(f"Failed to load map: {map_file}")
                return False

        except Exception as e:
            logger.error(f"Error loading map {map_file}: {e}")
            self.performance_stats['error_count'] += 1
            return False

    def _setup_fallback_map(self):
        """Create a fallback map when primary loading fails"""
        try:
            fallback_map = {
                "name": "Fallback Map",
                "grid_size": [config['GRID_WIDTH'], config['GRID_HEIGHT']],
                "permanent_obstacles": [
                    {"type": "outer_walls", "coords": self._create_simple_border()}
                ],
                "landmarks": {
                    "start": [2, 2],
                    "center": [config['GRID_WIDTH'] // 2, config['GRID_HEIGHT'] // 2]
                },
                "start_position": [2, 2]
            }

            # Try to load into MapManager, or build grid locally
            grid_width = config['GRID_WIDTH']
            grid_height = config['GRID_HEIGHT']
            grid = np.zeros((grid_height, grid_width), dtype=int)

            for x, y in fallback_map["permanent_obstacles"][0]["coords"]:
                if 0 <= x < grid_width and 0 <= y < grid_height:
                    grid[y, x] = 1

            if self.map_manager and hasattr(self.map_manager, 'load_from_dict'):
                self.map_manager.load_from_dict(fallback_map)
            elif self.map_manager and hasattr(self.map_manager, 'set_grid_data'):
                self.map_manager.set_grid_data(grid, fallback_map)
            else:
                # Minimal internal storage when MapManager lacks helpers
                self.map_manager = self.map_manager or MapManager()
                if hasattr(self.map_manager, 'set_grid_data'):
                    self.map_manager.set_grid_data(grid, fallback_map)

            with self.state_lock:
                self.current_position = (2, 2)
            self.system_status['map_loaded'] = True
            logger.info("Fallback map activated")
        except Exception as e:
            logger.error(f"Fallback map setup failed: {e}")
            self.system_status['last_error'] = "Map system completely failed"

    def _create_simple_border(self) -> List[Tuple[int, int]]:
        """Create simple border coordinates"""
        coords: List[Tuple[int, int]] = []
        for x in range(config['GRID_WIDTH']):
            coords.append((x, 0))
            coords.append((x, config['GRID_HEIGHT'] - 1))
        for y in range(1, config['GRID_HEIGHT'] - 1):
            coords.append((0, y))
            coords.append((config['GRID_WIDTH'] - 1, y))
        return coords

    def _initialize_pov_camera_with_fallback(self):
        """Initialize POV camera (mounted on wheelchair) for enhanced path-focused analysis"""
        try:
            if self.camera_processor and self.camera_processor.initialize_camera():
                # POV camera initialized successfully - essential for destination-oriented navigation
                self.system_status['camera_available'] = True
                self.simulation_mode = False
                # Calibrate POV camera for path-focused obstacle detection
                self.camera_processor.calibrate_detection()
                logger.info("✓ POV camera initialized successfully for path-focused analysis")
                logger.info("✓ Wheelchair-mounted camera ready for enhanced obstacle detection")
                logger.info("✓ Camera will analyze forward path for obstacles")
            else:
                # POV camera failed - cannot perform live navigation without camera
                logger.error("✗ POV camera initialization failed - navigation requires live camera feed")
                self._enter_simulation_mode()
                self._show_user_message("Camera Error", 
                    "POV camera not available. Navigation will use simulation mode.\n"
                    "Please connect camera for live path obstacle detection.")
        except Exception as e:
            logger.error(f"POV camera initialization error: {e}")
            self._enter_simulation_mode()

    def _enter_simulation_mode(self):
        """Enter simulation mode when camera is unavailable"""
        self.system_status['camera_available'] = False
        self.simulation_mode = True
        self.performance_stats['camera_failures'] += 1
        logger.warning("Entering simulation mode - camera unavailable")

        # Generate simulated obstacles
        self._generate_simulated_obstacles()

        # Update GUI status
        if hasattr(self, 'gui') and self.gui:
            try:
                self.gui.camera_status_var.set("Camera: SIMULATION MODE")
            except Exception:
                pass

    def _enter_degraded_mode(self):
        """Enter degraded mode when critical components fail"""
        self.system_status['degraded_mode'] = True
        logger.warning("Entering degraded mode - limited functionality available")

        # Stop any active navigation
        self.stop_navigation()

        # Update GUI if available
        if hasattr(self, 'gui') and self.gui:
            self.gui.show_error("Degraded Mode",
                                "System is running in degraded mode. Some features may be unavailable.")

    def _generate_simulated_obstacles(self):
        """Generate realistic simulated obstacles"""
        with self.obstacle_lock:
            self.simulated_obstacles = []

            # Add some fixed obstacles
            for i in range(10, min(config['GRID_WIDTH'], 40), 5):
                self.simulated_obstacles.append((i, min(20, config['GRID_HEIGHT'] - 1)))
                self.simulated_obstacles.append((i, min(40, config['GRID_HEIGHT'] - 1)))

            # Add some random obstacles that change over time
            if int(time.time()) % 10 < 5:  # Change every 5 seconds
                for _ in range(3):
                    x = np.random.randint(0, config['GRID_WIDTH'])
                    y = np.random.randint(0, config['GRID_HEIGHT'])
                    self.simulated_obstacles.append((x, y))

    def _start_system_monitor(self):
        """Start background system monitoring"""

        def monitor_loop():
            while True:
                try:
                    self._system_health_check()
                    time.sleep(5)  # Check every 5 seconds
                except Exception as e:
                    logger.error(f"System monitor error: {e}")
                    time.sleep(10)  # Wait longer on error

        monitor_thread = threading.Thread(target=monitor_loop, daemon=True, name="SystemMonitor")
        monitor_thread.start()
        logger.info("System monitor started")

    def start_navigation(self):
        """Start destination-oriented wheelchair navigation using live webcam feed"""
        try:
            if self.destination is None:
                self._show_user_message("Navigation Error", 
                    "Please set a destination first!\n"
                    "The wheelchair will navigate to the destination using live camera analysis.")
                return

            if not self.is_running:
                self.is_running = True
                self.is_paused = False
                self.system_status['emergency_stop'] = False

                # Plan initial path to destination
                if not self.plan_path():
                    self._show_user_message("Path Error", 
                        "No valid path to destination found!\n"
                        "Please check obstacles and try a different destination.")
                    self.is_running = False
                    return

                logger.info("=" * 60)
                logger.info("Starting Destination-Oriented Wheelchair Navigation")
                logger.info(f"Destination: {self.destination}")
                logger.info(f"Using live webcam feed for obstacle detection")
                logger.info("=" * 60)

                # Start navigation thread - moves wheelchair toward destination
                self.navigation_thread = threading.Thread(
                    target=self._navigation_loop,
                    daemon=True,
                    name="NavigationThread"
                )
                self.navigation_thread.start()

                # Start live webcam feed analysis thread - essential for navigation
                if self.system_status['camera_available']:
                    self.camera_thread = threading.Thread(
                        target=self._camera_processing_loop,
                        daemon=True,
                        name="CameraThread"
                    )
                    self.camera_thread.start()
                    logger.info("✓ Live webcam feed analysis started")
                else:
                    logger.warning("⚠ Webcam not available - using simulation mode")

                # Start command processor for wheelchair movement
                self.command_thread = threading.Thread(
                    target=self._command_processing_loop,
                    daemon=True,
                    name="CommandThread"
                )
                self.command_thread.start()

                self.system_status['navigation_active'] = True
                self._update_gui_command("NAVIGATING TO DESTINATION")
                self._update_navigation_status(f"Navigating to {self.destination}")
                logger.info("✓ Wheelchair navigation started successfully")

        except Exception as e:
            logger.error(f"Error starting navigation: {e}")
            self._handle_critical_error("NavigationStart", e)

    def _navigation_loop(self):
        """Enhanced navigation loop with comprehensive error handling"""
        loop_count = 0
        consecutive_errors = 0
        max_consecutive_errors = 5

        while self.is_running and not self.system_status['emergency_stop']:
            loop_start_time = time.time()
            try:
                # Skip if paused
                if self.is_paused:
                    time.sleep(0.1)
                    continue

                # Check system health
                if not self._pre_loop_checks():
                    time.sleep(0.1)
                    continue

                # Check if destination reached
                if self.has_reached_destination():
                    self._handle_destination_reached()
                    break

                # Get next position with path validation
                next_pos = self._get_valid_next_position()
                if not next_pos:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error("Too many consecutive navigation errors")
                        self.emergency_stop()
                        break
                    time.sleep(0.1)
                    continue

                consecutive_errors = 0  # Reset error counter on success

                # Update position and generate command
                self._update_position_and_command(next_pos)

                # Update performance stats
                loop_count += 1
                if loop_count % 10 == 0:
                    self._update_performance_stats()

                # Update GUI display
                self.update_gui_display()

                # Control loop speed with adaptive timing
                self._adaptive_sleep(loop_start_time)

            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Navigation loop error (attempt {consecutive_errors}): {e}")
                if consecutive_errors >= max_consecutive_errors:
                    logger.error("Critical navigation failure - emergency stop")
                    self.emergency_stop()
                    break
                time.sleep(0.5)  # Wait before retrying

        self._cleanup_navigation()

    def _pre_loop_checks(self) -> bool:
        """Perform pre-loop validation checks"""
        with self.path_lock:
            has_path = bool(self.path)
        if self.destination is None or not has_path:
            return False

        with self.state_lock:
            if not self.current_position:
                logger.warning("No current position available")
                return False

            # Validate current position is not blocked
            if not self._is_position_valid(self.current_position):
                logger.warning("Current position is blocked - attempting recovery")
                self._recover_from_blocked_position()
                return False

        return True

    def _get_valid_next_position(self) -> Optional[Tuple[int, int]]:
        """Get next position with validation"""
        if not self.path_planner:
            return None
        next_pos = self.path_planner.get_next_position()
        if next_pos and self._is_position_valid(next_pos):
            return next_pos

        # Position invalid, try to replan
        logger.warning("Next position invalid, attempting replan")
        if self.plan_path():
            next_pos = self.path_planner.get_next_position()
            return next_pos if (next_pos and self._is_position_valid(next_pos)) else None

        return None

    def _update_position_and_command(self, next_pos: Tuple[int, int]):
        """Update position and generate movement command"""
        with self.state_lock:
            old_position = self.current_position
            self.current_position = next_pos

        # Calculate distance traveled
        if old_position:
            dist = self._calculate_distance(old_position, self.current_position)
            self.performance_stats['total_distance_traveled'] += dist

        # Generate and queue movement command
        command = self.generate_movement_command(old_position, next_pos)
        self.current_direction = command

        # Queue command for processing
        self._queue_command(command)

        # Add to command history for smoothing
        self.command_history.append({
            'timestamp': time.time(),
            'command': command,
            'position': self.current_position
        })

    def _adaptive_sleep(self, loop_start_time: float):
        """Adaptive sleep based on processing time"""
        processing_time = time.time() - loop_start_time
        sleep_time = max(0.01, config['SIMULATION_SPEED'] - processing_time)
        # Adaptive sleep to maintain consistent timing
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            logger.warning(f"Navigation loop overloaded: {processing_time:.3f}s")

    def _camera_processing_loop(self):
        """Enhanced POV camera feed analysis loop for destination-oriented navigation"""
        consecutive_failures = 0
        max_consecutive_failures = 3

        logger.info("Starting enhanced POV camera feed analysis for wheelchair navigation...")

        while (self.is_running and
               self.system_status['camera_available'] and
               not self.system_status['emergency_stop']):
            try:
                # Get live frame from POV camera mounted on wheelchair
                frame = self.camera_processor.get_frame()
                if frame is None:
                    raise ValueError("No frame received from POV camera")

                # Get current path for path-focused obstacle analysis
                with self.path_lock:
                    current_path = self.path.copy()

                # Analyze POV feed to detect obstacles specifically in navigation path
                # This uses enhanced multi-method detection focused on path obstacles
                obstacles = self.camera_processor.analyze_path_obstacles(frame, current_path)

                # Update dynamic obstacles detected from POV feed
                with self.obstacle_lock:
                    self.dynamic_obstacles = obstacles

                # Real-time path replanning based on POV camera analysis
                # Replan if obstacles are detected in the path
                if (self.is_running and not self.is_paused and
                        self.destination and obstacles and current_path):
                    if self._is_path_blocked(obstacles, current_path):
                        logger.info("POV camera detected obstacle blocking path - replanning...")
                        self.plan_path()  # Replan path to reach destination

                # Update GUI with live POV camera feed and detection results
                self._queue_camera_frame(frame, obstacles, current_path)

                consecutive_failures = 0  # Reset failure counter

                # Process at ~30 FPS for real-time analysis
                time.sleep(0.033)

            except Exception as e:
                consecutive_failures += 1
                logger.warning(f"POV camera analysis error ({consecutive_failures}): {e}")
                if consecutive_failures >= max_consecutive_failures:
                    logger.error("POV camera feed failed repeatedly - cannot continue live navigation")
                    self._enter_simulation_mode()
                    break
                time.sleep(1)  # Wait before retrying

    def _command_processing_loop(self):
        """Process movement commands with smoothing and validation"""
        while self.is_running and not self.system_status['emergency_stop']:
            try:
                # Get command from queue with timeout
                command_data = self.command_queue.get(timeout=1.0)

                # Apply command smoothing
                smoothed_command = self._smooth_command(command_data)

                # Execute command
                self._execute_command(smoothed_command)

                self.command_queue.task_done()

            except queue.Empty:
                continue  # No commands to process
            except Exception as e:
                logger.error(f"Command processing error: {e}")

    def _smooth_command(self, command_data: Dict) -> str:
        """Apply smoothing to movement commands"""
        command = command_data.get('command', 'STOP')

        # Simple smoothing: avoid rapid direction changes
        if len(self.command_history) >= 2:
            last_commands = [cmd['command'] for cmd in list(self.command_history)[-2:]]

            # If last two commands are the same, and new is different, validate change
            if len(set(last_commands)) == 1 and command != last_commands[0]:
                if not self._is_direction_change_necessary(command):
                    logger.debug(f"Smoothed direction change: {command} -> {last_commands[0]}")
                    return last_commands[0]

        return command

    def _is_direction_change_necessary(self, new_command: str) -> bool:
        """Check if a direction change is necessary"""
        if not self.command_history:
            return True

        # Get recent positions
        recent_positions = [cmd['position'] for cmd in list(self.command_history)[-3:]]
        if len(recent_positions) < 2:
            return True

        # Simple check: if we're making progress, don't change direction abruptly
        return True  # Simplified for now

    def _execute_command(self, command: str):
        """Execute a movement command"""
        try:
            # Update GUI
            self._update_gui_command(command)

            # In a real system, this would send commands to hardware
            if command in ["RIGHT", "LEFT", "UP", "DOWN", "FORWARD"]:
                logger.debug(f"Executing command: {command}")
            elif command == "STOP":
                logger.debug("Stopping movement")
        except Exception as e:
            logger.error(f"Command execution error: {e}")

    def _queue_command(self, command: str):
        """Queue a command for processing"""
        try:
            command_data = {
                'timestamp': time.time(),
                'command': command,
                'position': self.current_position
            }
            if not self.command_queue.full():
                self.command_queue.put(command_data)
            else:
                logger.warning("Command queue full - dropping command")
        except Exception as e:
            logger.error(f"Error queueing command: {e}")

    def _queue_camera_frame(self, frame, obstacles, current_path=None):
        """Queue camera frame for GUI update with path overlay"""
        try:
            if not self.gui_update_queue.full():
                overlay_frame = self.camera_processor.draw_detection_overlay(
                    frame, obstacles, current_path
                )
                self.gui_update_queue.put(('camera_frame', overlay_frame))
        except Exception as e:
            logger.debug(f"Camera frame queue error: {e}")

    def plan_path(self) -> bool:
        """Plan path to destination using live camera-obstacle data for wheelchair navigation"""
        try:
            with self.state_lock:
                start = self.current_position
                dest = self.destination

            if start is None or dest is None:
                logger.warning("Cannot plan path: missing start or destination")
                return False

            # Validate positions
            if not self._is_position_valid(start):
                logger.error("Start position is invalid/blocked")
                return False

            if not self._is_position_valid(dest):
                logger.error("Destination position is invalid/blocked")
                return False

            # Combine all obstacles including those detected from live webcam feed
            # This ensures the path avoids obstacles detected by the wheelchair-mounted camera
            combined_grid = self.get_combined_obstacles_grid()

            # Plan path to destination using A* algorithm
            # Path incorporates obstacles detected from live camera feed
            new_path = self.path_planner.a_star_search(combined_grid, start, dest)
            if new_path:
                # Apply path smoothing for smoother wheelchair movement
                smoothed_path = self._smooth_path(new_path)
                with self.path_lock:
                    self.path = smoothed_path
                self.path_planner.set_path(self.path)
                self.path_history.append(smoothed_path)

                # Update performance stats
                self.performance_stats['total_replans'] += 1
                logger.info(f"✓ Path planned to destination: {len(self.path)} steps")
                logger.info(f"  Using obstacles from live camera feed: {len(self.dynamic_obstacles)} dynamic obstacles")
                # Print the path on an ASCII grid for console visibility
                self._print_path_grid(start, dest, self.path, combined_grid)
                return True
            else:
                logger.warning("✗ No valid path found to destination")
                logger.warning(f"  Consider obstacles detected by camera: {len(self.dynamic_obstacles)}")
                self.performance_stats['path_failures'] += 1
                self._handle_no_path_found()
                return False

        except Exception as e:
            logger.error(f"Path planning error: {e}")
            self.performance_stats['error_count'] += 1
            return False

    def _smooth_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Apply smoothing to the planned path"""
        if len(path) <= 2:
            return path

        smoothed_path = [path[0]]  # Always include start
        i = 0
        while i < len(path) - 1:
            # Find the furthest point we can reach directly
            j = len(path) - 1
            while j > i + 1:
                if self._is_direct_path_clear(path[i], path[j]):
                    smoothed_path.append(path[j])
                    i = j
                    break
                j -= 1
            else:
                # No direct path found, take next point
                smoothed_path.append(path[i + 1])
                i += 1

        logger.debug(f"Path smoothed: {len(path)} -> {len(smoothed_path)} points")
        return smoothed_path

    def _is_direct_path_clear(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """Check if direct path between two points is clear
        Simple implementation - in real system, use Bresenham's line algorithm
        For now, assume diagonal paths are clear if endpoints are clear
        """
        # Endpoints validation
        return self._is_position_valid(start) and self._is_position_valid(end)

    def _is_position_valid(self, position: Tuple[int, int]) -> bool:
        """Check if a position is valid and not blocked"""
        x, y = position
        # Check bounds
        if not (0 <= x < config['GRID_WIDTH'] and 0 <= y < config['GRID_HEIGHT']):
            return False

        # Check obstacles
        grid = self.get_combined_obstacles_grid()
        try:
            return grid[y, x] == 0
        except Exception:
            return False

    def _is_path_blocked(self, obstacles: List[Tuple[int, int]], 
                         current_path: List[Tuple[int, int]] = None) -> bool:
        """Enhanced path blocking detection using path-focused analysis"""
        if not self.path_planner:
            return False
        
        # Use provided path or get remaining path
        if current_path:
            path_to_check = current_path
        else:
            path_to_check = self.path_planner.get_remaining_path()
        
        if not path_to_check:
            return False

        # Create obstacle set for faster lookup
        obstacle_set = set(obstacles)

        # Check next few positions (more aggressive detection)
        check_depth = min(5, len(path_to_check))
        for i in range(check_depth):
            if path_to_check[i] in obstacle_set:
                logger.info(f"Path blocked at position {path_to_check[i]}")
                return True
        
        # Also check for obstacles within 1 cell of path (close obstacles)
        for path_pos in path_to_check[:check_depth]:
            for obstacle in obstacles:
                distance = abs(path_pos[0] - obstacle[0]) + abs(path_pos[1] - obstacle[1])
                if distance <= 1:
                    logger.info(f"Path blocked by nearby obstacle at {obstacle}")
                    return True
        
        return False

    def _recover_from_blocked_position(self):
        """Attempt to recover when current position is blocked"""
        logger.warning("Attempting recovery from blocked position")
        with self.state_lock:
            cur_x, cur_y = self.current_position

        # Try to find nearest valid position
        for distance in range(1, 4):  # Check up to 3 cells away
            for dx in range(-distance, distance + 1):
                for dy in range(-distance, distance + 1):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = cur_x + dx
                    new_y = cur_y + dy
                    if (0 <= new_x < config['GRID_WIDTH'] and
                            0 <= new_y < config['GRID_HEIGHT'] and
                            self._is_position_valid((new_x, new_y))):
                        logger.info(f"Recovery: moving to ({new_x}, {new_y})")
                        with self.state_lock:
                            self.current_position = (new_x, new_y)
                        self.performance_stats['recovery_count'] += 1
                        self.plan_path()  # Replan from new position
                        return

        logger.error("Could not recover from blocked position")
        self.emergency_stop()

    def _handle_no_path_found(self):
        """Handle scenario when no path is found"""
        self._update_gui_command("NO PATH")
        self._update_navigation_status("No Path Found")
        # Try to find alternative solutions
        self._attempt_path_recovery()

    def _attempt_path_recovery(self):
        """Attempt to recover when no path is found"""
        logger.info("Attempting path recovery...")

        # Strategy 1: Remove temporary obstacles and retry
        with self.obstacle_lock:
            original_obstacles = self.temporary_obstacles.copy()
            self.temporary_obstacles = []
        if self.plan_path():
            logger.info("Path recovery successful by removing temporary obstacles")
            return

        # Strategy 2: Restore obstacles and try intermediate points
        with self.obstacle_lock:
            self.temporary_obstacles = original_obstacles

        with self.state_lock:
            cur_x, cur_y = self.current_position

        # Try to find path to nearby points
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            intermediate_x = cur_x + dx
            intermediate_y = cur_y + dy
            if (0 <= intermediate_x < config['GRID_WIDTH'] and
                    0 <= intermediate_y < config['GRID_HEIGHT'] and
                    self._is_position_valid((intermediate_x, intermediate_y))):
                # Plan to intermediate point first
                grid = self.get_combined_obstacles_grid()
                if self.path_planner.a_star_search(grid, (cur_x, cur_y), (intermediate_x, intermediate_y)):
                    logger.info(f"Found path to intermediate point ({intermediate_x}, {intermediate_y})")
                    # Let next loop iteration continue from there
                    return

        logger.warning("Path recovery failed")

    def _handle_destination_reached(self):
        """Handle destination reached scenario"""
        logger.info("Destination reached successfully!")
        self._update_gui_command("DESTINATION REACHED")
        self._update_navigation_status("Destination Reached")

        # Stop navigation
        self.is_running = False

        # Show success message
        self._show_user_message("Success", "Destination reached!")

        # Log completion statistics
        self._log_navigation_completion()

    def _cleanup_navigation(self):
        """Clean up navigation resources"""
        self.is_running = False
        self.system_status['navigation_active'] = False
        logger.info("Navigation stopped")

    def emergency_stop(self):
        """Enhanced emergency stop with comprehensive safety measures"""
        logger.critical("EMERGENCY STOP ACTIVATED!")

        # Set emergency flags
        self.is_running = False
        self.is_paused = False
        self.system_status['emergency_stop'] = True
        self.system_status['navigation_active'] = False

        # Clear command queue
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
                self.command_queue.task_done()
            except queue.Empty:
                break

        # Update GUI
        self._update_gui_command("EMERGENCY STOP")
        self._update_navigation_status("Emergency Stop")

        # Force GUI update
        self.update_gui_display()

        # Log emergency stop
        self._log_emergency_stop()

        # Show emergency message
        self._show_user_message("Emergency Stop",
                                "Navigation stopped due to system error!\nPlease check system status.")

    def _handle_critical_error(self, error_type: str, error: Exception):
        """Handle critical errors with appropriate recovery"""
        logger.error(f"Critical {error_type} error: {error}")
        self.performance_stats['error_count'] += 1
        self.system_status['last_error'] = f"{error_type}: {str(error)}"

        # Determine recovery action based on error type
        if error_type == "NavigationStart":
            self._show_user_message("System Error", "Failed to start navigation")
        else:
            self.emergency_stop()

    def _show_user_message(self, title: str, message: str):
        """Safely show user message with error handling"""
        try:
            if hasattr(self, 'gui') and self.gui:
                self.gui.show_message(title, message)
            else:
                print(f"{title}: {message}")
        except Exception as e:
            logger.error(f"Error showing user message: {e}")

    def _update_gui_command(self, command: str):
        """Safely update GUI command display"""
        try:
            if hasattr(self, 'gui') and self.gui:
                self.gui.set_command(command)
        except Exception as e:
            logger.debug(f"Error updating GUI command: {e}")

    def _update_navigation_status(self, status: str):
        """Safely update navigation status"""
        try:
            if hasattr(self, 'gui') and self.gui:
                self.gui.update_navigation_status(status)
        except Exception as e:
            logger.debug(f"Error updating navigation status: {e}")

    def get_combined_obstacles_grid(self):
        """Get combined obstacles grid including permanent, dynamic, and temporary obstacles"""
        try:
            # Start with permanent obstacles from map
            if self.map_manager and hasattr(self.map_manager, 'get_grid_data'):
                base_grid = self.map_manager.get_grid_data()
                grid = base_grid.copy() if base_grid is not None else np.zeros(
                    (config['GRID_HEIGHT'], config['GRID_WIDTH']), dtype=int)
            else:
                grid = np.zeros((config['GRID_HEIGHT'], config['GRID_WIDTH']), dtype=int)

            # Add dynamic obstacles from camera
            with self.obstacle_lock:
                for obstacle in self.dynamic_obstacles:
                    x, y = obstacle
                    if 0 <= x < config['GRID_WIDTH'] and 0 <= y < config['GRID_HEIGHT']:
                        grid[y, x] = 1

                # Add temporary obstacles
                for obstacle in self.temporary_obstacles:
                    x, y = obstacle
                    if 0 <= x < config['GRID_WIDTH'] and 0 <= y < config['GRID_HEIGHT']:
                        grid[y, x] = 1

                # Add simulated obstacles in simulation mode
                if self.simulation_mode:
                    for obstacle in self.simulated_obstacles:
                        x, y = obstacle
                        if 0 <= x < config['GRID_WIDTH'] and 0 <= y < config['GRID_HEIGHT']:
                            grid[y, x] = 1

            return grid
        except Exception as e:
            logger.error(f"Error creating combined obstacles grid: {e}")
            return np.zeros((config['GRID_HEIGHT'], config['GRID_WIDTH']), dtype=int)

    def generate_movement_command(self, old_pos: Tuple[int, int], new_pos: Tuple[int, int]) -> str:
        """Generate movement command based on position change"""
        if old_pos is None or new_pos is None:
            return "STOP"

        dx = new_pos[0] - old_pos[0]
        dy = new_pos[1] - old_pos[1]

        if dx == 1 and dy == 0:
            return "RIGHT"
        elif dx == -1 and dy == 0:
            return "LEFT"
        elif dx == 0 and dy == -1:
            return "UP"
        elif dx == 0 and dy == 1:
            return "DOWN"
        elif dx == 0 and dy == 0:
            return "STOP"
        else:
            return "FORWARD"  # For diagonal or complex movements

    def has_reached_destination(self) -> bool:
        """Check if current position is at destination"""
        with self.state_lock:
            if self.current_position is None or self.destination is None:
                return False
            return (self.current_position[0] == self.destination[0] and
                    self.current_position[1] == self.destination[1])

    def update_gui_display(self):
        """Enhanced GUI display update with error handling"""
        try:
            # Get current camera frame if available
            camera_frame = None
            if self.system_status['camera_available'] and self.camera_processor:
                try:
                    frame = self.camera_processor.get_frame()
                    with self.path_lock:
                        current_path = self.path.copy() if self.path else []
                    camera_frame = self.camera_processor.draw_detection_overlay(
                        frame, self.dynamic_obstacles, current_path
                    )
                except Exception as e:
                    logger.debug(f"Error getting camera frame for GUI: {e}")

            # Use simulated obstacles in simulation mode
            with self.obstacle_lock:
                obstacles_to_show = self.dynamic_obstacles + list(self.temporary_obstacles)
                if self.simulation_mode:
                    obstacles_to_show += self.simulated_obstacles

            # Path and position snapshot
            with self.state_lock:
                current_pos = self.current_position
            path_remaining = self.path_planner.get_remaining_path() if self.path_planner else []

            # Put update in queue for GUI thread
            if not self.gui_update_queue.full():
                self.gui_update_queue.put(('display_update', {
                    'current_pos': current_pos,
                    'path': path_remaining,
                    'obstacles': obstacles_to_show,
                    'camera_frame': camera_frame
                }))
        except Exception as e:
            logger.error(f"GUI display update error: {e}")

    def process_gui_updates(self):
        """Process GUI updates with error handling"""
        try:
            # Process all pending GUI updates
            while not self.gui_update_queue.empty():
                try:
                    update_type, data = self.gui_update_queue.get_nowait()
                    if update_type == 'display_update' and hasattr(self, 'gui') and self.gui:
                        self.gui.update_display(
                            data['current_pos'],
                            data['path'],
                            data['obstacles'],
                            data['camera_frame']
                        )
                    elif update_type == 'camera_frame' and hasattr(self, 'gui') and self.gui:
                        self.gui.update_camera_feed(data)
                    self.gui_update_queue.task_done()
                except Exception as e:
                    logger.error(f"Error processing GUI update: {e}")
                    continue
        except queue.Empty:
            pass  # No updates to process
        except Exception as e:
            logger.error(f"GUI update processing error: {e}")

        # Schedule next update
        if hasattr(self, 'root') and self.root:
            try:
                self.root.after(config['REFRESH_RATE'], self.process_gui_updates)
            except Exception:
                # In case root is already destroyed
                pass

    def load_map(self, map_file: str) -> bool:
        """Public method for GUI to load a map file"""
        try:
            loaded = self._safe_load_map(map_file)
            if loaded:
                # Redraw and update GUI state
                self.update_gui_display()
            return loaded
        except Exception as e:
            logger.error(f"Load map request failed: {e}")
            return False

    def update_temporary_obstacles(self, obstacles: Any) -> None:
        """Update temporary obstacles from GUI interactions and optionally replan"""
        try:
            with self.obstacle_lock:
                # GUI may pass a set; normalize to list of tuples
                self.temporary_obstacles = [(int(x), int(y)) for x, y in obstacles]

            # If navigating or a destination is set, attempt to replan
            if self.destination is not None:
                self.plan_path()

            # Update GUI
            self.update_gui_display()
        except Exception as e:
            logger.error(f"Failed to update temporary obstacles: {e}")

    def reset_system(self) -> None:
        """Reset system state for a fresh start keeping current map"""
        try:
            # Stop any ongoing navigation
            self.stop_navigation()

            # Clear navigation state
            with self.state_lock:
                self.current_position = tuple(self.map_manager.get_start_position()) if self.map_manager else (0, 0)
                self.destination = None
                self.current_direction = "STOP"
            with self.path_lock:
                self.path = []
                if self.path_planner:
                    self.path_planner.set_path([])
            with self.obstacle_lock:
                self.dynamic_obstacles = []
                self.temporary_obstacles = []

            # Refresh GUI
            self.update_gui_display()
        except Exception as e:
            logger.error(f"System reset failed: {e}")

    def _system_health_check(self):
        """Comprehensive system health check"""
        try:
            status_checks: Dict[str, Any] = {}

            # Check camera status
            if self.camera_processor:
                camera_ok = (hasattr(self.camera_processor, 'camera') and
                             self.camera_processor.camera and
                             self.camera_processor.camera.isOpened())
                status_checks['camera'] = camera_ok
                if not camera_ok and self.system_status['camera_available']:
                    logger.warning("Camera connection lost")
                    self._enter_simulation_mode()

            # Check path planner status
            path_planner_ok = self.path_planner is not None
            status_checks['path_planner'] = path_planner_ok

            # Check map status
            map_ok = self.system_status['map_loaded']
            status_checks['map'] = map_ok

            # Check memory usage (simplified)
            status_checks['memory_ok'] = True

            # Check for system overload
            current_time = time.time()
            if not hasattr(self, 'last_health_check_time'):
                self.last_health_check_time = current_time
            else:
                time_diff = current_time - self.last_health_check_time
                status_checks['responsiveness'] = time_diff < 10  # Should be ~5 seconds
                self.last_health_check_time = current_time

            # Log status periodically
            if int(time.time()) % 30 == 0:  # Every 30 seconds
                logger.info(f"System health check: {status_checks}")
                logger.info(f"Performance stats: {self.performance_stats}")

            # Take action if critical components failed
            critical_failures = not (status_checks.get('path_planner', False) and
                                     status_checks.get('map', False))
            if critical_failures and not self.system_status['degraded_mode']:
                logger.error("Critical system failure detected!")
                self._enter_degraded_mode()

        except Exception as e:
            logger.error(f"Health check error: {e}")

    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate distance between two positions"""
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        return (dx ** 2 + dy ** 2) ** 0.5

    def _update_performance_stats(self):
        """Update performance statistics"""
        self.performance_stats['system_uptime'] = time.time() - getattr(self, 'start_time', time.time())

        # Get stats from modules
        if self.path_planner and hasattr(self.path_planner, 'get_search_statistics'):
            path_stats = self.path_planner.get_search_statistics()
            self.performance_stats['average_processing_time'] = path_stats.get('planning_time', 0.0)

        if self.camera_processor and hasattr(self.camera_processor, 'get_detection_statistics'):
            camera_stats = self.camera_processor.get_detection_statistics()
            self.performance_stats['total_frames_processed'] = camera_stats.get('frames_processed', 0)

    def _log_navigation_completion(self):
        """Log navigation completion statistics"""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'destination': self.destination,
            'total_distance': self.performance_stats['total_distance_traveled'],
            'total_replans': self.performance_stats['total_replans'],
            'total_errors': self.performance_stats['error_count'],
            'success': True
        }
        logger.info(f"Navigation completed: {stats}")

    def _log_emergency_stop(self):
        """Log emergency stop event"""
        stop_info = {
            'timestamp': datetime.now().isoformat(),
            'current_position': self.current_position,
            'destination': self.destination,
            'last_error': self.system_status['last_error'],
            'performance_stats': self.performance_stats.copy()
        }
        logger.critical(f"Emergency stop: {stop_info}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        with self.state_lock, self.obstacle_lock, self.path_lock:
            return {
                'system_status': self.system_status.copy(),
                'navigation_state': {
                    'active': self.is_running,
                    'paused': self.is_paused,
                    'current_position': self.current_position,
                    'destination': self.destination,
                    'path_length': len(self.path),
                    'current_direction': self.current_direction
                },
                'obstacles': {
                    'dynamic': len(self.dynamic_obstacles),
                    'temporary': len(self.temporary_obstacles),
                    'simulated': len(self.simulated_obstacles) if self.simulation_mode else 0
                },
                'performance': self.performance_stats.copy(),
                'simulation_mode': self.simulation_mode,
                'degraded_mode': self.system_status['degraded_mode']
            }

    def reset_navigation(self):
        """Reset navigation state"""
        with self.path_lock:
            self.path = []
        with self.state_lock:
            self.destination = None
        if self.path_planner:
            self.path_planner.set_path([])

    def stop_navigation(self):
        """Stop navigation gracefully"""
        self.is_running = False
        self.system_status['navigation_active'] = False
        logger.info("Navigation stopped by user")

    def pause_navigation(self):
        """Pause navigation"""
        self.is_paused = True
        self._update_navigation_status("Paused")
        logger.info("Navigation paused")

    def resume_navigation(self):
        """Resume paused navigation"""
        self.is_paused = False
        self._update_navigation_status("Navigating")
        logger.info("Navigation resumed")

    def run(self):
        """Start the main application"""
        self.start_time = time.time()
        logger.info("Starting Smart Wheelchair System...")
        logger.info("System features:")
        logger.info("- Comprehensive error handling and recovery")
        logger.info("- Graceful degradation on component failure")
        logger.info("- Automatic simulation mode when camera unavailable")
        logger.info("- Path smoothing and command queuing")
        logger.info("- Real-time performance monitoring")

        try:
            # Start the GUI main loop
            if hasattr(self, 'root') and self.root:
                self.root.mainloop()
            else:
                logger.error("GUI not available - cannot start system")
        except KeyboardInterrupt:
            logger.info("System shutdown requested by user")
        except Exception as e:
            logger.critical(f"Fatal error in main loop: {e}")
        finally:
            self.shutdown()

    def shutdown(self):
        """Enhanced clean shutdown of the system"""
        logger.info("Initiating system shutdown...")

        # Stop all threads
        self.is_running = False
        self.system_status['navigation_active'] = False

        # Release camera resources
        if self.camera_processor:
            try:
                self.camera_processor.release_camera()
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")

        # Wait for threads to finish with timeout
        threads_to_join = []
        if self.navigation_thread and self.navigation_thread.is_alive():
            threads_to_join.append(('Navigation', self.navigation_thread))
        if self.camera_thread and self.camera_thread.is_alive():
            threads_to_join.append(('Camera', self.camera_thread))
        if self.command_thread and self.command_thread.is_alive():
            threads_to_join.append(('Command', self.command_thread))

        for name, thread in threads_to_join:
            try:
                thread.join(timeout=3.0)
                if thread.is_alive():
                    logger.warning(f"{name} thread did not terminate gracefully")
            except Exception as e:
                logger.error(f"Error joining {name} thread: {e}")

        # Save final statistics
        self._save_final_stats()

        # Destroy GUI
        if hasattr(self, 'root') and self.root:
            try:
                self.root.destroy()
            except Exception:
                pass

        logger.info("System shutdown complete.")

    def _save_final_stats(self):
        """Save final system statistics to file"""
        try:
            stats = {
                'shutdown_time': datetime.now().isoformat(),
                'final_performance_stats': self.performance_stats.copy(),
                'system_status': self.system_status.copy(),
                'total_uptime': time.time() - getattr(self, 'start_time', time.time())
            }
            with open('system_stats.json', 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info("Final statistics saved to system_stats.json")
        except Exception as e:
            logger.error(f"Error saving final statistics: {e}")

    def _print_path_grid(self, start: Tuple[int, int], dest: Tuple[int, int], path: List[Tuple[int, int]], grid: np.ndarray) -> None:
        """Render and print an ASCII grid with obstacles and the planned path.
        Legend: '#' obstacle, '.' free, '*' path, 'S' start, 'D' destination
        """
        try:
            grid_h, grid_w = grid.shape
            # Build a set for quick membership
            path_set = set(path)
            lines = []
            for y in range(grid_h):
                row_chars = []
                for x in range(grid_w):
                    if (x, y) == tuple(start):
                        row_chars.append('S')
                    elif (x, y) == tuple(dest):
                        row_chars.append('D')
                    elif (x, y) in path_set:
                        row_chars.append('*')
                    else:
                        row_chars.append('#' if grid[y, x] == 1 else '.')
                lines.append(''.join(row_chars))
            print("\nPlanned Path Grid (S=start, D=dest, *=path, #=obstacle):")
            print('\n'.join(lines))
        except Exception as e:
            logger.debug(f"Failed to print path grid: {e}")


# Main entry point with enhanced error handling
if __name__ == "__main__":
    try:
        # Validate configuration first
        if hasattr(config, 'validate_config'):
            valid = config.validate_config()
        else:
            # If config is a dict, basic validation
            required_keys = ['GRID_WIDTH', 'GRID_HEIGHT', 'SIMULATION_SPEED', 'REFRESH_RATE']
            valid = all(k in config for k in required_keys)

        if not valid:
            logger.error("Configuration validation failed! Please check config.py")
            raise RuntimeError("Invalid configuration")

        # Create and run the application
        app = SmartWheelchairApp()
        app.run()
    except Exception as e:
        logger.critical(f"Failed to start application: {e}")
        logger.critical(traceback.format_exc())

        # Attempt to save crash information
        try:
            crash_info = {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            with open('crash_report.json', 'w') as f:
                json.dump(crash_info, f, indent=2)
            print("Crash report saved to crash_report.json")
        except Exception:
            pass
        exit(1)
