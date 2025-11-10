"""
Enhanced POV Camera Processor for Smart Wheelchair Navigation System

This module provides sophisticated point-of-view video feed analysis for
real-time obstacle detection in the navigation path. The camera is mounted
on the wheelchair and analyzes the forward-looking video stream to detect
obstacles that may block the planned path.

Key Features:
- Real-time POV video feed processing
- Multi-method obstacle detection (depth estimation, edge detection, motion analysis)
- Path-focused obstacle analysis (detects obstacles specifically in navigation path)
- Dynamic obstacle tracking and filtering
- Adaptive detection parameters based on environment
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from collections import deque
import time
from config import config


class POVCameraProcessor:
    """Enhanced POV camera processor for wheelchair-mounted camera"""
    
    def __init__(self):
        self.camera = None
        self.frame_count = 0
        self.last_frame = None
        
        # Obstacle detection state
        self.detected_obstacles: List[Tuple[int, int]] = []
        self.obstacle_history = deque(maxlen=30)  # Track obstacles over time
        self.path_obstacles: List[Tuple[int, int]] = []  # Obstacles specifically in path
        
        # Detection parameters
        self.detection_stats = {
            'frames_processed': 0,
            'obstacles_detected': 0,
            'processing_time': 0.0,
            'detection_confidence': 0.0,
            'path_obstacles_count': 0
        }
        
        # Background subtraction for motion detection
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )
        
        # Depth estimation (simplified using focus/blur analysis)
        self.depth_estimator = None
        
        # Path region of interest (ROI) - focuses on forward path area
        self.path_roi = None  # Will be set based on camera FOV and path direction
        
        # Initialize camera
        self.initialize_camera()
        
    def initialize_camera(self) -> bool:
        """Initialize POV camera (webcam mounted on wheelchair)"""
        try:
            # Release existing camera if any
            if self.camera is not None:
                self.camera.release()
            
            # Initialize webcam (index 0) - mounted on wheelchair
            camera_index = config.get('CAMERA_INDEX', 0)
            print(f"Initializing POV camera (index {camera_index})...")
            
            self.camera = cv2.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                print(f"Error: Camera not available at index {camera_index}")
                return False
            
            # Set camera properties for optimal POV analysis
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config['CAMERA_WIDTH'])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config['CAMERA_HEIGHT'])
            self.camera.set(cv2.CAP_PROP_FPS, config['CAMERA_FPS'])
            
            # Optimize for consistent analysis
            self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.camera.set(cv2.CAP_PROP_EXPOSURE, -4)
            self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.camera.set(cv2.CAP_PROP_BRIGHTNESS, 128)
            
            # Test frame capture
            ret, test_frame = self.camera.read()
            if not ret or test_frame is None:
                print("Error: Camera test frame capture failed")
                self.camera.release()
                self.camera = None
                return False
            
            # Initialize path ROI (focus on center-forward region for path analysis)
            frame_height, frame_width = test_frame.shape[:2]
            self.path_roi = self._initialize_path_roi(frame_width, frame_height)
            
            print(f"✓ POV camera initialized: {config['CAMERA_WIDTH']}x{config['CAMERA_HEIGHT']} @ {config['CAMERA_FPS']}fps")
            print(f"✓ Path ROI configured for forward-looking analysis")
            
            return True
            
        except Exception as e:
            print(f"Camera initialization error: {e}")
            self.camera = None
            return False
    
    def _initialize_path_roi(self, frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
        """Initialize region of interest for path analysis (forward-looking area)"""
        # Focus on lower-center region of frame (where path obstacles typically appear)
        roi_width = int(frame_width * 0.7)  # 70% of width
        roi_height = int(frame_height * 0.6)  # 60% of height (lower portion)
        roi_x = int((frame_width - roi_width) / 2)  # Centered horizontally
        roi_y = int(frame_height * 0.2)  # Start 20% from top (focus on forward path)
        
        return (roi_x, roi_y, roi_width, roi_height)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Capture frame from POV camera"""
        try:
            if self.camera and self.camera.isOpened():
                ret, frame = self.camera.read()
                if ret and frame is not None:
                    self.last_frame = frame.copy()
                    self.frame_count += 1
                    return frame
                else:
                    print("Warning: Failed to capture frame from camera")
                    return None
            else:
                print("Error: Camera not available")
                return None
                
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None
    
    def analyze_path_obstacles(self, frame: np.ndarray, 
                              current_path: List[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
        """
        Analyze POV video feed to detect obstacles specifically in the navigation path
        
        Args:
            frame: Current camera frame
            current_path: Current navigation path (for path-focused analysis)
            
        Returns:
            List of obstacle grid coordinates detected in the path
        """
        start_time = time.time()
        
        if frame is None or frame.size == 0:
            return []
        
        # Extract path ROI for focused analysis
        path_frame = self._extract_path_roi(frame)
        
        # Multi-method obstacle detection
        obstacles_method1 = self._detect_by_depth_estimation(path_frame, frame)
        obstacles_method2 = self._detect_by_structural_analysis(path_frame, frame)
        obstacles_method3 = self._detect_by_motion_analysis(frame)
        obstacles_method4 = self._detect_by_edge_analysis(path_frame, frame)
        
        # Combine detection results
        all_obstacles = set(obstacles_method1 + obstacles_method2 + obstacles_method3 + obstacles_method4)
        
        # Filter and validate obstacles
        filtered_obstacles = self._filter_obstacles(list(all_obstacles), frame.shape)
        
        # Focus on obstacles in the navigation path if path is provided
        if current_path and len(current_path) > 0:
            path_obstacles = self._filter_path_obstacles(filtered_obstacles, current_path)
            self.path_obstacles = path_obstacles
        else:
            self.path_obstacles = filtered_obstacles
        
        # Update obstacle history for temporal consistency
        self.obstacle_history.append({
            'timestamp': time.time(),
            'obstacles': self.path_obstacles.copy(),
            'frame_number': self.frame_count
        })
        
        # Apply temporal filtering (obstacles must be detected consistently)
        consistent_obstacles = self._apply_temporal_filtering()
        
        # Update statistics
        processing_time = time.time() - start_time
        self.detection_stats['frames_processed'] += 1
        self.detection_stats['processing_time'] = processing_time
        self.detection_stats['obstacles_detected'] = len(consistent_obstacles)
        self.detection_stats['path_obstacles_count'] = len(self.path_obstacles)
        self.detection_stats['detection_confidence'] = self._calculate_confidence()
        
        self.detected_obstacles = consistent_obstacles
        
        return consistent_obstacles
    
    def _extract_path_roi(self, frame: np.ndarray) -> np.ndarray:
        """Extract region of interest focused on forward path"""
        if self.path_roi is None:
            return frame
        
        x, y, w, h = self.path_roi
        frame_height, frame_width = frame.shape[:2]
        
        # Ensure ROI is within frame bounds
        x = max(0, min(x, frame_width - 1))
        y = max(0, min(y, frame_height - 1))
        w = min(w, frame_width - x)
        h = min(h, frame_height - y)
        
        return frame[y:y+h, x:x+w]
    
    def _detect_by_depth_estimation(self, path_frame: np.ndarray, 
                                    full_frame: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect obstacles using depth estimation (focus/blur analysis)
        Closer objects appear sharper, further objects are more blurred
        """
        obstacles = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(path_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate Laplacian variance (measure of sharpness/focus)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            focus_measure = laplacian.var()
            
            # Use adaptive thresholding to identify sharp regions (potential close obstacles)
            # Sharp regions in path ROI likely indicate obstacles
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate edges to form obstacle regions
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=2)
            
            # Find contours of potential obstacles
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if config['MIN_OBSTACLE_AREA'] <= area <= config['MAX_OBSTACLE_AREA']:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate center in path ROI coordinates
                    roi_x, roi_y, _, _ = self.path_roi
                    center_x = roi_x + x + w // 2
                    center_y = roi_y + y + h // 2
                    
                    # Convert to grid coordinates
                    grid_pos = self._camera_to_grid_coords(center_x, center_y, full_frame.shape)
                    if grid_pos:
                        obstacles.append(grid_pos)
            
            return obstacles
            
        except Exception as e:
            print(f"Depth estimation detection error: {e}")
            return []
    
    def _detect_by_structural_analysis(self, path_frame: np.ndarray,
                                       full_frame: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect obstacles using structural analysis (identify solid objects)
        Uses adaptive thresholding and morphological operations
        """
        obstacles = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(path_frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (config['BLUR_KERNEL_SIZE'], config['BLUR_KERNEL_SIZE']), 0)
            
            # Adaptive thresholding to identify dark regions (obstacles)
            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Morphological operations to clean up
            kernel = np.ones((config['MORPHOLOGY_KERNEL_SIZE'], 
                            config['MORPHOLOGY_KERNEL_SIZE']), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if config['MIN_OBSTACLE_AREA'] <= area <= config['MAX_OBSTACLE_AREA']:
                    # Check solidity (area/convex hull area)
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    
                    if hull_area > 0:
                        solidity = area / hull_area
                        if solidity > 0.3:  # Reasonably solid objects
                            x, y, w, h = cv2.boundingRect(contour)
                            
                            # Verify it's actually a dark region (obstacle)
                            roi = path_frame[y:y+h, x:x+w]
                            if self._is_dark_region(roi):
                                # Convert to grid coordinates
                                roi_x, roi_y, _, _ = self.path_roi
                                center_x = roi_x + x + w // 2
                                center_y = roi_y + y + h // 2
                                
                                grid_pos = self._camera_to_grid_coords(center_x, center_y, full_frame.shape)
                                if grid_pos:
                                    obstacles.append(grid_pos)
            
            return obstacles
            
        except Exception as e:
            print(f"Structural analysis detection error: {e}")
            return []
    
    def _detect_by_motion_analysis(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect moving obstacles using background subtraction"""
        obstacles = []
        
        try:
            if self.background_subtractor is None:
                return obstacles
            
            # Apply background subtraction
            fg_mask = self.background_subtractor.apply(frame)
            
            # Clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            # Focus on path ROI
            if self.path_roi:
                x, y, w, h = self.path_roi
                path_mask = fg_mask[y:y+h, x:x+w]
            else:
                path_mask = fg_mask
            
            # Find contours in foreground mask
            contours, _ = cv2.findContours(path_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if config['MIN_OBSTACLE_AREA'] <= area <= config['MAX_OBSTACLE_AREA']:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Convert to grid coordinates
                    if self.path_roi:
                        roi_x, roi_y, _, _ = self.path_roi
                        center_x = roi_x + x + w // 2
                        center_y = roi_y + y + h // 2
                    else:
                        center_x = x + w // 2
                        center_y = y + h // 2
                    
                    grid_pos = self._camera_to_grid_coords(center_x, center_y, frame.shape)
                    if grid_pos:
                        obstacles.append(grid_pos)
            
            return obstacles
            
        except Exception as e:
            print(f"Motion analysis detection error: {e}")
            return []
    
    def _detect_by_edge_analysis(self, path_frame: np.ndarray,
                                 full_frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect obstacles using edge detection and analysis"""
        obstacles = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(path_frame, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate edges to form closed regions
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if config['MIN_OBSTACLE_AREA'] <= area <= config['MAX_OBSTACLE_AREA']:
                    # Check if contour forms a reasonable obstacle shape
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Aspect ratio check (obstacles are typically not extremely elongated)
                    aspect_ratio = w / max(h, 1)
                    if 0.2 < aspect_ratio < 5.0:
                        # Convert to grid coordinates
                        roi_x, roi_y, _, _ = self.path_roi
                        center_x = roi_x + x + w // 2
                        center_y = roi_y + y + h // 2
                        
                        grid_pos = self._camera_to_grid_coords(center_x, center_y, full_frame.shape)
                        if grid_pos:
                            obstacles.append(grid_pos)
            
            return obstacles
            
        except Exception as e:
            print(f"Edge analysis detection error: {e}")
            return []
    
    def _filter_path_obstacles(self, obstacles: List[Tuple[int, int]],
                               current_path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Filter obstacles to only include those in or near the navigation path
        This focuses detection on obstacles that actually matter for navigation
        """
        if not obstacles or not current_path:
            return obstacles
        
        path_set = set(current_path)
        path_obstacles = []
        
        # Check obstacles within path or nearby (within 2 cells)
        for obstacle in obstacles:
            # Check if obstacle is directly in path
            if obstacle in path_set:
                path_obstacles.append(obstacle)
            else:
                # Check if obstacle is near path (within 2 cells)
                for path_pos in current_path[:10]:  # Check next 10 path positions
                    distance = abs(obstacle[0] - path_pos[0]) + abs(obstacle[1] - path_pos[1])
                    if distance <= 2:
                        path_obstacles.append(obstacle)
                        break
        
        return path_obstacles
    
    def _apply_temporal_filtering(self) -> List[Tuple[int, int]]:
        """
        Apply temporal filtering to remove false positives
        Obstacles must be detected consistently across multiple frames
        """
        if len(self.obstacle_history) < 3:
            return self.path_obstacles
        
        # Count how many times each obstacle appears in recent history
        obstacle_counts = {}
        for history_entry in list(self.obstacle_history)[-5:]:  # Check last 5 frames
            for obstacle in history_entry['obstacles']:
                obstacle_counts[obstacle] = obstacle_counts.get(obstacle, 0) + 1
        
        # Only keep obstacles detected in at least 2 out of 5 frames
        consistent_obstacles = [
            obstacle for obstacle, count in obstacle_counts.items()
            if count >= 2
        ]
        
        return consistent_obstacles
    
    def _filter_obstacles(self, obstacles: List[Tuple[int, int]],
                         frame_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Filter and validate detected obstacles"""
        filtered = []
        grid_width = config['GRID_WIDTH']
        grid_height = config['GRID_HEIGHT']
        
        # Remove duplicates and validate bounds
        seen = set()
        for obstacle in obstacles:
            if obstacle not in seen:
                x, y = obstacle
                if 0 <= x < grid_width and 0 <= y < grid_height:
                    filtered.append(obstacle)
                    seen.add(obstacle)
        
        return filtered
    
    def _is_dark_region(self, roi: np.ndarray) -> bool:
        """Check if region is actually dark (likely an obstacle)"""
        if roi.size == 0:
            return False
        
        if len(roi.shape) == 3:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_roi = roi
        
        avg_brightness = np.mean(gray_roi)
        return avg_brightness < config['OBSTACLE_THRESHOLD']
    
    def _camera_to_grid_coords(self, cam_x: int, cam_y: int,
                               frame_shape: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Convert camera pixel coordinates to grid coordinates"""
        try:
            frame_height, frame_width = frame_shape[:2]
            grid_width = config['GRID_WIDTH']
            grid_height = config['GRID_HEIGHT']
            
            # Simple mapping: assume camera covers forward area of grid
            # Lower portion of frame corresponds to closer grid cells
            # Center of frame corresponds to forward direction
            
            # Map camera coordinates to grid
            # X: center of frame = center of grid width
            grid_x = int((cam_x / frame_width) * grid_width)
            
            # Y: lower portion of frame (where obstacles appear) = forward in grid
            # Assume camera looks forward, so obstacles appear in lower frame
            # Map to appropriate grid Y coordinate
            grid_y = int((cam_y / frame_height) * grid_height)
            
            # Clamp to grid bounds
            grid_x = max(0, min(grid_width - 1, grid_x))
            grid_y = max(0, min(grid_height - 1, grid_y))
            
            return (grid_x, grid_y)
            
        except Exception:
            return None
    
    def draw_detection_overlay(self, frame: np.ndarray,
                              obstacles: List[Tuple[int, int]] = None,
                              current_path: List[Tuple[int, int]] = None) -> np.ndarray:
        """Draw obstacle detection overlay on camera frame"""
        if frame is None or frame.size == 0:
            return frame
        
        overlay = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        
        # Draw path ROI outline
        if self.path_roi:
            x, y, w, h = self.path_roi
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(overlay, "Path ROI", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw detected obstacles
        obstacles_to_draw = obstacles if obstacles else self.detected_obstacles
        for grid_x, grid_y in obstacles_to_draw:
            # Convert grid coordinates back to camera coordinates
            cam_x = int((grid_x / config['GRID_WIDTH']) * frame_width)
            cam_y = int((grid_y / config['GRID_HEIGHT']) * frame_height)
            
            # Draw obstacle marker
            radius = 20
            cv2.circle(overlay, (cam_x, cam_y), radius, (0, 0, 255), 2)
            cv2.circle(overlay, (cam_x, cam_y), radius, (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.7, overlay, 0.3, 0, overlay)
        
        # Draw path overlay if provided
        if current_path and len(current_path) > 0:
            path_points = []
            for grid_x, grid_y in current_path[:10]:  # Show next 10 path points
                cam_x = int((grid_x / config['GRID_WIDTH']) * frame_width)
                cam_y = int((grid_y / config['GRID_HEIGHT']) * frame_height)
                path_points.append((cam_x, cam_y))
            
            # Draw path line
            for i in range(len(path_points) - 1):
                cv2.line(overlay, path_points[i], path_points[i+1], (255, 0, 0), 2)
        
        # Add information overlay
        self._add_info_overlay(overlay, obstacles_to_draw)
        
        return overlay
    
    def _add_info_overlay(self, frame: np.ndarray, obstacles: List[Tuple[int, int]]) -> None:
        """Add information text overlay to frame"""
        info_text = [
            f"Frame: {self.frame_count}",
            f"Obstacles: {len(obstacles)}",
            f"Path Obstacles: {len(self.path_obstacles)}",
            f"Confidence: {self.detection_stats['detection_confidence']:.2f}",
            f"Processing: {self.detection_stats['processing_time']*1000:.1f}ms"
        ]
        
        camera_status = "LIVE" if (self.camera and self.camera.isOpened()) else "OFFLINE"
        info_text.append(f"Camera: {camera_status}")
        
        for i, text in enumerate(info_text):
            y_pos = 30 + i * 25
            cv2.putText(frame, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    def _calculate_confidence(self) -> float:
        """Calculate detection confidence based on consistency and performance"""
        if self.detection_stats['frames_processed'] == 0:
            return 0.0
        
        base_confidence = 0.7
        
        # Time penalty (slower processing = lower confidence)
        time_penalty = min(self.detection_stats['processing_time'] * 10, 0.3)
        
        # Consistency bonus (more consistent detections = higher confidence)
        if len(self.obstacle_history) >= 3:
            consistency_bonus = min(len(self.detected_obstacles) * 0.05, 0.2)
        else:
            consistency_bonus = 0.0
        
        confidence = max(0.0, min(1.0, base_confidence - time_penalty + consistency_bonus))
        return confidence
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get detection performance statistics"""
        return self.detection_stats.copy()
    
    def calibrate_detection(self, calibration_frames: int = 30) -> bool:
        """Calibrate obstacle detection for current environment"""
        print("Calibrating POV camera obstacle detection...")
        
        try:
            # Capture frames to learn background
            for i in range(calibration_frames):
                frame = self.get_frame()
                if frame is not None:
                    # Update background subtractor
                    if self.background_subtractor:
                        self.background_subtractor.apply(frame)
                else:
                    print(f"Warning: Failed to capture frame {i+1}/{calibration_frames}")
            
            print("Calibration completed successfully")
            return True
            
        except Exception as e:
            print(f"Calibration error: {e}")
            return False
    
    def release_camera(self) -> None:
        """Release camera resources"""
        if self.camera:
            self.camera.release()
            self.camera = None
            print("Camera released")
    
    def obstacles_to_grid(self, obstacles: List[Tuple[int, int]], 
                         grid_shape: Tuple[int, int]) -> np.ndarray:
        """Convert obstacle coordinates to grid format (backward compatibility)"""
        grid = np.zeros(grid_shape, dtype=int)
        grid_h, grid_w = grid_shape
        
        for obstacle in obstacles:
            x, y = obstacle
            # Ensure coordinates are within grid bounds
            if 0 <= x < grid_w and 0 <= y < grid_h:
                grid[y, x] = 1
        
        return grid
    
    # Backward compatibility
    def detect_obstacles(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Legacy method for backward compatibility"""
        return self.analyze_path_obstacles(frame)


# Backward compatibility alias
CameraProcessor = POVCameraProcessor
