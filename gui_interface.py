"""
GUI Interface for Smart Wheelchair Navigation System
Dual-panel layout with Tkinter for real-time visualization
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from PIL import Image, ImageTk
import cv2
import json
import os
from typing import List, Tuple, Optional, Dict, Any
from config import config

class WheelchairGUI:
    """Main GUI interface for smart wheelchair navigation system"""
    
    def __init__(self, root, app_controller):
        self.root = root
        self.controller = app_controller
        self.canvas = None
        self.camera_label = None
        
        # GUI state variables
        self.destination = None
        self.temporary_obstacles = set()
        self.current_position = None
        self.path = []
        self.dynamic_obstacles = []
        
        # Camera display variables
        self.current_camera_image = None
        
        # Setup the GUI
        self.setup_gui()
        
        print("GUI initialized successfully")
    
    def setup_gui(self):
        """Setup the main GUI window and components"""
        # Configure main window
        self.root.title(config['GUI_TITLE'])
        self.root.geometry(f"{config['GUI_WIDTH']}x{config['GUI_HEIGHT']}")
        self.root.configure(bg='#f0f0f0')
        
        # Make window resizable
        self.root.minsize(1000, 700)
        
        # Create main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create top section for panels
        top_section = ttk.Frame(main_container)
        top_section.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Setup left and right panels
        self.setup_left_panel(top_section)
        self.setup_right_panel(top_section)
        
        # Setup control panel
        self.setup_control_panel(main_container)
        
        # Setup status panel
        self.setup_status_panel(main_container)
        
        # Draw initial grid
        self.draw_grid()
    
    def setup_left_panel(self, parent):
        """Setup left panel for POV camera feed display"""
        # Left frame for POV camera feed (wheelchair-mounted camera)
        left_frame = ttk.LabelFrame(parent, text="POV Camera Feed - Path Obstacle Detection", 
                                   padding=15)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Camera display area
        camera_container = ttk.Frame(left_frame)
        camera_container.pack(fill=tk.BOTH, expand=True)
        
        # Camera label for displaying video feed
        self.camera_label = ttk.Label(camera_container, background='black', 
                                     relief='sunken', borderwidth=2)
        self.camera_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Camera info panel
        camera_info_frame = ttk.Frame(left_frame)
        camera_info_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Camera status
        self.camera_status_var = tk.StringVar(value="POV Camera: Initializing...")
        ttk.Label(camera_info_frame, textvariable=self.camera_status_var,
                 font=("Arial", 10)).pack(side=tk.LEFT)
        
        # Path obstacles count (more relevant than total obstacles)
        self.path_obstacles_var = tk.StringVar(value="Path Obstacles: 0")
        ttk.Label(camera_info_frame, textvariable=self.path_obstacles_var,
                 font=("Arial", 10, "bold"), foreground="red").pack(side=tk.RIGHT)
        
        # Total obstacle count
        self.obstacle_count_var = tk.StringVar(value="Total Obstacles: 0")
        ttk.Label(camera_info_frame, textvariable=self.obstacle_count_var,
                 font=("Arial", 10)).pack(side=tk.RIGHT, padx=(0, 20))
        
        # Detection confidence
        self.confidence_var = tk.StringVar(value="Confidence: 0%")
        ttk.Label(camera_info_frame, textvariable=self.confidence_var,
                 font=("Arial", 10)).pack(side=tk.RIGHT, padx=(0, 20))
    
    def setup_right_panel(self, parent):
        """Setup right panel for interactive floor map"""
        # Right frame for floor map showing destination-oriented navigation
        right_frame = ttk.LabelFrame(parent, text="Destination-Oriented Navigation Map", padding=15)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Map controls
        map_controls = ttk.Frame(right_frame)
        map_controls.pack(fill=tk.X, pady=(0, 10))
        
        # Map selection
        ttk.Label(map_controls, text="Map:").pack(side=tk.LEFT)
        self.map_var = tk.StringVar(value="hospital_floor1.json")
        self.map_combo = ttk.Combobox(map_controls, textvariable=self.map_var,
                                     values=["hospital_floor1.json", "office_floor1.json"],
                                     state="readonly", width=20)
        self.map_combo.pack(side=tk.LEFT, padx=5)
        self.map_combo.bind('<<ComboboxSelected>>', self.on_map_selected)
        
        # Landmark display
        self.landmark_var = tk.StringVar(value="Landmarks: None")
        ttk.Label(map_controls, textvariable=self.landmark_var,
                 font=("Arial", 9)).pack(side=tk.RIGHT)
        
        # Canvas for grid display
        canvas_frame = ttk.Frame(right_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create scrollable canvas
        self._setup_scrollable_canvas(canvas_frame)
        
        # Instructions
        instructions = ttk.Frame(right_frame)
        instructions.pack(fill=tk.X, pady=(10, 0))
        
        instruction_text = "Left-click: Set Destination | Right-click: Toggle Obstacle | POV Camera analyzes forward path"
        ttk.Label(instructions, text=instruction_text, font=("Arial", 9),
                 foreground="gray").pack()
    
    def _setup_scrollable_canvas(self, parent):
        """Setup scrollable canvas for large grids"""
        # Create frame for canvas and scrollbars
        canvas_container = ttk.Frame(parent)
        canvas_container.pack(fill=tk.BOTH, expand=True)
        
        # Calculate canvas size
        canvas_width = config['GRID_WIDTH'] * config['CELL_SIZE']
        canvas_height = config['GRID_HEIGHT'] * config['CELL_SIZE']
        
        # Create canvas
        self.canvas = tk.Canvas(canvas_container, 
                               width=canvas_width,
                               height=canvas_height,
                               bg=config['COLORS']['background'],
                               highlightthickness=1,
                               highlightbackground="#cccccc")
        
        # Add scrollbars if canvas is larger than available space
        if canvas_width > 400 or canvas_height > 400:
            v_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL, 
                                       command=self.canvas.yview)
            h_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL,
                                       command=self.canvas.xview)
            
            self.canvas.configure(yscrollcommand=v_scrollbar.set,
                                 xscrollcommand=h_scrollbar.set)
            
            # Grid layout for scrollbars
            self.canvas.grid(row=0, column=0, sticky='nsew')
            v_scrollbar.grid(row=0, column=1, sticky='ns')
            h_scrollbar.grid(row=1, column=0, sticky='ew')
            
            canvas_container.grid_rowconfigure(0, weight=1)
            canvas_container.grid_columnconfigure(0, weight=1)
        else:
            self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind click events
        self.canvas.bind("<Button-1>", self.on_map_click)  # Left click for destination
        self.canvas.bind("<Button-3>", self.on_map_right_click)  # Right click for obstacles
        self.canvas.bind("<Motion>", self.on_mouse_move)  # Mouse movement for coordinates
    
    def setup_control_panel(self, parent):
        """Setup control buttons panel"""
        control_frame = ttk.LabelFrame(parent, text="Navigation Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Main control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        # Navigation controls
        nav_frame = ttk.Frame(button_frame)
        nav_frame.pack(side=tk.LEFT)
        
        ttk.Button(nav_frame, text="Start Navigation", 
                  command=self.start_navigation,
                  style="Accent.TButton").pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Stop", 
                  command=self.stop_navigation).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Pause", 
                  command=self.pause_navigation).pack(side=tk.LEFT, padx=2)
        
        # Map controls
        map_frame = ttk.Frame(button_frame)
        map_frame.pack(side=tk.LEFT, padx=20)
        
        ttk.Button(map_frame, text="Load Map", 
                  command=self.load_map_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(map_frame, text="Save Map", 
                  command=self.save_map_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(map_frame, text="New Map", 
                  command=self.new_map_dialog).pack(side=tk.LEFT, padx=2)
        
        # System controls
        system_frame = ttk.Frame(button_frame)
        system_frame.pack(side=tk.RIGHT)
        
        ttk.Button(system_frame, text="Reset System", 
                  command=self.reset_system).pack(side=tk.LEFT, padx=2)
        ttk.Button(system_frame, text="Calibrate Camera", 
                  command=self.calibrate_camera).pack(side=tk.LEFT, padx=2)
        
        # Configuration controls
        config_frame = ttk.Frame(control_frame)
        config_frame.pack(fill=tk.X, pady=5)
        
        # Speed control
        ttk.Label(config_frame, text="Speed:").pack(side=tk.LEFT)
        self.speed_var = tk.DoubleVar(value=config['SIMULATION_SPEED'])
        speed_scale = ttk.Scale(config_frame, from_=0.1, to=2.0, 
                               variable=self.speed_var, orient=tk.HORIZONTAL,
                               command=self.on_speed_change)
        speed_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Detection sensitivity
        ttk.Label(config_frame, text="Sensitivity:").pack(side=tk.LEFT, padx=(20, 5))
        self.sensitivity_var = tk.IntVar(value=config['OBSTACLE_THRESHOLD'])
        sensitivity_scale = ttk.Scale(config_frame, from_=50, to=200,
                                     variable=self.sensitivity_var, orient=tk.HORIZONTAL,
                                     command=self.on_sensitivity_change)
        sensitivity_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    
    def setup_status_panel(self, parent):
        """Setup status information panel"""
        status_frame = ttk.LabelFrame(parent, text="System Status", padding=10)
        status_frame.pack(fill=tk.X)
        
        # Top status row
        top_status = ttk.Frame(status_frame)
        top_status.pack(fill=tk.X, pady=2)
        
        # Current command display
        ttk.Label(top_status, text="Current Command:", 
                 font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        self.command_var = tk.StringVar(value="READY")
        command_label = ttk.Label(top_status, textvariable=self.command_var,
                                 font=("Arial", 12, "bold"),
                                 foreground="blue")
        command_label.pack(side=tk.LEFT, padx=10)
        
        # Navigation status
        self.nav_status_var = tk.StringVar(value="Status: Idle")
        ttk.Label(top_status, textvariable=self.nav_status_var,
                 font=("Arial", 10)).pack(side=tk.RIGHT)
        
        # Bottom status row
        bottom_status = ttk.Frame(status_frame)
        bottom_status.pack(fill=tk.X, pady=2)
        
        # Position display
        self.position_var = tk.StringVar(value="Position: (0, 0)")
        ttk.Label(bottom_status, textvariable=self.position_var,
                 font=("Arial", 9)).pack(side=tk.LEFT)
        
        # Destination display
        self.destination_var = tk.StringVar(value="Destination: Not set")
        ttk.Label(bottom_status, textvariable=self.destination_var,
                 font=("Arial", 9)).pack(side=tk.LEFT, padx=20)
        
        # Path information
        self.path_info_var = tk.StringVar(value="Path: No path")
        ttk.Label(bottom_status, textvariable=self.path_info_var,
                 font=("Arial", 9)).pack(side=tk.RIGHT)
    
    def draw_grid(self):
        """Draw the grid on canvas with all elements"""
        if self.canvas is None:
            return
        
        self.canvas.delete("all")
        
        # Draw grid lines if enabled
        if config['SHOW_GRID_LINES']:
            for i in range(config['GRID_WIDTH'] + 1):
                x = i * config['CELL_SIZE']
                self.canvas.create_line(x, 0, x, config['GRID_HEIGHT'] * config['CELL_SIZE'], 
                                       fill=config['COLORS']['grid_line'], width=1)
            
            for i in range(config['GRID_HEIGHT'] + 1):
                y = i * config['CELL_SIZE']
                self.canvas.create_line(0, y, config['GRID_WIDTH'] * config['CELL_SIZE'], y, 
                                       fill=config['COLORS']['grid_line'], width=1)
        
        # Draw permanent obstacles from map
        if self.controller.map_manager and self.controller.map_manager.grid_data is not None:
            grid_data = self.controller.map_manager.grid_data
            for y in range(config['GRID_HEIGHT']):
                for x in range(config['GRID_WIDTH']):
                    if grid_data[y, x] == 1:
                        self.draw_cell(x, y, config['COLORS']['permanent_obs'], "permanent")
        
        # Draw temporary obstacles
        for x, y in self.temporary_obstacles:
            self.draw_cell(x, y, config['COLORS']['obstacle'], "temporary")
        
        # Draw dynamic obstacles from camera
        for x, y in self.dynamic_obstacles:
            self.draw_cell(x, y, config['COLORS']['dynamic_obs'], "dynamic")
        
        # Draw path
        if self.path:
            for i, (x, y) in enumerate(self.path):
                # Use different shades for path visualization
                if i == 0:
                    continue  # Skip start position (will be drawn separately)
                path_color = self._get_path_color(i, len(self.path))
                self.draw_cell(x, y, path_color, "path")
        
        # Draw destination
        if self.destination:
            x, y = self.destination
            self.draw_cell(x, y, config['COLORS']['destination'], "destination")
            # Add destination marker
            self._draw_destination_marker(x, y)
        
        # Draw current position
        if self.current_position:
            x, y = self.current_position
            self.draw_cell(x, y, config['COLORS']['current_pos'], "position")
            # Add position indicator
            self._draw_position_indicator(x, y)
        
        # Draw start position
        start_pos = self.controller.map_manager.get_start_position() if self.controller.map_manager else [2, 2]
        if start_pos:
            x, y = start_pos
            self.draw_cell(x, y, config['COLORS']['start'], "start")
        
        # Draw landmarks
        landmarks = self.controller.map_manager.get_landmarks() if self.controller.map_manager else {}
        for name, (x, y) in landmarks.items():
            self._draw_landmark(x, y, name)
    
    def draw_cell(self, x: int, y: int, color: str, cell_type: str = ""):
        """Draw a cell on the grid with appropriate styling"""
        x1 = x * config['CELL_SIZE']
        y1 = y * config['CELL_SIZE']
        x2 = x1 + config['CELL_SIZE']
        y2 = y1 + config['CELL_SIZE']
        
        # Create rectangle with appropriate border
        border_width = 1
        if cell_type in ["destination", "position"]:
            border_width = 2
        
        self.canvas.create_rectangle(x1, y1, x2, y2, 
                                   fill=color, 
                                   outline=config['COLORS']['grid_line'],
                                   width=border_width,
                                   tags=cell_type)
    
    def _draw_destination_marker(self, x: int, y: int):
        """Draw a special marker for destination"""
        center_x = x * config['CELL_SIZE'] + config['CELL_SIZE'] // 2
        center_y = y * config['CELL_SIZE'] + config['CELL_SIZE'] // 2
        radius = config['CELL_SIZE'] // 3
        
        self.canvas.create_oval(center_x - radius, center_y - radius,
                               center_x + radius, center_y + radius,
                               fill=config['COLORS']['destination'],
                               outline="white", width=2)
        
        # Add cross marker
        cross_size = radius // 2
        self.canvas.create_line(center_x - cross_size, center_y,
                               center_x + cross_size, center_y,
                               fill="white", width=2)
        self.canvas.create_line(center_x, center_y - cross_size,
                               center_x, center_y + cross_size,
                               fill="white", width=2)
    
    def _draw_position_indicator(self, x: int, y: int):
        """Draw a special indicator for current position"""
        center_x = x * config['CELL_SIZE'] + config['CELL_SIZE'] // 2
        center_y = y * config['CELL_SIZE'] + config['CELL_SIZE'] // 2
        radius = config['CELL_SIZE'] // 3
        
        # Draw direction indicator if available
        if hasattr(self.controller, 'current_direction'):
            direction = self.controller.current_direction
            self._draw_direction_indicator(center_x, center_y, radius, direction)
        else:
            # Simple circle for position
            self.canvas.create_oval(center_x - radius, center_y - radius,
                                   center_x + radius, center_y + radius,
                                   fill=config['COLORS']['current_pos'],
                                   outline="black", width=1)
    
    def _draw_direction_indicator(self, center_x: int, center_y: int, 
                                 radius: int, direction: str):
        """Draw direction indicator for current position"""
        if direction == "RIGHT":
            points = [
                center_x - radius, center_y - radius,
                center_x + radius, center_y,
                center_x - radius, center_y + radius
            ]
        elif direction == "LEFT":
            points = [
                center_x + radius, center_y - radius,
                center_x - radius, center_y,
                center_x + radius, center_y + radius
            ]
        elif direction == "UP":
            points = [
                center_x - radius, center_y + radius,
                center_x, center_y - radius,
                center_x + radius, center_y + radius
            ]
        elif direction == "DOWN":
            points = [
                center_x - radius, center_y - radius,
                center_x, center_y + radius,
                center_x + radius, center_y - radius
            ]
        else:  # STOP or unknown
            points = [
                center_x - radius, center_y - radius,
                center_x + radius, center_y - radius,
                center_x + radius, center_y + radius,
                center_x - radius, center_y + radius
            ]
        
        self.canvas.create_polygon(points, fill=config['COLORS']['current_pos'],
                                  outline="black", width=1)
    
    def _draw_landmark(self, x: int, y: int, name: str):
        """Draw a landmark on the map"""
        center_x = x * config['CELL_SIZE'] + config['CELL_SIZE'] // 2
        center_y = y * config['CELL_SIZE'] + config['CELL_SIZE'] // 2
        size = config['CELL_SIZE'] // 2
        
        # Different shapes for different landmark types
        if "elevator" in name.lower():
            # Rectangle for elevator
            self.canvas.create_rectangle(center_x - size, center_y - size,
                                       center_x + size, center_y + size,
                                       fill="purple", outline="white", width=1)
        elif "reception" in name.lower():
            # Triangle for reception
            points = [
                center_x, center_y - size,
                center_x - size, center_y + size,
                center_x + size, center_y + size
            ]
            self.canvas.create_polygon(points, fill="orange", outline="white", width=1)
        else:
            # Diamond for other landmarks
            points = [
                center_x, center_y - size,
                center_x + size, center_y,
                center_x, center_y + size,
                center_x - size, center_y
            ]
            self.canvas.create_polygon(points, fill="cyan", outline="white", width=1)
        
        # Add landmark label
        if config['CELL_SIZE'] >= 15:  # Only show labels if cells are large enough
            self.canvas.create_text(center_x, center_y + size + 8,
                                   text=name.split('_')[0],  # Shortened name
                                   fill=config['COLORS']['text'],
                                   font=("Arial", 7))
    
    def _get_path_color(self, index: int, total_length: int) -> str:
        """Get color for path segment based on position in path"""
        # Create gradient from light blue to dark blue
        base_color = config['COLORS']['path']
        if total_length <= 1:
            return base_color
        
        # Calculate intensity based on position in path
        intensity = 0.3 + 0.7 * (index / total_length)
        
        # Convert hex to RGB
        hex_color = base_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Adjust brightness
        adjusted_rgb = tuple(int(c * intensity) for c in rgb)
        
        # Convert back to hex
        return f'#{adjusted_rgb[0]:02x}{adjusted_rgb[1]:02x}{adjusted_rgb[2]:02x}'
    
    def update_display(self, current_pos: Tuple[int, int], path: List[Tuple[int, int]], 
                      obstacles: List[Tuple[int, int]], camera_frame: np.ndarray = None):
        """Update the entire display with current state"""
        # Update internal state
        self.current_position = current_pos
        self.path = path
        self.dynamic_obstacles = obstacles
        
        # Redraw grid
        self.draw_grid()
        
        # Update camera feed
        if camera_frame is not None:
            self.update_camera_feed(camera_frame)
        
        # Update status information
        self.update_status()
    
    def update_camera_feed(self, frame: np.ndarray):
        """Update the camera feed display"""
        if frame is None or frame.size == 0:
            return
        
        try:
            # Resize frame to fit panel while maintaining aspect ratio
            height, width = frame.shape[:2]
            max_width = 600  # Maximum width for display
            max_height = 400  # Maximum height for display
            
            # Calculate scaling factor
            scale_x = max_width / width
            scale_y = max_height / height
            scale = min(scale_x, scale_y)
            
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize frame
            resized_frame = cv2.resize(frame, (new_width, new_height))
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_frame)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image=pil_image)
            
            # Update label
            self.camera_label.configure(image=photo)
            self.camera_label.image = photo  # Keep a reference
            
        except Exception as e:
            print(f"Error updating camera feed: {e}")
    
    def update_status(self):
        """Update status information displays"""
        # Update position
        if self.current_position:
            x, y = self.current_position
            self.position_var.set(f"Position: ({x}, {y})")
        else:
            self.position_var.set("Position: Unknown")
        
        # Update destination
        if self.destination:
            x, y = self.destination
            self.destination_var.set(f"Destination: ({x}, {y})")
        else:
            self.destination_var.set("Destination: Not set")
        
        # Update path info
        if self.path:
            remaining_steps = len(self.path)
            self.path_info_var.set(f"Path: {remaining_steps} steps remaining")
        else:
            self.path_info_var.set("Path: No path")
        
        # Update obstacle counts
        total_obstacles = len(self.dynamic_obstacles) + len(self.temporary_obstacles)
        self.obstacle_count_var.set(f"Total Obstacles: {total_obstacles}")
        
        # Update path obstacles count (from POV camera analysis)
        if hasattr(self.controller.camera_processor, 'path_obstacles'):
            path_obstacles_count = len(self.controller.camera_processor.path_obstacles)
            self.path_obstacles_var.set(f"Path Obstacles: {path_obstacles_count}")
        else:
            self.path_obstacles_var.set(f"Path Obstacles: 0")
        
        # Update camera status
        camera_status = "LIVE" if (self.controller.camera_processor and 
                                 self.controller.camera_processor.camera and 
                                 self.controller.camera_processor.camera.isOpened()) else "SIMULATED"
        self.camera_status_var.set(f"POV Camera: {camera_status}")
        
        # Update confidence if available
        if hasattr(self.controller.camera_processor, 'detection_stats'):
            confidence = self.controller.camera_processor.detection_stats.get('detection_confidence', 0)
            self.confidence_var.set(f"Confidence: {confidence:.1%}")
    
    def on_map_click(self, event):
        """Handle map click for destination setting"""
        # Adjust for scroll position if applicable
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        grid_x = int(x // config['CELL_SIZE'])
        grid_y = int(y // config['CELL_SIZE'])
        
        if 0 <= grid_x < config['GRID_WIDTH'] and 0 <= grid_y < config['GRID_HEIGHT']:
            self.destination = (grid_x, grid_y)
            self.controller.set_destination(self.destination)
            print(f"Destination set to: ({grid_x}, {grid_y})")
            
            # Provide visual feedback
            self._show_click_feedback(grid_x, grid_y, "destination")
    
    def on_map_right_click(self, event):
        """Handle right click for obstacle toggling"""
        # Adjust for scroll position if applicable
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        grid_x = int(x // config['CELL_SIZE'])
        grid_y = int(y // config['CELL_SIZE'])
        
        if 0 <= grid_x < config['GRID_WIDTH'] and 0 <= grid_y < config['GRID_HEIGHT']:
            obstacle = (grid_x, grid_y)
            if obstacle in self.temporary_obstacles:
                self.temporary_obstacles.remove(obstacle)
                print(f"Temporary obstacle removed at: ({grid_x}, {grid_y})")
            else:
                self.temporary_obstacles.add(obstacle)
                print(f"Temporary obstacle added at: ({grid_x}, {grid_y})")
            
            self.controller.update_temporary_obstacles(self.temporary_obstacles)
            
            # Provide visual feedback
            self._show_click_feedback(grid_x, grid_y, "obstacle")
    
    def on_mouse_move(self, event):
        """Handle mouse movement for coordinate display"""
        # Could be used to show coordinates in status bar
        pass
    
    def _show_click_feedback(self, x: int, y: int, click_type: str):
        """Show visual feedback for map clicks"""
        center_x = x * config['CELL_SIZE'] + config['CELL_SIZE'] // 2
        center_y = y * config['CELL_SIZE'] + config['CELL_SIZE'] // 2
        radius = config['CELL_SIZE'] // 2
        
        if click_type == "destination":
            color = "green"
        else:  # obstacle
            color = "red"
        
        # Create a temporary circle that fades out
        feedback = self.canvas.create_oval(center_x - radius, center_y - radius,
                                         center_x + radius, center_y + radius,
                                         outline=color, width=2)
        
        # Animate the feedback
        self._animate_feedback(feedback, 0)
    
    def _animate_feedback(self, item, alpha):
        """Animate click feedback"""
        if alpha > 10:
            self.canvas.delete(item)
            return
        
        current_color = self.canvas.itemcget(item, "outline")
        self.canvas.itemconfig(item, outline=current_color)
        self.canvas.after(50, lambda: self._animate_feedback(item, alpha + 1))
    
    def on_map_selected(self, event):
        """Handle map selection from dropdown"""
        map_file = self.map_var.get()
        if map_file:
            self.controller.load_map(map_file)
            self.update_landmark_display()
    
    def update_landmark_display(self):
        """Update landmark information display"""
        landmarks = self.controller.map_manager.get_landmarks() if self.controller.map_manager else {}
        if landmarks:
            landmark_names = ", ".join(landmarks.keys())
            self.landmark_var.set(f"Landmarks: {landmark_names}")
        else:
            self.landmark_var.set("Landmarks: None")
    
    def on_speed_change(self, value):
        """Handle speed scale change"""
        config['SIMULATION_SPEED'] = float(value)
        if hasattr(self.controller, 'navigation_thread'):
            print(f"Speed changed to: {value}")
    
    def on_sensitivity_change(self, value):
        """Handle sensitivity scale change"""
        config['OBSTACLE_THRESHOLD'] = int(value)
        if self.controller.camera_processor:
            self.controller.camera_processor.adjust_detection_parameters(
                threshold=int(value)
            )
    
    # Control panel command methods
    def start_navigation(self):
        """Start navigation"""
        self.controller.start_navigation()
        self.nav_status_var.set("Status: Navigating")
        self.command_var.set("NAVIGATING")
    
    def stop_navigation(self):
        """Stop navigation"""
        self.controller.stop_navigation()
        self.nav_status_var.set("Status: Stopped")
        self.command_var.set("STOPPED")
    
    def pause_navigation(self):
        """Pause navigation"""
        self.nav_status_var.set("Status: Paused")
        self.command_var.set("PAUSED")
        # Implementation would go in controller
    
    def load_map_dialog(self):
        """Load map from file dialog"""
        map_file = self.map_var.get()
        if map_file:
            success = self.controller.load_map(map_file)
            if success:
                self.update_landmark_display()
                messagebox.showinfo("Success", f"Map '{map_file}' loaded successfully")
            else:
                messagebox.showerror("Error", f"Failed to load map '{map_file}'")
    
    def save_map_dialog(self):
        """Save current map state"""
        # This would open a file dialog and save the current map state
        messagebox.showinfo("Save Map", "Map save functionality would be implemented here")
    
    def new_map_dialog(self):
        """Create new map dialog"""
        response = messagebox.askyesno("New Map", "Create a new empty map?")
        if response:
            # Implementation for creating new map would go here
            print("Creating new map...")
    
    def reset_system(self):
        """Reset the system to initial state"""
        self.destination = None
        self.temporary_obstacles.clear()
        self.current_position = None
        self.path = []
        self.dynamic_obstacles = []
        
        self.controller.reset_system()
        self.update_status()
        self.draw_grid()
        
        self.nav_status_var.set("Status: Reset")
        self.command_var.set("READY")
        print("System reset complete")
    
    def calibrate_camera(self):
        """Calibrate camera detection"""
        if self.controller.camera_processor:
            success = self.controller.camera_processor.calibrate_detection()
            if success:
                messagebox.showinfo("Success", "Camera calibration completed successfully")
            else:
                messagebox.showerror("Error", "Camera calibration failed")
        else:
            messagebox.showwarning("Warning", "Camera processor not available")
    
    def set_command(self, command: str):
        """Set the current movement command"""
        self.command_var.set(command)
        
        # Update navigation status based on command
        if command in ["RIGHT", "LEFT", "UP", "DOWN", "FORWARD"]:
            self.nav_status_var.set("Status: Moving")
        elif command == "STOP":
            self.nav_status_var.set("Status: Stopped")
        elif command == "DESTINATION REACHED":
            self.nav_status_var.set("Status: Destination Reached")
    
    def show_message(self, title: str, message: str):
        """Show message dialog"""
        messagebox.showinfo(title, message)
    
    def show_error(self, title: str, message: str):
        """Show error dialog"""
        messagebox.showerror(title, message)
    
    def update_navigation_status(self, status: str):
        """Update navigation status display"""
        self.nav_status_var.set(f"Status: {status}")


# Custom style for accent buttons
def configure_styles():
    """Configure custom styles for the GUI"""
    style = ttk.Style()
    style.configure("Accent.TButton", foreground="white", background="#0078D7")
    return style


# Example usage and testing
if __name__ == "__main__":
    print("Testing GUI Interface...")
    
    # Create mock controller for testing
    class MockController:
        def __init__(self):
            self.map_manager = None
            self.camera_processor = None
        
        def start_navigation(self):
            print("Mock: Start navigation")
        
        def stop_navigation(self):
            print("Mock: Stop navigation")
        
        def load_map(self, map_file):
            print(f"Mock: Load map {map_file}")
            return True
        
        def set_destination(self, destination):
            print(f"Mock: Set destination {destination}")
        
        def update_temporary_obstacles(self, obstacles):
            print(f"Mock: Update obstacles {obstacles}")
        
        def reset_system(self):
            print("Mock: Reset system")
    
    # Test the GUI
    root = tk.Tk()
    configure_styles()
    
    controller = MockController()
    gui = WheelchairGUI(root, controller)
    
    # Test with some sample data
    gui.current_position = (5, 5)
    gui.destination = (45, 45)
    gui.path = [(5, 5), (10, 10), (20, 20), (30, 30), (40, 40), (45, 45)]
    gui.dynamic_obstacles = [(15, 15), (25, 25)]
    gui.temporary_obstacles = {(35, 35)}
    
    gui.update_display(gui.current_position, gui.path, gui.dynamic_obstacles)
    
    print("GUI test completed. Close the window to exit.")
    root.mainloop()