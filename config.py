"""
Configuration settings for Smart Wheelchair Navigation System
Complete with save/load functionality and validation
"""

import json
import os
from typing import Dict, Any

class Config:
    """Configuration manager for Smart Wheelchair System"""
    
    # Default configuration values
    DEFAULTS = {
        # Grid and Map Settings
        'GRID_WIDTH': 50,
        'GRID_HEIGHT': 50,
        'CELL_SIZE': 10,
        'MAP_DIR': "maps",
        
        # Color Configuration (Hex codes)
        'COLORS': {
            'start': '#00FF00',        # Green
            'destination': '#FF0000',  # Red
            'obstacle': '#000000',     # Black
            'path': '#0000FF',         # Blue
            'current_pos': '#FFFF00',  # Yellow
            'background': '#FFFFFF',   # White
            'grid_line': '#CCCCCC',    # Light Gray
            'permanent_obs': '#333333',# Dark Gray
            'dynamic_obs': '#FF4444',  # Light Red
            'text': '#000000',         # Black
        },
        
        # Webcam Settings (Wheelchair-Mounted Camera)
        # Note: System uses webcam (index 0) exclusively for live feed analysis
        'CAMERA_WIDTH': 1280,
        'CAMERA_HEIGHT': 720,
        'CAMERA_FPS': 30,
        'CAMERA_INDEX': 0,  # Webcam index - fixed to 0 for wheelchair-mounted camera
        'ENABLE_CAMERA': True,
        'USE_WEBCAM_ONLY': True,  # Enforce webcam-only usage
        # Perspective Mapping (optional homography)
        'USE_PERSPECTIVE_MAPPING': False,
        # 4 source points in camera pixel coords (top-left, top-right, bottom-right, bottom-left)
        'CAMERA_PERSPECTIVE_POINTS': [],  # e.g., [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        
        # Path Planning Settings (A* Algorithm)
        'MOVEMENT_COST': 1,
        'DIAGONAL_COST': 1.4,
        'ALLOW_DIAGONAL': True,
        'HEURISTIC_TYPE': 'diagonal',  # 'manhattan', 'euclidean', 'diagonal'
        
        # GUI Settings
        'GUI_WIDTH': 1200,
        'GUI_HEIGHT': 800,
        'REFRESH_RATE': 100,  # milliseconds
        'GUI_TITLE': "Smart Wheelchair Navigation System",
        'SHOW_GRID_LINES': True,
        'SHOW_COORDINATES': False,
        
        # Obstacle Detection Settings
        'OBSTACLE_THRESHOLD': 128,
        'MIN_OBSTACLE_AREA': 100,
        'MAX_OBSTACLE_AREA': 10000,
        'CONTOUR_APPROX_EPSILON': 0.02,
        'BLUR_KERNEL_SIZE': 5,
        'MORPHOLOGY_KERNEL_SIZE': 3,
        
        # Simulation Settings
        'SIMULATION_SPEED': 0.3,  # seconds per move
        'ENABLE_SIMULATION_MODE': True,
        'SIMULATION_OBSTACLE_PROBABILITY': 0.1,
        
        # System Settings
        'CONFIG_FILE': "config.json",
        'LOG_LEVEL': "INFO",  # DEBUG, INFO, WARNING, ERROR
        'AUTO_SAVE_CONFIG': True,
    }
    
    def __init__(self, config_file: str = None):
        """Initialize configuration manager"""
        self.config_file = config_file or self.DEFAULTS['CONFIG_FILE']
        self._config = self.DEFAULTS.copy()
        self.load_config()
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value"""
        return self._config.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set configuration value"""
        self._config[key] = value
        if self._config['AUTO_SAVE_CONFIG']:
            self.save_config()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default"""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        self.__setitem__(key, value)
    
    def load_config(self) -> bool:
        """Load configuration from JSON file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                
                # Deep merge loaded config with defaults
                self._deep_merge(self._config, loaded_config)
                
                # Validate the loaded configuration
                if self.validate_config():
                    print(f"Configuration loaded from {self.config_file}")
                    return True
                else:
                    print("Loaded configuration failed validation, using defaults")
                    self._config = self.DEFAULTS.copy()
                    return False
            else:
                print(f"Config file {self.config_file} not found, using defaults")
                return False
                
        except Exception as e:
            print(f"Error loading configuration: {e}")
            self._config = self.DEFAULTS.copy()
            return False
    
    def save_config(self) -> bool:
        """Save configuration to JSON file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_file) if os.path.dirname(self.config_file) else '.', exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                json.dump(self._config, f, indent=4, ensure_ascii=False)
            
            print(f"Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def _deep_merge(self, target: Dict, source: Dict) -> None:
        """Recursively merge source dictionary into target dictionary"""
        for key, value in source.items():
            if key in target:
                if isinstance(target[key], dict) and isinstance(value, dict):
                    self._deep_merge(target[key], value)
                else:
                    target[key] = value
            else:
                target[key] = value
    
    def validate_config(self) -> bool:
        """Validate all configuration parameters"""
        try:
            # Grid validation
            assert self._config['GRID_WIDTH'] > 0, "Grid width must be positive"
            assert self._config['GRID_HEIGHT'] > 0, "Grid height must be positive"
            assert self._config['CELL_SIZE'] > 0, "Cell size must be positive"
            
            # Color validation
            colors = self._config['COLORS']
            for color_name, hex_code in colors.items():
                assert isinstance(hex_code, str) and len(hex_code) == 7 and hex_code[0] == '#', \
                    f"Invalid color format for {color_name}: {hex_code}"
                # Validate hex code characters
                assert all(c in '0123456789ABCDEFabcdef' for c in hex_code[1:]), \
                    f"Invalid hex characters in color {color_name}: {hex_code}"
            
            # Camera validation
            assert self._config['CAMERA_WIDTH'] > 0, "Camera width must be positive"
            assert self._config['CAMERA_HEIGHT'] > 0, "Camera height must be positive"
            assert self._config['CAMERA_FPS'] > 0, "Camera FPS must be positive"
            assert 0 <= self._config['CAMERA_INDEX'] <= 10, "Camera index out of range"
            # Perspective mapping validation (if enabled)
            if self._config.get('USE_PERSPECTIVE_MAPPING', False):
                pts = self._config.get('CAMERA_PERSPECTIVE_POINTS', [])
                assert isinstance(pts, list) and len(pts) == 4, "CAMERA_PERSPECTIVE_POINTS must have 4 points"
                for p in pts:
                    assert isinstance(p, (list, tuple)) and len(p) == 2, "Each perspective point must be [x,y]"
            
            # Path planning validation
            assert self._config['MOVEMENT_COST'] > 0, "Movement cost must be positive"
            assert self._config['DIAGONAL_COST'] > 0, "Diagonal cost must be positive"
            assert self._config['DIAGONAL_COST'] >= self._config['MOVEMENT_COST'], \
                "Diagonal cost should be >= movement cost"
            assert self._config['HEURISTIC_TYPE'] in ['manhattan', 'euclidean', 'diagonal'], \
                "Invalid heuristic type"
            
            # GUI validation
            assert self._config['GUI_WIDTH'] > 0, "GUI width must be positive"
            assert self._config['GUI_HEIGHT'] > 0, "GUI height must be positive"
            assert self._config['REFRESH_RATE'] > 0, "Refresh rate must be positive"
            
            # Obstacle detection validation
            assert 0 <= self._config['OBSTACLE_THRESHOLD'] <= 255, \
                "Obstacle threshold must be between 0 and 255"
            assert self._config['MIN_OBSTACLE_AREA'] >= 0, \
                "Minimum obstacle area must be non-negative"
            assert self._config['MAX_OBSTACLE_AREA'] > self._config['MIN_OBSTACLE_AREA'], \
                "Maximum obstacle area must be greater than minimum"
            
            # Simulation validation
            assert self._config['SIMULATION_SPEED'] > 0, "Simulation speed must be positive"
            assert 0 <= self._config['SIMULATION_OBSTACLE_PROBABILITY'] <= 1, \
                "Obstacle probability must be between 0 and 1"
            
            print("Configuration validation successful")
            return True
            
        except AssertionError as e:
            print(f"Configuration validation failed: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error during configuration validation: {e}")
            return False
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values"""
        self._config = self.DEFAULTS.copy()
        if self._config['AUTO_SAVE_CONFIG']:
            self.save_config()
        print("Configuration reset to defaults")
    
    def get_grid_dimensions(self) -> tuple:
        """Get grid dimensions as tuple (width, height)"""
        return (self._config['GRID_WIDTH'], self._config['GRID_HEIGHT'])
    
    def get_camera_resolution(self) -> tuple:
        """Get camera resolution as tuple (width, height)"""
        return (self._config['CAMERA_WIDTH'], self._config['CAMERA_HEIGHT'])
    
    def get_gui_size(self) -> tuple:
        """Get GUI window size as tuple (width, height)"""
        return (self._config['GUI_WIDTH'], self._config['GUI_HEIGHT'])
    
    def get_canvas_size(self) -> tuple:
        """Get canvas size for grid display"""
        width = self._config['GRID_WIDTH'] * self._config['CELL_SIZE']
        height = self._config['GRID_HEIGHT'] * self._config['CELL_SIZE']
        return (width, height)
    
    def update_colors(self, new_colors: Dict[str, str]) -> None:
        """Update color configuration"""
        if isinstance(new_colors, dict):
            self._config['COLORS'].update(new_colors)
            if self._config['AUTO_SAVE_CONFIG']:
                self.save_config()
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary"""
        return self._config.copy()
    
    def __str__(self) -> str:
        """String representation of configuration"""
        config_summary = [
            "Smart Wheelchair Configuration:",
            f"Grid: {self._config['GRID_WIDTH']}x{self._config['GRID_HEIGHT']} ({self._config['CELL_SIZE']}px cells)",
            f"Camera: {self._config['CAMERA_WIDTH']}x{self._config['CAMERA_HEIGHT']} @ {self._config['CAMERA_FPS']}fps",
            f"GUI: {self._config['GUI_WIDTH']}x{self._config['GUI_HEIGHT']} ({self._config['REFRESH_RATE']}ms refresh)",
            f"Path Planning: A* with {self._config['HEURISTIC_TYPE']} heuristic",
            f"Simulation: {'Enabled' if self._config['ENABLE_SIMULATION_MODE'] else 'Disabled'}",
            f"Config File: {self.config_file}"
        ]
        return "\n".join(config_summary)


# Global configuration instance
config = Config()

# Convenience access to common parameters
GRID_WIDTH = config['GRID_WIDTH']
GRID_HEIGHT = config['GRID_HEIGHT']
CELL_SIZE = config['CELL_SIZE']
COLORS = config['COLORS']
CAMERA_WIDTH = config['CAMERA_WIDTH']
CAMERA_HEIGHT = config['CAMERA_HEIGHT']
CAMERA_FPS = config['CAMERA_FPS']
MOVEMENT_COST = config['MOVEMENT_COST']
DIAGONAL_COST = config['DIAGONAL_COST']
ALLOW_DIAGONAL = config['ALLOW_DIAGONAL']
GUI_WIDTH = config['GUI_WIDTH']
GUI_HEIGHT = config['GUI_HEIGHT']
REFRESH_RATE = config['REFRESH_RATE']
GUI_TITLE = config['GUI_TITLE']

# Validation function for external use
def validate_config():
    """Validate configuration (for external imports)"""
    return config.validate_config()

# Example usage and testing
if __name__ == "__main__":
    print("Testing Configuration Manager...")
    print(config)
    
    # Test saving and loading
    config.save_config()
    
    # Test modification
    config['CELL_SIZE'] = 15
    print(f"Updated CELL_SIZE: {config['CELL_SIZE']}")
    
    # Test validation
    config.validate_config()
    
    # Test reset
    config.reset_to_defaults()
    print(f"Reset CELL_SIZE: {config['CELL_SIZE']}")