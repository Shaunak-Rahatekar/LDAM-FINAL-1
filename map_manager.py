"""
Map Manager for Smart Wheelchair Navigation System
"""

import json
import os
import numpy as np
from config import config

class MapManager:
    def __init__(self):
        self.current_map = None
        self.grid_data = None
        self.ensure_map_directory()
    
    def ensure_map_directory(self):
        """Create maps directory if it doesn't exist"""
        if not os.path.exists(config['MAP_DIR']):
            os.makedirs(config['MAP_DIR'])
            print(f"Created maps directory: {config['MAP_DIR']}")
    
    def create_default_maps(self):
        """Create default map files"""
        print("Creating default maps...")
        
        # Hospital Map
        hospital_map = self._create_hospital_map()
        self._save_map("hospital_floor1.json", hospital_map)
        
        # Office Map
        office_map = self._create_office_map()
        self._save_map("office_floor1.json", office_map)
        
        print("✅ Default maps created successfully!")
        return True
    
    def _create_hospital_map(self):
        """Create a hospital floor map"""
        obstacles = []
        
        # Outer walls
        obstacles.append({
            "type": "outer_wall",
            "coords": self._create_rectangle_border(0, 0, 49, 49)
        })
        
        # Rooms
        room1 = self._create_room_with_door(5, 5, 20, 15, 'bottom')
        room2 = self._create_room_with_door(25, 5, 40, 15, 'bottom')
        room3 = self._create_room_with_door(5, 30, 20, 40, 'top')
        room4 = self._create_room_with_door(25, 30, 40, 40, 'top')
        
        obstacles.extend(room1)
        obstacles.extend(room2)
        obstacles.extend(room3)
        obstacles.extend(room4)
        
        # Corridor
        for x in range(10, 40):
            obstacles.append({"type": "corridor_wall", "coords": [[x, 20]]})
            obstacles.append({"type": "corridor_wall", "coords": [[x, 25]]})
        
        # Columns
        obstacles.append({"type": "column", "coords": [[15, 10]]})
        obstacles.append({"type": "column", "coords": [[35, 10]]})
        obstacles.append({"type": "column", "coords": [[15, 35]]})
        obstacles.append({"type": "column", "coords": [[35, 35]]})
        
        return {
            "name": "Hospital Floor 1",
            "grid_size": [50, 50],
            "permanent_obstacles": obstacles,
            "landmarks": {
                "reception": [8, 8],
                "elevator": [42, 42],
                "pharmacy": [35, 35]
            },
            "start_position": [5, 22],
            "description": "Hospital environment with rooms, corridors, and medical facilities"
        }
    
    def _create_office_map(self):
        """Create an office floor map"""
        obstacles = []
        
        # Outer walls
        obstacles.append({
            "type": "outer_wall",
            "coords": self._create_rectangle_border(0, 0, 49, 49)
        })
        
        # Conference room
        conference = self._create_room_with_door(10, 10, 30, 20, 'bottom')
        obstacles.extend(conference)
        
        # Offices
        office1 = self._create_room_with_door(5, 30, 15, 40, 'right')
        office2 = self._create_room_with_door(20, 30, 30, 40, 'right')
        office3 = self._create_room_with_door(35, 30, 45, 40, 'left')
        
        obstacles.extend(office1)
        obstacles.extend(office2)
        obstacles.extend(office3)
        
        # Cubicles
        for x in range(10, 40, 5):
            for y in range(25, 30):
                obstacles.append({"type": "cubicle", "coords": [[x, y]]})
        
        return {
            "name": "Office Floor 1",
            "grid_size": [50, 50],
            "permanent_obstacles": obstacles,
            "landmarks": {
                "reception": [3, 3],
                "conference_room": [20, 15],
                "break_room": [40, 35]
            },
            "start_position": [3, 3],
            "description": "Office environment with cubicles, meeting rooms, and workspaces"
        }
    
    def _create_rectangle_border(self, x1, y1, x2, y2):
        """Create rectangle border coordinates"""
        coords = []
        # Top and bottom walls
        for x in range(x1, x2 + 1):
            coords.append([x, y1])
            coords.append([x, y2])
        # Left and right walls (excluding corners)
        for y in range(y1 + 1, y2):
            coords.append([x1, y])
            coords.append([x2, y])
        return coords
    
    def _create_room_with_door(self, x1, y1, x2, y2, door_side):
        """Create a room with doorway"""
        walls = self._create_rectangle_border(x1, y1, x2, y2)
        
        # Remove door area (center 2 cells)
        room_width = x2 - x1 + 1
        door_start = x1 + (room_width - 2) // 2
        
        if door_side == 'bottom':
            walls = [coord for coord in walls if not (coord[1] == y2 and door_start <= coord[0] <= door_start + 1)]
        elif door_side == 'top':
            walls = [coord for coord in walls if not (coord[1] == y1 and door_start <= coord[0] <= door_start + 1)]
        elif door_side == 'left':
            walls = [coord for coord in walls if not (coord[0] == x1 and door_start <= coord[1] <= door_start + 1)]
        elif door_side == 'right':
            walls = [coord for coord in walls if not (coord[0] == x2 and door_start <= coord[1] <= door_start + 1)]
        
        return [{"type": "room_wall", "coords": walls}]
    
    def _save_map(self, filename, map_data):
        """Save map to JSON file"""
        try:
            filepath = os.path.join(config['MAP_DIR'], filename)
            with open(filepath, 'w') as f:
                json.dump(map_data, f, indent=2)
            print(f"✅ Map saved: {filename}")
            return True
        except Exception as e:
            print(f"❌ Error saving map {filename}: {e}")
            return False
    
    def load_map(self, filename):
        """Load map from JSON file"""
        try:
            filepath = os.path.join(config['MAP_DIR'], filename)
            if not os.path.exists(filepath):
                print(f"❌ Map file not found: {filepath}")
                return False
                
            with open(filepath, 'r') as f:
                self.current_map = json.load(f)
            
            self._convert_map_to_grid()
            print(f"✅ Map loaded: {self.current_map['name']}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading map {filename}: {e}")
            return False
    
    def _convert_map_to_grid(self):
        """Convert map obstacles to grid format"""
        self.grid_data = np.zeros((config['GRID_HEIGHT'], config['GRID_WIDTH']), dtype=int)
        
        if self.current_map and 'permanent_obstacles' in self.current_map:
            for obstacle in self.current_map['permanent_obstacles']:
                for coord in obstacle['coords']:
                    x, y = coord
                    if 0 <= x < config['GRID_WIDTH'] and 0 <= y < config['GRID_HEIGHT']:
                        self.grid_data[y, x] = 1
    
    def get_grid_data(self):
        """Get current grid data"""
        if self.grid_data is not None:
            return self.grid_data.copy()
        else:
            return np.zeros((config['GRID_HEIGHT'], config['GRID_WIDTH']), dtype=int)
    
    def get_start_position(self):
        """Get start position from current map"""
        if self.current_map and 'start_position' in self.current_map:
            pos = self.current_map['start_position']
            return (pos[0], pos[1])
        return (2, 2)  # Default start position
    
    def get_landmarks(self):
        """Get landmarks from current map"""
        if self.current_map and 'landmarks' in self.current_map:
            return self.current_map['landmarks'].copy()
        return {}

    # --- Validation and edit helpers used by tests ---
    def validate_map(self, map_data) -> bool:
        """Validate map structure and coordinates"""
        try:
            # Required fields
            if not isinstance(map_data, dict):
                return False
            required = ['name', 'grid_size', 'permanent_obstacles']
            if not all(k in map_data for k in required):
                return False

            grid_w, grid_h = map_data.get('grid_size', [0, 0])
            if not (isinstance(grid_w, int) and isinstance(grid_h, int) and grid_w > 0 and grid_h > 0):
                return False

            # Validate obstacles
            for obstacle in map_data.get('permanent_obstacles', []):
                coords = obstacle.get('coords', [])
                for xy in coords:
                    if not (isinstance(xy, (list, tuple)) and len(xy) == 2):
                        return False
                    x, y = xy
                    if not (0 <= x < grid_w and 0 <= y < grid_h):
                        return False

            # Validate landmarks (if present)
            landmarks = map_data.get('landmarks', {}) or {}
            for name, pos in landmarks.items():
                if not (isinstance(pos, (list, tuple)) and len(pos) == 2):
                    return False
                x, y = pos
                if not (0 <= x < grid_w and 0 <= y < grid_h):
                    return False

            return True
        except Exception:
            return False

    def add_obstacle(self, x: int, y: int, obstacle_type: str = "user") -> bool:
        """Add a permanent obstacle to current map and grid"""
        try:
            if self.grid_data is None:
                self._convert_map_to_grid()

            if not (0 <= x < config['GRID_WIDTH'] and 0 <= y < config['GRID_HEIGHT']):
                return False

            # Update grid
            self.grid_data[y, x] = 1

            # Update map structure
            if self.current_map is None:
                self.current_map = {
                    'name': 'Untitled',
                    'grid_size': [config['GRID_WIDTH'], config['GRID_HEIGHT']],
                    'permanent_obstacles': [],
                    'landmarks': {}
                }

            # Try to append to an existing obstacle group of same type
            for entry in self.current_map['permanent_obstacles']:
                if entry.get('type') == obstacle_type:
                    if [x, y] not in entry['coords']:
                        entry['coords'].append([x, y])
                    break
            else:
                self.current_map['permanent_obstacles'].append({
                    'type': obstacle_type,
                    'coords': [[x, y]]
                })

            return True
        except Exception:
            return False

    def remove_obstacle(self, x: int, y: int) -> bool:
        """Remove a permanent obstacle from current map and grid"""
        try:
            if self.grid_data is None or self.current_map is None:
                return False

            if not (0 <= x < config['GRID_WIDTH'] and 0 <= y < config['GRID_HEIGHT']):
                return False

            changed = False
            if self.grid_data[y, x] == 1:
                self.grid_data[y, x] = 0
                changed = True

            # Remove from map structure
            for entry in self.current_map.get('permanent_obstacles', []):
                coords = entry.get('coords', [])
                if [x, y] in coords:
                    coords.remove([x, y])
                    changed = True

            return changed
        except Exception:
            return False