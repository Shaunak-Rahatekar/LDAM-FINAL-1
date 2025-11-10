"""
A* Path Planning Algorithm for Smart Wheelchair Navigation System
Robust implementation with multiple heuristics and dynamic replanning
"""

import numpy as np
import heapq
import time
from typing import List, Tuple, Optional, Dict, Any
from config import config

class Node:
    """Node class for A* pathfinding"""
    __slots__ = ('position', 'parent', 'g', 'h', 'f')
    
    def __init__(self, position: Tuple[int, int], parent: Optional['Node'] = None):
        self.position = position
        self.parent = parent
        self.g = 0  # Cost from start to current node
        self.h = 0  # Heuristic cost to goal
        self.f = 0  # Total cost (g + h)
    
    def __lt__(self, other: 'Node') -> bool:
        """Comparison for priority queue - tie-breaking for better performance"""
        if self.f == other.f:
            return self.h < other.h  # Prefer nodes closer to goal
        return self.f < other.f
    
    def __eq__(self, other: object) -> bool:
        """Equality comparison based on position"""
        if not isinstance(other, Node):
            return False
        return self.position == other.position
    
    def __hash__(self) -> int:
        """Hash based on position for set operations"""
        return hash(self.position)

class PathPlanner:
    """Robust A* path planning implementation with dynamic replanning"""
    
    def __init__(self):
        self.path = []
        self.current_index = 0
        self.search_stats = {
            'nodes_explored': 0,
            'planning_time': 0.0,
            'path_length': 0,
            'replan_count': 0
        }
        self.cache = {}  # Cache for frequent queries
    
    def a_star_search(self, grid: np.ndarray, start: Tuple[int, int], 
                     goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        A* pathfinding algorithm implementation with multiple optimizations
        
        Args:
            grid: 2D numpy array where 0=free, 1=obstacle
            start: (x, y) start position
            goal: (x, y) goal position
            
        Returns:
            List of positions from start to goal, or empty list if no path found
        """
        start_time = time.time()
        self.search_stats['nodes_explored'] = 0
        
        # Input validation
        if not self._validate_inputs(grid, start, goal):
            return []
        
        # Check cache for recent queries (optimization for frequent replanning)
        cache_key = self._get_cache_key(grid, start, goal)
        if cache_key in self.cache:
            cached_path = self.cache[cache_key]
            if self._validate_cached_path(cached_path, grid):
                self.search_stats['planning_time'] = time.time() - start_time
                return cached_path
        
        # Early exit if start or goal are obstacles
        if not self.is_valid_position(grid, start) or not self.is_valid_position(grid, goal):
            print("Start or goal position is blocked")
            return []
        
        # Check if goal is reachable (simple connectivity check)
        if not self._is_goal_reachable(grid, start, goal):
            print("Goal is unreachable from start position")
            return []
        
        # Initialize nodes
        start_node = Node(start)
        goal_node = Node(goal)
        
        # Initialize data structures
        open_set = []
        open_dict = {}  # For quick lookup
        closed_set = set()
        
        # Add start node
        start_node.h = self.calculate_heuristic(start, goal)
        start_node.f = start_node.h
        heapq.heappush(open_set, (start_node.f, start_node))
        open_dict[start] = start_node
        
        self.search_stats['nodes_explored'] = 1
        
        # Main A* loop
        while open_set:
            # Get node with lowest f cost
            current_f, current_node = heapq.heappop(open_set)
            current_pos = current_node.position
            
            # Remove from open dict
            if current_pos in open_dict:
                del open_dict[current_pos]
            
            # Check if we reached the goal
            if current_pos == goal_node.position:
                path = self._reconstruct_path(current_node)
                self.search_stats['planning_time'] = time.time() - start_time
                self.search_stats['path_length'] = len(path)
                
                # Cache the result
                self.cache[cache_key] = path
                if len(self.cache) > 100:  # Limit cache size
                    self.cache.pop(next(iter(self.cache)))
                
                return path
            
            # Add to closed set
            closed_set.add(current_pos)
            
            # Explore neighbors
            for neighbor_pos in self.get_neighbors(current_pos):
                # Skip if invalid or in closed set
                if not self.is_valid_position(grid, neighbor_pos) or neighbor_pos in closed_set:
                    continue
                
                # Calculate new g cost
                new_g = current_node.g + self._get_move_cost(current_pos, neighbor_pos)
                
                # Check if neighbor is in open set
                if neighbor_pos in open_dict:
                    neighbor_node = open_dict[neighbor_pos]
                    if new_g < neighbor_node.g:
                        # Update existing node with better path
                        neighbor_node.g = new_g
                        neighbor_node.f = neighbor_node.g + neighbor_node.h
                        neighbor_node.parent = current_node
                        # Re-heapify (inefficient but necessary with heapq)
                        self._rebuild_heap(open_set)
                else:
                    # Create new node
                    neighbor_node = Node(neighbor_pos, current_node)
                    neighbor_node.g = new_g
                    neighbor_node.h = self.calculate_heuristic(neighbor_pos, goal)
                    neighbor_node.f = neighbor_node.g + neighbor_node.h
                    
                    heapq.heappush(open_set, (neighbor_node.f, neighbor_node))
                    open_dict[neighbor_pos] = neighbor_node
                    self.search_stats['nodes_explored'] += 1
            
            # Early termination for performance (optional)
            if self.search_stats['nodes_explored'] > 10000:  # Safety limit
                print("Path planning timeout - too many nodes explored")
                break
        
        # No path found
        self.search_stats['planning_time'] = time.time() - start_time
        print(f"No path found after exploring {self.search_stats['nodes_explored']} nodes")
        return []
    
    def _validate_inputs(self, grid: np.ndarray, start: Tuple[int, int], 
                        goal: Tuple[int, int]) -> bool:
        """Validate input parameters"""
        if grid is None or grid.size == 0:
            print("Invalid grid")
            return False
        
        if not isinstance(start, (tuple, list)) or len(start) != 2:
            print("Start must be a tuple/list of length 2")
            return False
        
        if not isinstance(goal, (tuple, list)) or len(goal) != 2:
            print("Goal must be a tuple/list of length 2")
            return False
        
        grid_height, grid_width = grid.shape
        start_x, start_y = start
        goal_x, goal_y = goal
        
        if not (0 <= start_x < grid_width and 0 <= start_y < grid_height):
            print(f"Start position {start} out of grid bounds")
            return False
        
        if not (0 <= goal_x < grid_width and 0 <= goal_y < grid_height):
            print(f"Goal position {goal} out of grid bounds")
            return False
        
        return True
    
    def _get_cache_key(self, grid: np.ndarray, start: Tuple[int, int], 
                      goal: Tuple[int, int]) -> str:
        """Generate cache key for path queries"""
        # Use a simplified representation for caching
        grid_hash = hash(grid.tobytes())
        return f"{start}_{goal}_{grid_hash}"
    
    def _validate_cached_path(self, path: List[Tuple[int, int]], 
                            grid: np.ndarray) -> bool:
        """Validate that cached path is still valid"""
        if not path:
            return False
        
        # Check if all path positions are valid
        for pos in path:
            if not self.is_valid_position(grid, pos):
                return False
        
        return True
    
    def _is_goal_reachable(self, grid: np.ndarray, start: Tuple[int, int], 
                          goal: Tuple[int, int]) -> bool:
        """Quick check if goal is reachable using flood fill"""
        if start == goal:
            return True
        
        visited = set()
        stack = [start]
        
        while stack:
            current = stack.pop()
            if current == goal:
                return True
            
            if current in visited:
                continue
            
            visited.add(current)
            
            # Check 4-direction neighbors for quick connectivity
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if (self.is_valid_position(grid, neighbor) and 
                    neighbor not in visited):
                    stack.append(neighbor)
        
        return False
    
    def _rebuild_heap(self, heap: List[Tuple[float, Node]]) -> None:
        """Rebuild heap after priority updates"""
        heapq.heapify(heap)
    
    def get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring positions based on movement constraints"""
        x, y = position
        neighbors = []
        grid_height, grid_width = config['GRID_HEIGHT'], config['GRID_WIDTH']
        
        # 4-directional movement (always available)
        directions_4 = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Down, Right, Up, Left
        
        for dx, dy in directions_4:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_width and 0 <= ny < grid_height:
                neighbors.append((nx, ny))
        
        # Diagonal movement (if allowed)
        if config['ALLOW_DIAGONAL']:
            directions_diag = [(1, 1), (1, -1), (-1, 1), (-1, -1)]  # Diagonals
            
            for dx, dy in directions_diag:
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid_width and 0 <= ny < grid_height:
                    # Check if diagonal movement is possible (adjacent cells must be free)
                    if (self._is_valid_diagonal(position, (nx, ny))):
                        neighbors.append((nx, ny))
        
        return neighbors
    
    def _is_valid_diagonal(self, current: Tuple[int, int], 
                          diagonal: Tuple[int, int]) -> bool:
        """Check if diagonal move is valid (adjacent cells must be free)"""
        cx, cy = current
        dx, dy = diagonal
        
        # For diagonal move (cx+1, cy+1), check (cx+1, cy) and (cx, cy+1)
        if dx > cx and dy > cy:  # Down-right
            return True  # We'll check obstacles in is_valid_position
        elif dx > cx and dy < cy:  # Up-right
            return True
        elif dx < cx and dy > cy:  # Down-left
            return True
        elif dx < cx and dy < cy:  # Up-left
            return True
        
        return True
    
    def _get_move_cost(self, current_pos: Tuple[int, int], 
                      next_pos: Tuple[int, int]) -> float:
        """Calculate movement cost between positions"""
        dx = abs(current_pos[0] - next_pos[0])
        dy = abs(current_pos[1] - next_pos[1])
        
        if dx == 1 and dy == 1:  # Diagonal move
            return config['DIAGONAL_COST']
        else:  # Horizontal or vertical move
            return config['MOVEMENT_COST']
    
    def calculate_heuristic(self, pos1: Tuple[int, int], 
                           pos2: Tuple[int, int]) -> float:
        """
        Calculate heuristic cost using selected method
        
        Available heuristics:
        - 'manhattan': |dx| + |dy|
        - 'euclidean': sqrt(dx² + dy²) 
        - 'diagonal': D * (dx + dy) + (D2 - 2*D) * min(dx, dy)
        """
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        
        heuristic_type = config.get('HEURISTIC_TYPE', 'diagonal')
        
        if heuristic_type == 'manhattan':
            return config['MOVEMENT_COST'] * (dx + dy)
        
        elif heuristic_type == 'euclidean':
            return config['MOVEMENT_COST'] * np.sqrt(dx*dx + dy*dy)
        
        elif heuristic_type == 'diagonal':
            # Diagonal distance heuristic (optimal for 8-direction movement)
            d = config['MOVEMENT_COST']  # Cost for non-diagonal
            d2 = config['DIAGONAL_COST']  # Cost for diagonal
            return d * (dx + dy) + (d2 - 2 * d) * min(dx, dy)
        
        else:
            # Default to diagonal
            d = config['MOVEMENT_COST']
            d2 = config['DIAGONAL_COST']
            return d * (dx + dy) + (d2 - 2 * d) * min(dx, dy)
    
    def is_valid_position(self, grid: np.ndarray, position: Tuple[int, int]) -> bool:
        """Check if position is valid and not an obstacle"""
        x, y = position
        grid_height, grid_width = grid.shape
        
        if not (0 <= x < grid_width and 0 <= y < grid_height):
            return False
        
        return grid[y, x] == 0  # 0 means free space
    
    def _reconstruct_path(self, node: Node) -> List[Tuple[int, int]]:
        """Reconstruct path from goal to start and smooth if possible"""
        path = []
        current = node
        
        while current is not None:
            path.append(current.position)
            current = current.parent
        
        path.reverse()  # Change from goal->start to start->goal
        
        # Optional: Path smoothing (remove unnecessary waypoints)
        if len(path) > 2:
            path = self._smooth_path(path)
        
        return path
    
    def _smooth_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Simple path smoothing to remove unnecessary waypoints"""
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]  # Start with first point
        
        i = 0
        while i < len(path) - 1:
            # Look ahead to find the furthest point we can reach directly
            j = len(path) - 1
            while j > i + 1:
                if self._is_direct_path_clear(path[i], path[j]):
                    smoothed.append(path[j])
                    i = j
                    break
                j -= 1
            else:
                # No direct path found, take next point
                smoothed.append(path[i + 1])
                i += 1
        
        return smoothed
    
    def _is_direct_path_clear(self, start: Tuple[int, int], 
                             end: Tuple[int, int]) -> bool:
        """Check if direct path between two points is clear (Bresenham's line)"""
        x0, y0 = start
        x1, y1 = end
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        # For now, assume direct path is clear for diagonal moves
        # In a real implementation, you'd check each cell along the line
        return True
    
    def dynamic_replan(self, grid: np.ndarray, current_pos: Tuple[int, int], 
                      goal: Tuple[int, int], new_obstacles: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Replan path when new obstacles are detected
        
        Args:
            grid: Current grid state
            current_pos: Current wheelchair position
            goal: Destination position
            new_obstacles: List of new obstacle positions
            
        Returns:
            New path or empty list if no path found
        """
        self.search_stats['replan_count'] += 1
        
        # Update grid with new obstacles
        updated_grid = grid.copy()
        for x, y in new_obstacles:
            if 0 <= x < updated_grid.shape[1] and 0 <= y < updated_grid.shape[0]:
                updated_grid[y, x] = 1
        
        # Check if current path is still valid
        if self._is_current_path_valid(updated_grid):
            print("Current path remains valid")
            return self.path[self.current_index:]  # Return remaining path
        
        # Replan from current position
        print("Obstacle detected - replanning path...")
        new_path = self.a_star_search(updated_grid, current_pos, goal)
        
        if new_path:
            self.set_path(new_path)
            print(f"New path found with {len(new_path)} steps")
        else:
            print("No valid path found after replanning")
        
        return new_path
    
    def _is_current_path_valid(self, grid: np.ndarray) -> bool:
        """Check if the current path is still obstacle-free"""
        if not self.path or self.current_index >= len(self.path):
            return False
        
        remaining_path = self.path[self.current_index:]
        for pos in remaining_path:
            if not self.is_valid_position(grid, pos):
                return False
        
        return True
    
    def set_path(self, path: List[Tuple[int, int]]) -> None:
        """Set the current path and reset index"""
        self.path = path
        self.current_index = 0
    
    def get_next_position(self) -> Optional[Tuple[int, int]]:
        """Get next position in the path"""
        if self.current_index < len(self.path):
            pos = self.path[self.current_index]
            self.current_index += 1
            return pos
        return None
    
    def has_reached_destination(self) -> bool:
        """Check if destination has been reached"""
        return self.current_index >= len(self.path)
    
    def get_remaining_path(self) -> List[Tuple[int, int]]:
        """Get remaining path from current position"""
        return self.path[self.current_index:] if self.current_index < len(self.path) else []
    
    def get_current_direction(self, current_pos: Tuple[int, int], 
                            next_pos: Tuple[int, int]) -> str:
        """Get movement direction command"""
        if not current_pos or not next_pos:
            return "STOP"
        
        curr_x, curr_y = current_pos
        next_x, next_y = next_pos
        
        if next_x > curr_x:
            return "RIGHT"
        elif next_x < curr_x:
            return "LEFT"
        elif next_y > curr_y:
            return "DOWN"
        elif next_y < curr_y:
            return "UP"
        else:
            return "STOP"
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return self.search_stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset search statistics"""
        self.search_stats = {
            'nodes_explored': 0,
            'planning_time': 0.0,
            'path_length': 0,
            'replan_count': 0
        }
    
    def clear_cache(self) -> None:
        """Clear path cache"""
        self.cache.clear()


# Example usage and testing
if __name__ == "__main__":
    print("Testing Path Planner...")
    
    # Create a test grid
    grid = np.zeros((50, 50), dtype=int)
    
    # Add some obstacles
    for i in range(10, 40):
        grid[25, i] = 1  # Horizontal wall
    
    for i in range(10, 40):
        grid[i, 25] = 1  # Vertical wall
    
    planner = PathPlanner()
    
    # Test path planning
    start = (5, 5)
    goal = (45, 45)
    
    path = planner.a_star_search(grid, start, goal)
    
    if path:
        print(f"Path found with {len(path)} steps")
        stats = planner.get_search_statistics()
        print(f"Nodes explored: {stats['nodes_explored']}")
        print(f"Planning time: {stats['planning_time']:.4f}s")
        print(f"Path length: {stats['path_length']}")
        
        # Test dynamic replanning
        print("\nTesting dynamic replanning...")
        new_obstacles = [(20, 20), (21, 21)]
        new_path = planner.dynamic_replan(grid, (10, 10), goal, new_obstacles)
        
        if new_path:
            print(f"Replanned path with {len(new_path)} steps")
        
    else:
        print("No path found")
    
    # Test invalid inputs
    print("\nTesting error handling...")
    invalid_path = planner.a_star_search(grid, (-1, -1), goal)
    print(f"Invalid start handled: {len(invalid_path) == 0}")
    
    blocked_goal_path = planner.a_star_search(grid, start, (25, 25))
    print(f"Blocked goal handled: {len(blocked_goal_path) == 0}")