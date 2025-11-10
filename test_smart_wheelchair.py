"""
Unit Tests for Smart Wheelchair Navigation System
Comprehensive testing of A* algorithm and obstacle detection
"""

import unittest
import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from path_planner import PathPlanner, Node
from camera_processor import CameraProcessor
from map_manager import MapManager
from config import config

class TestPathPlanner(unittest.TestCase):
    """Test cases for A* path planning algorithm"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.planner = PathPlanner()
        
        # Create test grid
        self.grid = np.zeros((50, 50), dtype=int)
        
        # Add some obstacles for testing
        for i in range(10, 20):
            self.grid[25, i] = 1  # Horizontal wall
        
        for i in range(10, 20):
            self.grid[i, 25] = 1  # Vertical wall
    
    def test_node_creation(self):
        """Test Node class creation and properties"""
        node = Node((5, 5))
        self.assertEqual(node.position, (5, 5))
        self.assertIsNone(node.parent)
        self.assertEqual(node.g, 0)
        self.assertEqual(node.h, 0)
        self.assertEqual(node.f, 0)
    
    def test_node_comparison(self):
        """Test Node comparison operators"""
        node1 = Node((1, 1))
        node1.f = 10
        
        node2 = Node((2, 2))
        node2.f = 5
        
        self.assertTrue(node2 < node1)  # Lower f cost should be "less than"
        self.assertFalse(node1 < node2)
    
    def test_a_star_direct_path(self):
        """Test A* with direct unobstructed path"""
        start = (5, 5)
        goal = (10, 10)
        
        path = self.planner.a_star_search(self.grid, start, goal)
        
        self.assertIsNotNone(path)
        self.assertGreater(len(path), 0)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)
    
    def test_a_star_obstacle_avoidance(self):
        """Test A* with obstacles requiring detour"""
        start = (5, 5)
        goal = (30, 30)
        
        path = self.planner.a_star_search(self.grid, start, goal)
        
        self.assertIsNotNone(path)
        self.assertGreater(len(path), 0)
        
        # Verify path doesn't go through obstacles
        for x, y in path:
            self.assertEqual(self.grid[y, x], 0, f"Path goes through obstacle at ({x}, {y})")
    
    def test_a_star_no_path(self):
        """Test A* when no path exists"""
        # Create completely blocked goal
        blocked_grid = self.grid.copy()
        blocked_grid[10:15, 10:15] = 1  # Block a region
        
        start = (5, 5)
        goal = (12, 12)  # Inside blocked region
        
        path = self.planner.a_star_search(blocked_grid, start, goal)
        
        self.assertEqual(len(path), 0)
    
    def test_a_star_same_start_goal(self):
        """Test A* with start and goal at same position"""
        start = (10, 10)
        goal = (10, 10)
        
        path = self.planner.a_star_search(self.grid, start, goal)
        
        self.assertEqual(len(path), 1)
        self.assertEqual(path[0], start)
    
    def test_neighbor_generation(self):
        """Test neighbor position generation"""
        position = (25, 25)
        neighbors = self.planner.get_neighbors(position)
        
        expected_directions = 8 if config['ALLOW_DIAGONAL'] else 4
        self.assertEqual(len(neighbors), expected_directions)
        
        # Check all neighbors are within bounds
        for nx, ny in neighbors:
            self.assertTrue(0 <= nx < self.grid.shape[1])
            self.assertTrue(0 <= ny < self.grid.shape[0])
    
    def test_position_validation(self):
        """Test position validation"""
        # Valid position
        self.assertTrue(self.planner.is_valid_position(self.grid, (5, 5)))
        
        # Obstacle position
        self.assertFalse(self.planner.is_valid_position(self.grid, (25, 15)))
        
        # Out of bounds position
        self.assertFalse(self.planner.is_valid_position(self.grid, (-1, -1)))
        self.assertFalse(self.planner.is_valid_position(self.grid, (100, 100)))
    
    def test_heuristic_calculation(self):
        """Test heuristic cost calculation"""
        pos1 = (5, 5)
        pos2 = (10, 10)
        
        heuristic = self.planner.calculate_heuristic(pos1, pos2)
        
        self.assertGreater(heuristic, 0)
        self.assertLessEqual(heuristic, 10 * config['DIAGONAL_COST'])  # Reasonable upper bound
    
    def test_dynamic_replanning(self):
        """Test dynamic path replanning"""
        start = (5, 5)
        goal = (30, 30)
        
        # Plan initial path
        initial_path = self.planner.a_star_search(self.grid, start, goal)
        self.assertGreater(len(initial_path), 0)
        
        # Add new obstacles
        new_obstacles = [(15, 15), (16, 16)]
        updated_grid = self.grid.copy()
        for x, y in new_obstacles:
            updated_grid[y, x] = 1
        
        # Replan with new obstacles
        new_path = self.planner.dynamic_replan(updated_grid, (10, 10), goal, new_obstacles)
        
        self.assertIsNotNone(new_path)
        
        # Verify new path avoids new obstacles
        for x, y in new_obstacles:
            self.assertNotIn((x, y), new_path)

class TestCameraProcessor(unittest.TestCase):
    """Test cases for camera-based obstacle detection"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.camera_processor = CameraProcessor()
        
        # Create test frame
        self.test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        # Add some dark regions for obstacle simulation
        self.test_frame[100:200, 100:200] = 50  # Dark rectangle
        self.test_frame[300:350, 400:450] = 30  # Very dark rectangle
    
    def test_frame_capture(self):
        """Test frame capture functionality"""
        frame = self.camera_processor.get_frame()
        
        self.assertIsNotNone(frame)
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(frame.dtype, np.uint8)
    
    def test_obstacle_detection(self):
        """Test obstacle detection on test frame"""
        obstacles = self.camera_processor.detect_obstacles(self.test_frame)
        
        self.assertIsInstance(obstacles, list)
        
        # Should detect at least the dark regions
        self.assertGreater(len(obstacles), 0)
    
    def test_obstacle_coordinate_conversion(self):
        """Test conversion of obstacles to grid coordinates"""
        obstacles = [(100, 100), (200, 200)]  # Camera coordinates
        grid_shape = (50, 50)
        
        grid = self.camera_processor.obstacles_to_grid(obstacles, grid_shape)
        
        self.assertEqual(grid.shape, grid_shape)
        self.assertEqual(grid.dtype, int)
        
        # Check that obstacle positions are marked
        for x, y in obstacles:
            grid_x = int((x / 1280) * grid_shape[1])
            grid_y = int((y / 720) * grid_shape[0])
            if 0 <= grid_x < grid_shape[1] and 0 <= grid_y < grid_shape[0]:
                self.assertEqual(grid[grid_y, grid_x], 1)
    
    def test_detection_overlay(self):
        """Test obstacle detection overlay creation"""
        obstacles = [(10, 10), (20, 20)]
        
        overlay = self.camera_processor.draw_detection_overlay(self.test_frame, obstacles)
        
        self.assertIsNotNone(overlay)
        self.assertEqual(overlay.shape, self.test_frame.shape)
        self.assertEqual(overlay.dtype, self.test_frame.dtype)
    
    def test_performance_statistics(self):
        """Test performance statistics tracking"""
        # Process a few frames to generate statistics
        for _ in range(3):
            frame = self.camera_processor.get_frame()
            self.camera_processor.detect_obstacles(frame)
        
        stats = self.camera_processor.get_detection_statistics()
        
        self.assertIn('frames_processed', stats)
        self.assertIn('processing_time', stats)
        self.assertIn('detection_confidence', stats)
        
        self.assertGreaterEqual(stats['frames_processed'], 3)

class TestMapManager(unittest.TestCase):
    """Test cases for map management functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.map_manager = MapManager()
    
    def test_map_creation(self):
        """Test map creation and validation"""
        test_map = {
            "name": "Test Map",
            "grid_size": [50, 50],
            "permanent_obstacles": [
                {"type": "wall", "coords": [[0, 0], [0, 49], [49, 0], [49, 49]]},
                {"type": "obstacle", "coords": [[10, 10]]}
            ],
            "landmarks": {
                "start": [5, 5],
                "destination": [45, 45]
            }
        }
        
        # Test validation
        is_valid = self.map_manager.validate_map(test_map)
        self.assertTrue(is_valid)
    
    def test_invalid_map_validation(self):
        """Test validation of invalid maps"""
        invalid_maps = [
            # Missing required field
            {
                "name": "Invalid Map",
                "grid_size": [50, 50],
                # Missing permanent_obstacles
                "landmarks": {}
            },
            # Invalid coordinates
            {
                "name": "Invalid Map",
                "grid_size": [50, 50],
                "permanent_obstacles": [
                    {"type": "wall", "coords": [[100, 100]]}  # Out of bounds
                ],
                "landmarks": {}
            }
        ]
        
        for invalid_map in invalid_maps:
            is_valid = self.map_manager.validate_map(invalid_map)
            self.assertFalse(is_valid)
    
    def test_obstacle_management(self):
        """Test adding and removing obstacles"""
        # Create a simple map
        simple_map = {
            "name": "Simple Test Map",
            "grid_size": [10, 10],
            "permanent_obstacles": [],
            "landmarks": {"start": [1, 1]}
        }
        
        self.map_manager.current_map = simple_map
        self.map_manager._convert_map_to_grid()
        
        # Test adding obstacle
        success = self.map_manager.add_obstacle(5, 5, "test_obstacle")
        self.assertTrue(success)
        self.assertEqual(self.map_manager.grid_data[5, 5], 1)
        
        # Test removing obstacle
        success = self.map_manager.remove_obstacle(5, 5)
        self.assertTrue(success)
        self.assertEqual(self.map_manager.grid_data[5, 5], 0)
        
        # Test removing non-existent obstacle
        success = self.map_manager.remove_obstacle(1, 1)
        self.assertFalse(success)

class TestIntegration(unittest.TestCase):
    """Integration tests for system components"""
    
    def test_system_integration(self):
        """Test integration of path planner and map manager"""
        map_manager = MapManager()
        path_planner = PathPlanner()
        
        # Create a simple test map
        test_map = {
            "name": "Integration Test Map",
            "grid_size": [20, 20],
            "permanent_obstacles": [
                {"type": "wall", "coords": [[5, 5], [5, 6], [5, 7], [6, 7], [7, 7]]}
            ],
            "landmarks": {"start": [2, 2], "destination": [15, 15]},
            "start_position": [2, 2]
        }
        
        # Load map
        map_manager.current_map = test_map
        map_manager._convert_map_to_grid()
        
        # Plan path
        start = map_manager.get_start_position()
        goal = (15, 15)
        path = path_planner.a_star_search(map_manager.grid_data, start, goal)
        
        self.assertGreater(len(path), 0)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)
        
        # Verify path avoids obstacles
        for x, y in path:
            self.assertEqual(map_manager.grid_data[y, x], 0)

def run_all_tests():
    """Run all unit tests and return results"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestPathPlanner))
    suite.addTests(loader.loadTestsFromTestCase(TestCameraProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestMapManager))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    # Run all tests
    success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)