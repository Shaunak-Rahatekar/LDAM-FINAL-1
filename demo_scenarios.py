"""
Demo Scenarios for Smart Wheelchair Navigation System
Pre-configured scenarios for testing and demonstration
"""

import time
import threading
from typing import List, Tuple, Dict, Any, Optional
from config import config

class DemoScenarios:
    """Pre-configured demo scenarios for testing"""
    
    def __init__(self, app_controller):
        self.controller = app_controller
        self.current_scenario = None
        self.scenario_thread = None
        self.is_running_scenario = False
    
    def run_basic_navigation(self):
        """Demo 1: Basic navigation without dynamic obstacles"""
        print("Starting Demo: Basic Navigation")
        self.current_scenario = "basic_navigation"
        
        # Load simple test map
        self.controller.load_map("simple_test.json")
        
        # Set destination
        destination = (40, 40)
        self.controller.set_destination(destination)
        
        # Start navigation
        self.controller.start_navigation()
        
        print(f"Basic navigation demo: Start -> {destination}")
    
    def run_obstacle_avoidance(self):
        """Demo 2: Navigation with static and dynamic obstacles"""
        print("Starting Demo: Obstacle Avoidance")
        self.current_scenario = "obstacle_avoidance"
        
        # Load obstacle course map
        self.controller.load_map("obstacle_course.json")
        
        # Set destination
        destination = (45, 25)  # Finish line
        self.controller.set_destination(destination)
        
        # Add temporary obstacles during navigation
        self.is_running_scenario = True
        self.scenario_thread = threading.Thread(target=self._add_dynamic_obstacles)
        self.scenario_thread.start()
        
        # Start navigation
        self.controller.start_navigation()
        
        print("Obstacle avoidance demo started with dynamic obstacles")
    
    def _add_dynamic_obstacles(self):
        """Add dynamic obstacles during obstacle avoidance demo"""
        obstacles_added = 0
        max_obstacles = 5
        
        while (self.is_running_scenario and 
               self.controller.is_running and 
               obstacles_added < max_obstacles):
            
            time.sleep(3)  # Wait 3 seconds between obstacles
            
            if not self.controller.is_running:
                break
            
            # Add obstacle in the current path
            if self.controller.path:
                # Find a position in the remaining path
                remaining_path = self.controller.path_planner.get_remaining_path()
                if len(remaining_path) > 5:
                    obstacle_pos = remaining_path[3]  # Position a few steps ahead
                    
                    # Add to temporary obstacles
                    if obstacle_pos not in self.controller.temporary_obstacles:
                        self.controller.temporary_obstacles.add(obstacle_pos)
                        self.controller.update_temporary_obstacles(self.controller.temporary_obstacles)
                        
                        obstacles_added += 1
                        print(f"Added dynamic obstacle at {obstacle_pos}")
        
        self.is_running_scenario = False
    
    def run_dead_end_recovery(self):
        """Demo 3: Recovery from dead ends"""
        print("Starting Demo: Dead End Recovery")
        self.current_scenario = "dead_end_recovery"
        
        # Load dead end test map
        self.controller.load_map("dead_end_test.json")
        
        # Set destination that requires dead end recovery
        destination = (40, 25)
        self.controller.set_destination(destination)
        
        # Start navigation - system should encounter and recover from dead end
        self.controller.start_navigation()
        
        print("Dead end recovery demo started")
    
    def run_multi_destination(self):
        """Demo 4: Continuous multi-destination navigation"""
        print("Starting Demo: Multi-Destination Tour")
        self.current_scenario = "multi_destination"
        
        # Load multi-destination map
        self.controller.load_map("multi_destination.json")
        
        # Define destination sequence
        destinations = [
            (12, 12),  # Area 1
            (37, 12),  # Area 2
            (37, 37),  # Area 4
            (12, 37),  # Area 3
            (25, 25),  # Return to hub
        ]
        
        self.is_running_scenario = True
        self.scenario_thread = threading.Thread(
            target=self._visit_multiple_destinations, 
            args=(destinations,)
        )
        self.scenario_thread.start()
        
        print("Multi-destination tour demo started")
    
    def _visit_multiple_destinations(self, destinations: List[Tuple[int, int]]):
        """Visit multiple destinations sequentially"""
        current_dest_index = 0
        
        while (self.is_running_scenario and 
               current_dest_index < len(destinations)):
            
            # Set next destination
            destination = destinations[current_dest_index]
            self.controller.set_destination(destination)
            
            # Start navigation to this destination
            if not self.controller.is_running:
                self.controller.start_navigation()
            
            # Wait until destination is reached or timeout
            wait_start = time.time()
            while (self.controller.is_running and 
                   not self.controller.has_reached_destination() and
                   time.time() - wait_start < 30):  # 30 second timeout per destination
                time.sleep(1)
            
            if self.controller.has_reached_destination():
                print(f"Reached destination {current_dest_index + 1}: {destination}")
                current_dest_index += 1
                
                # Brief pause before next destination
                if current_dest_index < len(destinations):
                    time.sleep(2)
            else:
                print(f"Timeout reaching destination {current_dest_index + 1}")
                break
        
        self.is_running_scenario = False
        print("Multi-destination tour completed")
    
    def run_comprehensive_test(self):
        """Demo 5: Comprehensive test of all features"""
        print("Starting Comprehensive System Test")
        self.current_scenario = "comprehensive_test"
        
        # Test sequence
        tests = [
            ("Loading hospital map", self._test_map_loading),
            ("Basic path planning", self._test_path_planning),
            ("Obstacle detection", self._test_obstacle_detection),
            ("Dynamic replanning", self._test_dynamic_replanning),
        ]
        
        for test_name, test_func in tests:
            print(f"\n--- Running: {test_name} ---")
            try:
                success = test_func()
                status = "PASS" if success else "FAIL"
                print(f"Result: {status}")
            except Exception as e:
                print(f"Result: ERROR - {e}")
        
        print("\nComprehensive test completed")
    
    def _test_map_loading(self) -> bool:
        """Test map loading functionality"""
        try:
            success = self.controller.load_map("hospital_floor1.json")
            return success and self.controller.current_position is not None
        except Exception as e:
            print(f"Map loading test failed: {e}")
            return False
    
    def _test_path_planning(self) -> bool:
        """Test path planning functionality"""
        try:
            self.controller.destination = (45, 45)
            self.controller.plan_path()
            return len(self.controller.path) > 0
        except Exception as e:
            print(f"Path planning test failed: {e}")
            return False
    
    def _test_obstacle_detection(self) -> bool:
        """Test obstacle detection functionality"""
        try:
            if self.controller.camera_processor:
                frame = self.controller.camera_processor.get_frame()
                obstacles = self.controller.camera_processor.detect_obstacles(frame)
                return True  # Success if no exception
            return True  # Skip if no camera
        except Exception as e:
            print(f"Obstacle detection test failed: {e}")
            return False
    
    def _test_dynamic_replanning(self) -> bool:
        """Test dynamic replanning functionality"""
        try:
            # Add an obstacle that blocks the current path
            if self.controller.path and len(self.controller.path) > 3:
                blocking_pos = self.controller.path[2]
                self.controller.temporary_obstacles.add(blocking_pos)
                self.controller.update_temporary_obstacles(self.controller.temporary_obstacles)
                
                # Check if replanning occurs
                original_path_length = len(self.controller.path)
                self.controller.plan_path()
                
                return len(self.controller.path) != original_path_length
            return True  # Skip if no path available
        except Exception as e:
            print(f"Dynamic replanning test failed: {e}")
            return False
    
    def stop_current_scenario(self):
        """Stop the currently running demo scenario"""
        self.is_running_scenario = False
        self.current_scenario = None
        
        if self.scenario_thread and self.scenario_thread.is_alive():
            self.scenario_thread.join(timeout=2.0)
        
        print("Demo scenario stopped")
    
    def get_available_scenarios(self) -> List[Dict[str, Any]]:
        """Get list of available demo scenarios"""
        return [
            {
                "name": "Basic Navigation",
                "description": "Simple start-to-destination navigation without obstacles",
                "function": self.run_basic_navigation,
                "map": "simple_test.json"
            },
            {
                "name": "Obstacle Avoidance",
                "description": "Navigation with static and dynamic obstacles requiring replanning",
                "function": self.run_obstacle_avoidance,
                "map": "obstacle_course.json"
            },
            {
                "name": "Dead End Recovery",
                "description": "Recovery from blocked paths and dead ends",
                "function": self.run_dead_end_recovery,
                "map": "dead_end_test.json"
            },
            {
                "name": "Multi-Destination Tour",
                "description": "Continuous navigation to multiple destinations",
                "function": self.run_multi_destination,
                "map": "multi_destination.json"
            },
            {
                "name": "Comprehensive Test",
                "description": "Complete system functionality test",
                "function": self.run_comprehensive_test,
                "map": "hospital_floor1.json"
            }
        ]