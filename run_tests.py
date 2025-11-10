#!/usr/bin/env python3
"""
Test Runner for Smart Wheelchair Navigation System
Run unit tests and demo scenarios
"""

import sys
import os
import argparse
from test_smart_wheelchair import run_all_tests

def main():
    parser = argparse.ArgumentParser(description='Smart Wheelchair Test Runner')
    parser.add_argument('--unit-tests', action='store_true', 
                       help='Run unit tests')
    parser.add_argument('--demo', choices=['basic', 'obstacle', 'deadend', 'multi', 'all'],
                       help='Run demo scenarios')
    parser.add_argument('--create-maps', action='store_true',
                       help='Create test maps')
    
    args = parser.parse_args()
    
    if args.unit_tests:
        print("Running Unit Tests...")
        print("=" * 50)
        success = run_all_tests()
        if success:
            print("\n✅ All unit tests passed!")
        else:
            print("\n❌ Some unit tests failed!")
            sys.exit(1)
    
    if args.create_maps:
        from map_manager import MapManager
        map_manager = MapManager()
        map_manager.create_default_maps()
        print("✅ Test maps created successfully!")
    
    if args.demo:
        from main import SmartWheelchairApp
        from demo_scenarios import DemoScenarios
        
        print(f"Running Demo: {args.demo}")
        print("=" * 50)
        
        # Note: Demos require GUI and are best run through the main application
        print("Please run demos through the main application GUI")
        print("or use: python main.py --demo")
    
    if not any(vars(args).values()):
        parser.print_help()

if __name__ == '__main__':
    main()