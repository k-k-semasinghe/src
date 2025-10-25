#!/usr/bin/env python3
"""
Automatic Spawn Position Finder
Tests positions from your maze map to find safe spawn locations
"""

import numpy as np
from PIL import Image
import os

def find_safe_spawn_positions(map_image_path, map_yaml_path):
    """
    Analyze maze map and find safe spawn positions
    """
    # Read map parameters from YAML
    resolution = 0.05  # from your maze_map.yaml
    origin = [-1.6, -1.55, 0]  # from your maze_map.yaml
    
    # Load the map image
    try:
        img = Image.open(map_image_path)
        map_array = np.array(img)
    except FileNotFoundError:
        print(f"❌ Could not find map image at: {map_image_path}")
        print(f"   Please run this script from the maze_world package directory")
        return []
    
    print("🗺️  MAP ANALYSIS")
    print("=" * 70)
    print(f"Map size: {map_array.shape}")
    print(f"Resolution: {resolution}m per pixel")
    print(f"Origin: {origin}")
    print()
    
    # Find free spaces (white pixels = 254 in grayscale)
    # In PGM: 254 = free, 0 = occupied, 205 = unknown
    free_space = (map_array >= 250)  # Free space
    
    # Robot radius in pixels (TurtleBot3 Burger is ~0.105m radius)
    robot_radius_m = 0.15  # Add safety margin
    robot_radius_px = int(robot_radius_m / resolution)
    
    print(f"Robot safety radius: {robot_radius_m}m ({robot_radius_px} pixels)")
    print()
    
    # Goal position (you said it's at [-0.36, -0.70])
    goal_world = np.array([-0.36, -0.70])
    
    # Convert world coordinates to pixel coordinates
    def world_to_pixel(x, y):
        px = int((x - origin[0]) / resolution)
        py = int((y - origin[1]) / resolution)
        return px, py
    
    def pixel_to_world(px, py):
        x = px * resolution + origin[0]
        y = py * resolution + origin[1]
        return x, y
    
    # Find safe positions
    safe_positions = []
    
    # Test a grid of positions
    print("🔍 Scanning for safe spawn positions...")
    print()
    
    test_positions_world = [
        # Original and nearby
        [0.3, 0.0],
        [0.5, 0.0],
        [0.7, 0.0],
        [0.2, 0.0],
        [0.0, 0.0],
        
        # Offset positions
        [0.3, 0.2],
        [0.3, -0.2],
        [0.5, 0.2],
        [0.5, -0.2],
        
        # Different x positions
        [-0.3, 0.0],
        [-0.5, 0.0],
        [0.0, 0.3],
        [0.0, -0.3],
        
        # Corners and edges
        [0.8, 0.0],
        [0.0, 0.8],
        [-0.8, 0.0],
        [0.0, -0.8],
    ]
    
    for world_pos in test_positions_world:
        x, y = world_pos
        px, py = world_to_pixel(x, y)
        
        # Check if position is within map bounds
        if px < robot_radius_px or px >= map_array.shape[1] - robot_radius_px:
            continue
        if py < robot_radius_px or py >= map_array.shape[0] - robot_radius_px:
            continue
        
        # Check if position and surroundings are free
        region = map_array[
            py - robot_radius_px:py + robot_radius_px,
            px - robot_radius_px:px + robot_radius_px
        ]
        
        # Position is safe if entire region is free space
        if np.all(region >= 250):
            # Calculate clearances
            min_clearance = robot_radius_m
            
            # Check clearances in 4 directions
            clearances = []
            for direction in ['front', 'back', 'left', 'right']:
                if direction == 'front':  # +x
                    check_region = map_array[py-5:py+5, px:px+50]
                elif direction == 'back':  # -x
                    check_region = map_array[py-5:py+5, px-50:px]
                elif direction == 'left':  # +y
                    check_region = map_array[py:py+50, px-5:px+5]
                else:  # right: -y
                    check_region = map_array[py-50:py, px-5:px+5]
                
                # Find first obstacle
                for i in range(check_region.shape[0] if direction in ['left', 'right'] else check_region.shape[1]):
                    if direction in ['front', 'back']:
                        if np.any(check_region[:, i] < 250):
                            clearances.append(i * resolution)
                            break
                    else:
                        if np.any(check_region[i, :] < 250):
                            clearances.append(i * resolution)
                            break
                else:
                    clearances.append(2.5)  # Max clearance
            
            front_clear, back_clear, left_clear, right_clear = clearances
            min_clearance = min(clearances)
            
            # Calculate distance to goal
            dist_to_goal = np.linalg.norm(np.array([x, y]) - goal_world)
            
            safe_positions.append({
                'position': [x, y],
                'front_clearance': front_clear,
                'back_clearance': back_clear,
                'left_clearance': left_clear,
                'right_clearance': right_clear,
                'min_clearance': min_clearance,
                'distance_to_goal': dist_to_goal
            })
    
    # Sort by clearance quality
    safe_positions.sort(key=lambda p: p['min_clearance'], reverse=True)
    
    print("✅ SAFE SPAWN POSITIONS FOUND:")
    print("=" * 70)
    print()
    
    if len(safe_positions) == 0:
        print("❌ No safe positions found!")
        print("   Try manually checking your maze in Gazebo")
        return []
    
    # Print top 10 positions
    for i, pos in enumerate(safe_positions[:10]):
        print(f"🎯 Option {i+1}: [{pos['position'][0]:.2f}, {pos['position'][1]:.2f}]")
        print(f"   Front:   {pos['front_clearance']:.3f}m  {'✅' if pos['front_clearance'] > 0.35 else '⚠️ ' if pos['front_clearance'] > 0.25 else '❌'}")
        print(f"   Back:    {pos['back_clearance']:.3f}m")
        print(f"   Left:    {pos['left_clearance']:.3f}m   {'✅' if pos['left_clearance'] > 0.35 else '⚠️ '}")
        print(f"   Right:   {pos['right_clearance']:.3f}m   {'✅' if pos['right_clearance'] > 0.35 else '⚠️ '}")
        print(f"   Min:     {pos['min_clearance']:.3f}m")
        print(f"   To goal: {pos['distance_to_goal']:.3f}m")
        print()
    
    print("=" * 70)
    print()
    print("📋 RECOMMENDED SPAWN POSITION:")
    best = safe_positions[0]
    print(f"   Position: [{best['position'][0]:.2f}, {best['position'][1]:.2f}]")
    print()
    print("📝 UPDATE THESE FILES:")
    print()
    print("1️⃣  maze.launch.py (around line 24-25):")
    print(f"   x_pose = LaunchConfiguration('x_pose', default='{best['position'][0]:.2f}')")
    print(f"   y_pose = LaunchConfiguration('y_pose', default='{best['position'][1]:.2f}')")
    print()
    print("2️⃣  dqn_trainer.py (around line 70):")
    print(f"   self.start_position = np.array([{best['position'][0]:.2f}, {best['position'][1]:.2f}])")
    print()
    
    return safe_positions


def main():
    print()
    print("🤖 AUTOMATIC SPAWN POSITION FINDER")
    print("=" * 70)
    print()
    
    # Common paths to check
    possible_paths = [
        # If running from package root
        "maps/maze_map.pgm",
        "../maps/maze_map.pgm",
        "../../maps/maze_map.pgm",
        
        # If running from scripts directory
        "../maps/maze_map.pgm",
        
        # Full workspace path (modify if needed)
        os.path.expanduser("~/turtlebot3_ws/src/maze_world/maps/maze_map.pgm"),
    ]
    
    map_path = None
    for path in possible_paths:
        if os.path.exists(path):
            map_path = path
            break
    
    if map_path is None:
        print("❌ Could not find maze_map.pgm automatically")
        print()
        print("Please provide the full path to your maze_map.pgm file:")
        map_path = input("Path: ").strip()
        
        if not os.path.exists(map_path):
            print(f"❌ File not found: {map_path}")
            return
    
    print(f"📁 Using map: {map_path}")
    print()
    
    yaml_path = map_path.replace('.pgm', '.yaml')
    
    safe_positions = find_safe_spawn_positions(map_path, yaml_path)
    
    if len(safe_positions) > 0:
        print("✅ SUCCESS! Use the recommended position above.")
    else:
        print("❌ FALLBACK: Try these common positions manually:")
        print("   [0.0, 0.0]")
        print("   [0.5, 0.0]")
        print("   [-0.5, 0.0]")


if __name__ == '__main__':
    main()