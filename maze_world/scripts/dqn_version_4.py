#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
import json
import time


class DQNetwork(nn.Module):
    """Network optimized for navigation"""
    def __init__(self, state_size, action_size, hidden_size=256):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class TightMazeDQNTrainer(Node):
    """DQN Trainer - SPAWN ORIENTATION FIXED"""
    
    def __init__(self):
        super().__init__('tight_maze_trainer')
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005
        self.batch_size = 64
        self.target_update_freq = 10
        
        # State and action space
        self.laser_bins = 16
        self.state_size = self.laser_bins + 5  # +1 for facing_wall flag
        self.action_size = 5  # left, slight_left, forward, slight_right, right
        
        # TurtleBot3 specs
        self.max_linear_vel = 0.22
        self.max_angular_vel = 2.84
        
        # Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNetwork(self.state_size, self.action_size).to(self.device)
        self.target_net = DQNetwork(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(capacity=50000)
        
        # ROS2 setup
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.reset_world_client = self.create_client(Empty, '/reset_world')
        
        # State variables
        self.current_odom = None
        self.current_scan = None
        self.goal_position = np.array([-0.36, -0.70])
        self.start_position = np.array([0.3, 0.0])
        self.previous_distance = None
        self.previous_position = None
        self.initial_orientation_done = False  # New: track if robot has oriented
        
        # Training variables
        self.episode = 0
        self.step = 0
        self.total_steps = 0
        self.episode_reward = 0
        self.episode_rewards = []
        self.success_count = 0
        self.collision_count = 0
        self.timeout_count = 0
        self.max_steps_per_episode = 300
        
        # Progress tracking
        self.best_distance = float('inf')
        self.recent_progress = deque(maxlen=50)
        
        # Action tracking
        self.action_counts = [0] * self.action_size
        
        # Results tracking
        self.results_dir = 'results'
        os.makedirs(self.results_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.results_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.get_logger().info(f'🗺️  SPAWN-AWARE Maze Trainer initialized. Device: {self.device}')
        self.get_logger().info(f'State size: {self.state_size}, Action size: {self.action_size}')
        self.get_logger().info('✅ Initial orientation phase enabled!')
        
    def odom_callback(self, msg):
        self.current_odom = msg
        
    def scan_callback(self, msg):
        self.current_scan = msg
        
    def is_facing_wall(self):
        """Check if robot is facing a wall (front clearance < 0.35m)"""
        if self.current_scan is None:
            return False
        
        ranges = np.array(self.current_scan.ranges)
        ranges = ranges[~np.isnan(ranges)]
        ranges = ranges[~np.isinf(ranges)]
        
        if len(ranges) == 0:
            return False
        
        # Check front 60 degrees
        front_range = len(ranges) // 6
        front_clearance = np.min(ranges[:front_range])
        
        return front_clearance < 0.35
    
    def get_robot_yaw(self):
        """Get robot's current yaw angle"""
        if self.current_odom is None:
            return 0.0
        
        orientation = self.current_odom.pose.pose.orientation
        robot_yaw = np.arctan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y**2 + orientation.z**2)
        )
        return robot_yaw
    
    def get_state(self):
        """Enhanced state with facing_wall flag"""
        if self.current_odom is None or self.current_scan is None:
            return None
            
        # Process laser scan
        ranges = np.array(self.current_scan.ranges)
        ranges[np.isinf(ranges)] = 3.5
        ranges[np.isnan(ranges)] = 0.0
        ranges = np.clip(ranges, 0, 3.5)
        
        # Create bins
        bin_size = len(ranges) // self.laser_bins
        laser_state = []
        for i in range(self.laser_bins):
            start_idx = i * bin_size
            end_idx = start_idx + bin_size
            if end_idx <= len(ranges):
                laser_state.append(np.min(ranges[start_idx:end_idx]))
            else:
                laser_state.append(np.min(ranges[start_idx:]))
        
        laser_state = np.array(laser_state) / 3.5  # Normalize
        
        # Robot position
        robot_pos = np.array([
            self.current_odom.pose.pose.position.x,
            self.current_odom.pose.pose.position.y
        ])
        
        # Goal info
        goal_vec = self.goal_position - robot_pos
        goal_dist = np.linalg.norm(goal_vec)
        goal_angle = np.arctan2(goal_vec[1], goal_vec[0])
        
        # Robot orientation
        robot_yaw = self.get_robot_yaw()
        
        relative_goal_angle = goal_angle - robot_yaw
        relative_goal_angle = np.arctan2(np.sin(relative_goal_angle), np.cos(relative_goal_angle))
        
        # Clearances
        front_range = len(ranges) // 8
        front_clearance = np.min(ranges[:front_range])
        
        left_idx = len(ranges) // 4
        right_idx = 3 * len(ranges) // 4
        side_clearance = min(
            np.min(ranges[left_idx-10:left_idx+10]) if left_idx < len(ranges) else 3.5,
            np.min(ranges[right_idx-10:right_idx+10]) if right_idx < len(ranges) else 3.5
        )
        
        # NEW: Facing wall flag
        facing_wall = 1.0 if front_clearance < 0.35 else 0.0
        
        state = np.concatenate([
            laser_state,
            [goal_dist / 2.0,
             relative_goal_angle / np.pi,
             front_clearance / 3.5,
             side_clearance / 3.5,
             facing_wall]
        ])
        
        return state
        
    def select_action(self, state):
        """Action selection with orientation awareness"""
        min_laser = self.get_min_laser_distance()
        front_clearance = state[-3] * 3.5  # Denormalize
        facing_wall = state[-1] > 0.5  # Binary flag
        
        # PHASE 1: Initial orientation (if facing wall)
        if facing_wall and not self.initial_orientation_done:
            # Force turning until not facing wall
            if random.random() < self.epsilon or self.epsilon > 0.3:
                # Random turn direction
                return random.choice([0, 4])  # Hard left or hard right
            else:
                # Let network decide which way to turn
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.policy_net(state_tensor)
                    # Block forward actions
                    q_values[0, 1] = -1000
                    q_values[0, 2] = -1000
                    q_values[0, 3] = -1000
                    action = q_values.argmax().item()
                    self.action_counts[action] += 1
                    return action
        
        # Mark orientation as done once we have clearance
        if not facing_wall and front_clearance > 0.30:
            if not self.initial_orientation_done:
                self.initial_orientation_done = True
                self.get_logger().info('✅ Initial orientation complete!')
        
        # PHASE 2: Normal navigation
        # Epsilon-greedy with safety
        if random.random() < self.epsilon:
            # Smart exploration
            if min_laser < 0.19:  # Very close - must turn
                return random.choice([0, 4])
            elif min_laser < 0.25:  # Close - prefer turning but allow slight turns with forward
                if random.random() < 0.5:
                    return random.choice([0, 4])  # Hard turns
                else:
                    return random.choice([1, 3])  # Slight turns (include forward)
            else:  # Safe - any action
                return random.randint(0, self.action_size - 1)
        
        # Exploitation
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            
            # ONLY block forward if VERY close
            if min_laser < 0.19:
                q_values[0, 2] = -1000
                
            action = q_values.argmax().item()
            self.action_counts[action] += 1
            return action
            
    def execute_action(self, action):
        """5 actions for smooth navigation"""
        cmd = Twist()
        
        if action == 0:  # Hard left
            cmd.linear.x = 0.0
            cmd.angular.z = 1.5
        elif action == 1:  # Slight left + forward
            cmd.linear.x = 0.15
            cmd.angular.z = 0.8
        elif action == 2:  # Forward
            cmd.linear.x = 0.18
            cmd.angular.z = 0.0
        elif action == 3:  # Slight right + forward
            cmd.linear.x = 0.15
            cmd.angular.z = -0.8
        elif action == 4:  # Hard right
            cmd.linear.x = 0.0
            cmd.angular.z = -1.5
            
        self.cmd_vel_pub.publish(cmd)
        
    def get_min_laser_distance(self):
        """Get minimum laser distance"""
        if self.current_scan is None:
            return float('inf')
            
        ranges = np.array(self.current_scan.ranges)
        ranges = ranges[~np.isnan(ranges)]
        ranges = ranges[~np.isinf(ranges)]
        
        return np.min(ranges) if len(ranges) > 0 else float('inf')
        
    def calculate_reward(self, state, next_state, done, action):
        """Reward system with orientation phase"""
        if self.current_odom is None or self.current_scan is None:
            return 0.0
            
        robot_pos = np.array([
            self.current_odom.pose.pose.position.x,
            self.current_odom.pose.pose.position.y
        ])
        
        current_dist = np.linalg.norm(self.goal_position - robot_pos)
        min_laser = self.get_min_laser_distance()
        front_clearance = next_state[-3] * 3.5
        facing_wall = next_state[-1] > 0.5
        
        if self.previous_distance is None:
            self.previous_distance = current_dist
        if self.previous_position is None:
            self.previous_position = robot_pos.copy()
        
        reward = 0.0
        
        # === TERMINAL REWARDS ===
        
        # GOAL REACHED
        if current_dist < 0.25:
            self.get_logger().info('🎉🎉🎉 GOAL REACHED!')
            self.success_count += 1
            return 1000.0
        
        # COLLISION
        if min_laser < 0.18:
            self.collision_count += 1
            return -100.0
        
        # === PHASE 1: ORIENTATION REWARDS (when facing wall) ===
        
        if facing_wall or not self.initial_orientation_done:
            # Reward turning away from wall
            if action in [0, 4]:  # Turning actions
                reward += 5.0
            
            # Big reward for getting clearance
            if front_clearance > 0.30:
                reward += 30.0
                
            # Reward for NOT colliding during orientation
            reward += 2.0
            
            # Small penalty for forward when facing wall
            if action == 2:
                reward -= 10.0
                
        # === PHASE 2: NAVIGATION REWARDS (after orientation) ===
        else:
            # 1. MASSIVE REWARD FOR PROGRESS
            distance_delta = self.previous_distance - current_dist
            if distance_delta > 0:
                reward += distance_delta * 600.0  # Huge multiplier
                
                self.recent_progress.append(distance_delta)
                
                if distance_delta > 0.02:
                    reward += 80.0
                    self.get_logger().info(f'🚀 Big progress! Δ{distance_delta:.4f}m')
                
                # New best
                if current_dist < self.best_distance:
                    improvement = self.best_distance - current_dist
                    self.best_distance = current_dist
                    reward += 300.0
                    progress_pct = (1.0 - current_dist / 0.96) * 100
                    self.get_logger().info(f'🏆 NEW BEST: {current_dist:.3f}m ({progress_pct:.1f}%)')
            else:
                # Penalty for moving away
                reward += distance_delta * 150.0
            
            # 2. ACTUAL MOVEMENT REWARD
            actual_movement = np.linalg.norm(robot_pos - self.previous_position)
            if actual_movement > 0.01:
                reward += actual_movement * 80.0
            
            # 3. FORWARD ACTION BONUS
            if action == 2 and min_laser > 0.25:
                reward += 15.0
            elif action in [1, 3] and min_laser > 0.23:  # Slight turns with forward
                reward += 10.0
            
            # 4. SURVIVAL
            reward += 1.0
            
            # 5. SAFETY
            if min_laser > 0.30:
                reward += 3.0
            elif min_laser < 0.22:
                reward -= 8.0
            
            # 6. HEADING ALIGNMENT (when safe and moving)
            if min_laser > 0.25 and actual_movement > 0.005:
                goal_angle_error = abs(next_state[self.laser_bins + 1])
                heading_reward = (1.0 - goal_angle_error) * 8.0
                reward += heading_reward
            
            # 7. MILESTONES
            if self.step == 50:
                reward += 40.0
                self.get_logger().info('💪 50 steps!')
            elif self.step == 100:
                reward += 70.0
                self.get_logger().info('💪💪 100 steps!')
            elif self.step == 150:
                reward += 100.0
                self.get_logger().info('💪💪💪 150 steps!')
        
        # Update tracking
        self.previous_distance = current_dist
        self.previous_position = robot_pos.copy()
        
        return reward
        
    def check_done(self):
        """Check if episode should end"""
        if self.current_odom is None or self.current_scan is None:
            return False
            
        robot_pos = np.array([
            self.current_odom.pose.pose.position.x,
            self.current_odom.pose.pose.position.y
        ])
        
        goal_dist = np.linalg.norm(self.goal_position - robot_pos)
        
        if goal_dist < 0.25:
            return True
        
        min_laser = self.get_min_laser_distance()
        if min_laser < 0.18:
            return True
            
        if self.step >= self.max_steps_per_episode:
            self.timeout_count += 1
            return True
            
        return False
        
    def reset_environment(self):
        """Reset environment"""
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)
        
        timeout_counter = 0
        while not self.reset_world_client.wait_for_service(timeout_sec=1.0):
            timeout_counter += 1
            if timeout_counter > 10:
                return False
        
        request = Empty.Request()
        future = self.reset_world_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        
        time.sleep(0.2)
        
        self.current_odom = None
        self.current_scan = None
        self.previous_distance = None
        self.previous_position = None
        self.initial_orientation_done = False  # Reset orientation flag
        
        wait_counter = 0
        while (self.current_odom is None or self.current_scan is None) and wait_counter < 50:
            rclpy.spin_once(self, timeout_sec=0.1)
            wait_counter += 1
        
        if self.current_odom is None or self.current_scan is None:
            return False
        
        self.step = 0
        self.episode_reward = 0
        
        return True
        
    def train_step(self):
        """Training step"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values (Double DQN)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Loss
        loss = nn.SmoothL1Loss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        
        return loss.item()
        
    def save_checkpoint(self):
        """Save checkpoint"""
        checkpoint = {
            'episode': self.episode,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'success_count': self.success_count,
            'collision_count': self.collision_count,
            'timeout_count': self.timeout_count,
            'best_distance': self.best_distance
        }
        
        path = os.path.join(self.checkpoint_dir, f'maze_ep{self.episode}.pth')
        torch.save(checkpoint, path)
        self.get_logger().info(f'💾 Saved: {path}')
        
        # Action distribution
        total_actions = sum(self.action_counts)
        action_dist = [count / max(1, total_actions) * 100 for count in self.action_counts]
        
        stats = {
            'episode': self.episode,
            'total_steps': self.total_steps,
            'epsilon': self.epsilon,
            'success_count': self.success_count,
            'collision_count': self.collision_count,
            'timeout_count': self.timeout_count,
            'best_distance': self.best_distance,
            'avg_reward_last_100': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards),
            'episode_rewards': self.episode_rewards,
            'action_distribution': {
                'hard_left': action_dist[0],
                'slight_left': action_dist[1],
                'forward': action_dist[2],
                'slight_right': action_dist[3],
                'hard_right': action_dist[4]
            }
        }
        
        stats_path = os.path.join(self.results_dir, 'training_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
            
    def train(self, num_episodes=2000):
        """Main training loop"""
        self.get_logger().info(f'🚀 Starting ORIENTATION-AWARE training for {num_episodes} episodes')
        self.get_logger().info('✅ Two-phase learning: 1) Turn from wall, 2) Navigate to goal')
        
        while self.current_odom is None or self.current_scan is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        self.get_logger().info('✔ Sensors ready!')
        
        for episode in range(num_episodes):
            self.episode = episode
            
            reset_success = self.reset_environment()
            if not reset_success:
                continue
            
            state = self.get_state()
            if state is None:
                continue
                
            episode_loss = 0
            loss_count = 0
            
            while not self.check_done():
                action = self.select_action(state)
                self.execute_action(action)
                
                time.sleep(0.15)
                rclpy.spin_once(self, timeout_sec=0.05)
                
                next_state = self.get_state()
                if next_state is None:
                    break
                    
                done = self.check_done()
                reward = self.calculate_reward(state, next_state, done, action)
                
                self.replay_buffer.push(state, action, reward, next_state, float(done))
                
                # Train every 4 steps
                if self.step % 4 == 0:
                    loss = self.train_step()
                    if loss is not None:
                        episode_loss += loss
                        loss_count += 1
                    
                self.episode_reward += reward
                state = next_state
                self.step += 1
                self.total_steps += 1
                
            self.episode_rewards.append(self.episode_reward)
            
            # Epsilon decay
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Update target network
            if episode % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                
            # Logging
            avg_loss = episode_loss / loss_count if loss_count > 0 else 0
            avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
            success_rate = self.success_count / max(1, episode + 1) * 100
            
            # Progress percentage
            progress_pct = (1.0 - self.best_distance / 0.96) * 100 if self.best_distance != float('inf') else 0.0
            
            # Recent progress
            avg_progress = np.mean(list(self.recent_progress)) if len(self.recent_progress) > 0 else 0
            
            self.get_logger().info(
                f'📊 Ep {episode} | Steps: {self.step} | R: {self.episode_reward:.1f} | '
                f'Avg: {avg_reward:.1f} | ✓{self.success_count} ✗{self.collision_count} '
                f'⏱️{self.timeout_count} | Best: {self.best_distance:.3f}m ({progress_pct:.1f}%) | '
                f'ε: {self.epsilon:.3f} | ΔAvg: {avg_progress:.5f}'
            )
            
            # Log action distribution every 50 episodes
            if (episode + 1) % 50 == 0:
                total_actions = sum(self.action_counts)
                if total_actions > 0:
                    self.get_logger().info(
                        f'🎯 Actions: L:{self.action_counts[0]/total_actions*100:.1f}% '
                        f'SL:{self.action_counts[1]/total_actions*100:.1f}% '
                        f'F:{self.action_counts[2]/total_actions*100:.1f}% '
                        f'SR:{self.action_counts[3]/total_actions*100:.1f}% '
                        f'R:{self.action_counts[4]/total_actions*100:.1f}%'
                    )
                self.action_counts = [0] * self.action_size
            
            # Save checkpoint
            if (episode + 1) % 100 == 0:
                self.save_checkpoint()
                
        self.get_logger().info(f'🏁 Training complete!')
        self.get_logger().info(f'Best distance achieved: {self.best_distance:.3f}m')
        self.save_checkpoint()


def main(args=None):
    rclpy.init(args=args)
    trainer = TightMazeDQNTrainer()
    
    try:
        trainer.train(num_episodes=2000)
    except KeyboardInterrupt:
        trainer.get_logger().info('⚠️ Interrupted')
    finally:
        stop_cmd = Twist()
        trainer.cmd_vel_pub.publish(stop_cmd)
        trainer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()