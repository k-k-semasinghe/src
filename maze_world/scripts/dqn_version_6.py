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
    """DQN with MANDATORY 90° TURN before forward movement"""
    
    def __init__(self):
        super().__init__('tight_maze_trainer')
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.10
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.0003
        self.batch_size = 64
        self.target_update_freq = 20
        
        # State and action space
        self.laser_bins = 16
        self.state_size = self.laser_bins + 5  # +1 for orientation_complete flag
        self.action_size = 3  # turn_left, forward, turn_right
        
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
        
        # ORIENTATION TRACKING
        self.spawn_yaw = None
        self.orientation_complete = False
        self.turned_enough = False
        
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
        self.consecutive_forward_steps = 0
        self.max_consecutive_forward = 0
        
        # Action tracking
        self.action_counts = [0] * self.action_size
        self.turn_direction_chosen = None  # Track which way robot is turning
        
        # Results tracking
        self.results_dir = 'results'
        os.makedirs(self.results_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.results_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.get_logger().info(f'🔄 MANDATORY 90° TURN Trainer initialized. Device: {self.device}')
        self.get_logger().info(f'State size: {self.state_size}, Actions: LEFT, FORWARD, RIGHT')
        self.get_logger().info('✅ Robot MUST turn 90° before moving forward!')
        
    def odom_callback(self, msg):
        self.current_odom = msg
        
    def scan_callback(self, msg):
        self.current_scan = msg
    
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
    
    def check_orientation_complete(self):
        """Check if robot has turned ~90 degrees from spawn"""
        if self.spawn_yaw is None or self.orientation_complete:
            return self.orientation_complete
        
        current_yaw = self.get_robot_yaw()
        yaw_diff = abs(current_yaw - self.spawn_yaw)
        
        # Normalize to [-pi, pi]
        while yaw_diff > np.pi:
            yaw_diff -= 2 * np.pi
        yaw_diff = abs(yaw_diff)
        
        # Check if turned approximately 90 degrees (1.57 radians = 90°)
        # Allow some tolerance: 70-110 degrees
        if 1.22 < yaw_diff < 1.92:  # 70° to 110°
            if not self.orientation_complete:
                self.orientation_complete = True
                self.get_logger().info(f'✅ Orientation complete! Turned {np.degrees(yaw_diff):.1f}°')
            return True
        
        # Mark as "turned enough" if > 45 degrees (can start considering forward)
        if yaw_diff > 0.79:  # 45°
            self.turned_enough = True
        
        return False
    
    def get_front_clearance(self):
        """Get front clearance distance"""
        if self.current_scan is None:
            return 0.0
        
        ranges = np.array(self.current_scan.ranges)
        ranges = ranges[~np.isnan(ranges)]
        ranges = ranges[~np.isinf(ranges)]
        
        if len(ranges) == 0:
            return 0.0
        
        # Front 60 degrees
        front_range = len(ranges) // 6
        return np.min(ranges[:front_range])
    
    def get_state(self):
        """State with orientation_complete flag"""
        if self.current_odom is None or self.current_scan is None:
            return None
            
        # Laser scan
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
        
        laser_state = np.array(laser_state) / 3.5
        
        # Robot position
        robot_pos = np.array([
            self.current_odom.pose.pose.position.x,
            self.current_odom.pose.pose.position.y
        ])
        
        # Goal info
        goal_vec = self.goal_position - robot_pos
        goal_dist = np.linalg.norm(goal_vec)
        goal_angle = np.arctan2(goal_vec[1], goal_vec[0])
        
        # Robot yaw
        robot_yaw = self.get_robot_yaw()
        
        relative_goal_angle = goal_angle - robot_yaw
        relative_goal_angle = np.arctan2(np.sin(relative_goal_angle), np.cos(relative_goal_angle))
        
        # Clearances
        front_clearance = self.get_front_clearance()
        
        # Orientation flag: 1.0 if turned 90°, 0.0 if not
        orientation_flag = 1.0 if self.orientation_complete else 0.0
        
        state = np.concatenate([
            laser_state,
            [goal_dist / 2.0,
             relative_goal_angle / np.pi,
             front_clearance / 3.5,
             self.consecutive_forward_steps / 50.0,
             orientation_flag]
        ])
        
        return state
        
    def select_action(self, state):
        """Action selection with MANDATORY initial turn"""
        min_laser = self.get_min_laser_distance()
        front_clearance = self.get_front_clearance()
        
        # Update orientation status
        self.check_orientation_complete()
        
        # PHASE 1: MANDATORY TURN (until 90° turned)
        if not self.orientation_complete:
            # BLOCK forward action completely
            if random.random() < self.epsilon:
                # Random turn (left or right)
                if self.turn_direction_chosen is None:
                    # First turn of episode - choose direction randomly
                    self.turn_direction_chosen = random.choice([0, 2])
                return self.turn_direction_chosen
            else:
                # Exploitation but still only allow turns
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.policy_net(state_tensor)
                    # BLOCK forward
                    q_values[0, 1] = -10000
                    
                    action = q_values.argmax().item()
                    self.action_counts[action] += 1
                    return action
        
        # PHASE 2: NORMAL NAVIGATION (after 90° turn)
        # Now front clearance should be good, can move forward
        
        # Epsilon-greedy
        if random.random() < self.epsilon:
            if min_laser > 0.25:
                # Prefer forward
                if random.random() < 0.6:
                    return 1
                return random.choice([0, 2])
            elif min_laser > 0.20:
                if random.random() < 0.4:
                    return 1
                return random.choice([0, 2])
            else:
                # Too close - turn only
                return random.choice([0, 2])
        
        # Exploitation with safety
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            
            # Safety: block forward if too close to obstacle
            if min_laser < 0.18:
                q_values[0, 1] = -1000
                
            action = q_values.argmax().item()
            self.action_counts[action] += 1
            return action
            
    def execute_action(self, action):
        """Execute action"""
        cmd = Twist()
        
        if action == 0:  # Turn left
            cmd.linear.x = 0.0
            cmd.angular.z = 1.2
        elif action == 1:  # Forward
            cmd.linear.x = 0.15
            cmd.angular.z = 0.0
        elif action == 2:  # Turn right
            cmd.linear.x = 0.0
            cmd.angular.z = -1.2
            
        self.cmd_vel_pub.publish(cmd)
        
        # Track forward steps
        if action == 1:
            self.consecutive_forward_steps += 1
            self.max_consecutive_forward = max(self.max_consecutive_forward, self.consecutive_forward_steps)
        else:
            self.consecutive_forward_steps = 0
        
    def get_min_laser_distance(self):
        """Get minimum laser distance"""
        if self.current_scan is None:
            return float('inf')
            
        ranges = np.array(self.current_scan.ranges)
        ranges = ranges[~np.isnan(ranges)]
        ranges = ranges[~np.isinf(ranges)]
        
        return np.min(ranges) if len(ranges) > 0 else float('inf')
        
    def calculate_reward(self, state, next_state, done, action):
        """Reward with two-phase system"""
        if self.current_odom is None or self.current_scan is None:
            return 0.0
            
        robot_pos = np.array([
            self.current_odom.pose.pose.position.x,
            self.current_odom.pose.pose.position.y
        ])
        
        current_dist = np.linalg.norm(self.goal_position - robot_pos)
        min_laser = self.get_min_laser_distance()
        front_clearance = self.get_front_clearance()
        
        if self.previous_distance is None:
            self.previous_distance = current_dist
        if self.previous_position is None:
            self.previous_position = robot_pos.copy()
        
        reward = 0.0
        
        # === TERMINAL REWARDS ===
        
        if current_dist < 0.25:
            self.get_logger().info('🎉🎉🎉 GOAL REACHED!')
            self.success_count += 1
            return 2000.0
        
        if min_laser < 0.17:
            self.collision_count += 1
            return -150.0
        
        # === PHASE 1: ORIENTATION REWARDS ===
        
        if not self.orientation_complete:
            # Reward for turning
            if action in [0, 2]:
                reward += 10.0
            
            # Big reward for reaching 90°
            if self.check_orientation_complete():
                reward += 100.0
                self.get_logger().info('🎯 90° turn complete! Now navigate to goal!')
            
            # Progressive rewards for partial turns
            if self.spawn_yaw is not None:
                current_yaw = self.get_robot_yaw()
                yaw_diff = abs(current_yaw - self.spawn_yaw)
                while yaw_diff > np.pi:
                    yaw_diff -= 2 * np.pi
                yaw_diff = abs(yaw_diff)
                
                # Reward proportional to how much turned
                turn_progress = min(yaw_diff / 1.57, 1.0)  # 1.57 = 90°
                reward += turn_progress * 20.0
            
            # Penalty for trying to go forward before orientation complete
            if action == 1:
                reward -= 50.0
            
            # Small survival bonus
            reward += 0.5
            
        # === PHASE 2: NAVIGATION REWARDS ===
        else:
            # 1. DISTANCE PROGRESS
            distance_delta = self.previous_distance - current_dist
            if distance_delta > 0:
                reward += distance_delta * 1000.0  # HUGE multiplier
                
                self.recent_progress.append(distance_delta)
                
                if distance_delta > 0.02:
                    reward += 150.0
                    self.get_logger().info(f'🚀 Big progress! Δ{distance_delta:.4f}m')
                
                if current_dist < self.best_distance:
                    improvement = self.best_distance - current_dist
                    self.best_distance = current_dist
                    reward += 600.0
                    progress_pct = (1.0 - current_dist / 0.96) * 100
                    self.get_logger().info(f'🏆 NEW BEST: {current_dist:.3f}m ({progress_pct:.1f}%)')
            else:
                reward += distance_delta * 250.0
            
            # 2. ACTUAL MOVEMENT
            actual_movement = np.linalg.norm(robot_pos - self.previous_position)
            if actual_movement > 0.01:
                reward += actual_movement * 300.0
            
            # 3. FORWARD ACTION BONUS
            if action == 1:
                if min_laser > 0.25:
                    reward += 30.0
                elif min_laser > 0.20:
                    reward += 15.0
                else:
                    reward -= 20.0
            
            # 4. FORWARD MOMENTUM
            if self.consecutive_forward_steps >= 3:
                reward += self.consecutive_forward_steps * 3.0
                
            if self.consecutive_forward_steps >= 10:
                reward += 70.0
                if self.consecutive_forward_steps == 10:
                    self.get_logger().info('🔥 10 consecutive forward steps!')
            
            # 5. SURVIVAL
            reward += 2.0
            
            # 6. SAFETY
            if min_laser > 0.30:
                reward += 5.0
            elif min_laser < 0.20:
                reward -= 12.0
            
            # 7. HEADING ALIGNMENT
            if min_laser > 0.25 and actual_movement > 0.005:
                goal_angle_error = abs(next_state[self.laser_bins + 1])
                heading_reward = (1.0 - goal_angle_error) * 10.0
                reward += heading_reward
            
            # 8. MILESTONES
            if self.step == 50:
                reward += 80.0
                self.get_logger().info('💪 50 steps!')
            elif self.step == 100:
                reward += 120.0
                self.get_logger().info('💪💪 100 steps!')
            elif self.step == 150:
                reward += 150.0
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
        if min_laser < 0.17:
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
        self.consecutive_forward_steps = 0
        
        # RESET ORIENTATION TRACKING
        self.spawn_yaw = None
        self.orientation_complete = False
        self.turned_enough = False
        self.turn_direction_chosen = None
        
        wait_counter = 0
        while (self.current_odom is None or self.current_scan is None) and wait_counter < 50:
            rclpy.spin_once(self, timeout_sec=0.1)
            wait_counter += 1
        
        if self.current_odom is None or self.current_scan is None:
            return False
        
        # Record spawn orientation
        self.spawn_yaw = self.get_robot_yaw()
        
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
            'best_distance': self.best_distance,
            'max_consecutive_forward': self.max_consecutive_forward
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
            'max_consecutive_forward': self.max_consecutive_forward,
            'avg_reward_last_100': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards),
            'episode_rewards': self.episode_rewards,
            'action_distribution': {
                'left': action_dist[0],
                'forward': action_dist[1],
                'right': action_dist[2]
            }
        }
        
        stats_path = os.path.join(self.results_dir, 'training_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
            
    def train(self, num_episodes=2000):
        """Main training loop"""
        self.get_logger().info(f'🚀 Starting MANDATORY 90° TURN training for {num_episodes} episodes')
        self.get_logger().info('🔄 Phase 1: Turn 90° | Phase 2: Navigate to goal')
        
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
                
                time.sleep(0.20)
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
            avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
            progress_pct = (1.0 - self.best_distance / 0.96) * 100 if self.best_distance != float('inf') else 0.0
            
            self.get_logger().info(
                f'📊 Ep {episode} | Steps: {self.step} | R: {self.episode_reward:.1f} | '
                f'Avg: {avg_reward:.1f} | ✓{self.success_count} ✗{self.collision_count} '
                f'⏱️{self.timeout_count} | Best: {self.best_distance:.3f}m ({progress_pct:.1f}%) | '
                f'MaxFwd: {self.max_consecutive_forward} | ε: {self.epsilon:.3f}'
            )
            
            # Log action distribution every 50 episodes
            if (episode + 1) % 50 == 0:
                total_actions = sum(self.action_counts)
                if total_actions > 0:
                    self.get_logger().info(
                        f'🎯 Actions: L:{self.action_counts[0]/total_actions*100:.1f}% '
                        f'F:{self.action_counts[1]/total_actions*100:.1f}% '
                        f'R:{self.action_counts[2]/total_actions*100:.1f}%'
                    )
                self.action_counts = [0] * self.action_size
            
            # Save checkpoint
            if (episode + 1) % 100 == 0:
                self.save_checkpoint()
                
        self.get_logger().info(f'🏁 Training complete!')
        self.get_logger().info(f'Best distance achieved: {self.best_distance:.3f}m')
        self.get_logger().info(f'Max consecutive forward steps: {self.max_consecutive_forward}')
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