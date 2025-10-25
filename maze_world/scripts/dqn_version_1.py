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
    """Smaller network for tight spaces"""
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity=30000):
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
    """DQN Trainer specialized for TIGHT mazes with no open space"""
    
    def __init__(self):
        super().__init__('tight_maze_trainer')
        
        # AGGRESSIVE hyperparameters for tight spaces
        self.gamma = 0.95  # Shorter horizon
        self.epsilon = 1.0
        self.epsilon_min = 0.3  # HIGH minimum - need exploration in tight spaces
        self.epsilon_decay = 0.9998  # Very slow decay
        self.learning_rate = 0.001  # Higher learning rate
        self.batch_size = 32  # Smaller batch for faster updates
        self.target_update_freq = 20  # More frequent updates
        
        # State and action space
        self.laser_bins = 12  # REDUCED - less info, faster learning
        self.state_size = self.laser_bins + 3  # laser + goal_dist, goal_angle, front_clearance
        self.action_size = 3  # ONLY 3 ACTIONS: rotate_left, forward, rotate_right
        
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
        self.replay_buffer = ReplayBuffer(capacity=30000)
        
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
        
        # Training variables
        self.episode = 0
        self.step = 0
        self.total_steps = 0
        self.episode_reward = 0
        self.episode_rewards = []
        self.success_count = 0
        self.collision_count = 0
        self.survival_count = 0  # Episodes that survive > 20 steps
        self.max_steps_per_episode = 200
        
        # Curriculum - reward for small achievements
        self.best_distance = float('inf')
        self.steps_survived = 0
        
        # Results tracking
        self.results_dir = 'results'
        os.makedirs(self.results_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.results_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.get_logger().info(f'🏗️  Tight Maze Trainer initialized. Device: {self.device}')
        self.get_logger().info(f'State size: {self.state_size}, Action size: {self.action_size}')
        self.get_logger().info('⚠️  Optimized for CRAMPED spaces with immediate obstacles')
        
    def odom_callback(self, msg):
        self.current_odom = msg
        
    def scan_callback(self, msg):
        self.current_scan = msg
        
    def get_state(self):
        """Simplified state for tight spaces"""
        if self.current_odom is None or self.current_scan is None:
            return None
            
        # Process laser scan - REDUCED bins
        ranges = np.array(self.current_scan.ranges)
        ranges[np.isinf(ranges)] = 3.5
        ranges[np.isnan(ranges)] = 0.0
        ranges = np.clip(ranges, 0, 3.5)
        
        # Only 12 bins - faster processing
        bin_size = len(ranges) // self.laser_bins
        laser_state = []
        for i in range(self.laser_bins):
            start_idx = i * bin_size
            end_idx = start_idx + bin_size
            if end_idx <= len(ranges):
                laser_state.append(np.min(ranges[start_idx:end_idx]))
            else:
                laser_state.append(np.min(ranges[start_idx:]))
        
        # Normalize with emphasis on close obstacles
        laser_state = np.array(laser_state)
        laser_state = 1.0 / (1.0 + laser_state)  # Inverse - closer = higher value
        
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
        orientation = self.current_odom.pose.pose.orientation
        robot_yaw = np.arctan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y**2 + orientation.z**2)
        )
        
        relative_goal_angle = goal_angle - robot_yaw
        relative_goal_angle = np.arctan2(np.sin(relative_goal_angle), np.cos(relative_goal_angle))
        
        # Front clearance - critical for tight spaces
        front_clearance = np.min(ranges[:30]) if len(ranges) > 30 else np.min(ranges[:10])
        
        state = np.concatenate([
            laser_state,
            [goal_dist / 5.0,
             relative_goal_angle / np.pi,
             front_clearance / 1.0]  # Normalized front clearance
        ])
        
        return state
        
    def select_action(self, state):
        """Action selection for tight spaces"""
        front_clearance = state[-1]  # Last element is front clearance
        
        # FORCED EXPLORATION in tight spaces
        if random.random() < self.epsilon:
            # If front is blocked, prefer rotation
            if front_clearance < 0.3:
                return random.choice([0, 2])  # Only rotate
            return random.randint(0, self.action_size - 1)
        
        # Exploitation
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            
            # HARD constraint: don't move forward if front blocked
            if front_clearance < 0.25:
                q_values[0, 1] = -1000  # Heavily penalize forward action
                
            return q_values.argmax().item()
            
    def execute_action(self, action):
        """3 SIMPLE actions for tight maneuvering"""
        cmd = Twist()
        
        if action == 0:  # Rotate left in place
            cmd.linear.x = 0.0
            cmd.angular.z = 1.2
        elif action == 1:  # Move forward slowly
            cmd.linear.x = 0.12
            cmd.angular.z = 0.0
        elif action == 2:  # Rotate right in place
            cmd.linear.x = 0.0
            cmd.angular.z = -1.2
            
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
        """Reward system for SURVIVAL in tight spaces"""
        if self.current_odom is None or self.current_scan is None:
            return 0.0
            
        robot_pos = np.array([
            self.current_odom.pose.pose.position.x,
            self.current_odom.pose.pose.position.y
        ])
        
        current_dist = np.linalg.norm(self.goal_position - robot_pos)
        min_laser = self.get_min_laser_distance()
        
        if self.previous_distance is None:
            self.previous_distance = current_dist
        
        reward = 0.0
        
        # === TERMINAL REWARDS ===
        
        # GOAL - huge reward
        if current_dist < 0.3:
            self.get_logger().info('🎉🎉🎉 GOAL REACHED!')
            self.success_count += 1
            return 500.0
        
        # COLLISION - moderate penalty
        if min_laser < 0.17:
            self.collision_count += 1
            return -20.0
        
        # === SURVIVAL REWARDS (Key for tight spaces!) ===
        
        # 1. REWARD FOR SURVIVING (most important early on!)
        reward += 0.5  # Small constant reward for not crashing
        
        # 2. DISTANCE IMPROVEMENT
        distance_delta = self.previous_distance - current_dist
        if distance_delta > 0:
            # HUGE reward for getting closer
            reward += distance_delta * 200.0
            
            # Track best distance
            if current_dist < self.best_distance:
                self.best_distance = current_dist
                reward += 20.0  # Bonus for new record
                self.get_logger().info(f'🏆 New best distance: {current_dist:.3f}m')
        else:
            # Small penalty for moving away
            reward += distance_delta * 10.0
        
        # 3. CLEARANCE REWARDS
        if min_laser > 0.25:
            reward += 2.0  # Reward safe navigation
        elif min_laser < 0.22:
            reward -= 8.0  # Penalty for danger
        
        # 4. HEADING REWARD (only if safe distance)
        if min_laser > 0.25:
            goal_angle_error = abs(next_state[self.laser_bins + 1])
            heading_reward = -(goal_angle_error ** 2) * 3.0
            reward += heading_reward
        
        # 5. ACTION-SPECIFIC
        # Penalize excessive rotation when path is clear
        if action in [0, 2] and min_laser > 0.5:
            reward -= 1.0
        
        # 6. MILESTONE REWARDS
        if self.step == 20:
            reward += 15.0  # Survived 20 steps!
            self.get_logger().info('💪 Survived 20 steps!')
        elif self.step == 50:
            reward += 25.0  # Survived 50 steps!
            self.get_logger().info('💪💪 Survived 50 steps!')
        elif self.step == 100:
            reward += 50.0  # Survived 100 steps!
            self.get_logger().info('💪💪💪 Survived 100 steps!')
        
        self.previous_distance = current_dist
        
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
        
        if goal_dist < 0.3:
            return True
        
        min_laser = self.get_min_laser_distance()
        if min_laser < 0.17:
            return True
            
        if self.step >= self.max_steps_per_episode:
            if self.step > 20:
                self.survival_count += 1
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
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
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
            'survival_count': self.survival_count,
            'best_distance': self.best_distance
        }
        
        path = os.path.join(self.checkpoint_dir, f'tight_maze_ep{self.episode}.pth')
        torch.save(checkpoint, path)
        self.get_logger().info(f'💾 Saved: {path}')
        
        stats = {
            'episode': self.episode,
            'total_steps': self.total_steps,
            'epsilon': self.epsilon,
            'success_count': self.success_count,
            'collision_count': self.collision_count,
            'survival_count': self.survival_count,
            'survival_rate': self.survival_count / max(1, self.episode) * 100,
            'best_distance': self.best_distance,
            'avg_reward_last_100': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards),
            'episode_rewards': self.episode_rewards
        }
        
        stats_path = os.path.join(self.results_dir, 'training_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
            
    def train(self, num_episodes=3000):
        """Main training loop"""
        self.get_logger().info(f'🚀 Starting TIGHT MAZE training for {num_episodes} episodes')
        self.get_logger().info('🎯 Phase 1 (0-1000): Learn to SURVIVE')
        self.get_logger().info('🎯 Phase 2 (1000-2000): Learn to NAVIGATE')
        self.get_logger().info('🎯 Phase 3 (2000+): Learn to REACH GOAL')
        
        while self.current_odom is None or self.current_scan is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        self.get_logger().info('✓ Sensors ready!')
        
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
                
                # SLOWER actions in tight space
                time.sleep(0.25)
                rclpy.spin_once(self, timeout_sec=0.05)
                
                next_state = self.get_state()
                if next_state is None:
                    break
                    
                done = self.check_done()
                reward = self.calculate_reward(state, next_state, done, action)
                
                self.replay_buffer.push(state, action, reward, next_state, float(done))
                
                # Train every step
                loss = self.train_step()
                if loss is not None:
                    episode_loss += loss
                    loss_count += 1
                    
                self.episode_reward += reward
                state = next_state
                self.step += 1
                self.total_steps += 1
                
            self.episode_rewards.append(self.episode_reward)
            
            # Very slow epsilon decay
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Update target network
            if episode % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                
            # Logging
            avg_loss = episode_loss / loss_count if loss_count > 0 else 0
            avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
            survival_rate = self.survival_count / max(1, episode + 1) * 100
            
            self.get_logger().info(
                f'📊 Ep {episode} | Steps: {self.step} | R: {self.episode_reward:.1f} | '
                f'Avg: {avg_reward:.1f} | ✓{self.success_count} ✗{self.collision_count} '
                f'💪{survival_rate:.0f}% | Best: {self.best_distance:.2f}m | ε: {self.epsilon:.2f}'
            )
            
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
        trainer.train(num_episodes=3000)
    except KeyboardInterrupt:
        trainer.get_logger().info('⚠️ Interrupted')
    finally:
        stop_cmd = Twist()
        trainer.cmd_vel_pub.publish(stop_cmd)
        trainer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()