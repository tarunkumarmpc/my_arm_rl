import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from gymnasium import spaces, Env
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
from visualization_msgs.msg import Marker
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from math import pi, sin, cos
import threading
import yaml
import os
from std_srvs.srv import Empty

class ArmEnv(Env):
    """Robotic Arm Environment with Gripper Control"""

    def __init__(self, config_path=None, max_attempts=10, noise_scale=0.05, timeout_steps=50,
                 urdf_path=None, debug_mode=False):
        super().__init__()
        self.debug_mode = debug_mode
        self.logger = rclpy.logging.get_logger('ArmEnv')
        if self.debug_mode:
            rclpy.logging.set_logger_level('ArmEnv', rclpy.logging.LoggingSeverity.DEBUG)
        else:
            rclpy.logging.set_logger_level('ArmEnv', rclpy.logging.LoggingSeverity.INFO)

        self.config = self._load_config(config_path)
        self.max_attempts = max_attempts
        self.noise_scale = noise_scale
        self.timeout_steps = timeout_steps
        self.urdf_path = urdf_path

        self.node = Node('arm_rl_env')
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self._start_spin_thread()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)

        self.joint_names, joint_limits, self.transforms = self._load_kinematics()
        self.joint_limits = np.array(joint_limits, dtype=np.float32)
        self.num_joints = len(self.joint_names)
        self.max_reach = 0.45
        self.action_space, self.observation_space = self._define_spaces()

        self.joint_positions = None
        self.joint_velocities = None
        self.joint_lock = threading.Lock()
        self.joint_sub = self.node.create_subscription(
            JointState, '/joint_states', self._joint_state_callback, 10)

        self.arm_client = ActionClient(
            self.node, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
        self.gripper_client = ActionClient(
            self.node, FollowJointTrajectory, '/gripper_controller/follow_joint_trajectory')
        if not self.arm_client.wait_for_server(timeout_sec=20.0):
            raise RuntimeError("Arm action server unavailable")
        if not self.gripper_client.wait_for_server(timeout_sec=20.0):
            raise RuntimeError("Gripper action server unavailable")

        self.target_pub = self.node.create_publisher(Marker, '/target_marker', 10)
        self.ee_marker_pub = self.node.create_publisher(Marker, '/ee_marker', 10)
        
        self.reset_sim_client = self.node.create_client(Empty, '/reset_simulation')

        self.target_joints = None
        self.target_ee_pos = None
        self.step_count = 0
        self.attempt_count = 0
        self.prev_distance = None

        # Fixed joint-space goals (5 arm joints + 1 gripper joint)
        self.fixed_goals = [
            np.array([1.345, -1.23, 0.264, -0.296, 0.389, 0.0], dtype=np.float32),  # Target 1: Gripper closed
            np.array([0.785, -0.523, 1.047, -0.349, 0.698, 0.5], dtype=np.float32),  # Target 2: Gripper open
            np.array([0.0, -pi/4, pi/4, 0.0, -pi/4, 0.0], dtype=np.float32),  # Target 3: Gripper closed
            np.array([1.0, -0.8, 0.5, -0.2, 0.3, 0.5], dtype=np.float32),  # Target 4: Gripper open
            np.array([-1.2, 0.6, -0.9, 0.4, -0.5, 0.0], dtype=np.float32),  # Target 5: Gripper closed
            np.array([0.5, -1.0, 1.2, -0.5, 0.7, 0.5], dtype=np.float32),  # Target 6: Gripper open
            np.array([1.5, -0.3, 0.8, -0.7, 0.2, 0.0], dtype=np.float32),  # Target 7: Gripper closed
            np.array([-0.8, 0.9, -1.1, 0.6, -0.4, 0.5], dtype=np.float32),  # Target 8: Gripper open
            np.array([0.3, -1.5, 0.7, -0.9, 0.5, 0.0], dtype=np.float32),  # Target 9: Gripper closed
            np.array([1.1, -0.7, 0.9, -0.3, 0.8, 0.5], dtype=np.float32),  # Target 10: Gripper open
        ]
        self.current_goal_idx = 0

        for i, goal in enumerate(self.fixed_goals):
            self.fixed_goals[i] = np.clip(goal, self.joint_limits[:, 0], self.joint_limits[:, 1])

        self.safe_home_joints = np.array([0.0, -pi/4, pi/4, 0.0, -pi/4, 0.0], dtype=np.float32)
        self.home_joints = self.safe_home_joints.copy()
        self.home_ee_pos, self.home_ee_rot = self._get_ee_position_from_joints(self.home_joints)
        self._initialize_home_position()

        self._last_ee_pos = None

    def _log(self, message, level="info"):
        if level == "debug" and self.debug_mode:
            self.logger.debug(message)
        elif level == "info":
            self.logger.info(message)
        elif level == "warn":
            self.logger.warn(message)
        elif level == "error":
            self.logger.error(message)

    def _load_config(self, config_path):
        default_config = {
            'collision_threshold': 0.05,
            'trajectory_duration': 5,
            'base_frame': 'base_link',
            'ee_frame': 'gripper_base',  # Updated to gripper frame
            'workspace_bounds': {'low': [-0.45, -0.45, 0.1], 'high': [0.45, 0.45, 0.45]}
        }
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return {**default_config, **yaml.safe_load(f)}
        return default_config

    def _load_kinematics(self):
        # Updated joint names: 5 arm joints + 1 gripper joint
        joint_names = [
            'link1_to_link2', 'link2_to_link3', 'link3_to_link4',
            'link4_to_link5', 'link5_to_link6', 'gripper_controller'
        ]
        # Joint limits: Arm joints + Gripper (assuming 0.0 to 0.5 for open/close)
        joint_limits = [
            [-2.879793, 2.879793], [-2.879793, 2.879793], [-2.879793, 2.879793],
            [-2.879793, 2.879793], [-2.879793, 2.879793], [0.0, 0.5]  # Gripper range
        ]
        # Transforms: Updated to include gripper (simplified, adjust based on URDF)
        transforms = [
            {'translation': [0.0, 0.0, 0.091], 'rpy': [0.0, 0.0, pi / 2]},
            {'translation': [0.0, 0.0, -0.001], 'rpy': [0.0, pi / 2, -pi / 2]},
            {'translation': [-0.1104, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0]},
            {'translation': [-0.096, 0.0, 0.06062], 'rpy': [0.0, 0.0, -pi / 2]},
            {'translation': [0.0, -0.07318, 0.0], 'rpy': [pi / 2, -pi / 2, 0.0]},
            {'translation': [0.0, 0.0456, 0.0], 'rpy': [0.0, 0.0, 0.0]},  # Gripper base
        ]
        self._log("Using kinematics for 5-DOF arm + gripper", level="info")
        return joint_names, joint_limits, transforms

    def _define_spaces(self):
        # Action space: 6 joints (5 arm + 1 gripper)
        action_space = spaces.Box(low=-0.2 * np.ones(self.num_joints, dtype=np.float32),
                                  high=0.2 * np.ones(self.num_joints, dtype=np.float32), dtype=np.float32)
        obs_low = np.concatenate([
            self.joint_limits[:, 0], self.joint_limits[:, 0],
            np.array(self.config['workspace_bounds']['low'], dtype=np.float32),
            -self.max_reach * np.ones(3, dtype=np.float32),
            np.array([0.0], dtype=np.float32)
        ])
        obs_high = np.concatenate([
            self.joint_limits[:, 1], self.joint_limits[:, 1],
            np.array(self.config['workspace_bounds']['high'], dtype=np.float32),
            self.max_reach * np.ones(3, dtype=np.float32),
            np.array([self.max_reach], dtype=np.float32)
        ])
        return action_space, spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    def _initialize_home_position(self):
        if self.home_reset(self.safe_home_joints):
            with self.joint_lock:
                self.home_joints = self.joint_positions.copy()
            self.home_ee_pos, self.home_ee_rot = self._get_ee_position_from_joints(self.home_joints)
            self._log(f"Home position initialized: Joints={self.home_joints}, EE={self.home_ee_pos}", level="info")
        else:
            self._log("Failed to initialize safe home, forcing position", level="error")
            self.home_joints = self.safe_home_joints.copy()
            self.home_ee_pos, self.home_ee_rot = self._get_ee_position_from_joints(self.home_joints)

    def _start_spin_thread(self):
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()

    def _joint_state_callback(self, msg):
        with self.joint_lock:
            try:
                indices = [msg.name.index(j) for j in self.joint_names]
                self.joint_positions = np.array(msg.position, dtype=np.float32)[indices]
                self.joint_velocities = np.array(msg.velocity, dtype=np.float32)[indices]
                self._log(f"Joint state update: Positions={self.joint_positions}", level="debug")
            except Exception as e:
                self._log(f"Joint state error: {str(e)}", level="warn")

    def _execute_trajectory(self, target_positions):
        with self.joint_lock:
            current_joints = self.joint_positions if self.joint_positions is not None else self.home_joints
        clamped_positions = np.clip(target_positions, self.joint_limits[:, 0], self.joint_limits[:, 1])
        if not np.allclose(target_positions, clamped_positions):
            self._log(f"Clamping action: {target_positions} -> {clamped_positions}", level="warn")
        
        self._log(f"Executing trajectory from {current_joints} to {clamped_positions}", level="debug")
        
        # Split into arm and gripper commands
        arm_positions = clamped_positions[:-1]  # First 5 joints (arm)
        gripper_position = clamped_positions[-1]  # Last joint (gripper)

        # Arm trajectory
        arm_goal_msg = FollowJointTrajectory.Goal()
        arm_point = JointTrajectoryPoint()
        arm_point.positions = arm_positions.tolist()
        arm_point.time_from_start = Duration(sec=self.config['trajectory_duration'])
        arm_goal_msg.trajectory.joint_names = self.joint_names[:-1]  # Arm joints
        arm_goal_msg.trajectory.points = [arm_point]

        # Gripper trajectory
        gripper_goal_msg = FollowJointTrajectory.Goal()
        gripper_point = JointTrajectoryPoint()
        gripper_point.positions = [float(gripper_position)]  # Single gripper joint
        gripper_point.time_from_start = Duration(sec=self.config['trajectory_duration'])
        gripper_goal_msg.trajectory.joint_names = [self.joint_names[-1]]  # Gripper joint
        gripper_goal_msg.trajectory.points = [gripper_point]

        # Send arm trajectory
        arm_future = self.arm_client.send_goal_async(arm_goal_msg)
        rclpy.spin_until_future_complete(self.node, arm_future)
        if not arm_future.done() or not arm_future.result().accepted:
            self._log("Arm trajectory goal rejected or timed out", level="error")
            return False
        arm_result_future = arm_future.result().get_result_async()
        rclpy.spin_until_future_complete(self.node, arm_result_future)
        arm_success = arm_result_future.result().result.error_code == FollowJointTrajectory.Result.SUCCESSFUL

        # Send gripper trajectory
        gripper_future = self.gripper_client.send_goal_async(gripper_goal_msg)
        rclpy.spin_until_future_complete(self.node, gripper_future)
        if not gripper_future.done() or not gripper_future.result().accepted:
            self._log("Gripper trajectory goal rejected or timed out", level="error")
            return False
        gripper_result_future = gripper_future.result().get_result_async()
        rclpy.spin_until_future_complete(self.node, gripper_result_future)
        gripper_success = gripper_result_future.result().result.error_code == FollowJointTrajectory.Result.SUCCESSFUL

        success = arm_success and gripper_success
        self._log(f"Trajectory result: {'Success' if success else 'Failed'}", level="debug")
        return success

    def _get_ee_position(self):
        if hasattr(self, '_last_ee_pos') and self.step_count % 5 != 0:
            return self._last_ee_pos
        for attempt in range(3):
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.config['base_frame'], self.config['ee_frame'], rclpy.time.Time())
                pos = np.array([
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z
                ], dtype=np.float32)
                self._last_ee_pos = pos
                self._log(f"TF EE position: {pos}", level="debug")
                return pos
            except TransformException as e:
                self._log(f"TF lookup failed (attempt {attempt + 1}/3): {str(e)}", level="warn")
                rclpy.spin_once(self.node, timeout_sec=0.05)
        self._log("Failed to get EE transform, using FK fallback", level="error")
        with self.joint_lock:
            joints = self.joint_positions if self.joint_positions is not None else self.home_joints
            pos, _ = self._get_ee_position_from_joints(joints)
        self._last_ee_pos = pos
        return pos

    def _create_transform_matrix(self, translation, rpy):
        tx, ty, tz = translation
        roll, pitch, yaw = rpy
        Rx = np.array([[1, 0, 0], [0, cos(roll), -sin(roll)], [0, sin(roll), cos(roll)]], dtype=np.float32)
        Ry = np.array([[cos(pitch), 0, sin(pitch)], [0, 1, 0], [-sin(pitch), 0, cos(pitch)]], dtype=np.float32)
        Rz = np.array([[cos(yaw), -sin(yaw), 0], [sin(yaw), cos(yaw), 0], [0, 0, 1]], dtype=np.float32)
        R = Rz @ Ry @ Rx
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]
        return T

    def _get_ee_position_from_joints(self, joint_angles):
        try:
            T = np.eye(4, dtype=np.float32)
            for i, theta in enumerate(joint_angles[:5]):  # Only arm joints for FK
                T_fixed = self._create_transform_matrix(self.transforms[i]['translation'], self.transforms[i]['rpy'])
                Rz_theta = np.array([
                    [cos(theta), -sin(theta), 0, 0],
                    [sin(theta), cos(theta), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ], dtype=np.float32)
                T = T @ T_fixed @ Rz_theta
            # Gripper transform (simplified, adjust based on actual gripper kinematics)
            T = T @ self._create_transform_matrix(self.transforms[5]['translation'], self.transforms[5]['rpy'])
            ee_pos = T[:3, 3]
            ee_rot = T[:3, :3]
            self._log(f"FK EE position: {ee_pos}, rotation: {ee_rot}", level="debug")
            return ee_pos, ee_rot
        except Exception as e:
            self._log(f"FK error: {str(e)}", level="error")
            return self.home_ee_pos, self.home_ee_rot

    def home_reset(self, target_joints=None, max_attempts=3, joint_tolerance=0.1, ee_tolerance=0.1):
        if target_joints is None:
            target_joints = self.safe_home_joints.copy()
        
        self._log(f"Home reset: Attempting to move to {target_joints}", level="info")
        expected_ee_pos, _ = self._get_ee_position_from_joints(target_joints)
        
        for attempt in range(max_attempts):
            success = self._execute_trajectory(target_joints)
            if not success:
                self._log(f"Home reset attempt {attempt + 1}/{max_attempts} failed", level="warn")
                continue
            
            rclpy.spin_once(self.node, timeout_sec=self.config['trajectory_duration'] + 2.0)
            
            with self.joint_lock:
                current_joints = self.joint_positions if self.joint_positions is not None else target_joints
            joint_error = np.max(np.abs(current_joints - target_joints))
            
            current_ee_pos = self._get_ee_position()
            ee_error = np.linalg.norm(current_ee_pos - expected_ee_pos)
            
            if joint_error < joint_tolerance and ee_error < ee_tolerance:
                with self.joint_lock:
                    self.home_joints = current_joints.copy()
                self.home_ee_pos, self.home_ee_rot = self._get_ee_position_from_joints(self.home_joints)
                self._log(f"Home reset successful: Joints={self.home_joints}, EE={self.home_ee_pos}", level="info")
                return True
        
        self._log("Home reset failed after all attempts", level="error")
        return False

    def reset(self, seed=None, options=None, goal_idx=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.attempt_count = 0
        self.prev_distance = None

        if not self.home_reset(self.safe_home_joints):
            self._log("Home reset failed, forcing simulation reset", level="error")
            if self.reset_sim_client.wait_for_service(timeout_sec=5.0):
                self.reset_sim_client.call_async(Empty.Request())
                rclpy.spin_once(self.node, timeout_sec=1.0)
            with self.joint_lock:
                self.joint_positions = self.safe_home_joints.copy()
            self.home_joints = self.safe_home_joints.copy()
            self.home_ee_pos, self.home_ee_rot = self._get_ee_position_from_joints(self.home_joints)

        if goal_idx is not None and 0 <= goal_idx < len(self.fixed_goals):
            self.current_goal_idx = goal_idx
        else:
            self.current_goal_idx = self.np_random.integers(0, len(self.fixed_goals))

        self.target_joints = self.fixed_goals[self.current_goal_idx]
        self.target_ee_pos, _ = self._get_ee_position_from_joints(self.target_joints)

        self._log(f"Reset: Selected goal {self.current_goal_idx}: Joints={self.target_joints}, EE={self.target_ee_pos}", level="info")
        print(f"Reset: Target {self.current_goal_idx} - Joints: {self.target_joints}, EE Position: {self.target_ee_pos}")

        self._publish_target_marker()
        self._publish_ee_marker()
        return self._get_observation(), {}

    def step(self, action):
        self.step_count += 1

        noisy_action = action + self.np_random.normal(0, self.noise_scale, size=action.shape).astype(np.float32)
        with self.joint_lock:
            current_joints = self.joint_positions if self.joint_positions is not None else self.home_joints
        
        target_positions = current_joints + noisy_action
        success = self._execute_trajectory(target_positions)
        self._log(f"Trajectory execution {'succeeded' if success else 'failed'}", level="debug")

        ee_pos_tf = self._get_ee_position()
        distance_tf = np.linalg.norm(ee_pos_tf - self.target_ee_pos)
        self._log(f"Step {self.step_count}: Distance to target: {distance_tf:.4f}", level="debug")
        print(f"Step {self.step_count}: Current EE position: {ee_pos_tf}, Target position: {self.target_ee_pos}, Distance: {distance_tf:.4f}")

        reward = 10 * np.exp(-distance_tf / 0.1)
        if self.prev_distance is not None:
            distance_change = self.prev_distance - distance_tf
            reward += 20 * distance_change
            if distance_change < 0:
                reward -= 5
        reward -= 0.05 * np.linalg.norm(noisy_action)

        # Additional reward for gripper state
        gripper_error = abs(self.joint_positions[-1] - self.target_joints[-1])
        reward += 5 * (1 - gripper_error / 0.5)  # Bonus for matching gripper state

        done = False
        truncated = False

        if distance_tf < self.config['collision_threshold'] and gripper_error < 0.05:
            reward += 50
            done = True
            self._log(f"Step {self.step_count}: Target {self.current_goal_idx} reached! Gripper error: {gripper_error:.4f}", level="info")

        elif self.step_count >= self.timeout_steps:
            reward -= 10
            truncated = True
            self._log(f"Step {self.step_count}: Timeout - exceeded {self.timeout_steps} steps", level="info")

        self.prev_distance = distance_tf

        self._publish_ee_marker()
        self._publish_target_marker()
        return self._get_observation(), reward, done, truncated, {}

    def render(self):
        pass

    def close(self):
        if rclpy.ok():
            self.node.destroy_node()
            rclpy.shutdown()
            if hasattr(self, 'spin_thread'):
                self.spin_thread.join(timeout=1.0)

    def _get_observation(self):
        ee_pos = self._get_ee_position()
        relative_pos = self.target_ee_pos - ee_pos
        distance = np.linalg.norm(relative_pos)
        with self.joint_lock:
            joint_pos = self.joint_positions if self.joint_positions is not None else self.home_joints
            joint_vel = self.joint_velocities if self.joint_velocities is not None else np.zeros(self.num_joints, dtype=np.float32)
        obs = np.concatenate([joint_pos, joint_vel, ee_pos, relative_pos, [distance]])
        return obs

    def _publish_target_marker(self):
        marker = Marker()
        marker.header.frame_id = self.config['base_frame']
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(self.target_ee_pos[0])
        marker.pose.position.y = float(self.target_ee_pos[1])
        marker.pose.position.z = float(self.target_ee_pos[2])
        marker.scale.x = marker.scale.y = marker.scale.z = 0.05
        marker.color.r = 1.0
        marker.color.a = 1.0
        self.target_pub.publish(marker)

    def _publish_ee_marker(self):
        ee_pos = self._get_ee_position()
        marker = Marker()
        marker.header.frame_id = self.config['base_frame']
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(ee_pos[0])
        marker.pose.position.y = float(ee_pos[1])
        marker.pose.position.z = float(ee_pos[2])
        marker.scale.x = marker.scale.y = marker.scale.z = 0.03
        marker.color.g = 1.0
        marker.color.a = 1.0
        self.ee_marker_pub.publish(marker)
