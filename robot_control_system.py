"""
Advanced Robotic Systems - EMS628U/728P
Queen Mary University of London - Final Year Project
Author: Samuel Adelufosi
Academic Year: 2024/25

Project: Modelling and Control of Robotic Systems Integration
Combining control theory with practical robotic implementation
"""

import numpy as np
import math

class RoboticSystemController:
    """
    Advanced robotic control system implementing concepts from
    EMS627U - Modelling and Control of Robotic Systems
    EMS628U - Advanced Robotic Systems
    """
    
    def __init__(self, dof=6):  # 6 degrees of freedom
        self.degrees_of_freedom = dof
        self.joint_angles = np.zeros(dof)
        self.joint_velocities = np.zeros(dof)
        self.end_effector_pos = np.zeros(3)  # [x, y, z]
        
    def forward_kinematics(self, joint_angles):
        """
        Calculate end-effector position from joint angles
        Implementation based on EMS627U coursework
        """
        # Simplified 3-DOF calculation for demonstration
        L1, L2, L3 = 1.0, 0.8, 0.6  # Link lengths
        
        theta1, theta2, theta3 = joint_angles[:3]
        
        # Forward kinematics equations
        x = L1*math.cos(theta1) + L2*math.cos(theta1+theta2) + L3*math.cos(theta1+theta2+theta3)
        y = L1*math.sin(theta1) + L2*math.sin(theta1+theta2) + L3*math.sin(theta1+theta2+theta3)
        z = 0.5  # Simplified for 2D workspace
        
        self.end_effector_pos = np.array([x, y, z])
        return self.end_effector_pos
        
    def inverse_kinematics(self, target_pos):
        """
        Calculate joint angles needed to reach target position
        Advanced topic from EMS628U Advanced Robotic Systems
        """
        x_target, y_target = target_pos[:2]
        
        # Simplified analytical solution for 2-DOF case
        L1, L2 = 1.0, 0.8
        
        # Distance to target
        r = math.sqrt(x_target**2 + y_target**2)
        
        # Check if target is reachable
        if r > (L1 + L2) or r < abs(L1 - L2):
            raise ValueError("Target position unreachable")
            
        # Calculate joint angles using inverse kinematics
        cos_theta2 = (r**2 - L1**2 - L2**2) / (2*L1*L2)
        theta2 = math.acos(cos_theta2)
        
        theta1 = math.atan2(y_target, x_target) - math.atan2(L2*math.sin(theta2), L1+L2*math.cos(theta2))
        
        return [theta1, theta2, 0, 0, 0, 0]  # Pad with zeros for 6-DOF
        
    def pid_control(self, target_angles, kp=10.0, ki=0.1, kd=1.0):
        """
        PID controller for joint position control
        Control theory from EMS627U Modelling and Control
        """
        current_angles = self.joint_angles
        error = np.array(target_angles) - current_angles
        
        # Simplified PID (would need integral and derivative terms in real implementation)
        control_signal = kp * error
        
        # Update joint angles (simplified dynamics)
        self.joint_angles += control_signal * 0.01  # Small time step
        
        return control_signal
        
    def trajectory_planning(self, start_pos, end_pos, duration=5.0, points=50):
        """
        Generate smooth trajectory between two points
        Advanced motion planning from EMS628U
        """
        trajectory = []
        
        for i in range(points):
            t = i / (points - 1)  # Normalized time [0, 1]
            
            # Cubic polynomial trajectory for smooth motion
            s = 3*t**2 - 2*t**3  # S-curve velocity profile
            
            # Interpolate position
            pos = start_pos + s * (end_pos - start_pos)
            trajectory.append(pos)
            
        return np.array(trajectory)

# Integration with Neural Networks (ECS659U coursework connection)
class RobotLearningSystem:
    """
    Combining robotics with neural networks from ECS659U
    Neural Networks and Deep Learning integration
    """
    
    def __init__(self, input_size=6, hidden_size=64, output_size=6):
        # Simplified neural network structure
        self.weights_input = np.random.randn(input_size, hidden_size) * 0.1
        self.weights_output = np.random.randn(hidden_size, output_size) * 0.1
        
    def forward_pass(self, sensor_data):
        """Neural network forward pass for adaptive control"""
        # Hidden layer with ReLU activation
        hidden = np.maximum(0, np.dot(sensor_data, self.weights_input))
        
        # Output layer
        output = np.dot(hidden, self.weights_output)
        
        return output
        
    def adaptive_control(self, robot_state, target_state):
        """Use neural network for adaptive robot control"""
        input_vector = np.concatenate([robot_state, target_state])
        control_adjustment = self.forward_pass(input_vector)
        
        return control_adjustment

# Example demonstration
def main():
    print("=== Queen Mary University Robotics Integration Project ===")
    print("Combining EMS627U, EMS628U, and ECS659U coursework\n")
    
    # Initialize systems
    robot = RoboticSystemController()
    learning_system = RobotLearningSystem()
    
    # Test forward kinematics
    print("1. Forward Kinematics Test:")
    test_angles = [0.5, 0.3, 0.2, 0, 0, 0]
    end_pos = robot.forward_kinematics(test_angles)
    print(f"Joint angles: {test_angles[:3]}")
    print(f"End-effector position: [{end_pos[0]:.3f}, {end_pos[1]:.3f}, {end_pos[2]:.3f}]")
    
    # Test inverse kinematics
    print("\n2. Inverse Kinematics Test:")
    target = [1.5, 1.2, 0.5]
    try:
        calculated_angles = robot.inverse_kinematics(target)
        print(f"Target position: {target[:2]}")
        print(f"Required joint angles: [{calculated_angles[0]:.3f}, {calculated_angles[1]:.3f}]")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Test trajectory planning
    print("\n3. Trajectory Planning:")
    start = np.array([0, 0, 0])
    end = np.array([1, 1, 0])
    trajectory = robot.trajectory_planning(start, end, points=10)
    print(f"Generated {len(trajectory)} trajectory points")
    print(f"Start: {start}")
    print(f"End: {end}")
    
    # Neural network adaptive control
    print("\n4. Neural Network Integration:")
    robot_state = robot.joint_angles
    target_state = np.array([0.1, 0.2, 0.1, 0, 0, 0])
    adaptation = learning_system.adaptive_control(robot_state, target_state)
    print(f"Current robot state: {robot_state[:3]}")
    print(f"Neural network adaptation: {adaptation[:3]}")

if __name__ == "__main__":
    main()
