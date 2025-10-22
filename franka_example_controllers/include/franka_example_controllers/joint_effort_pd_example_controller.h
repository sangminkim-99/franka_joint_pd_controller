// Copyright (c) 2023 Franka Robotics GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
//
// Modified by Gemini (2025) to implement an Effort-based PD Controller
// with Gravity Compensation and Torque Rate Saturation.

#pragma once

#include <array>
#include <string>
#include <vector>
#include <memory> // For std::unique_ptr

#include <controller_interface/multi_interface_controller.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <ros/node_handle.h>
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>

// For thread-safe, realtime-safe data buffering
#include <realtime_tools/realtime_buffer.h>

// Franka interfaces for Model (gravity) and State (torque rate)
#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>

// For matrix math (Eigen)
#include <Eigen/Core>

namespace franka_example_controllers {

/**
 * @brief An effort-based PD controller with Gravity Compensation.
 *
 * This controller receives desired joint positions, computes the necessary
 * torques using a PD control law, and adds gravity compensation.
 * It also saturates the torque rate for safety.
 */
class JointEffortPDController : public controller_interface::MultiInterfaceController<hardware_interface::EffortJointInterface,
                                            franka_hw::FrankaModelInterface,
                                            franka_hw::FrankaStateInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hardware, ros::NodeHandle& node_handle) override;
  void starting(const ros::Time& time) override;
  void update(const ros::Time& time, const ros::Duration& period) override;

 private:
  void jointPositionCallback(const sensor_msgs::JointStateConstPtr& msg);

  /**
   * @brief Saturates the commanded torque based on the maximum allowed rate.
   *
   * This is a safety function to prevent abrupt torque changes, based on
   * the implementation in franka_example_controllers.
   *
   * @param tau_d_calculated The desired torque command from the PD controller.
   * @param tau_J_d The last commanded torque (from the robot state).
   * @return The saturated torque command.
   */
  Eigen::Matrix<double, 7, 1> saturateTorqueRate(
      const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
      const Eigen::Matrix<double, 7, 1>& tau_J_d); // NOLINT (readability-identifier-naming)

  // --- Hardware Interfaces ---
  hardware_interface::EffortJointInterface* effort_joint_interface_{nullptr};
  std::vector<hardware_interface::JointHandle> effort_joint_handles_;

  // Franka Model and State Handles
  franka_hw::FrankaModelInterface* model_interface_{nullptr};
  std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
  
  franka_hw::FrankaStateInterface* state_interface_{nullptr};
  std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;


  // --- ROS Interface ---
  ros::Subscriber sub_joint_position_;

  // --- PD Gains (using Eigen Vectors) ---
  Eigen::Matrix<double, 7, 1> kp_gains_; // Proportional gains
  Eigen::Matrix<double, 7, 1> kd_gains_; // Derivative gains

  // --- Realtime-Safe Data Buffer ---
  realtime_tools::RealtimeBuffer<std::array<double, 7>> q_target_buffer_;
  
  // Max torque rate [Nm/ms] (same as [Nm/cycle] at 1kHz)
  static constexpr double kDeltaTauMax{1.0};
};

}  // namespace franka_example_controllers

