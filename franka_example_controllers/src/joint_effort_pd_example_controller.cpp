// Copyright (c) 2023 Franka Robotics GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
//
// Modified by Gemini (2025) to implement an Effort-based PD Controller

#include <franka_example_controllers/joint_effort_pd_example_controller.h>

#include <cmath>

#include <Eigen/Core> // For Eigen math

#include <controller_interface/multi_interface_controller.h>
#include <pluginlib/class_list_macros.h>

// Franka interfaces
#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>
#include <franka/robot_state.h> // For franka::RobotState


namespace franka_example_controllers {

bool JointEffortPDController::init(hardware_interface::RobotHW* robot_hardware,
                                     ros::NodeHandle& node_handle) {
  // --- 1. Initialize ROS Subscriber ---
  sub_joint_position_ = node_handle.subscribe(
      "/franka_joint_control/command", 1, &JointEffortPDController::jointPositionCallback, this,
      ros::TransportHints().reliable().tcpNoDelay());

  // --- 2. Get EffortJointInterface ---
  effort_joint_interface_ = robot_hardware->get<hardware_interface::EffortJointInterface>();
  if (effort_joint_interface_ == nullptr) {
    ROS_ERROR(
        "JointEffortPDController: Error getting effort joint interface from hardware!");
    return false;
  }

  model_interface_ = robot_hardware->get<franka_hw::FrankaModelInterface>();
  if (model_interface_ == nullptr) {
    ROS_ERROR("JointEffortPDController: Error getting model interface from hardware!");
    return false;
  }
  try {
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface_->getHandle("panda_model"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "JointEffortPDController: Exception getting model handle: " << ex.what());
    return false;
  }

  state_interface_ = robot_hardware->get<franka_hw::FrankaStateInterface>();
  if (state_interface_ == nullptr) {
    ROS_ERROR("JointEffortPDController: Error getting state interface from hardware!");
    return false;
  }
  try {
    state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
        state_interface_->getHandle("panda_robot"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "JointEffortPDController: Exception getting state handle: " << ex.what());
    return false;
  }


  // --- 4. Get Joint Names from Parameter Server ---
  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names)) {
    ROS_ERROR("JointEffortPDController: Could not parse joint names");
    return false;
  }
  if (joint_names.size() != 7) {
    ROS_ERROR_STREAM("JointEffortPDController: Wrong number of joint names, got "
                     << joint_names.size() << " instead of 7 names!");
    return false;
  }

  // --- 5. Get PD Gains from Parameter Server ---
  std::vector<double> kp_vec, kd_vec;
  if (!node_handle.getParam("kp_gains", kp_vec) || !node_handle.getParam("kd_gains", kd_vec)) {
    ROS_ERROR("JointEffortPDController: Could not parse PD gains (kp_gains or kd_gains)");
    return false;
  }
  if (kp_vec.size() != 7 || kd_vec.size() != 7) {
    ROS_ERROR_STREAM("JointEffortPDController: Wrong number of PD gains, got "
                     << kp_vec.size() << " Kp and " << kd_vec.size() << " Kd!");
    return false;
  }

  // Copy gains from std::vector (from param server) to Eigen::Matrix (member variable)
  kp_gains_ = Eigen::Map<Eigen::Matrix<double, 7, 1>>(kp_vec.data());
  kd_gains_ = Eigen::Map<Eigen::Matrix<double, 7, 1>>(kd_vec.data());

  // --- 6. Get Joint Handles ---
  effort_joint_handles_.resize(7);
  for (size_t i = 0; i < 7; ++i) {
    try {
      effort_joint_handles_[i] = effort_joint_interface_->getHandle(joint_names[i]);
    } catch (const hardware_interface::HardwareInterfaceException& e) {
      ROS_ERROR_STREAM(
          "JointEffortPDController: Exception getting joint handles: " << e.what());
      return false;
    }
  }

  // --- 7. Check Start Position (optional, but good practice) ---
  std::array<double, 7> q_start{{0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4}};
  for (size_t i = 0; i < q_start.size(); i++) {
    if (std::abs(effort_joint_handles_[i].getPosition() - q_start[i]) > 0.1) {
      ROS_ERROR_STREAM(
          "JointEffortPDController: Robot is not in the expected starting position for "
          "running this example. Run `roslaunch franka_example_controllers move_to_start.launch "
          "robot_ip:=<robot-ip> load_gripper:=<has-attached-gripper>` first.");
      return false;
    }
  }

  return true;
}

void JointEffortPDController::starting(const ros::Time& /* time */) {
  // Initialize the target position buffer with the current robot state.
  std::array<double, 7> q_current;
  for (size_t i = 0; i < 7; ++i) {
    q_current[i] = effort_joint_handles_[i].getPosition();
  }
  q_target_buffer_.initRT(q_current);
}

void JointEffortPDController::update(const ros::Time& /*time*/,
                                     const ros::Duration& /*period*/) {
  
  // --- Get Robot State and Model Data ---

  // Get current robot state (for tau_J_d)
  franka::RobotState robot_state = state_handle_->getRobotState();
  Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_J_d(robot_state.tau_J_d.data());

  // Get the latest target position from the realtime buffer
  std::array<double, 7>& q_target_array = *q_target_buffer_.readFromRT();
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q_target(q_target_array.data());

  // Get current state (q, dq) from joint handles
  Eigen::Matrix<double, 7, 1> q_current;
  Eigen::Matrix<double, 7, 1> dq_current;
  for (size_t i = 0; i < 7; ++i) {
    q_current(i) = effort_joint_handles_[i].getPosition();
    dq_current(i) = effort_joint_handles_[i].getVelocity();
  }

  // --- Compute Control Law ---

  // Gains as diagonal matrices
  Eigen::DiagonalMatrix<double, 7> Kp(kp_gains_);
  Eigen::DiagonalMatrix<double, 7> Kd(kd_gains_);

  // Calculate position error
  Eigen::Matrix<double, 7, 1> error_q = q_target - q_current;

  // Apply PD Control Law
  Eigen::Matrix<double, 7, 1> tau_command = Kp * error_q - Kd * dq_current;

  // --- Saturate and Command Torque ---

  // Saturate torque rate to avoid joint limits
  tau_command = saturateTorqueRate(tau_command, tau_J_d);

  // Send the computed torque (effort) command to the joint
  for (size_t i = 0; i < 7; ++i) {
    effort_joint_handles_[i].setCommand(tau_command(i));
  }
}

void JointEffortPDController::jointPositionCallback(
    const sensor_msgs::JointStateConstPtr& msg) {
  
  if (msg->position.size() != 7) {
    ROS_ERROR_STREAM("JointEffortPDController: Received joint state message with wrong number of positions, expected 7 but got "
                     << msg->position.size());
    return;
  }

  // Copy the received positions into a temporary array
  std::array<double, 7> q_target_msg;
  for (size_t i = 0; i < 7; ++i) {
    q_target_msg[i] = msg->position[i];
  }

  // Write the new target to the realtime buffer (non-realtime side)
  q_target_buffer_.writeFromNonRT(q_target_msg);
}

// --- Helper Function Implementation ---

Eigen::Matrix<double, 7, 1> JointEffortPDController::saturateTorqueRate(
    const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
    const Eigen::Matrix<double, 7, 1>& tau_J_d) { // NOLINT (readability-identifier-naming)

  Eigen::Matrix<double, 7, 1> tau_d_saturated;
  for (size_t i = 0; i < 7; i++) {
    double delta_tau = tau_d_calculated(i) - tau_J_d(i);
    // Saturate the change in torque to be within [-kDeltaTauMax, kDeltaTauMax]
    // (assuming 1kHz update rate, kDeltaTauMax is in Nm/ms)
    tau_d_saturated(i) =
        tau_J_d(i) + std::max(std::min(delta_tau, kDeltaTauMax), -kDeltaTauMax);
  }
  return tau_d_saturated;
}

}  // namespace franka_example_controllers

// Export the controller class as a plugin
PLUGINLIB_EXPORT_CLASS(franka_example_controllers::JointEffortPDController,
                       controller_interface::ControllerBase)

