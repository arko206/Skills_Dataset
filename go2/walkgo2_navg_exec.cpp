/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics Co. Ltd. All rights reserved.
 Rigorous Open-Loop Predictive Waypoint Controller for Unitree Go2
 ***********************************************************************/

#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <locale>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

// Toggle final leap / jump skill execution at goal waypoint
#define JUMP false

// Quasistatic settling sleep per single-shot impulse (seconds)
#define TIME_SLEEP 3.5

#define TOPIC_HIGHSTATE "rt/sportmodestate"

using namespace unitree::common;

// ===================================================================
// Physical Lower Bounds (Deadband) & Upper Bounds (Safety Limits)
// ===================================================================
#define MIN_V_LINEAR 0.08  // m/s
#define MIN_V_YAW    0.10  // rad/s

#define MAX_VX   1.0 //0.60  // m/s (Linear forward/backward cap)
#define MAX_VY   0.6 //0.40  // m/s (Linear lateral cap)
#define MAX_VYAW 1.00  // rad/s (~57.3 deg/s Angular yaw cap)

// ===================================================================
// Estimated Single-Shot Impulse Dynamics Model Matrices
// ===================================================================
static const double C_OFFSET[3] = {
    0.00060,   // c_x (m)
   -0.00547,   // c_y (m)
   -0.08325    // c_yaw (rad, -4.77 deg)
};

static const double B_IMPULSE[3][3] = {
    { 0.49006,  0.01829,  0.01400 },
    {-0.01689,  0.52700,  0.03446 },
    {-0.06484,  0.07759,  0.53092 }
};

static const double B_INV[3][3] = {
    { 2.03186, -0.06325, -0.04946 },
    { 0.04937,  1.91432, -0.12556 },
    { 0.24092, -0.28750,  1.89582 }
};

struct SE2Pose {
  double x = 0.0;
  double y = 0.0;
  double theta = 0.0;
};

struct VelocityCommand {
  double vx = 0.0;
  double vy = 0.0;
  double vyaw = 0.0;
  bool clamped = false;
  bool sub_threshold = false;
};

struct OpenLoopStep {
  std::size_t target_wp_idx = 0;
  SE2Pose start_virtual_pose;
  SE2Pose expected_landing_pose;
  VelocityCommand v_cmd;
  std::string status;
};

// Angle wrap to [-pi, pi]
static double wrap_angle(double a) {
  return std::atan2(std::sin(a), std::cos(a));
}

// Compute relative body displacement from p_start to p_target
static void diff_se2_body(const SE2Pose& p_start, const SE2Pose& p_target,
                          double& dx_b, double& dy_b, double& dth_b) {
  double dx_g = p_target.x - p_start.x;
  double dy_g = p_target.y - p_start.y;
  dth_b = wrap_angle(p_target.theta - p_start.theta);

  double cos_th = std::cos(p_start.theta);
  double sin_th = std::sin(p_start.theta);

  dx_b =  cos_th * dx_g + sin_th * dy_g;
  dy_b = -sin_th * dx_g + cos_th * dy_g;
}

// Predict pose update using Forward Dynamics Model: Delta_p_body = B_impulse * v_cmd + c
static SE2Pose apply_forward_model(const SE2Pose& p_start, const VelocityCommand& v_cmd) {
  double dp_b[3];
  dp_b[0] = B_IMPULSE[0][0] * v_cmd.vx + B_IMPULSE[0][1] * v_cmd.vy + B_IMPULSE[0][2] * v_cmd.vyaw + C_OFFSET[0];
  dp_b[1] = B_IMPULSE[1][0] * v_cmd.vx + B_IMPULSE[1][1] * v_cmd.vy + B_IMPULSE[1][2] * v_cmd.vyaw + C_OFFSET[1];
  dp_b[2] = B_IMPULSE[2][0] * v_cmd.vx + B_IMPULSE[2][1] * v_cmd.vy + B_IMPULSE[2][2] * v_cmd.vyaw + C_OFFSET[2];

  double cos_th = std::cos(p_start.theta);
  double sin_th = std::sin(p_start.theta);

  double dx_g = cos_th * dp_b[0] - sin_th * dp_b[1];
  double dy_g = sin_th * dp_b[0] + cos_th * dp_b[1];

  SE2Pose p_next;
  p_next.x = p_start.x + dx_g;
  p_next.y = p_start.y + dy_g;
  p_next.theta = wrap_angle(p_start.theta + dp_b[2]);

  return p_next;
}

// Compute raw inverse dynamics velocity
static VelocityCommand compute_raw_inverse_velocity(double target_dx, double target_dy, double target_dtheta) {
  double dx_adj = target_dx - C_OFFSET[0];
  double dy_adj = target_dy - C_OFFSET[1];
  double dth_adj = target_dtheta - C_OFFSET[2];

  VelocityCommand raw;
  raw.vx   = B_INV[0][0] * dx_adj + B_INV[0][1] * dy_adj + B_INV[0][2] * dth_adj;
  raw.vy   = B_INV[1][0] * dx_adj + B_INV[1][1] * dy_adj + B_INV[1][2] * dth_adj;
  raw.vyaw = B_INV[2][0] * dx_adj + B_INV[2][1] * dy_adj + B_INV[2][2] * dth_adj;

  if (std::abs(raw.vx) < MIN_V_LINEAR && std::abs(raw.vy) < MIN_V_LINEAR && std::abs(raw.vyaw) < MIN_V_YAW) {
    raw.sub_threshold = true;
  }

  return raw;
}

class Custom {
 public:
  Custom() {
    sport_client_.SetTimeout(10.0f);
    sport_client_.Init();

    suber_.reset(
        new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>(
            TOPIC_HIGHSTATE));
    suber_->InitChannel(
        std::bind(&Custom::HighStateHandler, this, std::placeholders::_1), 1);

    highstate_logfile_.open("Movement_HighState_pos_log.txt", std::ios::out | std::ios::app);
    highstate_logfile_.imbue(std::locale::classic());

    footpos_velocity_logfile_.open("Foot_log.txt", std::ios::out | std::ios::app);
    footpos_velocity_logfile_.imbue(std::locale::classic());
  }

  ~Custom() {
    StopMotion();
    if (highstate_logfile_.is_open()) highstate_logfile_.close();
    if (footpos_velocity_logfile_.is_open()) footpos_velocity_logfile_.close();
  }

  void GetInitState() {
    std::cout << "Waiting for robot state from " << TOPIC_HIGHSTATE << "..." << std::endl;
    while (!state_received_.load()) {
      usleep(10000);
    }

    SE2Pose p0 = GetCurrentPose();
    std::cout << "Recorded HighState pose: x0=" << p0.x << ", y0=" << p0.y
              << ", yaw0=" << p0.theta << " rad (" << (p0.theta * 180.0 / M_PI) << " deg)" << std::endl;

    sport_client_.BalanceStand();
    sleep(3);
    sport_client_.StaticWalk();

    std::cout << "Robot Mode: " << GetMode() << std::endl;
  }

  SE2Pose GetCurrentPose() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    SE2Pose p;
    p.x = state_.position()[0];
    p.y = state_.position()[1];
    p.theta = state_.imu_state().rpy()[2];
    return p;
  }

  // Load SE(2) waypoints from hardcoded deployment paths or local fallbacks
  std::vector<SE2Pose> LoadWaypoints() {
    std::vector<SE2Pose> waypoints;

    std::vector<std::string> candidate_paths = {
      "/home/unitree-arka/go2_planner_offline/se2_waypoints.txt",
      "/home/unitree-arka/go2_planner_offline/se2waypoints.txt",
      "se2_waypoints.txt",
      "se2waypoints.txt"
    };

    std::string loaded_path = "";
    for (const auto& path : candidate_paths) {
      std::ifstream f(path);
      if (f.is_open()) {
        std::string line;
        while (std::getline(f, line)) {
          if (line.empty() || line[0] == '#') continue;
          for (char& c : line) if (c == ',') c = ' ';
          std::istringstream ss(line);
          SE2Pose p;
          if (ss >> p.x >> p.y >> p.theta) {
            waypoints.push_back(p);
          }
        }
        f.close();
        if (!waypoints.empty()) {
          loaded_path = path;
          break;
        }
      }
    }

    if (!waypoints.empty()) {
      std::cout << "Successfully loaded " << waypoints.size() << " SE(2) waypoints from: " << loaded_path << std::endl;
    } else {
      std::cerr << "ERROR: Failed to load waypoints from any candidate path!" << std::endl;
    }

    return waypoints;
  }

  // Purely Open-Loop Predictive Model Solver
  std::vector<OpenLoopStep> GenerateOpenLoopPlan(const std::vector<SE2Pose>& waypoints) {
    std::vector<OpenLoopStep> steps;
    if (waypoints.size() < 2) return steps;

    SE2Pose P_virtual = waypoints[0];
    std::size_t i = 1;

    while (i < waypoints.size()) {
      SE2Pose target_wp = waypoints[i];

      // 1. Calculate body displacement from current virtual pose to target_wp
      double dx_b, dy_b, dth_b;
      diff_se2_body(P_virtual, target_wp, dx_b, dy_b, dth_b);

      // 2. Raw inverse model velocity
      VelocityCommand raw_cmd = compute_raw_inverse_velocity(dx_b, dy_b, dth_b);

      // 3. Lower Bounds Check: If sub-threshold, do NOT add to steps, skip WP_i and target WP_{i+1}
      if (raw_cmd.sub_threshold) {
        std::cout << "[Solver Note] Sub-threshold motion to WP" << i << ". Skipping WP" << i
                  << " and targeting WP" << (i + 1) << std::endl;
        i++;
        continue;
      }

      // 4. Upper Bounds Check: Clamp velocity to user-specified safety limits
      VelocityCommand v_exec = raw_cmd;
      bool is_clamped = false;

      if (std::abs(v_exec.vx) > MAX_VX) {
        v_exec.vx = std::copysign(MAX_VX, v_exec.vx);
        is_clamped = true;
      }
      if (std::abs(v_exec.vy) > MAX_VY) {
        v_exec.vy = std::copysign(MAX_VY, v_exec.vy);
        is_clamped = true;
      }
      if (std::abs(v_exec.vyaw) > MAX_VYAW) {
        v_exec.vyaw = std::copysign(MAX_VYAW, v_exec.vyaw);
        is_clamped = true;
      }

      // Filter sub-threshold individual components to zero
      if (std::abs(v_exec.vx) < MIN_V_LINEAR) v_exec.vx = 0.0;
      if (std::abs(v_exec.vy) < MIN_V_LINEAR) v_exec.vy = 0.0;
      if (std::abs(v_exec.vyaw) < MIN_V_YAW)   v_exec.vyaw = 0.0;

      // 5. Forward Dynamics Model: Estimate expected landing pose after execution
      SE2Pose P_next_virtual = apply_forward_model(P_virtual, v_exec);

      OpenLoopStep step;
      step.target_wp_idx = i;
      step.start_virtual_pose = P_virtual;
      step.v_cmd = v_exec;
      step.expected_landing_pose = P_next_virtual;
      step.status = is_clamped ? "CLAMPED" : "OK";

      steps.push_back(step);

      // Update virtual pose to expected landing pose
      P_virtual = P_next_virtual;

      // If clamped: do NOT advance i! Retry current waypoint i from new P_virtual.
      // If not clamped: advance i to i+1!
      if (is_clamped) {
        std::cout << "[Solver Note] Clamped step to WP" << i << ". Retrying WP" << i
                  << " from landing pose (" << P_virtual.x << ", " << P_virtual.y << ", " << P_virtual.theta << ")" << std::endl;
      } else {
        i++;
      }
    }

    return steps;
  }

  void ExecuteOpenLoopControl() {
    PrintHelp();

    std::vector<SE2Pose> waypoints = LoadWaypoints();
    if (waypoints.size() < 2) return;

    std::vector<OpenLoopStep> exec_plan = GenerateOpenLoopPlan(waypoints);
    if (exec_plan.empty()) {
      std::cerr << "ERROR: Failed to generate valid open-loop execution plan." << std::endl;
      return;
    }

    // --------------------------------------------------------
    // DISPLAY PURELY OPEN-LOOP EXECUTION PLAN PREVIEW
    // --------------------------------------------------------
    std::cout << "\n================================================================================";
    std::cout << "\nPURELY OPEN-LOOP MODEL-BASED EXECUTION PLAN PREVIEW";
    std::cout << "\n================================================================================";
    std::cout << "\nTotal Waypoints: " << waypoints.size() << " | Executable Impulse Steps: " << exec_plan.size();
    std::cout << "\nQuasistatic Settling Sleep: " << TIME_SLEEP << "s per single-shot Move() invocation";
    std::cout << "\nLower Bounds (Deadband): vx,vy >= " << MIN_V_LINEAR << " m/s | vyaw >= " << MIN_V_YAW << " rad/s";
    std::cout << "\nUpper Bounds (Safety)  : max_vx=" << MAX_VX << " m/s | max_vy=" << MAX_VY << " m/s | max_vyaw=" << MAX_VYAW << " rad/s";
#if defined(JUMP) && (JUMP == true)
    std::cout << "\nFinal Goal Action      : JUMP (FrontJump skill enabled)";
#else
    std::cout << "\nFinal Goal Action      : Normal stop";
#endif
    std::cout << "\n\n";

    std::cout << std::fixed << std::setprecision(4);
    std::cout << std::left
              << std::setw(6)  << "Step"
              << std::setw(8)  << "Target"
              << std::setw(12) << "Move() vx"
              << std::setw(12) << "Move() vy"
              << std::setw(14) << "Move() vyaw"
              << std::setw(10) << "Status"
              << std::setw(26) << "Virtual Landing Pose"
              << "\n";
    std::cout << std::string(88, '-') << "\n";

    for (std::size_t i = 0; i < exec_plan.size(); ++i) {
      const auto& s = exec_plan[i];
      std::ostringstream pose_ss;
      pose_ss << "(" << s.expected_landing_pose.x << ", " << s.expected_landing_pose.y << ", " << s.expected_landing_pose.theta << ")";

      std::cout << std::left
                << std::setw(6)  << i
                << "WP" << std::setw(6) << s.target_wp_idx
                << std::setw(12) << s.v_cmd.vx
                << std::setw(12) << s.v_cmd.vy
                << std::setw(14) << s.v_cmd.vyaw
                << std::setw(10) << s.status
                << std::setw(26) << pose_ss.str()
                << "\n";
    }

    std::cout << std::string(88, '-') << "\n";
    std::cout << "No controls have been sent to the robot yet.\n";
    std::cout << "================================================================================" << std::endl;

    // --------------------------------------------------------
    // REQUIRE EXPLICIT USER CONFIRMATION (ACCEPT Y / y PREFIX)
    // --------------------------------------------------------
    std::cout << "\nVerify robot clearance and safety surroundings.\n";
    std::cout << "Type Y / YES to execute open-loop Move() sequence: " << std::flush;

    std::string confirmation;
    if (!std::getline(std::cin, confirmation) || confirmation.empty() ||
        (confirmation[0] != 'Y' && confirmation[0] != 'y')) {
      std::cout << "Execution cancelled. No controls sent." << std::endl;
      StopMotion();
      return;
    }

    std::cout << "\nExecution confirmed. Starting open-loop single-shot impulse sequence...\n" << std::endl;

    // --------------------------------------------------------
    // PURELY OPEN-LOOP SINGLE-SHOT IMPULSE EXECUTION
    // --------------------------------------------------------
    for (std::size_t i = 0; i < exec_plan.size(); ++i) {
      const auto& s = exec_plan[i];

      std::cout << "[Step " << i << "/" << exec_plan.size() - 1 << "] Target WP" << s.target_wp_idx
                << " | Invoking single Move(" << s.v_cmd.vx << ", " << s.v_cmd.vy << ", " << s.v_cmd.vyaw << ")"
                << std::flush;

      // Invoke single Move() command -- ONCE
      sport_client_.Move(
          static_cast<float>(s.v_cmd.vx),
          static_cast<float>(s.v_cmd.vy),
          static_cast<float>(s.v_cmd.vyaw));

      // Quasistatic settling sleep to allow single-shot pulse to execute & come to rest
      sleep(TIME_SLEEP);

      std::cout << " -> Step completed." << std::endl;
    }

    std::cout << "================================================================================";
    std::cout << "\nMove() waypoint sequence finished successfully.";

#if defined(JUMP) && (JUMP == true)
    std::cout << "\n================================================================================";
    std::cout << "\nEXECUTING FINAL JUMP MANEUVER (FrontJump)...";
    std::cout << "\n================================================================================" << std::endl;

    sport_client_.FrontJump();
    sleep(5); // Allow crouch, forward leap, and standing recovery to complete
    std::cout << "Final FrontJump skill completed." << std::endl;
#endif

    std::cout << "================================================================================" << std::endl;
    StopMotion();
  }

 private:
  void StopMotion() {
    sport_client_.StopMove();
    std::cout << "Motion stopped." << std::endl;
  }

  int GetMode() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return static_cast<int>(state_.mode());
  }

  void PrintHelp() const {
    std::cout << "\nUnitree Go2 Rigorous Open-Loop Waypoint Controller\n"
              << "  Reads waypoints from /home/unitree-arka/go2_planner_offline/se2_waypoints.txt\n"
              << "  Applies estimated inverse impulse dynamics model B_inv * (Delta_p - c)\n"
              << "  Lower Bounds (Deadband): Skips sub-threshold commands, targets WP_{i+1}\n"
              << "  Upper Bounds (Safety)  : Clamps velocity, retries WP_i\n"
              << "  JUMP Flag Enabled      : Triggers sport_client_.FrontJump() after final waypoint\n"
              << "  Confirmation Prompt    : Accepts any input starting with 'Y' or 'y'\n"
              << "  Executes single Move() invocations open-loop with quasistatic settling sleeps\n"
              << std::endl;
  }

  void HighStateHandler(const void* message) {
    unitree_go::msg::dds_::SportModeState_ snapshot =
        *static_cast<const unitree_go::msg::dds_::SportModeState_*>(message);

    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      state_ = snapshot;
    }
    state_received_.store(true);

    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm local_tm{};
    localtime_r(&now_time, &local_tm);
    std::ostringstream time_stream;
    time_stream << "[" << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S") << "]";

    if (highstate_logfile_.is_open()) {
      highstate_logfile_ << time_stream.str() << " Position: "
                         << snapshot.position()[0] << ", "
                         << snapshot.position()[1] << ", "
                         << snapshot.position()[2] << " | IMU RPY: "
                         << snapshot.imu_state().rpy()[0] << ", "
                         << snapshot.imu_state().rpy()[1] << ", "
                         << snapshot.imu_state().rpy()[2] << " | Angular Vel: "
                         << snapshot.imu_state().gyroscope()[0] << ", "
                         << snapshot.imu_state().gyroscope()[1] << ", "
                         << snapshot.imu_state().gyroscope()[2]
                         << " | Acceleration: "
                         << snapshot.imu_state().accelerometer()[0] << ", "
                         << snapshot.imu_state().accelerometer()[1] << ", "
                         << snapshot.imu_state().accelerometer()[2]
                         << " | Quaternion: "
                         << snapshot.imu_state().quaternion()[0] << ","
                         << snapshot.imu_state().quaternion()[1] << ","
                         << snapshot.imu_state().quaternion()[2] << ","
                         << snapshot.imu_state().quaternion()[3] << " | Mode: "
                         << static_cast<int>(snapshot.mode()) << std::endl;
      highstate_logfile_ << std::flush;
    }

    if (footpos_velocity_logfile_.is_open()) {
      footpos_velocity_logfile_ << time_stream.str() << " Velocity: "
                                << "Vx=" << snapshot.velocity()[0] << ", "
                                << "Vy=" << snapshot.velocity()[1] << ", "
                                << "Vz=" << snapshot.velocity()[2] << ", "
                                << "FR(x,y,z)="
                                << snapshot.foot_position_body()[0] << ","
                                << snapshot.foot_position_body()[1] << ","
                                << snapshot.foot_position_body()[2] << ", "
                                << "FL(x,y,z)="
                                << snapshot.foot_position_body()[3] << ","
                                << snapshot.foot_position_body()[4] << ","
                                << snapshot.foot_position_body()[5] << ", "
                                << "RR(x,y,z)="
                                << snapshot.foot_position_body()[6] << ","
                                << snapshot.foot_position_body()[7] << ","
                                << snapshot.foot_position_body()[8] << ", "
                                << "RL(x,y,z)="
                                << snapshot.foot_position_body()[9] << ","
                                << snapshot.foot_position_body()[10] << ","
                                << snapshot.foot_position_body()[11]
                                << std::endl;
      footpos_velocity_logfile_ << std::flush;
    }
  }

  unitree_go::msg::dds_::SportModeState_ state_{};
  unitree::robot::go2::SportClient sport_client_;
  unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::SportModeState_> suber_;

  std::ofstream highstate_logfile_;
  std::ofstream footpos_velocity_logfile_;

  mutable std::mutex state_mutex_;
  std::atomic<bool> state_received_{false};
};

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " networkInterface" << std::endl;
    return -1;
  }

  unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);

  std::locale::global(std::locale::classic());
  std::cout.imbue(std::locale::classic());
  std::cerr.imbue(std::locale::classic());

  Custom custom;

  sleep(1);
  custom.GetInitState();
  custom.ExecuteOpenLoopControl();

  return 0;
}
