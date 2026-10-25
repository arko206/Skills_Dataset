/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics Co. Ltd. All rights reserved.
 Predictive Open-Loop HandStand Waypoint Executor for Unitree Go2
 ***********************************************************************/

#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <sstream>
#include <string>
#include <thread>
#include <vector>


// ===================================================================
// Skill-chain timing (same as go2_single_handstand_and_move.cpp)
// ===================================================================
static constexpr std::chrono::seconds HANDSTAND_ACTIVATION_WAIT{5};    // after HandStand(true)
static constexpr std::chrono::duration<double> BETWEEN_MOVE_WAIT{3.5}; // between consecutive Move() calls
static constexpr std::chrono::seconds BEFORE_RECOVERY_WAIT{2};         // after last Move(), before HandStand(false)

// ===================================================================
// Physical Lower Bounds (Deadband) & Upper Bounds (Safety Limits)
// ===================================================================
#define MIN_V_LINEAR 0.08  // m/s
#define MIN_V_YAW    0.10  // rad/s

// HandStand+Move admissible domain (same as skill_chain_planner.cc):
//   vx in [0, MAX_VX] (no reverse), vy in [-MAX_VY, +MAX_VY], vyaw in [-MAX_VYAW, +MAX_VYAW]
#define MAX_VX   0.6   // m/s   (forward only; negative vx is set to 0)
#define MAX_VY   0.4  // m/s   (signed lateral cap)
#define MAX_VYAW 0.5   // rad/s (signed yaw cap)



// C. HandStand+Move affine model:  Delta_p_body = B * V + C,   V = [vx, vy, vyaw]^T
static constexpr double SKILL_HSMOVE_B[3][3] = {
    { 0.181792246704, -0.030275430434, -0.000232524242},
    {-0.001526092267,  0.423638652233,  0.034211133410},
    {-0.026335041961, -0.152372699265,  0.190677912747}
};

static constexpr double SKILL_HSMOVE_C[3] = {
     0.136898489758,
    -0.106336208981,
    -0.099299686850
};

static constexpr double SKILL_HSMOVE_B_INV[3][3] = {
    { 5.495224046577,  0.371174824464, -0.059894397153},
    {-0.038979075478,  2.214774532143, -0.397418922204},
    { 0.727811663683,  1.821113275786,  4.918592148994}
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

// Predict pose update using the HandStand+Move forward model:
//   Delta_p_body = SKILL_HSMOVE_B * v_cmd + SKILL_HSMOVE_C
static SE2Pose apply_forward_model(const SE2Pose& p_start, const VelocityCommand& v_cmd) {
  double dp_b[3];
  dp_b[0] = SKILL_HSMOVE_B[0][0] * v_cmd.vx + SKILL_HSMOVE_B[0][1] * v_cmd.vy + SKILL_HSMOVE_B[0][2] * v_cmd.vyaw + SKILL_HSMOVE_C[0];
  dp_b[1] = SKILL_HSMOVE_B[1][0] * v_cmd.vx + SKILL_HSMOVE_B[1][1] * v_cmd.vy + SKILL_HSMOVE_B[1][2] * v_cmd.vyaw + SKILL_HSMOVE_C[1];
  dp_b[2] = SKILL_HSMOVE_B[2][0] * v_cmd.vx + SKILL_HSMOVE_B[2][1] * v_cmd.vy + SKILL_HSMOVE_B[2][2] * v_cmd.vyaw + SKILL_HSMOVE_C[2];

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

// Compute raw HandStand+Move inverse velocity:
//   V = SKILL_HSMOVE_B_INV * (Delta_p_body - SKILL_HSMOVE_C)
static VelocityCommand compute_raw_inverse_velocity(double target_dx, double target_dy, double target_dtheta) {
  double dx_adj = target_dx - SKILL_HSMOVE_C[0];
  double dy_adj = target_dy - SKILL_HSMOVE_C[1];
  double dth_adj = target_dtheta - SKILL_HSMOVE_C[2];

  VelocityCommand raw;
  raw.vx   = SKILL_HSMOVE_B_INV[0][0] * dx_adj + SKILL_HSMOVE_B_INV[0][1] * dy_adj + SKILL_HSMOVE_B_INV[0][2] * dth_adj;
  raw.vy   = SKILL_HSMOVE_B_INV[1][0] * dx_adj + SKILL_HSMOVE_B_INV[1][1] * dy_adj + SKILL_HSMOVE_B_INV[1][2] * dth_adj;
  raw.vyaw = SKILL_HSMOVE_B_INV[2][0] * dx_adj + SKILL_HSMOVE_B_INV[2][1] * dy_adj + SKILL_HSMOVE_B_INV[2][2] * dth_adj;

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
  }

  ~Custom() {
    StopMotion();
  }

  bool GetInitState() {
    const int32_t static_walk_ret = sport_client_.StaticWalk();
    std::cout << "[INIT] StaticWalk(), return=" << static_walk_ret << std::endl;

    if (static_walk_ret != 0) {
      std::cerr << "[ERROR] StaticWalk() failed. No HandStand skill chain will be executed." << std::endl;
      return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    return true;
  }

  // Load planner SE(2) waypoints (x, y, theta per row).
  // The first candidate file that opens is used. If its contents are invalid,
  // loading aborts (returns empty) instead of falling back to another file.
  std::vector<SE2Pose> LoadWaypoints() {
    std::vector<SE2Pose> waypoints;

    std::vector<std::string> candidate_paths = {
      "/home/unitree-arka/go2_planner_offline/skill_chain_se2_waypoints.txt",
      "skill_chain_se2_waypoints.txt"
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

  // Predictive open-loop HandStand+Move solver (virtual state, no measured feedback).
  //   WP0 -> WP1         : STAND_TO_HANDSTAND = HandStand(true), no Move()
  //   WP2 ... WP(N-2)    : HANDSTAND_MOVE targets, one or more Move() each
  //   WP(N-2) -> WP(N-1) : HANDSTAND_TO_STAND = HandStand(false), no Move()
  std::vector<OpenLoopStep> GenerateOpenLoopPlan(const std::vector<SE2Pose>& waypoints) {
    std::vector<OpenLoopStep> steps;
    if (waypoints.size() < 4) {
      std::cerr << "ERROR: Need at least 4 waypoints (STAND, HANDSTAND, >=1 HANDSTAND_MOVE target, final STAND), got "
                << waypoints.size() << "." << std::endl;
      return steps;
    }

    // Start from the planner-predicted pose after HandStand(true)
    SE2Pose P_virtual = waypoints[1];
    std::size_t i = 2;

    // The last waypoint is reached by HandStand(false), never by Move()
    while (i < waypoints.size() - 1) {
      SE2Pose target_wp = waypoints[i];

      // 1. Calculate body displacement from current virtual pose to target_wp
      double dx_b, dy_b, dth_b;
      diff_se2_body(P_virtual, target_wp, dx_b, dy_b, dth_b);
      const double POS_TOL = 1e-4;
      const double YAW_TOL = 1e-4;

      if (std::hypot(dx_b, dy_b) < POS_TOL &&
          std::abs(dth_b) < YAW_TOL) {

          std::cout << "[Solver Note] WP" << i
                    << " already reached by virtual pose. "
                    << "No Move() command generated."
                    << std::endl;

          i++;
          continue;
      }

      // 2. Raw HandStand+Move inverse model velocity
      VelocityCommand raw_cmd = compute_raw_inverse_velocity(dx_b, dy_b, dth_b);

      // 3. Lower Bounds Check: If all components are sub-threshold, do NOT add a step for WP_i
      if (raw_cmd.sub_threshold) {
        std::cout << "[Solver Note] Sub-threshold HSMove command to WP" << i
                  << " (all components below deadband). No Move() generated; advancing past WP" << i << std::endl;
        i++;
        continue;
      }

      // 4. Enforce HSMove control domain: vx in [0, MAX_VX], vy/vyaw symmetric
      VelocityCommand v_exec = raw_cmd;
      bool is_clamped = false;

      if (v_exec.vx < 0.0) {
        v_exec.vx = 0.0;   // reverse HandStand motion is not allowed
        is_clamped = true;
      } else if (v_exec.vx > MAX_VX) {
        v_exec.vx = MAX_VX;
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
      v_exec.clamped = is_clamped;

      // Filter sub-threshold individual components to zero
      if (std::abs(v_exec.vx) < MIN_V_LINEAR) v_exec.vx = 0.0;
      if (std::abs(v_exec.vy) < MIN_V_LINEAR) v_exec.vy = 0.0;
      if (std::abs(v_exec.vyaw) < MIN_V_YAW)   v_exec.vyaw = 0.0;

      // 5. HandStand+Move forward model: Estimate expected landing pose after execution
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

  void ExecuteSkillChain() {
    PrintHelp();

    const std::vector<SE2Pose> waypoints = LoadWaypoints();
    if (waypoints.size() < 4) {
      std::cerr << "ERROR: Need at least 4 planner waypoints (got " << waypoints.size()
                << "). No robot command has been sent." << std::endl;
      return;
    }

    const std::vector<OpenLoopStep> exec_plan = GenerateOpenLoopPlan(waypoints);
    if (exec_plan.empty()) {
      std::cerr << "ERROR: Failed to generate a valid predictive HSMove execution plan. "
                << "No robot command has been sent." << std::endl;
      return;
    }

    // --------------------------------------------------------
    // DISPLAY PREDICTIVE EXECUTION PLAN PREVIEW
    // --------------------------------------------------------
    std::cout << "\n==========================================================================================================";
    std::cout << "\nPREDICTIVE HANDSTAND WAYPOINT EXECUTION PREVIEW";
    std::cout << "\n==========================================================================================================";
    std::cout << "\nNumber of planner waypoints        : " << waypoints.size();
    std::cout << "\nNumber of generated HSMove commands: " << exec_plan.size();
    std::cout << "\nWP0 -> WP1 : HandStand(true)  (no Move)";
    std::cout << "\nWP2 ... WP" << waypoints.size() - 2 << " : HSMove inverse-model targets";
    std::cout << "\nWP" << waypoints.size() - 2 << " -> WP" << waypoints.size() - 1 << " : HandStand(false) (no Move)";
    std::cout << "\n\n";

    std::cout << std::fixed << std::setprecision(4);
    std::cout << std::left
              << std::setw(6)  << "Step"
              << std::setw(8)  << "Target"
              << std::setw(10) << "vx"
              << std::setw(10) << "vy"
              << std::setw(10) << "vyaw"
              << std::setw(9)  << "Status"
              << std::setw(30) << "Virtual Start Pose"
              << "Predicted Landing Pose"
              << "\n";
    std::cout << std::string(106, '-') << "\n";

    for (std::size_t i = 0; i < exec_plan.size(); ++i) {
      const auto& s = exec_plan[i];
      std::ostringstream start_ss, landing_ss;
      start_ss.imbue(std::locale::classic());
      landing_ss.imbue(std::locale::classic());
      start_ss << std::fixed << std::setprecision(4) << "(" << s.start_virtual_pose.x << ", "
               << s.start_virtual_pose.y << ", " << s.start_virtual_pose.theta << ")";
      landing_ss << std::fixed << std::setprecision(4) << "(" << s.expected_landing_pose.x << ", "
                 << s.expected_landing_pose.y << ", " << s.expected_landing_pose.theta << ")";

      std::cout << std::left
                << std::setw(6)  << i
                << "WP" << std::setw(6) << s.target_wp_idx
                << std::setw(10) << s.v_cmd.vx
                << std::setw(10) << s.v_cmd.vy
                << std::setw(10) << s.v_cmd.vyaw
                << std::setw(9)  << s.status
                << std::setw(30) << start_ss.str()
                << landing_ss.str()
                << "\n";
    }
    std::cout << std::string(106, '-') << "\n";

    std::cout << std::setprecision(1)
              << "HandStand activation wait : "
              << std::chrono::duration<double>(HANDSTAND_ACTIVATION_WAIT).count() << " s"
              << "\nBetween Move commands     : " << BETWEEN_MOVE_WAIT.count() << " s"
              << "\nBefore HandStand recovery : "
              << std::chrono::duration<double>(BEFORE_RECOVERY_WAIT).count() << " s"
              << "\n\nNo HandStand/Move execution has started yet.";
    std::cout << "\n==========================================================================================================" << std::endl;
    std::cout << std::setprecision(4);

    // --------------------------------------------------------
    // REQUIRE EXPLICIT USER CONFIRMATION (ACCEPT Y / y PREFIX)
    // --------------------------------------------------------
    std::cout << "\nVerify robot clearance and safety surroundings.\n";
    std::cout << "Type Y / YES to execute predictive HandStand waypoint sequence: " << std::flush;

    std::string confirmation;
    if (!std::getline(std::cin, confirmation) || confirmation.empty() ||
        (confirmation[0] != 'Y' && confirmation[0] != 'y')) {
      std::cout << "Execution cancelled. No controls sent." << std::endl;
      StopMotion();
      return;
    }

    std::cout << "\nExecution confirmed. Starting predictive HandStand waypoint sequence...\n" << std::endl;

    // --------------------------------------------------------
    // A. STAND -> HANDSTAND (WP0 -> WP1, exactly once)
    // --------------------------------------------------------
    std::cout << "[HANDSTAND] HandStand(true)" << std::endl;
    const int32_t handstand_ret = sport_client_.HandStand(true);
    if (handstand_ret != 0) {
      std::cerr << "[ERROR] SportClient::HandStand(true) failed, return=" << handstand_ret
                << ". Aborting skill chain." << std::endl;
      return;
    }
    std::this_thread::sleep_for(HANDSTAND_ACTIVATION_WAIT);

    // --------------------------------------------------------
    // B/C. HANDSTAND_MOVE SEQUENCE: one Move() per OpenLoopStep, in plan order
    // --------------------------------------------------------
    const std::size_t n = exec_plan.size();
    for (std::size_t i = 0; i < n; ++i) {
      const OpenLoopStep& s = exec_plan[i];

      std::cout << "[Step " << i << "/" << n - 1 << "] Target WP" << s.target_wp_idx << " | Move("
                << s.v_cmd.vx << ", " << s.v_cmd.vy << ", " << s.v_cmd.vyaw << ")" << std::endl;

      const int32_t move_ret = sport_client_.Move(
          static_cast<float>(s.v_cmd.vx),
          static_cast<float>(s.v_cmd.vy),
          static_cast<float>(s.v_cmd.vyaw));

      if (move_ret != 0) {
        std::cerr << "[ERROR] SportClient::Move() failed at step " << i << " (target WP" << s.target_wp_idx
                  << "), return=" << move_ret << ". Aborting remaining Move sequence." << std::endl;
        const int32_t close_ret = sport_client_.HandStand(false);
        std::cerr << "[CLEANUP] HandStand(false), return=" << close_ret << std::endl;
        return;
      }

      if (i + 1 < n) {
        std::this_thread::sleep_for(BETWEEN_MOVE_WAIT);
      }
    }

    // --------------------------------------------------------
    // D. HANDSTAND -> STAND (WP(N-2) -> WP(N-1), exactly once)
    // --------------------------------------------------------
    std::this_thread::sleep_for(BEFORE_RECOVERY_WAIT);

    std::cout << "[HANDSTAND] HandStand(false)" << std::endl;
    const int32_t recover_ret = sport_client_.HandStand(false);
    if (recover_ret != 0) {
      std::cerr << "[ERROR] SportClient::HandStand(false) failed, return=" << recover_ret << std::endl;
      return;
    }

    std::cout << "============================================================";
    std::cout << "\nPredictive HandStand waypoint sequence finished successfully.";
    std::cout << "\n============================================================" << std::endl;
    StopMotion();
  }

 private:
  void StopMotion() {
    sport_client_.StopMove();
    std::cout << "Motion stopped." << std::endl;
  }

  void PrintHelp() const {
    std::cout << "\nUnitree Go2 Predictive HandStand Waypoint Executor\n"
              << "  Reads planner SE(2) waypoints (x, y, theta) from\n"
              << "    /home/unitree-arka/go2_planner_offline/skill_chain_se2_waypoints.txt\n"
              << "  WP0 -> WP1         : executed as HandStand(true), no Move() command\n"
              << "  WP2 ... WP(N-2)    : HSMove affine inverse model V = B_inv * (Delta_p - C)\n"
              << "  Virtual landing    : HSMove affine forward model Delta_p = B * V + C\n"
              << "  Limits             : 0 <= vx <= " << MAX_VX << " (negative vx -> 0), |vy| <= " << MAX_VY
              << ", |vyaw| <= " << MAX_VYAW << "\n"
              << "  Deadbands          : then zero |vx|,|vy| < " << MIN_V_LINEAR << ", |vyaw| < " << MIN_V_YAW << "\n"
              << "  Clamped command    : waypoint retried virtually from the predicted landing pose\n"
              << "  WP(N-2) -> WP(N-1) : executed as HandStand(false), no Move() command\n"
              << "  Execution is predictive open-loop (no measured pose feedback)\n"
              << "  Confirmation Prompt: accepts any input starting with 'Y' or 'y'\n"
              << std::endl;
  }

  unitree::robot::go2::SportClient sport_client_;
};

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " networkInterface" << std::endl;
    std::cout << "Example: " << argv[0] << " wlx1cbfced3bbc7" << std::endl;
    return -1;
  }

  unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);

  std::locale::global(std::locale::classic());
  std::cout.imbue(std::locale::classic());
  std::cerr.imbue(std::locale::classic());

  Custom custom;

  if (!custom.GetInitState()) {
    return 1;
  }
  custom.ExecuteSkillChain();

  return 0;
}
