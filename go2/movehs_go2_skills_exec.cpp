/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics Co. Ltd. All rights reserved.
 Open-Loop Move + HandStand Skill-Chain Executor for Unitree Go2
 ***********************************************************************/

#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>

#include <algorithm>
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

// Planner skill-chain CSV (Move_HS_skill_chain_planner.cc). Another file can be given on the command line.
static const std::string DEFAULT_CSV_PATH =
    "/home/unitree-arka/go2_planner_offline/move_hs_skill_chain_steps.csv";

// The CSV must start with these columns, in this order. Later columns (edge_cost, cost) are ignored.
static const std::string CSV_HEADER = "wp,x,y,theta,mode,incoming_skill,vx,vy,vyaw,has_velocity_cmd";

// ===================================================================
// Skill-chain timing (same as go2_single_handstand_and_move.cpp)
//   Move() and HandStand() return immediately; these waits let each motion finish.
// ===================================================================
static constexpr std::chrono::duration<double> HANDSTAND_ACTIVATION_WAIT{5.0};  // after HandStand(true)
static constexpr std::chrono::duration<double> BETWEEN_MOVE_WAIT{3.5};          // after a Move(), before the next skill
static constexpr std::chrono::duration<double> BEFORE_RECOVERY_WAIT{2.0};       // after the last HandStand Move(), before HandStand(false)
static constexpr std::chrono::duration<double> AFTER_RECOVERY_WAIT{3.5};        // after HandStand(false), before the next skill

enum class RobotMode { STAND, HANDSTAND };
enum class SkillPrimitive { START, MOVE, STAND_TO_HANDSTAND, HANDSTAND_MOVE, HANDSTAND_TO_STAND };

// One planner-generated skill (one row of the CSV)
struct SkillStep {
  int wp = 0;
  double x = 0.0;            // planner-predicted pose after the skill (display only)
  double y = 0.0;
  double theta = 0.0;
  std::string mode_name;     // mode after the skill, as written in the CSV
  std::string skill_name;    // incoming_skill, as written in the CSV
  RobotMode mode = RobotMode::STAND;
  SkillPrimitive skill = SkillPrimitive::START;
  double vx = 0.0;           // Move() command, sent unmodified
  double vy = 0.0;
  double vyaw = 0.0;
  bool has_velocity_cmd = false;
};

// Load the planner skill chain, one SkillStep per CSV row.
// A malformed row aborts loading (returns empty).
std::vector<SkillStep> LoadSkillChain(const std::string& path) {
  std::ifstream f(path);
  if (!f.is_open()) {
    std::cerr << "ERROR: Failed to open skill-chain CSV: " << path << std::endl;
    return {};
  }

  std::vector<SkillStep> steps;
  std::string line;
  std::size_t line_no = 0;
  std::size_t n_columns = 0;  // set by the header row

  auto fail = [&](const std::string& message) {
    std::cerr << "ERROR: " << path << ":" << line_no << ": " << message << " Aborting." << std::endl;
    return std::vector<SkillStep>{};
  };

  while (std::getline(f, line)) {
    ++line_no;
    const auto first = line.find_first_not_of(" \t\r\n");
    if (first == std::string::npos || line[first] == '#') continue;

    if (n_columns == 0) {
      if (line.compare(0, CSV_HEADER.size(), CSV_HEADER) != 0) {
        return fail("header must start with '" + CSV_HEADER + "'.");
      }
      n_columns = static_cast<std::size_t>(std::count(line.begin(), line.end(), ',')) + 1;
      continue;
    }

    for (char& c : line) if (c == ',') c = ' ';
    std::istringstream ss(line);
    ss.imbue(std::locale::classic());

    SkillStep step;
    std::string flag;
    if (!(ss >> step.wp >> step.x >> step.y >> step.theta >> step.mode_name >> step.skill_name
             >> step.vx >> step.vy >> step.vyaw >> flag)) {
      return fail("expected 'wp, x, y, theta, mode, incoming_skill, vx, vy, vyaw, has_velocity_cmd'.");
    }

    // Every row must have as many values as the header (catches empty fields)
    std::size_t n_values = 10;
    for (std::string extra; ss >> extra;) ++n_values;
    if (n_values != n_columns) {
      return fail("expected " + std::to_string(n_columns) + " values, found " + std::to_string(n_values) + ".");
    }

    if (flag != "0" && flag != "1") return fail("has_velocity_cmd must be 0 or 1.");
    step.has_velocity_cmd = (flag == "1");

    if (!std::isfinite(step.vx) || !std::isfinite(step.vy) || !std::isfinite(step.vyaw)) {
      return fail("NaN or infinite velocity value.");
    }
    if (step.wp != static_cast<int>(steps.size())) {
      return fail("waypoint index must be " + std::to_string(steps.size()) + ".");
    }

    if (step.mode_name == "STAND") step.mode = RobotMode::STAND;
    else if (step.mode_name == "HANDSTAND") step.mode = RobotMode::HANDSTAND;
    else return fail("unknown mode '" + step.mode_name + "'.");

    if (step.skill_name == "START") step.skill = SkillPrimitive::START;
    else if (step.skill_name == "MOVE") step.skill = SkillPrimitive::MOVE;
    else if (step.skill_name == "STAND_TO_HANDSTAND") step.skill = SkillPrimitive::STAND_TO_HANDSTAND;
    else if (step.skill_name == "HANDSTAND_MOVE") step.skill = SkillPrimitive::HANDSTAND_MOVE;
    else if (step.skill_name == "HANDSTAND_TO_STAND") step.skill = SkillPrimitive::HANDSTAND_TO_STAND;
    else return fail("unknown incoming_skill '" + step.skill_name + "'.");

    steps.push_back(step);
  }

  if (steps.empty()) {
    std::cerr << "ERROR: " << path << " contains zero skill rows. Aborting." << std::endl;
    return {};
  }

  std::cout << "Successfully loaded " << steps.size() << " skill steps from: " << path << std::endl;
  return steps;
}

// Check the skill sequence before any robot communication:
//   the first row is START in STAND, every skill matches the mode change in the CSV,
//   only MOVE / HANDSTAND_MOVE rows carry a Move() command, and the chain ends in STAND.
bool ValidateSkillChain(const std::vector<SkillStep>& steps) {
  if (steps.size() < 2 || steps[0].skill != SkillPrimitive::START || steps[0].mode != RobotMode::STAND ||
      steps[0].has_velocity_cmd) {
    std::cerr << "ERROR: The first row must be START (mode STAND, has_velocity_cmd 0), "
              << "followed by at least one skill." << std::endl;
    return false;
  }

  for (std::size_t i = 1; i < steps.size(); ++i) {
    const SkillStep& s = steps[i];
    const RobotMode before = steps[i - 1].mode;

    bool transition_ok = false;
    switch (s.skill) {
      case SkillPrimitive::MOVE:               transition_ok = (before == RobotMode::STAND && s.mode == RobotMode::STAND); break;
      case SkillPrimitive::STAND_TO_HANDSTAND: transition_ok = (before == RobotMode::STAND && s.mode == RobotMode::HANDSTAND); break;
      case SkillPrimitive::HANDSTAND_MOVE:     transition_ok = (before == RobotMode::HANDSTAND && s.mode == RobotMode::HANDSTAND); break;
      case SkillPrimitive::HANDSTAND_TO_STAND: transition_ok = (before == RobotMode::HANDSTAND && s.mode == RobotMode::STAND); break;
      case SkillPrimitive::START:              break;  // only allowed in the first row
    }
    const bool is_move = (s.skill == SkillPrimitive::MOVE || s.skill == SkillPrimitive::HANDSTAND_MOVE);

    if (!transition_ok) {
      std::cerr << "ERROR: WP " << s.wp << ": " << s.skill_name << " cannot go from "
                << steps[i - 1].mode_name << " to " << s.mode_name << "." << std::endl;
      return false;
    }
    if (s.has_velocity_cmd != is_move) {
      std::cerr << "ERROR: WP " << s.wp << ": " << s.skill_name << " must have has_velocity_cmd = "
                << (is_move ? 1 : 0) << "." << std::endl;
      return false;
    }
    if (is_move && s.vx == 0.0 && s.vy == 0.0 && s.vyaw == 0.0) {
      std::cerr << "ERROR: WP " << s.wp << ": " << s.skill_name << " has a zero Move command." << std::endl;
      return false;
    }
  }

  if (steps.back().mode != RobotMode::STAND) {
    std::cerr << "ERROR: The last row must have mode STAND." << std::endl;
    return false;
  }
  return true;
}

// Print the skills and their commands in execution order
void PrintSkillChain(const std::vector<SkillStep>& steps) {
  std::cout << "\n============================================================";
  std::cout << "\nMOVE + HANDSTAND SKILL-CHAIN EXECUTION PREVIEW";
  std::cout << "\n============================================================\n\n";

  std::cout << std::left
            << std::setw(5)  << "WP"
            << std::setw(21) << "Incoming skill"
            << std::setw(12) << "Mode after"
            << std::setw(28) << "Command"
            << "Predicted x, y, theta\n";

  int n_move = 0, n_hsmove = 0, n_s2h = 0, n_h2s = 0;
  for (const SkillStep& s : steps) {
    std::ostringstream command;
    switch (s.skill) {
      case SkillPrimitive::START:              command << "None"; break;
      case SkillPrimitive::MOVE:               command << "Move(" << s.vx << ", " << s.vy << ", " << s.vyaw << ")"; ++n_move; break;
      case SkillPrimitive::STAND_TO_HANDSTAND: command << "HandStand(true)"; ++n_s2h; break;
      case SkillPrimitive::HANDSTAND_MOVE:     command << "Move(" << s.vx << ", " << s.vy << ", " << s.vyaw << ")"; ++n_hsmove; break;
      case SkillPrimitive::HANDSTAND_TO_STAND: command << "HandStand(false)"; ++n_h2s; break;
    }
    std::cout << std::left
              << std::setw(5)  << s.wp
              << std::setw(21) << s.skill_name
              << std::setw(12) << s.mode_name
              << std::setw(28) << command.str()
              << s.x << ", " << s.y << ", " << s.theta << "\n";
  }

  std::cout << "\n------------------------------------------------------------"
            << "\nMOVE commands              : " << n_move
            << "\nHANDSTAND_MOVE commands    : " << n_hsmove
            << "\nSTAND_TO_HANDSTAND         : " << n_s2h
            << "\nHANDSTAND_TO_STAND         : " << n_h2s
            << "\n------------------------------------------------------------"
            << "\nHandStand activation wait  : " << HANDSTAND_ACTIVATION_WAIT.count() << " s"
            << "\nBetween Move commands      : " << BETWEEN_MOVE_WAIT.count() << " s"
            << "\nBefore HandStand recovery  : " << BEFORE_RECOVERY_WAIT.count() << " s"
            << "\nAfter HandStand recovery   : " << AFTER_RECOVERY_WAIT.count() << " s"
            << "\n============================================================" << std::endl;
}

class Custom {
 public:
  Custom() {
    sport_client_.SetTimeout(10.0f);
    sport_client_.Init();
  }

  bool GetInitState() {
    const int32_t static_walk_ret = sport_client_.StaticWalk();
    std::cout << "[INIT] StaticWalk(), return=" << static_walk_ret << std::endl;

    if (static_walk_ret != 0) {
      std::cerr << "[ERROR] StaticWalk() failed. No skill chain will be executed." << std::endl;
      return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    return true;
  }

  // Execute the CSV rows in order, dispatched on incoming_skill. Stops at the first failed API call.
  bool ExecuteSkillChain(const std::vector<SkillStep>& steps) {
    RobotMode current_mode = RobotMode::STAND;  // mode the executor has commanded (not measured)
    const std::size_t n = steps.size();

    for (std::size_t i = 0; i < n; ++i) {
      const SkillStep& step = steps[i];

      std::cout << "[WP " << step.wp << "/" << n - 1 << "] " << step.skill_name << " -> " << step.mode_name
                << ", predicted pose (" << step.x << ", " << step.y << ", " << step.theta << ")" << std::endl;

      // The commanded mode must be the mode the CSV expects before this skill
      if (i > 0 && current_mode != steps[i - 1].mode) {
        std::cerr << "[ERROR] WP " << step.wp << ": commanded mode does not match the CSV. Stopping." << std::endl;
        return false;
      }

      switch (step.skill) {
        // START: initial robot configuration, nothing to execute
        case SkillPrimitive::START:
          break;

        // MOVE (walking in STAND) and HANDSTAND_MOVE (moving in HANDSTAND) are the same single
        // Move() call with the CSV values; the robot's current mode decides how it moves.
        // HandStand(true) is not called again for HANDSTAND_MOVE.
        case SkillPrimitive::MOVE:
        case SkillPrimitive::HANDSTAND_MOVE: {
          std::cout << "  Move(" << step.vx << ", " << step.vy << ", " << step.vyaw << ")" << std::endl;
          const int32_t move_ret = sport_client_.Move(
              static_cast<float>(step.vx),
              static_cast<float>(step.vy),
              static_cast<float>(step.vyaw));

          if (move_ret != 0) {
            std::cerr << "[ERROR] SportClient::Move() failed at WP " << step.wp << ", return=" << move_ret
                      << ". Stopping the skill chain." << std::endl;
            return false;
          }

          const bool next_is_recovery = (i + 1 < n) && steps[i + 1].skill == SkillPrimitive::HANDSTAND_TO_STAND;
          std::this_thread::sleep_for(next_is_recovery ? BEFORE_RECOVERY_WAIT : BETWEEN_MOVE_WAIT);
          break;
        }

        // STAND -> HANDSTAND
        case SkillPrimitive::STAND_TO_HANDSTAND: {
          std::cout << "  HandStand(true)" << std::endl;
          const int32_t handstand_ret = sport_client_.HandStand(true);
          if (handstand_ret != 0) {
            std::cerr << "[ERROR] SportClient::HandStand(true) failed at WP " << step.wp
                      << ", return=" << handstand_ret << ". Stopping the skill chain." << std::endl;
            return false;
          }
          std::this_thread::sleep_for(HANDSTAND_ACTIVATION_WAIT);
          current_mode = RobotMode::HANDSTAND;
          break;
        }

        // HANDSTAND -> STAND
        case SkillPrimitive::HANDSTAND_TO_STAND: {
          std::cout << "  HandStand(false)" << std::endl;
          const int32_t recover_ret = sport_client_.HandStand(false);
          if (recover_ret != 0) {
            std::cerr << "[ERROR] SportClient::HandStand(false) failed at WP " << step.wp
                      << ", return=" << recover_ret << ". Stopping the skill chain." << std::endl;
            return false;
          }
          std::this_thread::sleep_for(AFTER_RECOVERY_WAIT);
          current_mode = RobotMode::STAND;
          break;
        }
      }
    }

    std::cout << "============================================================";
    std::cout << "\nSkill chain completed";
    std::cout << "\n============================================================" << std::endl;
    StopMotion();  // the chain always ends in STAND
    return true;
  }

 private:
  void StopMotion() {
    sport_client_.StopMove();
    std::cout << "Motion stopped." << std::endl;
  }

  unitree::robot::go2::SportClient sport_client_;
};

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " networkInterface [csvFile]" << std::endl;
    std::cout << "       " << argv[0] << " --dry-run [csvFile]" << std::endl;
    std::cout << "Example: " << argv[0] << " wlx1cbfced3bbc7" << std::endl;
    return -1;
  }

  std::locale::global(std::locale::classic());
  std::cout.imbue(std::locale::classic());
  std::cerr.imbue(std::locale::classic());

  const bool dry_run = (std::string(argv[1]) == "--dry-run");
  const std::string csv_path = (argc >= 3) ? argv[2] : DEFAULT_CSV_PATH;

  // --------------------------------------------------------
  // 1-3. READ, VALIDATE AND PRINT THE SKILL CHAIN
  // --------------------------------------------------------
  const std::vector<SkillStep> steps = LoadSkillChain(csv_path);
  if (steps.empty() || !ValidateSkillChain(steps)) {
    std::cerr << "Invalid skill-chain CSV. No robot command has been sent." << std::endl;
    return 1;
  }

  PrintSkillChain(steps);

  if (dry_run) {
    std::cout << "\nDry run: robot communication not initialized, no robot command sent." << std::endl;
    return 0;
  }

  // --------------------------------------------------------
  // 4. REQUIRE EXPLICIT USER CONFIRMATION (ACCEPT Y / y PREFIX)
  // --------------------------------------------------------
  std::cout << "\nVerify robot clearance and safety surroundings.\n";
  std::cout << "Type Y / YES to execute the skill chain: " << std::flush;

  std::string confirmation;
  if (!std::getline(std::cin, confirmation) || confirmation.empty() ||
      (confirmation[0] != 'Y' && confirmation[0] != 'y')) {
    std::cout << "Execution cancelled. No controls sent." << std::endl;
    return 0;
  }

  std::cout << "\nExecution confirmed. Starting skill chain...\n" << std::endl;

  // --------------------------------------------------------
  // 5-6. ROBOT COMMUNICATION AND StaticWalk()
  // --------------------------------------------------------
  unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);

  Custom custom;

  if (!custom.GetInitState()) {
    return 1;
  }

  // --------------------------------------------------------
  // 7-10. EXECUTE THE CSV ROWS IN ORDER
  // --------------------------------------------------------
  return custom.ExecuteSkillChain(steps) ? 0 : 1;
}
