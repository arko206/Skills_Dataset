/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics Co. Ltd. All rights reserved.
 Open-Loop HandStand Skill-Chain Executor for Unitree Go2
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
// Planner command limits (same as skill_chain_planner.cc HSMOVE_MAX_*)
//   Used ONLY to validate the loaded file. Commands are never clamped here.
// ===================================================================
// #define MAX_VX   1.0  // m/s
// #define MAX_VY   0.6  // m/s
// #define MAX_VYAW 1.0  // rad/s

// ===================================================================
// Skill-chain timing (same as go2_single_handstand_and_move.cpp)
// ===================================================================
static constexpr std::chrono::seconds HANDSTAND_ACTIVATION_WAIT{5};    // after HandStand(true)
static constexpr std::chrono::duration<double> BETWEEN_MOVE_WAIT{3.5}; // between consecutive Move() calls
static constexpr std::chrono::seconds BEFORE_RECOVERY_WAIT{2};         // after last Move(), before HandStand(false)

// One planner-generated HandStand+Move command (one row of the velocity file)
struct VelocityCommand {
  double vx = 0.0;
  double vy = 0.0;
  double vyaw = 0.0;
};

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

  // Load planner-generated HandStand+Move commands (vx, vy, vyaw per row).
  // The first candidate file that opens is used. If its contents are invalid,
  // loading aborts (returns empty) instead of falling back to another file.
  std::vector<VelocityCommand> LoadVelocityCommands() {
    std::vector<VelocityCommand> commands;

    std::vector<std::string> candidate_paths = {
      "/home/unitree-arka/go2_planner_offline/skill_chain_velocity_commands.txt",
      "skill_chain_velocity_commands.txt"
    };

    for (const auto& path : candidate_paths) {
      std::ifstream f(path);
      if (!f.is_open()) continue;

      std::string line;
      std::size_t line_no = 0;
      while (std::getline(f, line)) {
        ++line_no;
        const auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#') continue;

        for (char& c : line) if (c == ',') c = ' ';
        std::istringstream ss(line);
        ss.imbue(std::locale::classic());

        VelocityCommand cmd;
        std::string extra;
        if (!(ss >> cmd.vx >> cmd.vy >> cmd.vyaw) || (ss >> extra)) {
          std::cerr << "ERROR: " << path << ":" << line_no
                    << ": expected exactly three values 'vx, vy, vyaw'. Aborting." << std::endl;
          return {};
        }

        if (!std::isfinite(cmd.vx) || !std::isfinite(cmd.vy) || !std::isfinite(cmd.vyaw)) {
          std::cerr << "ERROR: " << path << ":" << line_no
                    << ": NaN or infinite velocity value. Aborting." << std::endl;
          return {};
        }

        // if (std::abs(cmd.vx) > MAX_VX || std::abs(cmd.vy) > MAX_VY || std::abs(cmd.vyaw) > MAX_VYAW) {
        //   std::cerr << "ERROR: " << path << ":" << line_no
        //             << ": command (" << cmd.vx << ", " << cmd.vy << ", " << cmd.vyaw
        //             << ") exceeds planner limits |vx|<=" << MAX_VX << ", |vy|<=" << MAX_VY
        //             << ", |vyaw|<=" << MAX_VYAW << ". Aborting." << std::endl;
        //   return {};
        // }

        commands.push_back(cmd);
      }

      if (commands.empty()) {
        std::cerr << "ERROR: " << path << " contains zero velocity commands. Aborting." << std::endl;
        return {};
      }

      std::cout << "Successfully loaded " << commands.size() << " velocity commands from: " << path << std::endl;
      return commands;
    }

    std::cerr << "ERROR: Failed to open a velocity-command file from any candidate path!" << std::endl;
    return {};
  }

  void ExecuteSkillChain() {
    PrintHelp();

    const std::vector<VelocityCommand> commands = LoadVelocityCommands();
    if (commands.empty()) {
      std::cerr << "No valid velocity commands. No robot command has been sent." << std::endl;
      return;
    }

    // --------------------------------------------------------
    // DISPLAY EXECUTION PREVIEW
    // --------------------------------------------------------
    std::cout << "\n============================================================";
    std::cout << "\nHANDSTAND SKILL-CHAIN EXECUTION PREVIEW";
    std::cout << "\n============================================================";
    std::cout << "\nNumber of HandStand+Move commands: " << commands.size() << "\n\n";

    std::cout << std::fixed << std::setprecision(4);
    std::cout << std::left
              << std::setw(8)  << "Step"
              << std::setw(12) << "vx"
              << std::setw(12) << "vy"
              << std::setw(12) << "vyaw"
              << "\n";
    for (std::size_t i = 0; i < commands.size(); ++i) {
      std::cout << std::left
                << std::setw(8)  << i
                << std::setw(12) << commands[i].vx
                << std::setw(12) << commands[i].vy
                << std::setw(12) << commands[i].vyaw
                << "\n";
    }

    std::cout << std::setprecision(1)
              << "\nHandStand activation wait : "
              << std::chrono::duration<double>(HANDSTAND_ACTIVATION_WAIT).count() << " s"
              << "\nBetween Move commands     : " << BETWEEN_MOVE_WAIT.count() << " s"
              << "\nBefore HandStand recovery : "
              << std::chrono::duration<double>(BEFORE_RECOVERY_WAIT).count() << " s"
              << "\n\nNo robot command has been sent yet.";
    std::cout << "\n============================================================" << std::endl;
    std::cout << std::setprecision(4);

    // --------------------------------------------------------
    // REQUIRE EXPLICIT USER CONFIRMATION (ACCEPT Y / y PREFIX)
    // --------------------------------------------------------
    std::cout << "\nVerify robot clearance and safety surroundings.\n";
    std::cout << "Type Y / YES to execute HandStand skill chain: " << std::flush;

    std::string confirmation;
    if (!std::getline(std::cin, confirmation) || confirmation.empty() ||
        (confirmation[0] != 'Y' && confirmation[0] != 'y')) {
      std::cout << "Execution cancelled. No controls sent." << std::endl;
      StopMotion();
      return;
    }

    std::cout << "\nExecution confirmed. Starting HandStand skill chain...\n" << std::endl;

    // --------------------------------------------------------
    // A. STAND -> HANDSTAND (exactly once)
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
    // B/C. HANDSTAND_MOVE SEQUENCE: one Move() per file row, in file order
    // --------------------------------------------------------
    const std::size_t n = commands.size();
    for (std::size_t i = 0; i < n; ++i) {
      const VelocityCommand& cmd = commands[i];

      std::cout << "[Step " << i << "/" << n - 1 << "] Move("
                << cmd.vx << ", " << cmd.vy << ", " << cmd.vyaw << ")" << std::endl;

      const int32_t move_ret = sport_client_.Move(
          static_cast<float>(cmd.vx),
          static_cast<float>(cmd.vy),
          static_cast<float>(cmd.vyaw));

      if (move_ret != 0) {
        std::cerr << "[ERROR] SportClient::Move() failed at step " << i << ", return=" << move_ret
                  << ". Aborting remaining Move sequence." << std::endl;
        const int32_t close_ret = sport_client_.HandStand(false);
        std::cerr << "[CLEANUP] HandStand(false), return=" << close_ret << std::endl;
        return;
      }

      if (i + 1 < n) {
        std::this_thread::sleep_for(BETWEEN_MOVE_WAIT);
      }
    }

    // --------------------------------------------------------
    // D. HANDSTAND -> STAND (exactly once)
    // --------------------------------------------------------
    std::this_thread::sleep_for(BEFORE_RECOVERY_WAIT);

    std::cout << "[HANDSTAND] HandStand(false)" << std::endl;
    const int32_t recover_ret = sport_client_.HandStand(false);
    if (recover_ret != 0) {
      std::cerr << "[ERROR] SportClient::HandStand(false) failed, return=" << recover_ret << std::endl;
      return;
    }

    std::cout << "============================================================";
    std::cout << "\nHandStand skill chain finished successfully.";
    std::cout << "\n============================================================" << std::endl;
    StopMotion();
  }

 private:
  void StopMotion() {
    sport_client_.StopMove();
    std::cout << "Motion stopped." << std::endl;
  }

  void PrintHelp() const {
    std::cout << "\nUnitree Go2 Open-Loop HandStand Skill-Chain Executor\n"
              << "  Reads planner-generated HandStand+Move commands (vx, vy, vyaw) from\n"
              << "    /home/unitree-arka/go2_planner_offline/skill_chain_velocity_commands.txt\n"
              << "  Executes them directly in file order (no recomputation, clamping or deadband)\n"
              << "  HandStand(true) is called once before the sequence\n"
              << "  Every file row produces exactly one Move() call\n"
              << "  HandStand(false) is called once after the sequence\n"
              << "  Execution is open-loop (no state feedback)\n"
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
