/**********************************************************************
 Unitree Go2 single-Move-command experiment.

 - Uses SportClient only.
 - Sends exactly one non-zero Move(vx, vy, vyaw) command.
 - Measures the Move() API-call execution duration.
 - Sends no further Move() command.
 - Does not send Move(0,0,0).
 - Does not call StopMove().
 - - Exits after sending the single Move() command and saving
  the command timing events.
***********************************************************************/

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <locale>
#include <mutex>
#include <sstream>
#include <thread>
#include <memory>
#include <sys/stat.h>
#include <sys/types.h>

#include <unitree/robot/go2/sport/sport_client.hpp>
#include <unitree/robot/channel/channel_factory.hpp>

// Root directory shared with demo_test_log_motion.py.
static constexpr const char *DATASET_ROOT =
    "/home/unitree-arka/Go2_Walk_Base_Data_Sensor";

std::string VelocityToken(float value)
{
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << std::fixed << std::setprecision(3) << value;

  std::string text = stream.str();

  while (!text.empty() && text.back() == '0')
  {
    text.pop_back();
  }

  if (!text.empty() && text.back() == '.')
  {
    text.push_back('0');
  }

  if (text == "-0.0")
  {
    text = "0.0";
  }

  return text;
}

bool EnsureDirectory(const std::string &directory)
{
  if (::mkdir(directory.c_str(), 0755) == 0)
  {
    return true;
  }

  if (errno == EEXIST)
  {
    return true;
  }

  std::cerr
      << "[ERROR] Could not create directory: "
      << directory
      << ", errno="
      << errno
      << std::endl;

  return false;
}

std::atomic<bool> g_stop_requested{false};

void SignalHandler(int)
{
  // Do not call robot APIs inside the signal handler.
  // The normal control flow will notice this flag and stop the robot.
  g_stop_requested.store(true);
}

class Custom
{
public:
  explicit Custom(
      const std::string &command_event_file)
      : command_event_file_(command_event_file)
  {
    // Configure the sport client and initialize its communication.
    sport_client_.SetTimeout(1.0f);
    sport_client_.Init();

    // Save the movement command timing data to a CSV file.
    command_event_logfile_.open(
        command_event_file_,
        std::ios::out | std::ios::trunc);

    if (!command_event_logfile_.is_open())
    {
      std::cerr
          << "[ERROR] Failed to open command timestamp file: "
          << command_event_file_
          << std::endl;
    }
    else
    {
      command_event_logfile_.imbue(std::locale::classic());

      command_event_logfile_
          << "event,"
          << "wall_time_ns,"
          << "timestamp_iso,"
          << "vx,"
          << "vy,"
          << "vyaw\n"
          << std::flush;
    }

  }

  ~Custom()
  {


    if (command_event_logfile_.is_open())
    {
      command_event_logfile_.close();
    }
  }

  bool GetInitState()
  {
    // Prepare the robot for walking mode.
    const int32_t static_walk_ret =
        sport_client_.StaticWalk();

    std::cout << "[INIT] StaticWalk(), return="
              << static_walk_ret << std::endl;

    if (static_walk_ret != 0)
    {
      std::cerr
          << "[ERROR] StaticWalk() failed. "
             "No velocity command will be sent."
          << std::endl;
      return false;
    }

    // Pause this thread for 500 milliseconds so the robot has time to
    // settle into walk-ready mode before we start sending motion commands.
    // This is C++ thread sleep time, not Python time. It suspends only this
    // thread in the current process.
    std::this_thread::sleep_for(
        std::chrono::milliseconds(500));

    motion_ready_ = true;
    return true;
  }

void RunSingleMove(
    float vx,
    float vy,
    float vyaw)
{
  if (!motion_ready_)
  {
    std::cerr
        << "[ERROR] Motion mode is not ready. "
        << "The command will not be sent."
        << std::endl;
    return;
  }

  if (!command_event_logfile_.is_open())
  {
    std::cerr
        << "[ERROR] Command timestamp file is unavailable."
        << std::endl;
    return;
  }

  using SteadyClock = std::chrono::steady_clock;
  using WallClock = std::chrono::system_clock;

  std::cout
      << "[MOVE] Sending exactly one SportClient::Move("
      << vx << ", " << vy << ", " << vyaw
      << ") command."
      << std::endl;

  // Wall-clock timestamp for synchronization with the TF logger.
  const auto move_call_start_wall_time =
      WallClock::now();

  // Monotonic timestamp for measuring the API-call duration.
  const auto move_call_start_steady_time =
      SteadyClock::now();

  
  const int32_t move_ret =
    sport_client_.Move(vx, vy, vyaw);

  // Captured immediately after Move() returns.
  const auto move_call_return_steady_time =
      SteadyClock::now();

  const auto move_call_return_wall_time =
      WallClock::now();

  const auto move_call_duration =
      move_call_return_steady_time - move_call_start_steady_time;

  const int64_t move_call_duration_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          move_call_duration)
          .count();

  const double move_call_duration_us =
      static_cast<double>(move_call_duration_ns) / 1.0e3;

  const double move_call_duration_ms =
      static_cast<double>(move_call_duration_ns) / 1.0e6;

  const int64_t start_steady_time_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          move_call_start_steady_time.time_since_epoch())
          .count();

  const int64_t return_steady_time_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          move_call_return_steady_time.time_since_epoch())
          .count();

  std::cout
      << "[MOVE] start_steady_time_ns = "
      << start_steady_time_ns
      << std::endl;

  std::cout
      << "[MOVE] return_steady_time_ns = "
      << return_steady_time_ns
      << std::endl;

  std::cout
      << "[MOVE] move_call_duration_ns = "
      << move_call_duration_ns
      << " ns"
      << std::endl;

  std::cout
      << "[MOVE] move_call_duration_us = "
      << std::fixed
      << std::setprecision(6)
      << move_call_duration_us
      << " us"
      << std::endl;

  std::cout
      << "[MOVE] move_call_duration_ms = "
      << std::fixed
      << std::setprecision(6)
      << move_call_duration_ms
      << " ms"
      << std::endl;




  if (move_ret != 0)
  {
    std::cerr
        << "[ERROR] SportClient::Move() failed, return="
        << move_ret
        << std::endl;
    return;
  }

  

  WriteCommandEvent(
      "MOVE_START",
      move_call_start_wall_time,
      vx,
      vy,
      vyaw);

  WriteCommandEvent(
      "MOVE_CALL_RETURN",
      move_call_return_wall_time,
      vx,
      vy,
      vyaw);

  std::cout
      << "[MOVE] Move() returned successfully."
      << std::endl;
}
  

private:

  // Convert a system clock timestamp to a raw integer number of nanoseconds
  // since the Unix epoch (1970-01-01). This is useful for storing precise
  // timestamps in logs or CSV files.
  static int64_t SystemTimeToNanoseconds(
      const std::chrono::system_clock::time_point &time_point)
  {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               time_point.time_since_epoch())
        .count();
  }

  // Convert a system clock timestamp to a human-readable ISO 8601 string,
  // such as: 2026-08-04T12:34:56.789
  //
  // The function:
  // 1. Converts the time point to a calendar time value.
  // 2. Converts it to local time (for the current machine timezone).
  // 3. Extracts the millisecond part and appends it after the seconds.
  static std::string SystemTimeToIso(
      const std::chrono::system_clock::time_point &time_point)
  {
    // Convert the time point to a C-style time_t value.
    const std::time_t time_value =
        std::chrono::system_clock::to_time_t(time_point);

    // Break the time_t value into year/month/day/hour/minute/second parts.
    std::tm local_time{};
    localtime_r(&time_value, &local_time);

    // Get the total milliseconds since the epoch so we can keep the fraction.
    const int64_t milliseconds_since_epoch =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            time_point.time_since_epoch())
            .count();

    // Keep only the last 3 digits, which represent the milliseconds part.
    const int milliseconds_part =
        static_cast<int>(milliseconds_since_epoch % 1000);

    std::ostringstream stream;

    // Build a string like: YYYY-MM-DDTHH:MM:SS.mmm
    stream << std::put_time(&local_time, "%Y-%m-%dT%H:%M:%S")
           << "."
           << std::setfill('0')
           << std::setw(3)
           << milliseconds_part;

    return stream.str();
  }

  void WriteCommandEvent(
      const std::string &event_name,
      const std::chrono::system_clock::time_point &time_point,
      float vx,
      float vy,
      float vyaw)
    {
      if (!command_event_logfile_.is_open())
      {
        std::cerr
            << "[ERROR] Cannot write command event "
            << event_name
            << ": timestamp file is not open."
            << std::endl;
        return;
      }

      command_event_logfile_
          << event_name << ","
          << SystemTimeToNanoseconds(time_point) << ","
          << SystemTimeToIso(time_point) << ","
          << vx << ","
          << vy << ","
          << vyaw << "\n"
          << std::flush;

    }

  unitree::robot::go2::SportClient sport_client_;

  bool motion_ready_ = false;

  std::string command_event_file_;
  std::ofstream command_event_logfile_;
};

int main(int argc, char **argv)
{
  if (argc < 5)
  {
    std::cerr
        << "Usage: " << argv[0]
        << " networkInterface vx vy vyaw"
        << std::endl;
    return 1;
  }

  std::locale::global(std::locale::classic());
  std::cout.imbue(std::locale::classic());
  std::cerr.imbue(std::locale::classic());

  float vx = 0.0f;
  float vy = 0.0f;
  float vyaw = 0.0f;

  try
  {
    vx = std::stof(argv[2]);
    vy = std::stof(argv[3]);
    vyaw = std::stof(argv[4]);
  }
  catch (const std::exception &error)
  {
    std::cerr
        << "[ERROR] vx, vy, and vyaw must be valid numbers: "
        << error.what()
        << std::endl;
    return 1;
  }

  if (
      !std::isfinite(vx)
      || !std::isfinite(vy)
      || !std::isfinite(vyaw))
  {
    std::cerr
        << "[ERROR] vx, vy, and vyaw must be finite."
        << std::endl;
    return 1;
  }

  const std::string velocity_suffix =
      VelocityToken(vx)
      + "_"
      + VelocityToken(vy)
      + "_"
      + VelocityToken(vyaw);

  const std::string trial_directory =
      std::string(DATASET_ROOT)
      + "/Fd_"
      + velocity_suffix;

  if (!EnsureDirectory(DATASET_ROOT))
  {
    return 1;
  }

  if (!EnsureDirectory(trial_directory))
  {
    return 1;
  }

  const std::string command_event_file =
      trial_directory
      + "/cmd_w_"
      + velocity_suffix
      + ".csv";

  std::cout
      << "[TRIAL] vx=" << vx
      << ", vy=" << vy
      << ", vyaw=" << vyaw
      << std::endl;

  std::cout
      << "[TRIAL] Directory: "
      << trial_directory
      << std::endl;

  std::cout
      << "[TRIAL] Command event CSV: "
      << command_event_file
      << std::endl;

  std::signal(SIGINT, SignalHandler);
  std::signal(SIGTERM, SignalHandler);

  unitree::robot::ChannelFactory::Instance()->Init(
      0,
      argv[1]);

  Custom custom(command_event_file);

  if (!custom.GetInitState())
  {
    return 1;
  }

  custom.RunSingleMove(
      vx,
      vy,
      vyaw);

  std::cout
      << "Program finished after the single Move command."
      << std::endl;

  return 0;
}
