/**********************************************************************
 Unitree Go2 single-HandStand experiment.

 - Uses SportClient only.
 - Sends exactly one HandStand() command.
 - Measures the HandStand() API-call execution duration.
 - Records HANDSTAND_START and HANDSTAND_CALL_RETURN wall-clock timestamps.
 - Exits after sending the single HandStand command.
***********************************************************************/


#include <cerrno>
#include <chrono>
#include <cmath>
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
    "/home/unitree-arka/Go2_Skill_Base_Data_Sensor";



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



class Custom
{
    public:
    explicit Custom(
        const std::string &command_event_file,
        const std::string &mode,
        int trial_number)
        : command_event_file_(command_event_file),
        mode_(mode),
        trial_number_(trial_number)
    {
        sport_client_.SetTimeout(1.0f);
        sport_client_.Init();

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
                << "mode,"
                << "trial_number\n"
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
        << "No HandStand command will be sent."
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

void Trigger_HandStand()
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
      << "[HANDSTAND] Sending Handstand Command."
      << std::endl;

  // Wall-clock timestamp for synchronization with the TF logger.
  const auto handstand_call_start_wall_time =
      WallClock::now();

  // Monotonic timestamp for measuring the API-call duration.
  const auto handstand_call_start_steady_time =
      SteadyClock::now();

  WriteCommandEvent(
      "HANDSTAND_START",
      handstand_call_start_wall_time);

  const int32_t handstand_ret =
    sport_client_.HandStand(true);

   if (handstand_ret != 0)
    {
        std::cerr
            << "[ERROR] SportClient::HandStand(true) failed, return="
            << handstand_ret
            << std::endl;

        return;
    }

    std::this_thread::sleep_for(
        std::chrono::seconds(5)
    );

   const int32_t handstand_ret2 =
        sport_client_.HandStand(false);

   if (handstand_ret2 != 0)
    {
        std::cerr
            << "[ERROR] SportClient::HandStand(false) failed, return="
            << handstand_ret2
            << std::endl;

        return;
    }


  // Captured immediately after Handstand() returns.
  const auto handstand_call_return_steady_time =
      SteadyClock::now();

  const auto handstand_call_return_wall_time =
      WallClock::now();

  WriteCommandEvent(
      "HANDSTAND_CALL_RETURN",
      handstand_call_return_wall_time);

  const auto handstand_call_duration =
      handstand_call_return_steady_time - handstand_call_start_steady_time;

  const int64_t handstand_call_duration_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          handstand_call_duration)
          .count();

  const double handstand_call_duration_us =
      static_cast<double>(handstand_call_duration_ns) / 1.0e3;

  const double handstand_call_duration_ms =
      static_cast<double>(handstand_call_duration_ns) / 1.0e6;

  const int64_t start_steady_time_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          handstand_call_start_steady_time.time_since_epoch())
          .count();

  const int64_t return_steady_time_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          handstand_call_return_steady_time.time_since_epoch())
          .count();

  std::cout
      << "[HANDSTAND] start_steady_time_ns = "
      << start_steady_time_ns
      << std::endl;

  std::cout
      << "[HANDSTAND] return_steady_time_ns = "
      << return_steady_time_ns
      << std::endl;

  std::cout
      << "[HANDSTAND] handstand_call_duration_ns = "
      << handstand_call_duration_ns
      << " ns"
      << std::endl;

  std::cout
      << "[HANDSTAND] handstand_call_duration_us = "
      << std::fixed
      << std::setprecision(6)
      << handstand_call_duration_us
      << " us"
      << std::endl;

  std::cout
      << "[HANDSTAND] handstand_call_duration_ms = "
      << std::fixed
      << std::setprecision(6)
      << handstand_call_duration_ms
      << " ms"
      << std::endl;



  std::cout
      << "[HANDSTAND] HandStand() returned successfully."
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
    const std::chrono::system_clock::time_point &time_point)
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
        << mode_ << ","
        << trial_number_ << "\n"
        << std::flush;
}

  unitree::robot::go2::SportClient sport_client_;

  bool motion_ready_ = false;

  std::string command_event_file_;
  std::ofstream command_event_logfile_;

  std::string mode_;
  int trial_number_ = 0;


};

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cerr
            << "Usage: " << argv[0]
            << " networkInterface mode trialNumber"
            << std::endl;

        std::cerr
            << "Example: " << argv[0]
            << " wlx1cbfced3bbc7 HandStand 1"
            << std::endl;

        return 1;
    }

    std::locale::global(std::locale::classic());
    std::cout.imbue(std::locale::classic());
    std::cerr.imbue(std::locale::classic());

    const std::string mode = argv[2];

    if (mode != "HandStand")
    {
        std::cerr
            << "[ERROR] go2_single_handstand supports only "
            << "mode=HandStand. Received: "
            << mode
            << std::endl;

        return 1;
    }

    int trial_number = 0;

    try
    {
        trial_number = std::stoi(argv[3]);
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "[ERROR] trialNumber must be an integer: "
            << error.what()
            << std::endl;

        return 1;
    }

    if (trial_number < 1)
    {
        std::cerr
            << "[ERROR] trialNumber must be >= 1."
            << std::endl;

        return 1;
    }

    const std::string skill_directory =
        std::string(DATASET_ROOT)
        + "/"
        + mode;

    const std::string trial_directory =
        skill_directory
        + "/Trial_"
        + std::to_string(trial_number);

    if (!EnsureDirectory(DATASET_ROOT))
    {
        return 1;
    }

    if (!EnsureDirectory(skill_directory))
    {
        return 1;
    }

    if (!EnsureDirectory(trial_directory))
    {
        return 1;
    }

    const std::string command_event_file =
        trial_directory
        + "/cmd_"
        + mode
        + "_Trial_"
        + std::to_string(trial_number)
        + ".csv";

    std::cout
        << "[TRIAL] mode="
        << mode
        << ", trial_number="
        << trial_number
        << std::endl;

    std::cout
        << "[TRIAL] Directory: "
        << trial_directory
        << std::endl;

    std::cout
        << "[TRIAL] Command event CSV: "
        << command_event_file
        << std::endl;

    // std::signal(SIGINT, SignalHandler);
    // std::signal(SIGTERM, SignalHandler);

    unitree::robot::ChannelFactory::Instance()->Init(
        0,
        argv[1]);

    Custom custom(
        command_event_file,
        mode,
        trial_number);

    if (!custom.GetInitState())
    {
        return 1;
    }

    custom.Trigger_HandStand();

    std::cout
        << "Program finished after the single HandStand command."
        << std::endl;

    return 0;
}
