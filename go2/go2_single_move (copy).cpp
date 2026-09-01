/**********************************************************************
 Unitree Go2 single-Move-command experiment.

 - Uses SportClient only.
 - Sends exactly one non-zero Move(vx, vy, vyaw) command.
 - Measures the Move() API-call execution duration.
 - Sends no further Move() command.
 - Does not send Move(0,0,0).
 - Does not call StopMove().
 - Keeps the process alive for three seconds to observe the robot's
   response and record high-state feedback.
***********************************************************************/

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
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

#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>
#include <unitree/robot/channel/channel_factory.hpp>

#define TOPIC_HIGHSTATE "rt/sportmodestate"

//--- Path to the CSV file that saves the movement command window data.--//
static constexpr const char *COMMAND_EVENT_FILE =
    "/home/unitree-arka/Go2_Walk_Base_Data_Sensor/"
    "First_Single_command_window.csv";

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
  Custom()
  {
    // Configure the sport client and initialize its communication.
    sport_client_.SetTimeout(1.0f);
    sport_client_.Init();

    // Create a subscriber for the robot's high-state topic.
    // This will receive feedback such as position, IMU, velocity, and foot data.
    highstate_subscriber_.reset(
        new unitree::robot::ChannelSubscriber<
            unitree_go::msg::dds_::SportModeState_>(
            TOPIC_HIGHSTATE));

    highstate_subscriber_->InitChannel(
        std::bind(
            &Custom::HighStateHandler,
            this,
            std::placeholders::_1),
        1);

    highstate_logfile_.open(
        "Movement_HighState_pos_log.txt",
        std::ios::out | std::ios::trunc);

    footpos_velocity_logfile_.open(
        "Foot_log.txt",
        std::ios::out | std::ios::trunc);
    
    //--- saving the movement command window data to a CSV file. ---//
    command_event_logfile_.open(
        COMMAND_EVENT_FILE,
        std::ios::out | std::ios::trunc);

    if (!command_event_logfile_.is_open())
    {
      std::cerr
          << "[ERROR] Failed to open command timestamp file: "
          << COMMAND_EVENT_FILE
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

    if (!highstate_logfile_.is_open())
    {
      std::cerr << "[ERROR] Failed to open "
                   "Movement_HighState_pos_log.txt"
                << std::endl;
    }
    else
    {
      highstate_logfile_.imbue(std::locale::classic());
    }

    if (!footpos_velocity_logfile_.is_open())
    {
      std::cerr << "[ERROR] Failed to open Foot_log.txt"
                << std::endl;
    }
    else
    {
      footpos_velocity_logfile_.imbue(
          std::locale::classic());
    }
  }

  ~Custom()
  {


    if (highstate_logfile_.is_open())
    {
      highstate_logfile_.close();
    }

    if (footpos_velocity_logfile_.is_open())
    {
      footpos_velocity_logfile_.close();
    }

    if (command_event_logfile_.is_open())
    {
      command_event_logfile_.close();
    }
  }

  bool GetInitState()
  {
    // Wait until the first robot state message arrives, so we know the robot is online.
    if (!WaitForHighState(5.0))
    {
      std::cerr
          << "[ERROR] No rt/sportmodestate message was received "
             "within five seconds. No motion command will be sent."
          << std::endl;
      return false;
    }

    // Take a local copy of the latest received robot state in a thread-safe way.
    // The high-state callback may update `state_` from another thread,
    // so we protect access with `state_mutex_` to avoid data races.
    unitree_go::msg::dds_::SportModeState_ state_snapshot;

    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      state_snapshot = state_;
    }

    // Save the initial pose from the first valid state message.
    px0_ = state_snapshot.position()[0];
    py0_ = state_snapshot.position()[1];
    yaw0_ = state_snapshot.imu_state().rpy()[2];

    std::cout << "Initial position: x0=" << px0_
              << ", y0=" << py0_
              << ", yaw0=" << yaw0_
              << std::endl;

    // Retained from the user's original navigation code.
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

  
  sport_client_.Move(vx, vy, vyaw);

  // Captured immediately after Move() returns.
  const auto move_call_return_steady_time =
      SteadyClock::now();

  const auto move_call_return_wall_time =
      WallClock::now();

  const double move_call_duration_ms =
      std::chrono::duration<double, std::milli>(
          move_call_return_steady_time
          - move_call_start_steady_time)
          .count();
  

  std::cout
      << "[MOVE] Move() API statrt steady time: "
      << std::fixed
      << std::setprecision(6)
      << move_call_start_steady_time
      << " ms"
      << std::endl;


  
  std::cout
      << "[MOVE] Move() API stop steady time: "
      << std::fixed
      << std::setprecision(6)
      << move_call_return_steady_time
      << " ms"
      << std::endl;





    


  WriteCommandEvent(
      "MOVE_CALL_START",
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

  std::cout
      << "[MOVE] Move() API-call duration: "
      << std::fixed
      << std::setprecision(6)
      << move_call_duration_ms
      << " ms"
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

      std::cout
          << "[TIMESTAMP] "
          << event_name
          << " at "
          << SystemTimeToIso(time_point)
          << std::endl;
    }

  bool WaitForHighState(double timeout_seconds)
  {
    using Clock = std::chrono::steady_clock;

    const auto deadline =
        Clock::now()
        + std::chrono::duration<double>(timeout_seconds);

    while (!highstate_received_.load()
           && !g_stop_requested.load())
    {
      if (Clock::now() >= deadline)
      {
        return false;
      }

      std::this_thread::sleep_for(
          std::chrono::milliseconds(20));
    }

    return highstate_received_.load();
  }

  static std::string CurrentTimestamp()
  {
    const auto now =
        std::chrono::system_clock::now();

    const std::time_t now_time =
        std::chrono::system_clock::to_time_t(now);

    std::tm local_tm{};
    localtime_r(&now_time, &local_tm);

    std::ostringstream stream;
    stream << "["
           << std::put_time(
                  &local_tm,
                  "%Y-%m-%d %H:%M:%S")
           << "]";

    return stream.str();
  }

  void HighStateHandler(const void *message)
  {
    // This callback receives periodic robot state updates.
    if (message == nullptr)
    {
      return;
    }

    const auto state_snapshot =
        *static_cast<
            const unitree_go::msg::dds_::SportModeState_ *>(
            message);

    {
      // Save the latest high-state in a thread-safe way.
      std::lock_guard<std::mutex> lock(state_mutex_);
      state_ = state_snapshot;
    }

    highstate_received_.store(true);

    const std::string timestamp =
        CurrentTimestamp();

    if (highstate_logfile_.is_open())
    {
      highstate_logfile_
          << timestamp << " Position: "
          << state_snapshot.position()[0] << ", "
          << state_snapshot.position()[1] << ", "
          << state_snapshot.position()[2]
          << " | IMU RPY: "
          << state_snapshot.imu_state().rpy()[0] << ", "
          << state_snapshot.imu_state().rpy()[1] << ", "
          << state_snapshot.imu_state().rpy()[2]
          << " | Angular Vel: "
          << state_snapshot.imu_state().gyroscope()[0] << ", "
          << state_snapshot.imu_state().gyroscope()[1] << ", "
          << state_snapshot.imu_state().gyroscope()[2]
          << " | Acceleration: "
          << state_snapshot.imu_state().accelerometer()[0] << ", "
          << state_snapshot.imu_state().accelerometer()[1] << ", "
          << state_snapshot.imu_state().accelerometer()[2]
          << " | Quaternion: "
          << state_snapshot.imu_state().quaternion()[0] << ","
          << state_snapshot.imu_state().quaternion()[1] << ","
          << state_snapshot.imu_state().quaternion()[2] << ","
          << state_snapshot.imu_state().quaternion()[3]
          << " | Mode: "
          << static_cast<int>(state_snapshot.mode())
          << '\n'
          << std::flush;
    }

    if (footpos_velocity_logfile_.is_open())
    {
      footpos_velocity_logfile_
          << timestamp
          << " Velocity:"
          << " Velocity along x-axis:"
          << state_snapshot.velocity()[0] << ","
          << " Velocity along y-axis:"
          << state_snapshot.velocity()[1] << ","
          << " Velocity along z-axis:"
          << state_snapshot.velocity()[2] << ","
          << " FR(x,y,z)="
          << state_snapshot.foot_position_body()[0] << ","
          << state_snapshot.foot_position_body()[1] << ","
          << state_snapshot.foot_position_body()[2] << ","
          << " FL(x,y,z)="
          << state_snapshot.foot_position_body()[3] << ","
          << state_snapshot.foot_position_body()[4] << ","
          << state_snapshot.foot_position_body()[5] << ","
          << " RR(x,y,z)="
          << state_snapshot.foot_position_body()[6] << ","
          << state_snapshot.foot_position_body()[7] << ","
          << state_snapshot.foot_position_body()[8] << ","
          << " RL(x,y,z)="
          << state_snapshot.foot_position_body()[9] << ","
          << state_snapshot.foot_position_body()[10] << ","
          << state_snapshot.foot_position_body()[11]
          << '\n'
          << std::flush;
    }
  }

  unitree::robot::go2::SportClient sport_client_;

  unitree::robot::ChannelSubscriberPtr<
    unitree_go::msg::dds_::SportModeState_>
    highstate_subscriber_;

  unitree_go::msg::dds_::SportModeState_ state_;
  std::mutex state_mutex_;

  std::atomic<bool> highstate_received_{false};
  std::atomic<bool> stop_sequence_started_{false};

  bool motion_ready_ = false;

  bool movement_command_started_ = false;

  float active_vx_ = 0.0f;
  float active_vy_ = 0.0f;
  float active_vyaw_ = 0.0f;
  

  double px0_ = 0.0;
  double py0_ = 0.0;
  double yaw0_ = 0.0;

  std::ofstream highstate_logfile_;
  std::ofstream footpos_velocity_logfile_;
  std::ofstream command_event_logfile_;
};

int main(int argc, char **argv)
{
  if (argc < 2)
  {
    std::cerr
        << "Usage: " << argv[0]
        << " networkInterface"
        << std::endl;
    return 1;
  }

  std::locale::global(std::locale::classic());
  std::cout.imbue(std::locale::classic());
  std::cerr.imbue(std::locale::classic());

  std::signal(SIGINT, SignalHandler);
  std::signal(SIGTERM, SignalHandler);

  unitree::robot::ChannelFactory::Instance()->Init(
      0,
      argv[1]);

  Custom custom;

  if (!custom.GetInitState())
  {
    return 1;
  }

  constexpr float kVx = 0.2f;
  constexpr float kVy = 0.0f;
  constexpr float kVyaw = 0.0f;
  custom.RunSingleMove(
      kVx,
      kVy,
      kVyaw);

  std::cout
      << "Program finished after the explicit stop sequence."
      << std::endl;

  return 0;
}