/**********************************************************************
 Navigation-only Unitree Go2 movement and state logger.

 - Uses SportClient only.
 - No ObstaclesAvoidClient.
 - Commands vx = 0.2 m/s for one real second.
 - Sends zero velocity and StopMove() when the duration is reached.
 - Ctrl+C requests the same explicit stop sequence.
***********************************************************************/

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <sstream>
#include <thread>
#include <filesystem>
#include <string>

#include <unitree/robot/go2/sport/sport_client.hpp>
#include <unitree/robot/channel/channel_factory.hpp>



std::atomic<bool> g_stop_requested{false};

void SignalHandler(int)
{
  // Do not call robot APIs inside the signal handler.
  // The normal control flow will notice this flag and stop the robot.
  g_stop_requested.store(true);
}

// Converts a floating-point velocity value to a string with three decimal places,
// removing trailing zeros and ensuring a minimum of one decimal place.--##
std::string VelocityToken(double value)
{
    std::ostringstream stream;

    stream
        << std::fixed
        << std::setprecision(3)
        << value;

    std::string text = stream.str();
    //-- back() = “look at the last character”
    // --pop_back() = “delete the last character”
    while (
        text.size() > 1
        && text.back() == '0')
    {
        text.pop_back();
    }

    if (!text.empty() && text.back() == '.')
    {
        text += '0';
    }

    if (text == "-0.0")
    {
        text = "0.0";
    }

    return text;
}


class Custom
{
public:
  Custom(const std::string &command_event_file)
  {
    // Configure the sport client and initialize its communication.
    sport_client_.SetTimeout(1.0f);
    sport_client_.Init();

    //--- saving the movement command window data to a CSV file. ---//
    command_event_logfile_.open(
        command_event_file,
        std::ios::out | std::ios::trunc);

    if (!command_event_logfile_.is_open())
    {
      std::cerr
          << "[ERROR] Failed to open command timestamp file: "
          << command_event_file
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
    // Idempotent fallback in case normal control flow exits early.
    StopRobot();

    if (command_event_logfile_.is_open())
    {
      command_event_logfile_.close();
    }
  }

  bool GetInitState()
  {
    std::cout
        << "[INIT] Robot initialized; preparing to send movement commands."
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
  
  //---Check this Function for the RunForwardFor() function, which commands the robot to move forward for a specified duration.---//  
  void RunForwardFor(
      double duration_seconds,
      float vx,
      float vy,
      float vyaw)
  {
    // Confirm that initialization succeeded before sending motion commands.
    if (!motion_ready_)
    {
      std::cerr
          << "[ERROR] Motion mode is not ready. "
             "The robot will not be commanded."
          << std::endl;
      StopRobot();
      return;
    }

    if (duration_seconds <= 0.0)
    {
      std::cerr
          << "[ERROR] duration_seconds must be positive."
          << std::endl;
      StopRobot();
      return;
    }

    if (!command_event_logfile_.is_open())
    {
      std::cerr
          << "[ERROR] Command timestamp file is unavailable. "
            "The robot will not be commanded."
          << std::endl;

      StopRobot();
      return;
    }

  

    // Log the motion request before entering the control loop.
    std::cout
        << "[MOVE] Commanding SportClient::Move("
        << vx << ", " << vy << ", " << vyaw
        << ") for " << duration_seconds
        << " real seconds."
        << std::endl;



    using Clock = std::chrono::steady_clock;

    const auto start_time = Clock::now();
    auto next_status_time = start_time;
    bool first_move_command = true;

   

    while (!g_stop_requested.load())
    {
      const auto now = Clock::now();

      const double elapsed_seconds =
          std::chrono::duration<double>(
              now - start_time)
              .count();

      // Stop once the requested movement duration has elapsed.
      if (elapsed_seconds >= duration_seconds)
      {
        break;
      }

      std::chrono::system_clock::time_point
        first_move_command_timestamp;

      if (first_move_command)
      {
        // Captured immediately before the first nonzero Move() request.
        first_move_command_timestamp =
            std::chrono::system_clock::now();
      }

      // Send a velocity command to the robot at 50 Hz.
      const int32_t move_ret =
          sport_client_.Move(vx, vy, vyaw);

      if (move_ret != 0)
      {
        std::cerr
            << "[ERROR] SportClient::Move() failed, return="
            << move_ret
            << ". Executing the stop sequence."
            << std::endl;

        g_stop_requested.store(true);
        break;
      }
      
      

      if (first_move_command)
      {
        movement_command_started_ = true;

        active_vx_ = vx;
        active_vy_ = vy;
        active_vyaw_ = vyaw;

        // The timestamp was captured immediately before Move().
        // The row is written only after Move() returned successfully.
        WriteCommandEvent(
            "MOVE_START",
            first_move_command_timestamp,
            vx,
            vy,
            vyaw);

        first_move_command = false;
      }




      if (now >= next_status_time)
      {
          std::cout
              << "[MOVE] Current motion time: "
              << std::fixed
              << std::setprecision(3)
              << elapsed_seconds
              << " s / "
              << duration_seconds
              << " s"
              << std::endl;

          next_status_time =
              now + std::chrono::milliseconds(100);
      }

      // Send high-level velocity commands at 50 Hz.
      std::this_thread::sleep_for(
          std::chrono::milliseconds(20));
    }

    if (g_stop_requested.load())
    {
      std::cout
          << "[MOVE] Interrupt or communication error requested a stop."
          << std::endl;
    }
    else
    {
      std::cout
          << "[MOVE] Requested duration reached."
          << std::endl;
    }

    StopRobot();
  }

  void StopRobot()
  {
    // Ensure that only one path executes the stop sequence.
    if (stop_sequence_started_.exchange(true))
    {
      return;
    }

    // If we previously started moving, record the stop event.
    if (movement_command_started_)
    {
      // Captured before the first zero-velocity command.
      const auto stop_command_timestamp =
          std::chrono::system_clock::now();

      WriteCommandEvent(
          "MOVE_STOP",
          stop_command_timestamp,
          0.0f,
          0.0f,
          0.0f);

      movement_command_started_ = false;
    }


    std::cout
        << "[STOP] Sending zero velocity through SportClient..."
        << std::endl;

    // Send zero velocity more than once to improve robustness
    // over the Wi-Fi connection.
    for (int attempt = 1; attempt <= 3; ++attempt)
    {
      const int32_t zero_ret =
          sport_client_.Move(0.0f, 0.0f, 0.0f);

      std::cout
          << "[STOP] SportClient::Move(0,0,0) "
          << attempt << "/3, return="
          << zero_ret << std::endl;

      std::this_thread::sleep_for(
          std::chrono::milliseconds(20));
    }
    // const int32_t stop_ret =
    //     sport_client_.StopMove();

    // std::cout
    //     << "[STOP] SportClient::StopMove(), return="
    //     << stop_ret << std::endl;

    motion_ready_ = false;
    std::cout
        << "[STOP] Stop sequence completed."
        << std::endl;
  }

private:

  static int64_t SystemTimeToNanoseconds(
    const std::chrono::system_clock::time_point &time_point)
    {
      return std::chrono::duration_cast<std::chrono::nanoseconds>(
                time_point.time_since_epoch())
          .count();
    }

  static std::string SystemTimeToIso(
      const std::chrono::system_clock::time_point &time_point)
      {
        const std::time_t time_value =
            std::chrono::system_clock::to_time_t(time_point);

        std::tm local_time{};
        localtime_r(&time_value, &local_time);

        const int64_t milliseconds_since_epoch =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                time_point.time_since_epoch())
                .count();

        const int milliseconds_part =
            static_cast<int>(milliseconds_since_epoch % 1000);

        std::ostringstream stream;

        stream << std::put_time(
                      &local_time,
                      "%Y-%m-%dT%H:%M:%S")
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

  unitree::robot::go2::SportClient sport_client_;

  std::atomic<bool> stop_sequence_started_{false};

  bool motion_ready_ = false;

  bool movement_command_started_ = false;

  float active_vx_ = 0.0f;
  float active_vy_ = 0.0f;
  float active_vyaw_ = 0.0f;

  std::ofstream command_event_logfile_;
};

int main(int argc, char **argv)
{
  if (argc < 7)
  {
      std::cerr
          << "Usage: "
          << argv[0]
          << " networkInterface"
          << " trialNumber"
          << " vx"
          << " vy"
          << " vyaw"
          << " durationSeconds"
          << std::endl;

      return 1;
  }

 

  std::locale::global(std::locale::classic());
  std::cout.imbue(std::locale::classic());
  std::cerr.imbue(std::locale::classic());

  std::signal(SIGINT, SignalHandler);
  std::signal(SIGTERM, SignalHandler);

 // Validate and parse command-line arguments.
  int trial_number;
  float vx;
  float vy;
  float vyaw;
  double duration_seconds;

  try
  {
      trial_number =
          std::stoi(argv[2]);

      vx =
          std::stof(argv[3]);

      vy =
          std::stof(argv[4]);

      vyaw =
          std::stof(argv[5]);

      duration_seconds =
          std::stod(argv[6]);
  }
  catch (const std::exception &error)
  {
      std::cerr
          << "[ERROR] Invalid command-line argument: "
          << error.what()
          << std::endl;

      return 1;
  }

   // Parse command-line arguments and validate them.//
  if (
    !std::isfinite(vx)
    || !std::isfinite(vy)
    || !std::isfinite(vyaw)
    || !std::isfinite(duration_seconds))
  {
      std::cerr
          << "[ERROR] vx, vy, vyaw, and durationSeconds "
          << "must all be finite numbers."
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

  if (duration_seconds <= 0.0)
  {
      std::cerr
          << "[ERROR] durationSeconds must be positive."
          << std::endl;

      return 1;
  }

  // Construct the output directory path based on the trial number. 
  // and velcoity parameters.
  const std::string vx_token =
    VelocityToken(vx);

  const std::string vy_token =
      VelocityToken(vy);

  const std::string vyaw_token =
      VelocityToken(vyaw);

  const std::string velocity_suffix =
      vx_token
      + "_"
      + vy_token
      + "_"
      + vyaw_token;

  const std::string dataset_root =
    "/home/unitree-arka/"
    "Go2_Walk_Base_Data_Sensor";

  const std::string trials_directory =
      dataset_root
      + "/Trials_"
      + velocity_suffix;

  const std::string trial_directory =
      trials_directory
      + "/Trial_"
      + std::to_string(trial_number)
      + "_"
      + velocity_suffix;
  try
  {
      std::filesystem::create_directories(
          trial_directory);
  }
  catch (const std::filesystem::filesystem_error &error)
  {
      std::cerr
          << "[ERROR] Could not create trial directory: "
          << error.what()
          << std::endl;

      return 1;
  }

  const std::string command_event_file =
    trial_directory
    + "/cmd_w_"
    + velocity_suffix
    + ".csv";
    

  unitree::robot::ChannelFactory::Instance()->Init(
      0,
      argv[1]);

  Custom custom(command_event_file);

  if (!custom.GetInitState())
  {
    custom.StopRobot();
    return 1;
  }

    std::cout
    << "[TRIAL] Trial number: "
    << trial_number
    << std::endl;

  std::cout
      << "[TRIAL] Command: ("
      << vx << ", "
      << vy << ", "
      << vyaw << ")"
      << std::endl;

  std::cout
      << "[TRIAL] Command duration: "
      << duration_seconds
      << " s"
      << std::endl;

  std::cout
      << "[TRIAL] Output directory: "
      << trial_directory
      << std::endl;

  std::cout
      << "[TRIAL] Command CSV: "
      << command_event_file
      << std::endl;

  custom.RunForwardFor(
      duration_seconds,
      vx,
      vy,
      vyaw);


  return 0;
}