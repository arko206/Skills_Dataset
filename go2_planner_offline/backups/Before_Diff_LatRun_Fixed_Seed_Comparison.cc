#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include <sys/wait.h>

namespace fs = std::filesystem;

struct Waypoint
{
    double x = 0.0;
    double y = 0.0;
    double theta = 0.0;
};

struct ObstaclePoint
{
    double x = 0.0;
    double y = 0.0;
};

struct EnvironmentData
{
    double goal_x = 0.0;
    double goal_y = 0.0;
    std::map<int, ObstaclePoint> obstacles;
};

struct TrialResult
{
    bool success = false;
    double path_length_m = std::numeric_limits<double>::quiet_NaN();
    double wall_runtime_s = 0.0;
    int process_exit_code = -1;

    std::string search_termination = "not_reported";
    std::string primary_failure_reason = "not_reported";

    long duplicate_nodes_found = -1;
    long candidate_nodes_collision_checked = -1;
    long collision_check_calls = -1;
    long collision_rejections = -1;
    long endpoint_collision_rejections = -1;
    long rotation_collision_rejections = -1;
    long segment_collision_rejections = -1;
    long angular_no_progress_rejections = -1;
    long positional_no_progress_rejections = -1;
    long local_target_failures = -1;
    long observation_failures = -1;
    long inference_failures = -1;
    long diffusion_action_requests = -1;
    long accepted_rollout_nodes = -1;
    long successful_extensions = -1;
    long failed_extensions = -1;
    long sample_reached_stops = -1;
    long tree_nodes_generated = -1;
};

static void trim(std::string &text)
{
    const std::size_t first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
    {
        text.clear();
        return;
    }

    const std::size_t last = text.find_last_not_of(" \t\r\n");
    text = text.substr(first, last - first + 1);
}

static std::string shell_quote(const std::string &text)
{
    std::string quoted = "'";
    for (char c : text)
    {
        if (c == '\'')
            quoted += "'\\''";
        else
            quoted += c;
    }
    quoted += "'";
    return quoted;
}

static bool is_nonempty_regular_file(const fs::path &path)
{
    std::error_code ec;
    return fs::exists(path, ec) &&
           fs::is_regular_file(path, ec) &&
           fs::file_size(path, ec) > 0;
}

static std::vector<Waypoint> read_waypoints(const fs::path &path)
{
    std::vector<Waypoint> waypoints;
    std::ifstream input(path);
    if (!input)
        return waypoints;

    std::string line;
    while (std::getline(input, line))
    {
        for (char &c : line)
        {
            if (c == ',')
                c = ' ';
        }

        std::istringstream parser(line);
        Waypoint waypoint;
        if (parser >> waypoint.x >> waypoint.y >> waypoint.theta)
            waypoints.push_back(waypoint);
    }

    return waypoints;
}

static double compute_planar_path_length(const std::vector<Waypoint> &waypoints)
{
    double length = 0.0;
    for (std::size_t i = 1; i < waypoints.size(); ++i)
    {
        const double dx = waypoints[i].x - waypoints[i - 1].x;
        const double dy = waypoints[i].y - waypoints[i - 1].y;
        length += std::sqrt(dx * dx + dy * dy);
    }
    return length;
}

static bool copy_file_if_present(const fs::path &source,
                                 const fs::path &destination)
{
    if (!is_nonempty_regular_file(source))
    {
        std::cerr << "WARNING: output file was not produced: "
                  << source << "\n";
        return false;
    }

    std::error_code ec;
    fs::copy_file(source,
                  destination,
                  fs::copy_options::overwrite_existing,
                  ec);
    if (ec)
    {
        std::cerr << "ERROR: cannot copy " << source
                  << " to " << destination
                  << ": " << ec.message() << "\n";
        return false;
    }

    return true;
}

static bool parse_environment_config(const fs::path &config_path,
                                     EnvironmentData &environment,
                                     std::string *query_file_name)
{
    std::ifstream input(config_path);
    if (!input)
    {
        std::cerr << "ERROR: cannot open config file "
                  << config_path << "\n";
        return false;
    }

    std::string line;
    while (std::getline(input, line))
    {
        const std::size_t comment = line.find('#');
        if (comment != std::string::npos)
            line.erase(comment);

        const std::size_t equals = line.find('=');
        if (equals == std::string::npos)
            continue;

        std::string key = line.substr(0, equals);
        std::string value = line.substr(equals + 1);
        trim(key);
        trim(value);

        if (key.empty() || value.empty())
            continue;

        try
        {
            if (key == "goal_x")
            {
                environment.goal_x = std::stod(value);
            }
            else if (key == "goal_y")
            {
                environment.goal_y = std::stod(value);
            }
            else if (key == "query_file" && query_file_name != nullptr)
            {
                *query_file_name = value;
            }
            else if (key.rfind("obs.", 0) == 0)
            {
                const std::size_t second_dot = key.find('.', 4);
                if (second_dot == std::string::npos)
                    continue;

                const int obstacle_index =
                    std::stoi(key.substr(4, second_dot - 4));
                const std::string field = key.substr(second_dot + 1);

                if (field == "x")
                    environment.obstacles[obstacle_index].x =
                        std::stod(value);
                else if (field == "y")
                    environment.obstacles[obstacle_index].y =
                        std::stod(value);
            }
        }
        catch (const std::exception &error)
        {
            std::cerr << "ERROR: config parse failure for key '"
                      << key << "' in " << config_path
                      << ": " << error.what() << "\n";
            return false;
        }
    }

    return true;
}

static bool load_environment(const fs::path &planner_directory,
                             const fs::path &planner_config,
                             EnvironmentData &environment,
                             fs::path *resolved_query_path)
{
    std::string query_file_name = "query.cfg";

    if (!parse_environment_config(planner_config,
                                  environment,
                                  &query_file_name))
    {
        return false;
    }

    fs::path query_path(query_file_name);
    if (query_path.is_relative())
        query_path = planner_directory / query_path;

    if (!parse_environment_config(query_path,
                                  environment,
                                  nullptr))
    {
        return false;
    }

    if (resolved_query_path != nullptr)
        *resolved_query_path = query_path;

    if (environment.obstacles.empty())
    {
        std::cerr << "WARNING: no obstacle centres were read from "
                  << query_path << "\n";
    }

    return true;
}

static bool write_local_obstacle_file(
    const fs::path &destination,
    const std::vector<Waypoint> &waypoints,
    const EnvironmentData &environment)
{
    if (waypoints.size() < 2)
        return false;

    std::ofstream output(destination, std::ios::trunc);
    if (!output)
    {
        std::cerr << "ERROR: cannot write " << destination << "\n";
        return false;
    }

    output << std::fixed << std::setprecision(6);

    for (std::size_t i = 0; i + 1 < waypoints.size(); ++i)
    {
        const Waypoint &pose = waypoints[i];
        const double cos_theta = std::cos(pose.theta);
        const double sin_theta = std::sin(pose.theta);

        output << "[";
        std::size_t obstacle_counter = 0;

        for (const auto &entry : environment.obstacles)
        {
            const ObstaclePoint &obstacle = entry.second;
            const double dx = obstacle.x - pose.x;
            const double dy = obstacle.y - pose.y;

            const double relative_x =
                dx * cos_theta + dy * sin_theta;
            const double relative_y =
                -dx * sin_theta + dy * cos_theta;
            const double distance =
                std::sqrt(relative_x * relative_x +
                          relative_y * relative_y);

            output << "["
                   << relative_x << ", "
                   << relative_y << ", "
                   << distance << "]";

            ++obstacle_counter;
            if (obstacle_counter < environment.obstacles.size())
                output << ", ";
        }

        output << "]\n";
    }

    return true;
}

static bool write_local_goal_file(
    const fs::path &destination,
    const std::vector<Waypoint> &waypoints,
    const EnvironmentData &environment)
{
    if (waypoints.size() < 2)
        return false;

    std::ofstream output(destination, std::ios::trunc);
    if (!output)
    {
        std::cerr << "ERROR: cannot write " << destination << "\n";
        return false;
    }

    output << std::fixed << std::setprecision(6);

    for (std::size_t i = 0; i + 1 < waypoints.size(); ++i)
    {
        const Waypoint &pose = waypoints[i];
        const double dx = environment.goal_x - pose.x;
        const double dy = environment.goal_y - pose.y;

        const double cos_theta = std::cos(pose.theta);
        const double sin_theta = std::sin(pose.theta);

        const double relative_x =
            dx * cos_theta + dy * sin_theta;
        const double relative_y =
            -dx * sin_theta + dy * cos_theta;
        const double distance =
            std::sqrt(relative_x * relative_x +
                      relative_y * relative_y);

        output << relative_x << ", "
               << relative_y << ", "
               << distance << "\n";
    }

    return true;
}


static bool parse_long_value(const std::string &text, long &value)
{
    try
    {
        std::size_t consumed = 0;
        const long parsed = std::stol(text, &consumed);
        if (consumed != text.size())
            return false;
        value = parsed;
        return true;
    }
    catch (const std::exception &)
    {
        return false;
    }
}

static void parse_planner_diagnostics(
    const fs::path &log_file,
    TrialResult &result)
{
    std::ifstream input(log_file);
    if (!input)
        return;

    std::string line;
    while (std::getline(input, line))
    {
        if (line.rfind("DIAG_", 0) != 0)
            continue;

        const std::size_t equals = line.find('=');
        if (equals == std::string::npos)
            continue;

        const std::string key =
            line.substr(5, equals - 5);
        const std::string value =
            line.substr(equals + 1);

        if (key == "search_termination")
            result.search_termination = value;
        else if (key == "primary_failure_reason")
            result.primary_failure_reason = value;
        else if (key == "duplicate_nodes_found")
            parse_long_value(value, result.duplicate_nodes_found);
        else if (key == "candidate_nodes_collision_checked")
            parse_long_value(
                value,
                result.candidate_nodes_collision_checked);
        else if (key == "collision_check_calls")
            parse_long_value(value, result.collision_check_calls);
        else if (key == "collision_rejections")
            parse_long_value(value, result.collision_rejections);
        else if (key == "endpoint_collision_rejections")
            parse_long_value(
                value,
                result.endpoint_collision_rejections);
        else if (key == "rotation_collision_rejections")
            parse_long_value(
                value,
                result.rotation_collision_rejections);
        else if (key == "segment_collision_rejections")
            parse_long_value(
                value,
                result.segment_collision_rejections);
        else if (key == "angular_no_progress_rejections")
            parse_long_value(
                value,
                result.angular_no_progress_rejections);
        else if (key == "positional_no_progress_rejections")
            parse_long_value(
                value,
                result.positional_no_progress_rejections);
        else if (key == "local_target_failures")
            parse_long_value(value, result.local_target_failures);
        else if (key == "observation_failures")
            parse_long_value(value, result.observation_failures);
        else if (key == "inference_failures")
            parse_long_value(value, result.inference_failures);
        else if (key == "diffusion_action_requests")
            parse_long_value(
                value,
                result.diffusion_action_requests);
        else if (key == "accepted_rollout_nodes")
            parse_long_value(
                value,
                result.accepted_rollout_nodes);
        else if (key == "successful_extensions")
            parse_long_value(value, result.successful_extensions);
        else if (key == "failed_extensions")
            parse_long_value(value, result.failed_extensions);
        else if (key == "sample_reached_stops")
            parse_long_value(value, result.sample_reached_stops);
        else if (key == "tree_nodes_generated")
            parse_long_value(value, result.tree_nodes_generated);
    }
}

static bool compile_source(const fs::path &working_directory,
                           const fs::path &source,
                           const fs::path &executable)
{
    const std::string command =
        "cd " + shell_quote(working_directory.string()) +
        " && g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic " +
        shell_quote(source.filename().string()) +
        " -o " + shell_quote(executable.filename().string());

    std::cout << "Compiling " << source.filename() << " ...\n";
    const int result = std::system(command.c_str());

    if (result == -1 || !WIFEXITED(result) ||
        WEXITSTATUS(result) != 0)
    {
        std::cerr << "ERROR: compilation failed for "
                  << source << "\n";
        return false;
    }

    return true;
}

static TrialResult run_one_planner(
    const fs::path &planner_directory,
    const fs::path &executable,
    const fs::path &config_file,
    unsigned int seed,
    int run_idx,
    const fs::path &waypoint_file,
    const fs::path &log_file,
    const std::vector<fs::path> &files_to_remove)
{
    TrialResult result;

    std::error_code ec;
    for (const fs::path &file : files_to_remove)
    {
        fs::remove(file, ec);
        ec.clear();
    }

    fs::remove(log_file, ec);

    const std::string command =
        "cd " + shell_quote(planner_directory.string()) +
        " && " + shell_quote(executable.string()) +
        " --config " + shell_quote(config_file.filename().string()) +
        " --run_idx " + std::to_string(run_idx) +
        " --seed " + std::to_string(seed) +
        " > " + shell_quote(log_file.string()) + " 2>&1";

    const auto start = std::chrono::steady_clock::now();
    const int system_result = std::system(command.c_str());
    const auto finish = std::chrono::steady_clock::now();

    result.wall_runtime_s =
        std::chrono::duration<double>(finish - start).count();

    if (system_result != -1 && WIFEXITED(system_result))
        result.process_exit_code = WEXITSTATUS(system_result);

    // Read the machine-readable DIAG_* lines written by the planner.
    // This is done before the waypoint check so failed trials still
    // retain their diagnostic information.
    parse_planner_diagnostics(log_file, result);

    if (!is_nonempty_regular_file(waypoint_file))
        return result;

    const std::vector<Waypoint> waypoints =
        read_waypoints(waypoint_file);
    if (waypoints.size() < 2)
        return result;

    result.success = true;
    result.path_length_m =
        compute_planar_path_length(waypoints);
    return result;
}

static bool archive_trial_outputs(
    const fs::path &trial_directory,
    const std::string &file_suffix,
    const fs::path &source_controls,
    const fs::path &source_waypoints,
    const TrialResult &result,
    unsigned int seed,
    int run_idx,
    const EnvironmentData &environment)
{
    try
    {
        fs::create_directories(trial_directory);
    }
    catch (const fs::filesystem_error &error)
    {
        std::cerr << "ERROR: cannot create " << trial_directory
                  << ": " << error.what() << "\n";
        return false;
    }

    const fs::path archived_controls =
        trial_directory /
        ("controls_" + std::to_string(run_idx) +
         "_" + file_suffix + ".txt");

    const fs::path archived_waypoints =
        trial_directory /
        ("se2_waypoints_" + std::to_string(run_idx) +
         "_" + file_suffix + ".txt");

    const fs::path archived_obstacles =
        trial_directory /
        ("local_obs_" + std::to_string(run_idx) +
         "_" + file_suffix + ".txt");

    const fs::path archived_goal =
        trial_directory /
        ("local_goal_" + std::to_string(run_idx) +
         "_" + file_suffix + ".txt");

    const fs::path status_file =
        trial_directory /
        ("status_" + std::to_string(run_idx) +
         "_" + file_suffix + ".txt");

    std::ofstream status(status_file, std::ios::trunc);
    if (!status)
    {
        std::cerr << "ERROR: cannot write " << status_file << "\n";
        return false;
    }

    status << "run_idx=" << run_idx << "\n"
           << "seed=" << seed << "\n"
           << "success=" << (result.success ? 1 : 0) << "\n"
           << "process_exit_code=" << result.process_exit_code << "\n"
           << std::fixed << std::setprecision(6)
           << "path_length_m=";
    if (result.success)
        status << result.path_length_m;
    else
        status << "nan";
    status << "\nwall_runtime_s=" << result.wall_runtime_s << "\n"
           << "search_termination="
           << result.search_termination << "\n"
           << "primary_failure_reason="
           << result.primary_failure_reason << "\n"
           << "duplicate_nodes_found="
           << result.duplicate_nodes_found << "\n"
           << "candidate_nodes_collision_checked="
           << result.candidate_nodes_collision_checked << "\n"
           << "collision_check_calls="
           << result.collision_check_calls << "\n"
           << "collision_rejections="
           << result.collision_rejections << "\n"
           << "endpoint_collision_rejections="
           << result.endpoint_collision_rejections << "\n"
           << "rotation_collision_rejections="
           << result.rotation_collision_rejections << "\n"
           << "segment_collision_rejections="
           << result.segment_collision_rejections << "\n"
           << "angular_no_progress_rejections="
           << result.angular_no_progress_rejections << "\n"
           << "positional_no_progress_rejections="
           << result.positional_no_progress_rejections << "\n"
           << "local_target_failures="
           << result.local_target_failures << "\n"
           << "observation_failures="
           << result.observation_failures << "\n"
           << "inference_failures="
           << result.inference_failures << "\n"
           << "diffusion_action_requests="
           << result.diffusion_action_requests << "\n"
           << "accepted_rollout_nodes="
           << result.accepted_rollout_nodes << "\n"
           << "successful_extensions="
           << result.successful_extensions << "\n"
           << "failed_extensions="
           << result.failed_extensions << "\n"
           << "sample_reached_stops="
           << result.sample_reached_stops << "\n"
           << "tree_nodes_generated="
           << result.tree_nodes_generated << "\n";
    status.close();

    if (!result.success)
        return true;

    const bool controls_copied =
        copy_file_if_present(source_controls, archived_controls);
    const bool waypoints_copied =
        copy_file_if_present(source_waypoints, archived_waypoints);

    const std::vector<Waypoint> waypoints =
        read_waypoints(source_waypoints);

    const bool obstacles_written =
        write_local_obstacle_file(archived_obstacles,
                                  waypoints,
                                  environment);
    const bool goal_written =
        write_local_goal_file(archived_goal,
                              waypoints,
                              environment);

    return controls_copied &&
           waypoints_copied &&
           obstacles_written &&
           goal_written;
}

static void print_usage(const char *program_name)
{
    std::cout
        << "Usage:\n"
        << "  " << program_name
        << " --comparison_idx <positive_integer> [--overwrite]\n"
        << "  " << program_name
        << " <positive_integer> [--overwrite]\n\n"
        << "Example:\n"
        << "  " << program_name
        << " --comparison_idx 1\n";
}

int main(int argc, char **argv)
{
    int comparison_idx = -1;
    bool overwrite_existing = false;

    for (int argument = 1; argument < argc; ++argument)
    {
        const std::string option = argv[argument];

        if (option == "--comparison_idx" &&
            argument + 1 < argc)
        {
            comparison_idx = std::stoi(argv[++argument]);
        }
        else if (option == "--overwrite")
        {
            overwrite_existing = true;
        }
        else if (option == "-h" || option == "--help")
        {
            print_usage(argv[0]);
            return 0;
        }
        else if (!option.empty() && option.front() != '-' &&
                 comparison_idx < 0)
        {
            comparison_idx = std::stoi(option);
        }
        else
        {
            std::cerr << "ERROR: unknown or incomplete argument: "
                      << option << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (comparison_idx <= 0)
    {
        std::cerr
            << "ERROR: comparison_idx must be a positive integer.\n";
        print_usage(argv[0]);
        return 1;
    }

    // Ten paired fixed seeds.
    const std::array<unsigned int, 10> fixed_seeds = {
        101U, 211U, 307U, 401U, 503U,
        601U, 701U, 809U, 907U, 1009U};

    const char *home_environment = std::getenv("HOME");
    if (home_environment == nullptr)
    {
        std::cerr << "ERROR: HOME is not defined.\n";
        return 1;
    }

    const fs::path home(home_environment);
    const fs::path planner_directory =
        home / "go2_planner_offline";
    const fs::path planner_config =
        planner_directory / "planner.cfg";

    const fs::path classical_source =
        planner_directory / "planner_Single_Trial.cc";
    const fs::path diffusion_source =
        planner_directory /
        "Diffusion_Planner_Single_Trial.cc";

    const fs::path classical_executable =
        planner_directory / "planner_single_trial_exec";
    const fs::path diffusion_executable =
        planner_directory /
        "diffusion_planner_single_trial_exec";

    const fs::path dataset_root =
        home / "Navigation_Dataset";

    const std::string comparison_name =
        "Comparison_" + std::to_string(comparison_idx);

    const fs::path comparison_values_root =
        dataset_root / "Comparison_Values";
    const fs::path comparison_directory =
        comparison_values_root / comparison_name;

    const fs::path rrt_values_directory =
        comparison_directory /
        ("RRT_" + std::to_string(comparison_idx));
    const fs::path diffusion_values_directory =
        comparison_directory /
        ("Diffusion_RRT_" +
         std::to_string(comparison_idx));

    if (fs::exists(comparison_directory) &&
        !overwrite_existing)
    {
        std::cerr
            << "ERROR: " << comparison_directory
            << " already exists.\n"
            << "Use a new --comparison_idx, or add --overwrite "
            << "to replace this comparison.\n";
        return 1;
    }

    if (overwrite_existing)
    {
        std::error_code ec;
        fs::remove_all(comparison_directory, ec);
    }

    try
    {
        fs::create_directories(rrt_values_directory);
        fs::create_directories(diffusion_values_directory);
    }
    catch (const fs::filesystem_error &error)
    {
        std::cerr << "ERROR: cannot create result folders: "
                  << error.what() << "\n";
        return 1;
    }

    if (!fs::exists(classical_source) ||
        !fs::exists(diffusion_source) ||
        !fs::exists(planner_config))
    {
        std::cerr
            << "ERROR: planner source/config files are missing in "
            << planner_directory << "\n";
        return 1;
    }

    EnvironmentData environment;
    fs::path resolved_query_file;
    if (!load_environment(planner_directory,
                          planner_config,
                          environment,
                          &resolved_query_file))
    {
        return 1;
    }

    const fs::path archived_query_file =
        comparison_directory /
        ("query_" + std::to_string(comparison_idx) + ".cfg");

    if (!copy_file_if_present(resolved_query_file,
                              archived_query_file))
    {
        std::cerr
            << "ERROR: failed to archive the current query file.\n";
        return 1;
    }

    std::cout
        << "Archived query file: "
        << archived_query_file << "\n";

    if (!compile_source(planner_directory,
                        classical_source,
                        classical_executable) ||
        !compile_source(planner_directory,
                        diffusion_source,
                        diffusion_executable))
    {
        return 1;
    }

    const fs::path summary_file =
        comparison_directory /
        ("Fixed_Seed_Planner_Comparison_" +
         std::to_string(comparison_idx) + ".csv");

    std::ofstream summary(summary_file, std::ios::trunc);
    if (!summary)
    {
        std::cerr << "ERROR: cannot create "
                  << summary_file << "\n";
        return 1;
    }

    summary
        << "comparison_idx,trial,seed,"
        << "rrt_run_idx,rrt_success,rrt_path_length_m,"
        << "rrt_wall_runtime_s,diffusion_run_idx,"
        << "diffusion_success,diffusion_path_length_m,"
        << "diffusion_wall_runtime_s\n";

    int rrt_success_count = 0;
    int diffusion_success_count = 0;

    // Comparison 1 uses 3001-3010 and 4001-4010.
    // Comparison 2 uses 3011-3020 and 4011-4020, etc.
    const int run_offset = (comparison_idx - 1) * 10;

    for (std::size_t index = 0;
         index < fixed_seeds.size();
         ++index)
    {
        const int trial =
            static_cast<int>(index) + 1;
        const unsigned int seed = fixed_seeds[index];

        const int rrt_run_idx =
            3000 + run_offset + trial;
        const int diffusion_run_idx =
            4000 + run_offset + trial;

        const fs::path rrt_waypoint_file =
            dataset_root / "waypoints" /
            ("se2_waypoints_" +
             std::to_string(rrt_run_idx) + ".txt");
        const fs::path rrt_controls_file =
            dataset_root / "controls" /
            ("controls_" +
             std::to_string(rrt_run_idx) + ".txt");

        const fs::path diffusion_waypoint_file =
            dataset_root /
            "Diffusion_Policy_Navigation" /
            "Diffusion_waypoints" /
            ("Diffusion_se2_waypoints_" +
             std::to_string(diffusion_run_idx) + ".txt");
        const fs::path diffusion_controls_file =
            dataset_root /
            "Diffusion_Policy_Navigation" /
            "Diffusion_controls" /
            ("Diffusion_controls_" +
             std::to_string(diffusion_run_idx) + ".txt");

        // Keep each trial's log and archived outputs together inside
        // its corresponding Comparison_Values run directory.
        const fs::path rrt_trial_directory =
            rrt_values_directory /
            ("rrt_run_" +
             std::to_string(rrt_run_idx));
        const fs::path diffusion_trial_directory =
            diffusion_values_directory /
            ("diff_rrt_" +
             std::to_string(diffusion_run_idx));

        try
        {
            fs::create_directories(rrt_trial_directory);
            fs::create_directories(diffusion_trial_directory);
        }
        catch (const fs::filesystem_error &error)
        {
            std::cerr
                << "ERROR: cannot create trial directories: "
                << error.what() << "\n";
            return 1;
        }

        const fs::path rrt_log_file =
            rrt_trial_directory /
            ("planner_log_" +
             std::to_string(rrt_run_idx) +
             "_rrt.log");

        const fs::path diffusion_log_file =
            diffusion_trial_directory /
            ("planner_log_" +
             std::to_string(diffusion_run_idx) +
             "_diff_rrt.log");

        std::cout
            << "\n========================================\n"
            << "Comparison " << comparison_idx
            << ", trial " << trial
            << " / 10, seed = " << seed << "\n"
            << "========================================\n";

        std::cout << "Running classical RRT ...\n";
        const TrialResult rrt_result =
            run_one_planner(
                planner_directory,
                classical_executable,
                planner_config,
                seed,
                rrt_run_idx,
                rrt_waypoint_file,
                rrt_log_file,
                {rrt_waypoint_file, rrt_controls_file});

        std::cout
            << "Running diffusion-steering RRT ...\n";
        const TrialResult diffusion_result =
            run_one_planner(
                planner_directory,
                diffusion_executable,
                planner_config,
                seed,
                diffusion_run_idx,
                diffusion_waypoint_file,
                diffusion_log_file,
                {diffusion_waypoint_file,
                 diffusion_controls_file});

        if (rrt_result.success)
            ++rrt_success_count;
        if (diffusion_result.success)
            ++diffusion_success_count;

        if (!archive_trial_outputs(
                rrt_trial_directory,
                "rrt",
                rrt_controls_file,
                rrt_waypoint_file,
                rrt_result,
                seed,
                rrt_run_idx,
                environment) ||
            !archive_trial_outputs(
                diffusion_trial_directory,
                "diff_rrt",
                diffusion_controls_file,
                diffusion_waypoint_file,
                diffusion_result,
                seed,
                diffusion_run_idx,
                environment))
        {
            return 1;
        }

        // The single-trial planners first write indexed waypoint and
        // control files to their normal dataset folders. After those
        // files have been copied into Comparison_Values, remove only
        // the temporary indexed copies to avoid duplicate storage.
        {
            const std::array<fs::path, 4> temporary_outputs = {
                rrt_waypoint_file,
                rrt_controls_file,
                diffusion_waypoint_file,
                diffusion_controls_file};

            std::error_code cleanup_error;
            for (const fs::path &temporary_output : temporary_outputs)
            {
                fs::remove(temporary_output, cleanup_error);

                if (cleanup_error)
                {
                    std::cerr
                        << "WARNING: could not remove temporary output "
                        << temporary_output
                        << ": " << cleanup_error.message() << "\n";
                    cleanup_error.clear();
                }
            }
        }

        summary << comparison_idx << ','
                << trial << ','
                << seed << ','
                << rrt_run_idx << ','
                << (rrt_result.success ? 1 : 0) << ',';

        if (rrt_result.success)
        {
            summary << std::fixed << std::setprecision(6)
                    << rrt_result.path_length_m;
        }
        else
        {
            summary << "nan";
        }

        summary << ','
                << std::fixed << std::setprecision(6)
                << rrt_result.wall_runtime_s << ','
                << diffusion_run_idx << ','
                << (diffusion_result.success ? 1 : 0)
                << ',';

        if (diffusion_result.success)
        {
            summary << std::fixed << std::setprecision(6)
                    << diffusion_result.path_length_m;
        }
        else
        {
            summary << "nan";
        }

        summary << ','
                << std::fixed << std::setprecision(6)
                << diffusion_result.wall_runtime_s
                << '\n';
        summary.flush();

        std::cout
            << "Classical RRT: success="
            << rrt_result.success
            << ", path_length=";
        if (rrt_result.success)
            std::cout << rrt_result.path_length_m << " m";
        else
            std::cout << "nan";
        std::cout
            << ", runtime="
            << rrt_result.wall_runtime_s << " s\n";

        std::cout
            << "Diffusion RRT: success="
            << diffusion_result.success
            << ", path_length=";
        if (diffusion_result.success)
            std::cout
                << diffusion_result.path_length_m << " m";
        else
            std::cout << "nan";
        std::cout
            << ", runtime="
            << diffusion_result.wall_runtime_s << " s\n";
    }

    summary.close();

    std::cout
        << "\n========================================\n"
        << "Fixed-seed comparison complete\n"
        << "========================================\n"
        << "Comparison index        : "
        << comparison_idx << "\n"
        << "Classical RRT successes : "
        << rrt_success_count << " / 10\n"
        << "Diffusion RRT successes : "
        << diffusion_success_count << " / 10\n"
        << "Comparison values       : "
        << comparison_directory << "\n"
        << "Archived query          : "
        << archived_query_file << "\n"
        << "Summary                 : "
        << summary_file << "\n";

    return 0;
}
