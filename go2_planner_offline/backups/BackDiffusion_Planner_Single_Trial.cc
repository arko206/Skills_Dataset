// planner.cc — SE(2) RRT Blossom Planner + Controller
//
// Configuration Files:
//   planner.cfg  — Master settings (search budget, robot parameters, paths)
//   query.cfg    — Dynamic problem settings (start/goal/obstacles)
//
// Execution:  ./astar --config planner.cfg
//             (Writes waypoints and controls as specified in config)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <ctime>
#include <chrono>

//-- Added the Header Files-----//
#include <cerrno>
#include <iomanip>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define MIN_DIST 0.25
#define MIN_ROT (M_PI/4)
#define MAX_V 0.5
#define MAX_VTH (M_PI/3)


// ============================================================
// Diffusion extension settings
// ============================================================

// Stop the rollout when the generated pose is within 0.20 m
// of the fixed RRT sampled position.
#define DIFF_SAMPLE_TOL 0.20

// Maximum number of diffusion-generated transitions inside
// one RRT extension.
#define DIFF_MAX_ROLLOUT_STEPS 12

// Minimum required progress during translation.
#define DIFF_MIN_POS_PROGRESS 1e-3

// Minimum required angular progress during rotation.
#define DIFF_MIN_ANG_PROGRESS 1e-3

// Small tolerance for rejecting unintended mixed actions.
// #define DIFF_TRANSLATION_LEAK_TOL 0.05
// #define DIFF_ROTATION_LEAK_TOL (5.0 * M_PI / 180.0)



#define MAX_ITERS 250
#define AOX_RUNS 1
#define GOAL_BIAS_PERCENT 0
#define LATERAL_BIAS_PERCENT 0
#define ROT_ERR 10



// ============================================================
// Geometry helpers (forward declarations)
// ============================================================
static inline double wrap_angle(double a);

// ============================================================
// SE(2) Configuration
// ============================================================
struct Configuration
{
    double x = 0.0;
    double y = 0.0;
    double theta = 0.0;

    static double weight_theta;

    double distance(const Configuration& other) const {
        double dx = other.x - x;
        double dy = other.y - y;
        double d_euc = std::sqrt(dx * dx + dy * dy);
        double d_theta = std::fabs(wrap_angle(other.theta - theta));
        return d_euc + weight_theta * d_theta;
    }

    double distance(const Configuration& other, int dimid) const {
        if (dimid == 0) return std::fabs(other.x - x);
        if (dimid == 1) return std::fabs(other.y - y);
        if (dimid == 2) return std::fabs(wrap_angle(other.theta - theta));
        return 0.0;
    }

    double distance(const Configuration& other, const std::vector<int>& dims) const {
        if (dims.size() == 3) {
            bool has0 = false, has1 = false, has2 = false;
            for (int d : dims) { if (d==0) has0=true; if (d==1) has1=true; if (d==2) has2=true; }
            if (has0 && has1 && has2) return distance(other);
        }

        double d_total = 0.0;
        bool has_x = false, has_y = false;
        for (int d : dims) {
            if (d == 0) has_x = true;
            else if (d == 1) has_y = true;
            else if (d == 2) d_total += weight_theta * std::fabs(wrap_angle(other.theta - theta));
        }
        if (has_x && has_y) {
            double dx = other.x - x;
            double dy = other.y - y;
            d_total += std::sqrt(dx * dx + dy * dy);
        }
        else if (has_x) d_total += std::fabs(other.x - x);
        else if (has_y) d_total += std::fabs(other.y - y);
        return d_total;
    }
};

double Configuration::weight_theta = 0.5;

// ============================================================
// Rectangular obstacle
// ============================================================
struct Obstacle
{
    Configuration pose;
    double length = 0.35;
    double width = 0.25;
};

// ============================================================
// Full planner configuration
// ============================================================
struct PlannerConfig
{
    // Map bounds (metres)
    double x_min = -1.5;
    double x_max = 1.5;
    double y_min = -1.5;
    double y_max = 2.75;

    // Search settings
    int blossom_number = 10;
    int seed = -1;
    double w_theta = 1;
    bool smooth_path = false;
    bool allow_reverse = false;
    bool allow_lateral = true;

    // Goal tolerance (Euclidean, metres)
    double goal_tol = 0.2;

    // Robot geometry
    double robot_radius = 0.5;
    double robot_length = 0.4;
    double robot_width = 0.35;

    // Obstacles
    std::vector<Obstacle> obstacles;
    double safety_margin = 0.1;

    // File paths
    std::string waypoints_out = "se2_waypoints.txt";
    std::string controls_out = "controls.txt";
    std::string query_file = "query_1.cfg";

    // Poses
    Configuration start;
    Configuration goal;

    // Controller settings
    double step_time = 1.0;
};

// ============================================================
// Geometry helpers
// ============================================================
static inline double wrap_angle(double a)
{
    return std::atan2(std::sin(a), std::cos(a));
}

static inline double hypot2(double dx, double dy)
{
    return std::sqrt(dx * dx + dy * dy);
}

struct Vec2 {
    double x, y;
};

static inline Vec2 rotate_vec(const Vec2& v, double th) {
    return {v.x * std::cos(th) - v.y * std::sin(th),
            v.x * std::sin(th) + v.y * std::cos(th)};
}

static inline bool overlap(double min1, double max1, double min2, double max2) {
    return std::max(min1, min2) <= std::min(max1, max2);
}

static inline bool rect_intersects_rect(double x1, double y1, double th1, double l1, double w1,
                                        double x2, double y2, double th2, double l2, double w2,
                                        double safety)
{
    // Apply safety margin to BOTH rectangles (effectively inflating the footprint)
    // Actually, user said only inflate obstacles by safety margin.
    // So we'll treat obstacle as (L2 + 2*safety, W2 + 2*safety).

    //--(a) safety margin applied as inflation to obstacle rectangle
    double L1 = l1, W1 = w1;
    double L2 = l2 + 2*safety, W2 = w2 + 2*safety;
    
    //-(b) Get corners of both rectangles in world frame
    Vec2 corners1[4], corners2[4];
    double hl1 = L1/2.0, hw1 = W1/2.0;
    double hl2 = L2/2.0, hw2 = W2/2.0;
    
    //-- (c) Define corners in local frame (centered at rectangle center, aligned with rectangle axes)
    Vec2 raw1[4] = {{-hl1, -hw1}, {hl1, -hw1}, {hl1, hw1}, {-hl1, hw1}};
    Vec2 raw2[4] = {{-hl2, -hw2}, {hl2, -hw2}, {hl2, hw2}, {-hl2, hw2}};
    
    //-(d) Rotate and translate corners to world frame
    for(int i=0; i<4; ++i) {
        Vec2 r1 = rotate_vec(raw1[i], th1);
        corners1[i] = {x1 + r1.x, y1 + r1.y};
        Vec2 r2 = rotate_vec(raw2[i], th2);
        corners2[i] = {x2 + r2.x, y2 + r2.y};
    }

    // --(e) Axes to test: Normals to the 4 sides (2 from each rect)
    Vec2 axes[4];

    //-- (f) axes[0] is normal to the rectangle-1 along it heading (th1), while axes [1]` is the normal to rectangle-1 along its width.
    // Similarly for axes[2] and axes[3] for rectangle-2.
    axes[0] = {std::cos(th1), std::sin(th1)};
    axes[1] = {-std::sin(th1), std::cos(th1)};
    axes[2] = {std::cos(th2), std::sin(th2)};
    axes[3] = {-std::sin(th2), std::cos(th2)};
    
    // --(g) projecting both rectangles onto each axis and checking for overlap (Separating Axis Theorem)
    for (int i=0; i<4; ++i) {
        double min1 = 1e18, max1 = -1e18;
        double min2 = 1e18, max2 = -1e18;
        for (int j=0; j<4; ++j) {
            double p1 = corners1[j].x * axes[i].x + corners1[j].y * axes[i].y;
            min1 = std::min(min1, p1); max1 = std::max(max1, p1);
            double p2 = corners2[j].x * axes[i].x + corners2[j].y * axes[i].y;
            min2 = std::min(min2, p2); max2 = std::max(max2, p2);
        }
        //--(h) Check for overlap on this axis [sandard i-dimensonal overalap check between (min1, max1) and (min2, max2)]
        if (!overlap(min1, max1, min2, max2)) return false;
    }
    return true;
}
// Checking if robot at (px, py, pth) collides with any obstacle in the config (with safety margin)
static inline bool point_hits_any_obstacle(double px, double py, double pth,
                                           const PlannerConfig &cfg)
{
    for (const auto &ob : cfg.obstacles)
        if (rect_intersects_rect(px, py, pth, cfg.robot_length, cfg.robot_width,
                                 ob.pose.x, ob.pose.y, ob.pose.theta, ob.length, ob.width,
                                 cfg.safety_margin))
            return true;
    return false;
}

// ============================================================
// Collision checking
// ============================================================
static bool is_free(double px, double py, double pth, const PlannerConfig &cfg)
{
    // Check map bounds
    double RL = cfg.robot_length, RW = cfg.robot_width;
    double r = std::max(RL, RW) / 2.0; // Conservative radius for simple bounds check


    //Your code checks the negation of that:

    // (1) if left side crosses boundary
    // (2) or right side crosses boundary
    // (3) or bottom side crosses boundary
    // (4) or top side crosses boundary
    if (px - r < cfg.x_min || px + r > cfg.x_max ||
        py - r < cfg.y_min || py + r > cfg.y_max)
        return false;
    return !point_hits_any_obstacle(px, py, pth, cfg);
}

// ============================================================
// Config-file loader (key = value, # comments)
// ============================================================
static bool load_config_file(const std::string &path, PlannerConfig &cfg)
{
    std::ifstream f(path);
    if (!f)
    {
        std::cerr << "Cannot open config: " << path << "\n";
        return false;
    }

    auto trim = [](std::string &s)
    {
        auto a = s.find_first_not_of(" \t\r\n");
        auto b = s.find_last_not_of(" \t\r\n");
        s = (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
    };

    std::string line;
    while (std::getline(f, line))
    {
        auto sharp = line.find('#');
        if (sharp != std::string::npos)
            line.erase(sharp);
        auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos)
            continue;

        auto eq = line.find('=');
        if (eq == std::string::npos)
            continue;

        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        trim(key);
        trim(val);
        if (key.empty() || val.empty())
            continue;

        try
        {
            if (key == "x_min")
                cfg.x_min = std::stod(val);
            else if (key == "x_max")
                cfg.x_max = std::stod(val);
            else if (key == "y_min")
                cfg.y_min = std::stod(val);
            else if (key == "y_max")
                cfg.y_max = std::stod(val);
            else if (key == "goal_tol")
                cfg.goal_tol = std::stod(val);
            else if (key == "robot_radius")
                cfg.robot_radius = std::stod(val);
            else if (key == "robot_length")
                cfg.robot_length = std::stod(val);
            else if (key == "robot_width")
                cfg.robot_width = std::stod(val);
            else if (key == "start_x")
                cfg.start.x = std::stod(val);
            else if (key == "start_y")
                cfg.start.y = std::stod(val);
            else if (key == "start_theta")
                cfg.start.theta = std::stod(val);
            else if (key == "goal_x")
                cfg.goal.x = std::stod(val);
            else if (key == "goal_y")
                cfg.goal.y = std::stod(val);
            else if (key == "goal_theta")
                cfg.goal.theta = std::stod(val);
            else if (key == "safety_margin")
                cfg.safety_margin = std::stod(val);
            else if (key == "blossom_number")
                cfg.blossom_number = std::stoi(val);
            else if (key == "smooth_path")
                cfg.smooth_path = (val == "true" || val == "1");
            else if (key == "allow_reverse")
                cfg.allow_reverse = (val == "true" || val == "1");
            else if (key == "allow_lateral")
                cfg.allow_lateral = (val == "true" || val == "1");
            else if (key == "seed")
                cfg.seed = std::stoi(val);
            else if (key == "w_theta")
                cfg.w_theta = std::stod(val);
            else if (key == "waypoints_out")
                cfg.waypoints_out = val;
            else if (key == "controls_out")
                cfg.controls_out = val;
            else if (key == "query_file")
                cfg.query_file = val;
            else if (key == "step_time")
                cfg.step_time = std::stod(val);
            else if (key.rfind("obs.", 0) == 0)
            {
                auto dot1 = key.find('.', 4);
                if (dot1 != std::string::npos)
                {
                    int idx = std::stoi(key.substr(4, dot1 - 4));
                    auto fld = key.substr(dot1 + 1);
                    if (idx < 0)
                        throw std::runtime_error("negative obstacle index");
                    auto uidx = static_cast<std::size_t>(idx);
                    if (uidx >= cfg.obstacles.size())
                        cfg.obstacles.resize(uidx + 1);
                    auto &ob = cfg.obstacles[uidx];
                    if (fld == "x")
                        ob.pose.x = std::stod(val);
                    else if (fld == "y")
                        ob.pose.y = std::stod(val);
                    else if (fld == "theta")
                        ob.pose.theta = std::stod(val);
                    else if (fld == "length")
                        ob.length = std::stod(val);
                    else if (fld == "width")
                        ob.width = std::stod(val);
                }
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Config parse error for key '" << key
                      << "': " << e.what() << "\n";
        }
    }
    return true;
}

// ============================================================
// Expansion Logic (Blossom)
// ============================================================

// ============================================================
// Collision checking
// ============================================================
// --- Collision Check (Robotic Footprint) ---

static bool rotation_free(double x, double y, double th0, double th1,
                         const PlannerConfig &cfg)
{
    double dth = std::fabs(wrap_angle(th1 - th0));
    int n = std::max(2, static_cast<int>(std::ceil(dth / (M_PI / 18.0)))); // Every 10 degrees
    for (int i = 0; i <= n; ++i)
    {
        double t = static_cast<double>(i) / n;
        double pth = wrap_angle(th0 + t * wrap_angle(th1 - th0));
        if (!is_free(x, y, pth, cfg))
            return false;
    }
    return true;
}

static bool segment_free(double x0, double y0,
                         double x1, double y1,
                         double pth,
                         const PlannerConfig &cfg)
{
    double dx = x1 - x0;
    double dy = y1 - y0;
    double d = std::sqrt(dx * dx + dy * dy);
    int n = std::max(2, static_cast<int>(std::ceil(d / 0.01)));
    for (int i = 0; i <= n; ++i)
    {
        double t = static_cast<double>(i) / n;
        double px = x0 + t * (x1 - x0);
        double py = y0 + t * (y1 - y0);
        if (!is_free(px, py, pth, cfg))
            return false;
    }
    return true;
}

// is_free forward declaration removed

// ============================================================
// RRT types
// ============================================================
struct RRTNode
{
    Configuration point;
    int parent = -1;
    double cost = 0.0;
};

//-- modified to adapt to diffusion model returns---///
struct RotTranslateEdge
{
    std::vector<Configuration> rollout;
};

struct DiffusionAction
{
    double dx_robot = 0.0;
    double dy_robot = 0.0;
    double dtheta = 0.0;
};

// Machine-readable diagnostics collected during one diffusion-RRT attempt.
struct PlannerDiagnostics
{
    long diffusion_action_requests = 0;
    long accepted_rollout_nodes = 0;

    // One candidate node is counted when a diffusion action has been
    // converted to a world-frame pose and enters collision validation.
    long candidate_nodes_collision_checked = 0;

    // Counts individual endpoint, rotation-sweep, and segment checks.
    long collision_check_calls = 0;

    long endpoint_collision_rejections = 0;
    long rotation_collision_rejections = 0;
    long segment_collision_rejections = 0;

    long angular_no_progress_rejections = 0;
    long positional_no_progress_rejections = 0;

    long local_target_failures = 0;
    long observation_failures = 0;
    long inference_failures = 0;
    long invalid_nearest_failures = 0;

    long successful_extensions = 0;
    long failed_extensions = 0;
    long sample_reached_stops = 0;

    long duplicate_nodes_found = 0;
};

// ============================================================
// Persistent Python diffusion-policy bridge
// ============================================================
class PythonDiffusionBridge
{
public:
    PythonDiffusionBridge() = default;

    PythonDiffusionBridge(
        const PythonDiffusionBridge &) = delete;

    PythonDiffusionBridge &operator=(
        const PythonDiffusionBridge &) = delete;

    ~PythonDiffusionBridge()
    {
        stop();
    }

    bool start(
        const std::string &python_executable,
        const std::string &script_path,
        const std::string &device,
        unsigned int diffusion_seed)
    {
        if (child_pid_ > 0)
        {
            std::cerr
                << "Diffusion bridge is already running.\n";
            return false;
        }

        const std::string diffusion_seed_text =
                  std::to_string(diffusion_seed);

        // parent_to_child:
        // C++ writes observations -> Python reads stdin.
        int parent_to_child[2] = {-1, -1};

        // child_to_parent:
        // Python writes actions -> C++ reads stdout.
        int child_to_parent[2] = {-1, -1};

        if (pipe(parent_to_child) == -1)
        {
            std::perror("pipe parent_to_child");
            return false;
        }

        if (pipe(child_to_parent) == -1)
        {
            std::perror("pipe child_to_parent");

            close(parent_to_child[0]);
            close(parent_to_child[1]);

            return false;
        }

        child_pid_ = fork();

        if (child_pid_ == -1)
        {
            std::perror("fork");

            close(parent_to_child[0]);
            close(parent_to_child[1]);

            close(child_to_parent[0]);
            close(child_to_parent[1]);

            child_pid_ = -1;
            return false;
        }

        // ====================================================
        // Child process: launch Eval_DM_navg.py
        // ====================================================
        if (child_pid_ == 0)
        {
            // Replace Python stdin with the C++ -> Python pipe.
            if (dup2(
                    parent_to_child[0],
                    STDIN_FILENO) == -1)
            {
                std::perror("dup2 Python stdin");
                _exit(127);
            }

            // Replace Python stdout with the Python -> C++ pipe.
            if (dup2(
                    child_to_parent[1],
                    STDOUT_FILENO) == -1)
            {
                std::perror("dup2 Python stdout");
                _exit(127);
            }

            // Close all inherited original pipe descriptors.
            close(parent_to_child[0]);
            close(parent_to_child[1]);

            close(child_to_parent[0]);
            close(child_to_parent[1]);

            // Equivalent command: venv/bin/python -u Eval_DM_navg.py \  --device cuda \ --seed 42
            execl(
                python_executable.c_str(),
                python_executable.c_str(),
                "-u",
                script_path.c_str(),
                "--device",
                device.c_str(),
                "--seed",
                diffusion_seed_text.c_str(),
                static_cast<char *>(nullptr));

            // This line is reached only when execl fails.
            std::perror(
                "execl Python diffusion server");

            _exit(127);
        }

        // ====================================================
        // Parent process: retain only the required pipe ends
        // ====================================================

        // Parent does not read its own outgoing pipe.
        close(parent_to_child[0]);

        // Parent does not write into Python's outgoing pipe.
        close(child_to_parent[1]);

        to_python_ = fdopen(
            parent_to_child[1],
            "w");

        if (to_python_ == nullptr)
        {
            std::perror("fdopen to_python");

            close(parent_to_child[1]);
            close(child_to_parent[0]);

            terminate_child();
            return false;
        }

        from_python_ = fdopen(
            child_to_parent[0],
            "r");

        if (from_python_ == nullptr)
        {
            std::perror("fdopen from_python");

            std::fclose(to_python_);
            to_python_ = nullptr;

            close(child_to_parent[0]);

            terminate_child();
            return false;
        }

        // Flush each complete line sent to Python.
        setvbuf(
            to_python_,
            nullptr,
            _IOLBF,
            0);

        // ====================================================
        // Wait for Python's READY handshake
        // ====================================================
        char response[256];

        if (std::fgets(
                response,
                sizeof(response),
                from_python_) == nullptr)
        {
            std::cerr
                << "ERROR: Python diffusion server "
                << "exited before sending READY.\n";

            stop();
            return false;
        }

        std::string ready(response);

        // Remove newline characters.
        while (!ready.empty() &&
               (ready.back() == '\n' ||
                ready.back() == '\r'))
        {
            ready.pop_back();
        }

        if (ready != "READY")
        {
            std::cerr
                << "ERROR: expected READY from Python, "
                << "but received: "
                << ready << "\n";

            stop();
            return false;
        }

        std::cout
            << "Python diffusion server handshake completed.\n";

        return true;
    }

    bool predict(
        const std::array<double, 12> &observation,
        DiffusionAction &action)
    {
        if (child_pid_ <= 0 ||
            to_python_ == nullptr ||
            from_python_ == nullptr)
        {
            std::cerr
                << "ERROR: diffusion bridge is not running.\n";

            return false;
        }

        // ====================================================
        // Send exactly 12 observation values to Python
        // ====================================================
        for (std::size_t i = 0;
             i < observation.size();
             ++i)
        {
            if (!std::isfinite(observation[i]))
            {
                std::cerr
                    << "ERROR: observation contains "
                    << "a non-finite value.\n";

                return false;
            }

            if (i > 0)
            {
                if (std::fputc(
                        ' ',
                        to_python_) == EOF)
                {
                    std::perror(
                        "write separator to Python");

                    return false;
                }
            }

            if (std::fprintf(
                    to_python_,
                    "%.17g",
                    observation[i]) < 0)
            {
                std::perror(
                    "write observation to Python");

                return false;
            }
        }

        // Complete and flush the request line.
        if (std::fputc(
                '\n',
                to_python_) == EOF ||
            std::fflush(
                to_python_) == EOF)
        {
            std::perror(
                "flush observation to Python");

            return false;
        }

        // ====================================================
        // Read one three-dimensional action from Python
        // ====================================================
        char response[512];

        if (std::fgets(
                response,
                sizeof(response),
                from_python_) == nullptr)
        {
            std::cerr
                << "ERROR: failed to read diffusion "
                << "action from Python.\n";

            return false;
        }

        std::istringstream input(response);

        double dx_robot = 0.0;
        double dy_robot = 0.0;
        double dtheta = 0.0;

        std::string extra_token;

        // Require exactly three values.
        if (!(input >>
              dx_robot >>
              dy_robot >>
              dtheta) ||
            (input >> extra_token))
        {
            std::cerr
                << "ERROR: malformed Python action response: "
                << response;

            return false;
        }

        if (!std::isfinite(dx_robot) ||
            !std::isfinite(dy_robot) ||
            !std::isfinite(dtheta))
        {
            std::cerr
                << "ERROR: Python returned a "
                << "non-finite action: "
                << response;

            return false;
        }

        action.dx_robot = dx_robot;
        action.dy_robot = dy_robot;
        action.dtheta = dtheta;

        return true;
    }

    void stop()
    {
        // Ask Python to exit normally.
        if (to_python_ != nullptr)
        {
            std::fputs(
                "QUIT\n",
                to_python_);

            std::fflush(to_python_);
            std::fclose(to_python_);

            to_python_ = nullptr;
        }

        if (from_python_ != nullptr)
        {
            std::fclose(from_python_);
            from_python_ = nullptr;
        }

        if (child_pid_ > 0)
        {
            int status = 0;

            while (
                waitpid(
                    child_pid_,
                    &status,
                    0) == -1 &&
                errno == EINTR)
            {
            }

            child_pid_ = -1;
        }
    }

private:
    void terminate_child()
    {
        if (child_pid_ > 0)
        {
            kill(
                child_pid_,
                SIGTERM);

            int status = 0;

            while (
                waitpid(
                    child_pid_,
                    &status,
                    0) == -1 &&
                errno == EINTR)
            {
            }

            child_pid_ = -1;
        }
    }

    pid_t child_pid_ = -1;

    FILE *to_python_ = nullptr;
    FILE *from_python_ = nullptr;
};


// One persistent bridge instance for the complete planner run.
static PythonDiffusionBridge diffusion_python_process;





enum class LocalTargetMode
{
    Rotation,
    Translation
};

///---- The Function added to provide local target to Diffusion Policy-----///
///-- (a) The target can be only translational----///
///--- (b) The target first rotates and translates ---///
static bool build_local_target(
    const PlannerConfig &cfg,
    const Configuration &current_pose,
    const Configuration &sample_pose,
    Configuration &local_target,
    LocalTargetMode &mode)
{
    double dx = sample_pose.x - current_pose.x;
    double dy = sample_pose.y - current_pose.y;

    double distance_to_sample =
        std::sqrt(dx * dx + dy * dy);

    // The current pose is already sufficiently close
    // to the fixed RRT sample.
    if (distance_to_sample <= DIFF_SAMPLE_TOL)
    {
        return false;
    }

    double target_heading = current_pose.theta;
    bool use_reverse = false;

    if (distance_to_sample > 1e-6)
    {
        double forward_heading =
            std::atan2(dy, dx);

        if (cfg.allow_reverse)
        {
            double reverse_heading =
                wrap_angle(forward_heading + M_PI);

            double forward_rotation =
                std::fabs(
                    wrap_angle(
                        forward_heading -
                        current_pose.theta));

            double reverse_rotation =
                std::fabs(
                    wrap_angle(
                        reverse_heading -
                        current_pose.theta));

            if (reverse_rotation < forward_rotation)
            {
                target_heading = reverse_heading;
                use_reverse = true;
            }
            else
            {
                target_heading = forward_heading;
                use_reverse = false;
            }
        }
        else
        {
            target_heading = forward_heading;
            use_reverse = false;
        }
    }

    // ========================================================
    // Case 1: Construct a pure rotation target
    // ========================================================
    double desired_rotation =
        wrap_angle(
            target_heading -
            current_pose.theta);

    double bounded_rotation =
        std::min(
            std::fabs(desired_rotation),
            static_cast<double>(MAX_VTH));

    if (bounded_rotation >= MIN_ROT)
    {
        double signed_rotation =
            (desired_rotation < 0.0)
                ? -bounded_rotation
                : bounded_rotation;

        local_target.x = current_pose.x;
        local_target.y = current_pose.y;

        local_target.theta =
            wrap_angle(
                current_pose.theta +
                signed_rotation);

        mode = LocalTargetMode::Rotation;
        return true;
    }

    // ========================================================
    // Case 2: Construct a pure forward/backward target
    // ========================================================
    double local_step =
        std::min(
            distance_to_sample,
            static_cast<double>(MAX_V));

    double signed_step =
        use_reverse
            ? -local_step
            : local_step;

    // Move along the robot's current heading.
    local_target.x =
        current_pose.x +
        signed_step *
            std::cos(current_pose.theta);

    local_target.y =
        current_pose.y +
        signed_step *
            std::sin(current_pose.theta);

    local_target.theta =
        current_pose.theta;

    mode = LocalTargetMode::Translation;
    return true;
}

////---- end of implementing the function------////


///--- constructing the exact 12-dimensional observation for diffusion policy ------////
static bool build_diffusion_observation(
    const PlannerConfig &cfg,
    const Configuration &current_pose,
    const Configuration &local_target,
    std::array<double, 12> &observation)
{
    // Your trained model expects exactly two obstacles:
    //
    // 2 obstacles × [rel_x, rel_y, distance] = 6 values.
    if (cfg.obstacles.size() != 2)
    {
        std::cerr
            << "Diffusion observation requires exactly 2 obstacles, but "
            << cfg.obstacles.size()
            << " were provided.\n";

        return false;
    }

    observation[0] = current_pose.x;
    observation[1] = current_pose.y;
    observation[2] = current_pose.theta;

    observation[3] = local_target.x;
    observation[4] = local_target.y;
    observation[5] = local_target.theta;

    double cos_theta = std::cos(current_pose.theta);
    double sin_theta = std::sin(current_pose.theta);

    for (std::size_t obstacle_idx = 0;
         obstacle_idx < 2;
         ++obstacle_idx)
    {
        const auto &obstacle = cfg.obstacles[obstacle_idx];

        double dx_world = obstacle.pose.x - current_pose.x;
        double dy_world = obstacle.pose.y - current_pose.y;

        double rel_x =
            dx_world * cos_theta +
            dy_world * sin_theta;

        double rel_y =
            -dx_world * sin_theta +
            dy_world * cos_theta;

        double rel_distance =
            std::sqrt(rel_x * rel_x + rel_y * rel_y);

        std::size_t base = 6 + 3 * obstacle_idx;

        observation[base + 0] = rel_x;
        observation[base + 1] = rel_y;
        observation[base + 2] = rel_distance;
    }

    return true;
}

///----- end of implementation-----////

//---Temporary diffusion placeholder addition------///
static bool request_diffusion_action(
    const Configuration &current_pose,
    const Configuration &local_target,
    const std::array<double, 12> &observation,
    DiffusionAction &action)
{
    (void)current_pose;
    (void)local_target;

    return diffusion_python_process.predict(
        observation,
        action
    );
}

///-- Converting the diffusion model generated poses in world frame from robot frame-----///
static bool diffusion_action_to_world_pose(
    const PlannerConfig &cfg,
    const Configuration &current_pose,
    const DiffusionAction &action,
    Configuration &candidate_pose,
    PlannerDiagnostics &diagnostics)
{
    // Convert predicted robot-frame translation to world frame.
    double cos_theta = std::cos(current_pose.theta);
    double sin_theta = std::sin(current_pose.theta);

    double dx_world =
        cos_theta * action.dx_robot -
        sin_theta * action.dy_robot;

    double dy_world =
        sin_theta * action.dx_robot +
        cos_theta * action.dy_robot;

    // Construct the generated world-frame pose.
    candidate_pose.x =
        current_pose.x + dx_world;

    candidate_pose.y =
        current_pose.y + dy_world;

    candidate_pose.theta =
        wrap_angle(
            current_pose.theta +
            action.dtheta);

    // Count this generated candidate as a node that entered collision checking.
    diagnostics.candidate_nodes_collision_checked++;

    // Check the generated final pose first.
    diagnostics.collision_check_calls++;
    if (!is_free(
            candidate_pose.x,
            candidate_pose.y,
            candidate_pose.theta,
            cfg))
    {
        diagnostics.endpoint_collision_rejections++;
        return false;
    }

    // If the model predicts a heading change, validate rotation
    // at the current position.
    if (std::fabs(action.dtheta) > 1e-6)
    {
        diagnostics.collision_check_calls++;
        if (!rotation_free(
                current_pose.x,
                current_pose.y,
                current_pose.theta,
                candidate_pose.theta,
                cfg))
        {
            diagnostics.rotation_collision_rejections++;
            return false;
        }
    }

    // If the model predicts translation, validate the complete
    // segment using the generated heading.
    double translation_distance =
        std::sqrt(
            dx_world * dx_world +
            dy_world * dy_world);

    if (translation_distance > 1e-6)
    {
        diagnostics.collision_check_calls++;
        if (!segment_free(
                current_pose.x,
                current_pose.y,
                candidate_pose.x,
                candidate_pose.y,
                candidate_pose.theta,
                cfg))
        {
            diagnostics.segment_collision_rejections++;
            return false;
        }
    }

    return true;
}

static bool extend(
    const PlannerConfig &cfg,
    const std::vector<RRTNode> &tree,
    int best_idx,
    const Configuration &r_sample,
    bool sampling_goal,
    int iter,
    RotTranslateEdge &out_best_cand,
    PlannerDiagnostics &diagnostics)
{
    (void)sampling_goal;
    (void)iter;

    if (best_idx < 0 ||
        best_idx >= static_cast<int>(tree.size()))
    {
        diagnostics.invalid_nearest_failures++;
        diagnostics.failed_extensions++;
        return false;
    }

    out_best_cand.rollout.clear();
    // out_best_cand.parent_idx = best_idx;
    // out_best_cand.eval_dist = 1e18;

    Configuration current_pose =
        tree[best_idx].point;

    for (int rollout_step = 0;
         rollout_step < DIFF_MAX_ROLLOUT_STEPS;
         ++rollout_step)
    {
        double previous_sample_distance =
            current_pose.distance(r_sample, {0, 1});

        // The fixed random sample has been reached closely enough.
        if (previous_sample_distance <= DIFF_SAMPLE_TOL)
        {
            diagnostics.sample_reached_stops++;
            break;
        }

        Configuration local_target;
        LocalTargetMode target_mode;

        if (!build_local_target(
                cfg,
                current_pose,
                r_sample,
                local_target,
                target_mode))
        {
            diagnostics.local_target_failures++;
            break;
        }

        std::array<double, 12> observation;

        if (!build_diffusion_observation(
                cfg,
                current_pose,
                local_target,
                observation))
        {
            diagnostics.observation_failures++;
            break;
        }

        DiffusionAction predicted_action;

        diagnostics.diffusion_action_requests++;
        if (!request_diffusion_action(
                current_pose,
                local_target,
                observation,
                predicted_action))
        {
            diagnostics.inference_failures++;
            break;
        }

        // std::cout
        //     << "[Diffusion raw action]"
        //     << " mode="
        //     << (target_mode == LocalTargetMode::Rotation
        //             ? "Rotation"
        //             : "Translation")
        //     << " dx_robot=" << predicted_action.dx_robot
        //     << " dy_robot=" << predicted_action.dy_robot
        //     << " dtheta=" << predicted_action.dtheta
        //     << "\n";



        // Preserve the pure motion primitive selected by build_local_target().
        if (target_mode == LocalTargetMode::Rotation)
        {
            predicted_action.dx_robot = 0.0;
            predicted_action.dy_robot = 0.0;
        }
        else
        {
            predicted_action.dtheta = 0.0;
            predicted_action.dy_robot = 0.0;
        }

        Configuration candidate_pose;

        if (!diffusion_action_to_world_pose(
                cfg,
                current_pose,
                predicted_action,
                candidate_pose,
                diagnostics))
        {
            break;
        }

        // ----------------------------------------------------
        // Progress validation
        // ----------------------------------------------------
        if (target_mode == LocalTargetMode::Rotation)
        {
            double previous_angle_error =
                std::fabs(
                    wrap_angle(
                        local_target.theta -
                        current_pose.theta));

            double new_angle_error =
                std::fabs(
                    wrap_angle(
                        local_target.theta -
                        candidate_pose.theta));
            // if the obtained pose have been obtained in the wrong direction, diffusion policy rollout stops--//
            if (new_angle_error >
                previous_angle_error -
                    DIFF_MIN_ANG_PROGRESS)
            {
                diagnostics.angular_no_progress_rejections++;
                break;
            }
        }
        else
        {
            double new_sample_distance =
                candidate_pose.distance(
                    r_sample,
                    {0, 1});
            // if the newly obtained pose is far way from the sample target pose--//
            if (new_sample_distance >
                previous_sample_distance -
                    DIFF_MIN_POS_PROGRESS)
            {
                diagnostics.positional_no_progress_rejections++;
                break;
            }
        }
        //--- the line stores each successful generated pose----///
        out_best_cand.rollout.push_back(
            candidate_pose);
        diagnostics.accepted_rollout_nodes++;

        current_pose = candidate_pose;
    }
    // if no collision-free rollouts are obtained from diffusion policy----///
    if (out_best_cand.rollout.empty())
    {
        diagnostics.failed_extensions++;
        return false;
    }

    diagnostics.successful_extensions++;
    return true;
}


static bool update_best_goal(const std::vector<RRTNode> &tree, int idx,
                             const PlannerConfig &cfg, int &goal_idx,
                             double &best_goal_cost, int iter)
{
    double dg = tree[idx].point.distance(cfg.goal, {0, 1});
    if (dg <= cfg.goal_tol)
    {
        if (tree[idx].cost < best_goal_cost)
        {
            best_goal_cost = tree[idx].cost;
            goal_idx = idx;
            std::cout << "New best goal cost found: " << best_goal_cost
                      << " (node " << idx << ") at iter " << iter << "\n";

            return true;
        }
    }

    return false;
}

struct KDNode
{
    int idx;
    double x, y;
    int left = -1;
    int right = -1;
};

static void kdtree_insert(std::vector<KDNode> &nodes, int &root_idx, int new_node_idx, const Configuration &point, int depth = 0)
{
    if (root_idx == -1)
    {
        root_idx = static_cast<int>(nodes.size());
        nodes.push_back({new_node_idx, point.x, point.y, -1, -1});
        return;
    }

    bool split_x = (depth % 2 == 0);
    double val = split_x ? point.x : point.y;
    double root_val = split_x ? nodes[root_idx].x : nodes[root_idx].y;

    if (val < root_val)
        kdtree_insert(nodes, nodes[root_idx].left, new_node_idx, point, depth + 1);
    else
        kdtree_insert(nodes, nodes[root_idx].right, new_node_idx, point, depth + 1);
}

static void kdtree_query(const std::vector<KDNode> &nodes, int root_idx, const Configuration &q_sample,
                         const std::vector<RRTNode> &tree, const PlannerConfig &cfg, int goal_idx,
                         int &best_idx, double &min_dist_sq, int depth = 0)
{
    if (root_idx == -1) return;

    const auto &kd = nodes[root_idx];
    const auto &rrt_node = tree[kd.idx];

    // 1. Goal optimization check (optional for find_nearest, usually skip for duplicates)
    double dg_node = rrt_node.point.distance(cfg.goal, {0, 1});
    if (goal_idx == -2 || !(dg_node <= cfg.goal_tol && goal_idx != -1))
    {
        // 2. Full SE(2) distance check
        double d = rrt_node.point.distance(q_sample);
        if (d < std::sqrt(min_dist_sq))
        {
            min_dist_sq = d * d;
            best_idx = kd.idx;
        }
    }

    bool split_x = (depth % 2 == 0);
    double q_val = split_x ? q_sample.x : q_sample.y;
    double kd_val = split_x ? kd.x : kd.y;

    int near = (q_val < kd_val) ? kd.left : kd.right;
    int far = (q_val < kd_val) ? kd.right : kd.left;

    kdtree_query(nodes, near, q_sample, tree, cfg, goal_idx, best_idx, min_dist_sq, depth + 1);

    double diff = std::fabs(q_val - kd_val);
    if (diff * diff < min_dist_sq)
    {
        kdtree_query(nodes, far, q_sample, tree, cfg, goal_idx, best_idx, min_dist_sq, depth + 1);
    }
}

static int find_nearest(const std::vector<RRTNode> &tree, const std::vector<KDNode> &kd_nodes, int kd_root,
                        const Configuration &q_sample, const PlannerConfig &cfg, int goal_idx)
{
    int best_idx = -1;
    double min_dist_sq = 1e36;
    kdtree_query(kd_nodes, kd_root, q_sample, tree, cfg, goal_idx, best_idx, min_dist_sq);
    return best_idx;
}

static int find_duplicate(const std::vector<RRTNode> &tree, const std::vector<KDNode> &kd_nodes, int kd_root, const Configuration &q)
{
    if (kd_root == -1) return -1;
    
    // Use KD-tree to find nearest, then check if it's within tolerance
    int best_idx = -1;
    double min_dist_sq = 1e36;
    PlannerConfig dummy_cfg; // Goal optimization not needed for duplicate check
    kdtree_query(kd_nodes, kd_root, q, tree, dummy_cfg, -2, best_idx, min_dist_sq);
    
    if (best_idx != -1) {
        const double eps_pos = 1e-3;
        const double eps_th = 1e-2;
        if (tree[best_idx].point.distance(q, {0, 1}) < eps_pos &&
            tree[best_idx].point.distance(q, 2) < eps_th)
            return best_idx;
    }
    return -1;
}

static int add_node(std::vector<RRTNode> &tree, std::vector<KDNode> &kd_nodes, int &kd_root,
                   const Configuration &point, int parent, double cost, long &rej_dup)
{
    int dup_idx = find_duplicate(tree, kd_nodes, kd_root, point);
    if (dup_idx != -1) {
        if (cost < tree[dup_idx].cost) {
            tree[dup_idx].cost = cost;
            tree[dup_idx].parent = parent;
        }
        rej_dup++;
        return dup_idx;
    }
    tree.push_back({point, parent, cost});
    int new_idx = static_cast<int>(tree.size() - 1);
    kdtree_insert(kd_nodes, kd_root, new_idx, point);
    return new_idx;
}

// ============================================================
// Random Sampler
// ============================================================
static Configuration sample(const PlannerConfig &cfg, int iter, int goal_bias_percent, bool &sampling_goal)
{
    Configuration q;
    sampling_goal = (iter > 2000) && ((rand() % 100) < goal_bias_percent); // Start goal bias after 2000 iters for richer viz
    if (sampling_goal)
    {
        q = cfg.goal;
    }
    else
    {
        q.x = cfg.x_min + (cfg.x_max - cfg.x_min) * ((double)rand() / RAND_MAX);
        q.y = cfg.y_min + (cfg.y_max - cfg.y_min) * ((double)rand() / RAND_MAX);
        q.theta = -M_PI + 2.0 * M_PI * ((double)rand() / RAND_MAX);
    }
    return q;
}



struct RRTResult {
    bool success = false;
    double cost = 1e18;
    std::vector<Configuration> path;
    std::vector<RRTNode> tree;
    unsigned int seed;
    long rej_col = 0, rej_dup = 0;
    long tree_nodes_generated = 0;
    PlannerDiagnostics diagnostics;

    // Time taken within this RRT run to find the best goal-reaching path
    double time_to_best_goal_sec = -1.0;
    int best_goal_iter = -1;
};

// ============================================================
// RRT search
// ============================================================
//-- cfg contains planner settings, seed initializes random sampling---///
static RRTResult run_rrt(const PlannerConfig &cfg, unsigned int seed)
{   
    //--(a)creates an empty result structure
    RRTResult result;

    //--(b)fixes the random seed so the same seed reproduces the same random tree
    result.seed = seed;
    std::srand(seed);

    auto rrt_start_time = std::chrono::steady_clock::now();

    //--(c)heading difference Δθ
    // --(i) Δθ is scaled by cfg.w_theta
    // --(ii) so orientation affects nearest-neighbor search and perhaps extension quality
    //--- (iii) tuning w_theta allows us to control how much the planner cares about heading similarity vs position similarity when finding nearest neighbors and evaluating extensions.
    // If large, the planner cares more about heading similarity.
    // If small, it mostly cares about position.

    Configuration::weight_theta = cfg.w_theta;



    std::vector<RRTNode> tree;
    tree.reserve(MAX_ITERS);

    //-- kd_nodes--> This is probably the KD-tree structure used for fast nearest-neighbor lookup.--//
    std::vector<KDNode> kd_nodes;
    kd_nodes.reserve(MAX_ITERS);
    int kd_root = -1;

    //goal_idx = -1 means no goal-reaching node has been found yet
    // best_goal_cost stores the best path cost to goal found so far
    // the rej_* counters count rejected expansion attempts

    int goal_idx = -1;
    double best_goal_cost = 1e18;

    long rej_dup = 0;
    PlannerDiagnostics diagnostics;

    
    add_node(tree, kd_nodes, kd_root, cfg.start, -1, 0.0, rej_dup);

    //-- (10) This repeats the grow-tree process up to MAX_ITERS.---//
    for (int iter = 0; iter < MAX_ITERS; ++iter)
    {
        //-- (a) near_idx = index of nearest tree node selected for expansion
        int near_idx = -1;
        //--(b) cand = candidate motion edge returned by extension--//
        RotTranslateEdge cand;
        //--(c) found = whether the extension was successful (collision-free, respects motion limits, etc.)--//
        bool found = false;

        bool sampling_goal = false;

        Configuration r_sample = sample(cfg, iter, GOAL_BIAS_PERCENT, sampling_goal);

        //--sampling_goal records whether this sample was the goal-biased one.--//
        near_idx = find_nearest(tree, kd_nodes, kd_root, r_sample, cfg, goal_idx);

        //-- This finds the tree node closest to the sampled target.--//

        // -- Mathematically:

        // -- qnear=arg⁡min⁡qi∈Vd(qi,qsample)

        //--So this is the node from which the planner tries to extend.
        if (near_idx != -1) {
            found = extend(
                cfg,
                tree,
                near_idx,
                r_sample,
                sampling_goal,
                iter,
                cand,
                diagnostics);
        }
        //-- This is the heart of the motion-generation logic.--//
        if (found)
        {
            // ========================================================
            // Case 1: Iterative diffusion rollout
            // ========================================================
            if (!cand.rollout.empty())
            {
                int last_parent = near_idx;

                for (const Configuration &generated_pose :
                    cand.rollout)
                {
                    // Copy before add_node(), because the vector may
                    // reallocate when a new node is inserted.
                    Configuration parent_pose =
                        tree[last_parent].point;

                    double parent_cost =
                        tree[last_parent].cost;

                    // Weighted SE(2) edge cost:
                    // planar distance + w_theta * angular difference.
                    double edge_cost =
                        parent_pose.distance(generated_pose);

                    int new_idx =
                        add_node(
                            tree,
                            kd_nodes,
                            kd_root,
                            generated_pose,
                            last_parent,
                            parent_cost + edge_cost,
                            rej_dup);

                    bool improved_goal =
                        update_best_goal(
                            tree,
                            new_idx,
                            cfg,
                            goal_idx,
                            best_goal_cost,
                            iter);

                    if (improved_goal)
                    {
                        auto now =
                            std::chrono::steady_clock::now();

                        result.time_to_best_goal_sec =
                            std::chrono::duration<double>(
                                now - rrt_start_time).count();

                        result.best_goal_iter = iter;
                    }

                    last_parent = new_idx;
                }
            }

        }

    }

    diagnostics.duplicate_nodes_found = rej_dup;


    result.rej_col =
        diagnostics.endpoint_collision_rejections +
        diagnostics.rotation_collision_rejections +
        diagnostics.segment_collision_rejections;
    result.rej_dup = rej_dup;
    result.tree_nodes_generated = static_cast<long>(tree.size());
    result.diagnostics = diagnostics;

    if (goal_idx != -1)
    {
        result.success = true;
        result.cost = best_goal_cost;

        std::vector<int> path_idx;
        int curr = goal_idx;
        while (curr != -1)
        {
            path_idx.push_back(curr);
            curr = tree[curr].parent;
        }
        std::reverse(path_idx.begin(), path_idx.end());

        for (int idx : path_idx) {
            result.path.push_back(tree[idx].point);
        }
        
    }
    result.tree = std::move(tree);

    return result;
}


static long total_collision_rejections(
    const PlannerDiagnostics &diagnostics)
{
    return diagnostics.endpoint_collision_rejections +
           diagnostics.rotation_collision_rejections +
           diagnostics.segment_collision_rejections;
}

static std::string dominant_extension_failure(
    const PlannerDiagnostics &diagnostics)
{
    std::vector<std::pair<long, std::string>> reasons = {
        {total_collision_rejections(diagnostics), "collision_rejection"},
        {diagnostics.angular_no_progress_rejections,
         "angular_no_progress"},
        {diagnostics.positional_no_progress_rejections,
         "positional_no_progress"},
        {diagnostics.local_target_failures,
         "local_target_construction_failure"},
        {diagnostics.observation_failures,
         "observation_construction_failure"},
        {diagnostics.inference_failures,
         "diffusion_inference_failure"},
        {diagnostics.invalid_nearest_failures,
         "invalid_nearest_node"}};

    const auto best = std::max_element(
        reasons.begin(),
        reasons.end(),
        [](const auto &lhs, const auto &rhs)
        {
            return lhs.first < rhs.first;
        });

    if (best == reasons.end() || best->first == 0)
        return "iteration_budget_exhausted_without_dominant_rejection";

    return best->second;
}

static void print_machine_readable_diagnostics(
    const RRTResult &result)
{
    const PlannerDiagnostics &d = result.diagnostics;

    std::cout
        << "DIAG_search_termination="
        << (result.success
                ? "goal_reached"
                : "max_iterations_exhausted_without_goal")
        << "\n";

    std::cout
        << "DIAG_primary_failure_reason="
        << (result.success
                ? "none"
                : dominant_extension_failure(d))
        << "\n";

    std::cout
        << "DIAG_duplicate_nodes_found="
        << d.duplicate_nodes_found << "\n";

    std::cout
        << "DIAG_candidate_nodes_collision_checked="
        << d.candidate_nodes_collision_checked << "\n";

    std::cout
        << "DIAG_collision_check_calls="
        << d.collision_check_calls << "\n";

    std::cout
        << "DIAG_collision_rejections="
        << total_collision_rejections(d) << "\n";

    std::cout
        << "DIAG_endpoint_collision_rejections="
        << d.endpoint_collision_rejections << "\n";

    std::cout
        << "DIAG_rotation_collision_rejections="
        << d.rotation_collision_rejections << "\n";

    std::cout
        << "DIAG_segment_collision_rejections="
        << d.segment_collision_rejections << "\n";

    std::cout
        << "DIAG_angular_no_progress_rejections="
        << d.angular_no_progress_rejections << "\n";

    std::cout
        << "DIAG_positional_no_progress_rejections="
        << d.positional_no_progress_rejections << "\n";

    std::cout
        << "DIAG_local_target_failures="
        << d.local_target_failures << "\n";

    std::cout
        << "DIAG_observation_failures="
        << d.observation_failures << "\n";

    std::cout
        << "DIAG_inference_failures="
        << d.inference_failures << "\n";

    std::cout
        << "DIAG_diffusion_action_requests="
        << d.diffusion_action_requests << "\n";

    std::cout
        << "DIAG_accepted_rollout_nodes="
        << d.accepted_rollout_nodes << "\n";

    std::cout
        << "DIAG_successful_extensions="
        << d.successful_extensions << "\n";

    std::cout
        << "DIAG_failed_extensions="
        << d.failed_extensions << "\n";

    std::cout
        << "DIAG_sample_reached_stops="
        << d.sample_reached_stops << "\n";

    std::cout
        << "DIAG_tree_nodes_generated="
        << result.tree_nodes_generated << "\n";
}

static bool write_rrt_tree(const std::vector<RRTNode> &tree, unsigned int seed, const std::string &filename)
{
    std::ofstream tree_file(filename);
    if (!tree_file) return false;
    tree_file << "Seed: " << seed << "\n";
    for (const auto &node : tree) {
        tree_file << node.point.x << " " << node.point.y << " " << node.point.theta << " " << node.parent << " " << node.cost << "\n";
    }
    tree_file.close();
    return true;
}

// ============================================================
// Controller: waypoints -> velocity commands
// ============================================================
struct ControlCmd
{
    double vx;       // forward velocity (m/s)
    double vy;       // lateral velocity (m/s)
    double vtheta;   // angular velocity (rad/s)
    double duration; // how long to hold this command (s)
};

static std::vector<ControlCmd> compute_controls(
    const std::vector<Configuration> &path,
    double step_time)
{
    std::vector<ControlCmd> cmds;
    if (path.size() < 2)
        return cmds;

    for (std::size_t i = 1; i < path.size(); ++i)
    {
        const auto &p1 = path[i - 1];
        const auto &p2 = path[i];

        double dx = p2.x - p1.x;
        double dy = p2.y - p1.y;
        double dtheta = wrap_angle(p2.theta - p1.theta);

        ControlCmd cmd;
        double cos_th = std::cos(p1.theta);
        double sin_th = std::sin(p1.theta);
        double d_fwd = dx * cos_th + dy * sin_th;
        double d_lat = -dx * sin_th + dy * cos_th;

        if (std::abs(dtheta) > 1e-6) {
            // Unicycle Rotation
            cmd.vx = 0.0; cmd.vy = 0.0; cmd.vtheta = dtheta / step_time;
        } else if (std::abs(d_fwd) > 1e-6 && std::abs(d_lat) < 1e-3) {
            // Unicycle Forward/Backward
            cmd.vx = d_fwd / step_time; cmd.vy = 0.0; cmd.vtheta = 0.0;
        } else if (std::abs(d_lat) > 1e-6) {
            // Lateral Translation
            cmd.vx = 0.0; cmd.vy = d_lat / step_time; cmd.vtheta = 0.0;
        } else {
            // No movement or mixed (should not happen with current motion primitives)
            cmd.vx = 0.0; cmd.vy = 0.0; cmd.vtheta = 0.0;
        }
        
        cmd.duration = step_time;
        cmds.push_back(cmd);
    }

    return cmds;
}

// ============================================================
// Write SE2 waypoints to file (x, y, theta per line)
// ============================================================
static bool write_waypoints_file(const std::vector<Configuration> &path,
                                 const std::string &filepath)
{
    std::ofstream f(filepath);
    if (!f)
    {
        std::cerr << "ERROR: cannot open " << filepath << " for writing\n";
        return false;
    }
    for (const auto &p : path)
    {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%f, %f, %f", p.x, p.y, p.theta);
        f << buf << "\n";
    }
    std::printf("Wrote %zu waypoints to %s\n", path.size(), filepath.c_str());
    return true;
}

// ============================================================
// Controls generation logic
// ============================================================

// ============================================================
// Write controls to file (vx, vy, vtheta per line)
// ============================================================
static bool write_controls_file(const std::vector<ControlCmd> &cmds,
                                const std::string &path)
{
    std::ofstream f(path);
    if (!f)
    {
        std::cerr << "ERROR: cannot open " << path << " for writing\n";
        return false;
    }
    for (const auto &c : cmds)
    {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%f, %f, %f", c.vx, c.vy, c.vtheta);
        f << buf << "\n";
    }
    std::printf("Wrote %zu controls to %s\n", cmds.size(), path.c_str());
    return true;
}

// ============================================================
// Safety audit
// ============================================================
static void collision_check_path(const std::vector<Configuration> &path,
                                 const PlannerConfig &cfg)
{
    std::printf("\n=== Final Collision Check (Rectangular Footprint) ===\n");
    std::printf("  %-4s  %-8s %-8s %-8s %-10s\n", "WP", "x", "y", "theta", "Status");

    bool any_violation = false;
    for (std::size_t i = 0; i < path.size(); ++i)
    {
        double px = path[i].x, py = path[i].y, pth = path[i].theta;
        bool free = is_free(px, py, pth, cfg);
        std::printf("  %-4zu  %7.3f  %7.3f  %7.3f  %-10s\n", i, px, py, pth, free ? "Free" : "COLLISION");
        if (!free)
            any_violation = true;
    }
    if (any_violation)
        std::printf("  *** WARNING: path has safety violations! ***\n");
    else
        std::printf("  All waypoints are collision-free (Rectangular Footprint).\n");
}

// ============================================================
// Navigation dataset generation
// ============================================================

// ============================================================
// Navigation dataset generation
// ============================================================

static bool directory_exists(const std::string &path)
{
    struct stat info;
    return (stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR));
}

static bool ensure_directory(const std::string &path)
{
    if (directory_exists(path))
        return true;

    if (mkdir(path.c_str(), 0775) == 0)
        return true;

    if (errno == EEXIST)
        return true;

    std::cerr << "ERROR: cannot create directory: " << path << "\n";
    return false;
}

static std::string get_navigation_dataset_root()
{
    const char *home = std::getenv("HOME");
    if (home == nullptr)
    {
        std::cerr << "ERROR: HOME environment variable is not set.\n";
        return "";
    }

    return std::string(home) +"/Single_Step_Position_Change/Navigation_Dataset";
}

static std::string indexed_filename(const std::string &prefix,
                                    int run_idx,
                                    const std::string &extension)
{
    return prefix + "_" + std::to_string(run_idx) + extension;
}

static bool copy_text_file(const std::string &src,
                           const std::string &dst)
{
    std::ifstream in(src);
    if (!in)
    {
        std::cerr << "ERROR: cannot open source file for copying: "
                  << src << "\n";
        return false;
    }

    std::ofstream out(dst);
    if (!out)
    {
        std::cerr << "ERROR: cannot open destination file for copying: "
                  << dst << "\n";
        return false;
    }

    out << in.rdbuf();
    return true;
}




static bool write_run_local_target_pose_file(const std::vector<Configuration> &path,
                                             const std::string &filepath)
{
    if (path.size() < 2)
    {
        std::cerr << "WARNING: path has fewer than 2 waypoints. No local target file saved.\n";
        return false;
    }

    std::ofstream f(filepath);
    if (!f)
    {
        std::cerr << "ERROR: cannot open " << filepath << " for writing\n";
        return false;
    }

    f << std::fixed << std::setprecision(6);

    // Design Choice [1]:
    // If path = p1 -> p2 -> p3 -> p4 -> p5,
    // this file stores:
    // p2
    // p3
    // p4
    // p5
    //
    // No extra goal row is added.

    for (std::size_t i = 0; i + 1 < path.size(); ++i)
    {
        const Configuration &target_pose = path[i + 1];

        f << target_pose.x << ", "
          << target_pose.y << ", "
          << target_pose.theta << "\n";
    }

    return true;
}

static bool write_run_local_obstacle_info_file(const std::vector<Configuration> &path,
                                               const PlannerConfig &cfg,
                                               const std::string &filepath)
{
    if (path.size() < 2)
    {
        std::cerr << "WARNING: path has fewer than 2 waypoints. No local obstacle file saved.\n";
        return false;
    }

    std::ofstream f(filepath);
    if (!f)
    {
        std::cerr << "ERROR: cannot open " << filepath << " for writing\n";
        return false;
    }

    f << std::fixed << std::setprecision(6);

    // Format:
    // One row per current robot pose.
    //
    // If path = p1 -> p2 -> p3 -> p4 -> p5,
    // this file stores:
    //
    // row 1: obstacle features relative to p1
    // row 2: obstacle features relative to p2
    // row 3: obstacle features relative to p3
    // row 4: obstacle features relative to p4
    //
    // Each row format:
    // [[obs0_rel_x, obs0_rel_y, obs0_dist], [obs1_rel_x, obs1_rel_y, obs1_dist], ...]

    for (std::size_t i = 0; i+1 < path.size(); ++i)
    {
        const Configuration &current_pose = path[i];

        f << "[";

        for (std::size_t j = 0; j < cfg.obstacles.size(); ++j)
        {
            const auto &ob = cfg.obstacles[j];

            double dx = ob.pose.x - current_pose.x;
            double dy = ob.pose.y - current_pose.y;

            double cos_th = std::cos(current_pose.theta);
            double sin_th = std::sin(current_pose.theta);

            double rel_x = dx * cos_th + dy * sin_th;
            double rel_y = -dx * sin_th + dy * cos_th;
            double rel_dist = std::sqrt(rel_x * rel_x + rel_y * rel_y);

            f << "["
              << rel_x << ", "
              << rel_y << ", "
              << rel_dist
              << "]";

            if (j + 1 < cfg.obstacles.size())
                f << ", ";
        }

        f << "]\n";
    }

    return true;
}

static bool write_run_current_pose_file(const std::vector<Configuration> &path,
                                        const std::string &filepath)
{
    if (path.size() < 2)
    {
        std::cerr << "WARNING: path has fewer than 2 waypoints. No current pose file saved.\n";
        return false;
    }

    std::ofstream f(filepath);
    if (!f)
    {
        std::cerr << "ERROR: cannot open " << filepath << " for writing\n";
        return false;
    }

    f << std::fixed << std::setprecision(6);

    // Format:
    // segment_id, current_x, current_y, current_theta
    //
    // If path = p1 -> p2 -> p3 -> p4 -> p5,
    // this file stores:
    // 1, p1
    // 2, p2
    // 3, p3
    // 4, p4

    for (std::size_t i = 0; i+1 < path.size(); ++i)
    {
        const Configuration &current_pose = path[i];

        f << current_pose.x << ", "
          << current_pose.y << ", "
          << current_pose.theta << "\n";
    }

    return true;
}


static bool write_run_local_goal_info_file(const std::vector<Configuration> &path,
                                           const PlannerConfig &cfg,
                                           const std::string &filepath)
{
    if (path.size() < 2)
    {
        std::cerr << "WARNING: path has fewer than 2 waypoints. No local goal info file saved.\n";
        return false;
    }

    std::ofstream f(filepath);
    if (!f)
    {
        std::cerr << "ERROR: cannot open " << filepath << " for writing\n";
        return false;
    }

    f << std::fixed << std::setprecision(6);

    // Format:
    // segment_id, goal_relative_x_robot_frame, goal_relative_y_robot_frame, goal_relative_euclidean_distance
    //
    // If path = p1 -> p2 -> p3 -> p4 -> p5,
    // this file stores goal information relative to:
    // segment 1: p1
    // segment 2: p2
    // segment 3: p3
    // segment 4: p4

    for (std::size_t i = 0; i+1  < path.size(); ++i)
    {
        
        const Configuration &current_pose = path[i];

        double dx = cfg.goal.x - current_pose.x;
        double dy = cfg.goal.y - current_pose.y;

        double cos_th = std::cos(current_pose.theta);
        double sin_th = std::sin(current_pose.theta);

        double goal_rel_x = dx * cos_th + dy * sin_th;
        double goal_rel_y = -dx * sin_th + dy * cos_th;
        double goal_rel_dist = std::sqrt(goal_rel_x * goal_rel_x +
                                         goal_rel_y * goal_rel_y);

        f << goal_rel_x << ", "
          << goal_rel_y << ", "
          << goal_rel_dist << "\n";
    }

    return true;
}

static bool write_run_plan_time_file(double planning_time_sec,
                                     const std::string &filepath)
{
    std::ofstream f(filepath);
    if (!f)
    {
        std::cerr << "ERROR: cannot open " << filepath << " for writing\n";
        return false;
    }

    f << std::fixed << std::setprecision(6);
    f << planning_time_sec << "\n";

    return true;
}

static bool save_navigation_dataset_for_run(const std::vector<Configuration> &path,
                                            const PlannerConfig &cfg,
                                            int run_idx,
                                            double planning_time_sec)
{
    if (run_idx <= 0)
    {
        std::cerr << "ERROR: run_idx must be positive. Example: --run_idx 1\n";
        return false;
    }

    std::string root_dir = get_navigation_dataset_root();
    if (root_dir.empty())
        return false;

    std::string curr_pose_dir =
    root_dir +
    "/Diffusion_Policy_Navigation/Diffusion_Curr_Robot_Pose";

    std::string target_dir =
        root_dir +
        "/Diffusion_Policy_Navigation/Diffusion_Target_Pose";

    std::string obs_dir =
        root_dir +
        "/Diffusion_Policy_Navigation/Diffusion_Local_Obstacle_Info";

    std::string goal_dir =
        root_dir +
        "/Diffusion_Policy_Navigation/Diffusion_Local_Goal_Info";

    std::string timestep_dir =
        root_dir +
        "/Diffusion_Policy_Navigation/Diffusion_Timestep";

    std::string query_dir =
        root_dir +
        "/Diffusion_Policy_Navigation/Diffusion_Query";

    if (!ensure_directory(root_dir))
        return false;
    if (!ensure_directory(curr_pose_dir))
        return false;
    if (!ensure_directory(target_dir))
        return false;
    if (!ensure_directory(obs_dir))
        return false;
    if (!ensure_directory(goal_dir))
        return false;
    if (!ensure_directory(timestep_dir))
        return false;
    if (!ensure_directory(query_dir))
    return false;

    std::string curr_pose_file =
        curr_pose_dir + "/diff_curr_pose_" + std::to_string(run_idx) + ".txt";

    std::string target_file =
        target_dir + "/diff_local_target_pose_" + std::to_string(run_idx) + ".txt";

    std::string obs_file =
        obs_dir + "/diff_local_obs_" + std::to_string(run_idx) + ".txt";

    std::string goal_file =
        goal_dir + "/diff_local_goal_info_" + std::to_string(run_idx) + ".txt";

    std::string plan_time_file =
        timestep_dir + "/diff_plan_time_" + std::to_string(run_idx) + ".txt";

    std::string query_file =
    query_dir + "/diff_query_" + std::to_string(run_idx) + ".cfg";

    if (!write_run_current_pose_file(path, curr_pose_file))
        return false;

   if (!write_run_local_target_pose_file(path, target_file))
        return false;

    if (!write_run_local_obstacle_info_file(path, cfg, obs_file))
        return false;

    if (!write_run_local_goal_info_file(path, cfg, goal_file))
        return false;

    if (!write_run_plan_time_file(planning_time_sec, plan_time_file))
        return false;

    if (!copy_text_file(cfg.query_file, query_file))
        return false;

    std::printf("Saved navigation dataset for run_idx=%d\n", run_idx);
    std::printf("  Current poses       : %s\n", curr_pose_file.c_str());
    std::printf("  Target poses        : %s\n", target_file.c_str());
    std::printf("  Local obstacle info : %s\n", obs_file.c_str());
    std::printf("  Local goal info     : %s\n", goal_file.c_str());
    std::printf("  Planning time       : %s\n", plan_time_file.c_str());
    std::printf("  Query config        : %s\n", query_file.c_str());

    return true;
}

// ============================================================
// Print configuration summary
// ============================================================
static void print_config(const PlannerConfig &cfg)
{
    std::printf("\n=== SE(2) RRT Blossom Planner ===\n");
    std::printf("  Start      : (%.3f, %.3f, %.3f)\n",
                cfg.start.x, cfg.start.y, cfg.start.theta);
    std::printf("  Goal       : (%.3f, %.3f, [Any])\n",
                cfg.goal.x, cfg.goal.y);
    std::printf("  Robot R    : %.3f m\n", cfg.robot_radius);
    std::printf("  Safety     : %.3f m\n", cfg.safety_margin);
    std::printf("  Inflation  : %.3f m\n", cfg.robot_radius + cfg.safety_margin);
    std::printf("  Step time  : %.1f s\n", cfg.step_time);
    std::printf("  Obstacles  : %zu\n", cfg.obstacles.size());
    for (std::size_t i = 0; i < cfg.obstacles.size(); ++i)
    {
        const auto &ob = cfg.obstacles[i];
        std::printf("    [%zu] centre=(%.3f, %.3f) th=%.3f  L=%.3f W=%.3f\n",
                    i, ob.pose.x, ob.pose.y, ob.pose.theta,
                    ob.length, ob.width);
    }
    std::printf("  Map        : X[%.1f, %.1f]  Y[%.1f, %.1f]\n",
                cfg.x_min, cfg.x_max, cfg.y_min, cfg.y_max);
    std::printf("  Goal Tol   : %.3f m\n\n", cfg.goal_tol);
}

// ============================================================
// Load both config files
// ============================================================
static bool load_configs(const std::string &path, PlannerConfig &cfg)
{
    if (path.empty()) return true;
    std::printf("Loading config: %s\n", path.c_str());
    if (!load_config_file(path, cfg))
    {
        std::cerr << "ERROR: cannot load config: " << path << "\n";
        return false;
    }
    cfg.start.theta = wrap_angle(cfg.start.theta);
    return true;
}




int main(int argc, char **argv)
{
    PlannerConfig cfg;
    std::string config_file = "planner.cfg";
    int run_idx = -1;
    int seed_override = -1;
    int diffusion_seed = 42;
    std::string output_dir_override;

    // Parse CLI for the master config only
    for (int i = 1; i < argc; ++i)
    {
        if (
            std::strcmp(argv[i], "--config") == 0 &&
            i + 1 < argc)
        {
            config_file = argv[++i];
        }
        else if (
            std::strcmp(argv[i], "--run_idx") == 0 &&
            i + 1 < argc)
        {
            run_idx = std::stoi(argv[++i]);
        }
        else if (
            std::strcmp(argv[i], "--seed") == 0 &&
            i + 1 < argc)
        {
            seed_override = std::stoi(argv[++i]);
        }
        else if (
            std::strcmp(argv[i], "--diffusion_seed") == 0 &&
            i + 1 < argc)
        {
            diffusion_seed = std::stoi(argv[++i]);
        }
        else if (
            std::strcmp(argv[i], "--output_dir") == 0 &&
            i + 1 < argc)
        {
            output_dir_override = argv[++i];
        }
        else if (
            std::strcmp(argv[i], "-h") == 0 ||
            std::strcmp(argv[i], "--help") == 0)
        {
            std::printf(
                "Usage: %s [options]\n"
                "\n"
                "Options:\n"
                "  --config <path>             Master config\n"
                "  --run_idx <integer>         Experiment index\n"
                "  --seed <integer>            Fixed RRT random seed\n"
                "  --diffusion_seed <integer>  Fixed diffusion seed\n"
                "  --output_dir <path>         Override output directory\n"
                "  -h, --help                  This message\n",
                argv[0]);

            return 0;
        }
        else
        {
            std::cerr
                << "ERROR: unknown or incomplete argument: "
                << argv[i] << "\n";
            return 1;
        }
    }

    if (diffusion_seed < 0)
    {
        std::cerr
            << "ERROR: diffusion seed must be non-negative.\n";
        return 1;
    }
    
    ////--- To validate for only positive running index-----------------///
    if (run_idx <= 0)
    {
        std::cerr << "ERROR: Please provide a positive run index.\n";
        std::cerr << "Example:\n";
        std::cerr << "  ./pla2exec --config planner.cfg --run_idx 1\n";
        return 1;
    }

    // Load static config first to get query_file path
    if (!load_configs(config_file, cfg))
        return 1;

    // Now load the query config (start/goal/obstacles)
    if (!load_configs(cfg.query_file, cfg))
        return 1;

    // Command-line seed has priority over planner.cfg and query.cfg.
    if (seed_override >= 0)
    {
        cfg.seed = seed_override;
    }


    // Use command-line run index for all output files
    // cfg.waypoints_out = indexed_filename("se2_waypoints", run_idx, ".txt");
    // cfg.controls_out = indexed_filename("controls", run_idx, ".txt");
    // std::string rrt_tree_out = indexed_filename("rrt_tree", run_idx, ".txt");

    // Use command-line run index for all output files
    // Save planner outputs inside Navigation_Dataset subfolders

    std::string dataset_root =
    get_navigation_dataset_root();

    if (dataset_root.empty())
    {
        return 1;
    }

    std::string rrt_tree_out;

    if (!output_dir_override.empty())
    {
        // =====================================================
        // Fixed-seed comparison mode
        //
        // All outputs are written directly inside the current
        // Comparison_Values diffusion run directory.
        // =====================================================
        if (!ensure_directory(output_dir_override))
        {
            std::cerr
                << "ERROR: cannot create comparison output directory: "
                << output_dir_override << "\n";
            return 1;
        }

        cfg.waypoints_out =
            output_dir_override + "/" +
            indexed_filename(
                "Diffusion_se2_waypoints",
                run_idx,
                ".txt");

        cfg.controls_out =
            output_dir_override + "/" +
            indexed_filename(
                "Diffusion_controls",
                run_idx,
                ".txt");

        rrt_tree_out =
            output_dir_override + "/" +
            indexed_filename(
                "Diffusion_rrt_tree",
                run_idx,
                ".txt");

        std::cout
            << "Using diffusion comparison output directory: "
            << output_dir_override << "\n";
    }
    else
    {
        // =====================================================
        // Standalone mode
        //
        // Retain the previous behaviour when this planner is run
        // manually without --output_dir.
        // =====================================================
        const std::string waypoints_dir =
            dataset_root +
            "/Diffusion_Policy_Navigation/"
            "Diffusion_waypoints";

        const std::string controls_dir =
            dataset_root +
            "/Diffusion_Policy_Navigation/"
            "Diffusion_controls";

        const std::string planning_tree_dir =
            dataset_root +
            "/Diffusion_Policy_Navigation/"
            "Diffusion_planning_tree";

        const std::string planner_logs_dir =
            dataset_root +
            "/Diffusion_Policy_Navigation/"
            "Diffusion_planner_logs";

        if (!ensure_directory(dataset_root))
            return 1;

        if (!ensure_directory(waypoints_dir))
            return 1;

        if (!ensure_directory(controls_dir))
            return 1;

        if (!ensure_directory(planning_tree_dir))
            return 1;

        if (!ensure_directory(planner_logs_dir))
            return 1;

        cfg.waypoints_out =
            waypoints_dir + "/" +
            indexed_filename(
                "Diffusion_se2_waypoints",
                run_idx,
                ".txt");

        cfg.controls_out =
            controls_dir + "/" +
            indexed_filename(
                "Diffusion_controls",
                run_idx,
                ".txt");

        rrt_tree_out =
            planning_tree_dir + "/" +
            indexed_filename(
                "Diffusion_rrt_tree",
                run_idx,
                ".txt");

        std::cout
            << "Using standalone diffusion output folders under: "
            << dataset_root << "\n";
    }



    // Initialize random seed from config or time
    unsigned int final_seed;
    if (cfg.seed != -1) {
        final_seed = static_cast<unsigned int>(cfg.seed);
    } else {
        final_seed = static_cast<unsigned int>(std::time(nullptr));
    }
    std::srand(final_seed);

    print_config(cfg);

    // ── Cleanup old files ──
    std::remove(cfg.waypoints_out.c_str());
    std::remove(cfg.controls_out.c_str());
    std::remove(rrt_tree_out.c_str());
    //std::remove(current_controls_file.c_str());

    // ── Pre-search Safety Check ──
    if (!is_free(cfg.start.x, cfg.start.y, cfg.start.theta, cfg)) {
        std::printf("ERROR: Start position is in collision!\n");
        return 0; // Return 0 so viz can show the setup
    }
    if (!is_free(cfg.goal.x, cfg.goal.y, 0.0, cfg)) {
        std::printf("ERROR: Goal position (%.3f, %.3f) is in collision at θ=0!\n",
                    cfg.goal.x, cfg.goal.y);
        return 0; // Return 0 so viz can show the setup
    }

    // ========================================================
    // Start the persistent Python diffusion-policy server once
    // ========================================================

    const std::string diffusion_policy_directory =
        dataset_root +
        "/Diffusion_Policy_Navigation";

    const std::string python_executable =
        diffusion_policy_directory +
        "/nav_diff_env/bin/python";

    const std::string diffusion_script =
        diffusion_policy_directory +
        "/Eval_DM_navg.py";
    // Prevent the planner from being terminated by SIGPIPE
    // if the Python process closes unexpectedly.
    std::signal(SIGPIPE, SIG_IGN);


    std::cout
        << "Starting diffusion server with seed "
        << diffusion_seed
        << ".\n";

    if (!diffusion_python_process.start(
            python_executable,
            diffusion_script,
            "cuda",
            static_cast<unsigned int>(
                diffusion_seed)))
        {
            std::cerr
                << "ERROR: failed to start the "
                << "Python diffusion server.\n";

            return 1;
    }

    // ── Plan ──
    int num_runs = AOX_RUNS;
    RRTResult best_overall;
    best_overall.cost = 1e18;

    std::printf("\n=== Executing %d RRT search attempts ===\n", num_runs);
    for (int i = 0; i < num_runs; ++i) {
        unsigned int run_seed = (cfg.seed == -1) ? (final_seed + i) : (unsigned int)cfg.seed;
        RRTResult res = run_rrt(cfg, run_seed);
        print_machine_readable_diagnostics(res);

        if (res.success) {
            std::printf("  Run %2d: cost = %10.4f (seed=%u)\n", i, res.cost, run_seed);
            if (res.cost < best_overall.cost) {
                best_overall = std::move(res);
            }
        } else {
            std::printf("  Run %2d: failed\n", i);
        }
        if (cfg.seed != -1) break; // Don't repeat if seed is fixed
    }

    if (!best_overall.success) {
        std::printf("RRT search failed to find a path in all attempts.\n");
        return 0; // Still return 0 so user can see the setup in viz
    }

    double planning_time_sec = best_overall.time_to_best_goal_sec;

    std::printf("Best path found at iter %d in %.6f seconds within the winning RRT run.\n",
                best_overall.best_goal_iter,
                planning_time_sec);

    std::printf("\nGlobally best cost: %.4f (seed=%u)\n", best_overall.cost, best_overall.seed);
    std::printf("Best run rejections: col=%ld, dup=%ld\n", best_overall.rej_col, best_overall.rej_dup);

    write_rrt_tree(best_overall.tree, best_overall.seed, rrt_tree_out);
    std::vector<Configuration> path = best_overall.path;

    
    collision_check_path(path, cfg);


    if (!write_waypoints_file(path, cfg.waypoints_out))
    return 1;

    // ── Save Navigation Dataset for this run index ──
    // if (!save_navigation_dataset_for_run(path, cfg, run_idx, planning_time_sec))
    // return 1;

    // ── Generate Controls ──
    cfg.step_time = 1.0;
    auto cmds = compute_controls(path, cfg.step_time);

    std::printf("\n=== Controls (%zu commands, step_time=%.1fs) ===\n",
                cmds.size(), cfg.step_time);
    std::printf("  %-4s  %10s  %10s  %10s  %8s\n",
                "Seg", "vx(m/s)", "vy(m/s)", "vth(rad/s)", "dur(s)");
    for (std::size_t i = 0; i < cmds.size(); ++i)
    {
        std::printf("  %-4zu  %10.4f  %10.4f  %10.4f  %8.1f\n",
                    i, cmds[i].vx, cmds[i].vy, cmds[i].vtheta, cmds[i].duration);
    }

    if (!write_controls_file(
        cmds,
        cfg.controls_out))
    {
        return 1;
    }


    diffusion_python_process.stop();

    return 0;

}