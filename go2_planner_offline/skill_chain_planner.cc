// planner.cc — SE(2) RRT Blossom Planner + Controller
//
// Configuration Files:
//   planner.cfg  — Master settings (search budget, robot parameters, paths)
//   query.cfg    — Dynamic problem settings (start/goal/obstacles)
//
// Execution:  ./astar --config planner.cfg
//             (Writes waypoints and controls as specified in config)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <ctime>


#define MAX_ITERS 2000
#define AOX_RUNS 100
#define GOAL_BIAS_PERCENT 0
#define ROT_ERR 10

// ============================================================
// Discrete robot mode (skill-chain planner)
// ============================================================
enum class RobotMode
{
    STAND,
    HANDSTAND
};

// ============================================================
// Skill on the tree edge parent -> child (skill-chain planner)
// ============================================================
enum class SkillPrimitive
{
    START,               // root node, no incoming edge
    STAND_TO_HANDSTAND,  // fixed learned transition (no q_rand)
    HANDSTAND_MOVE,      // learned affine model steered toward q_rand
    HANDSTAND_TO_STAND   // fixed learned transition (no q_rand)
};

// HANDSTAND footprint (STAND footprint still comes from planner.cfg)
static constexpr double HANDSTAND_LENGTH = 0.135;//0.385;   // m
static constexpr double HANDSTAND_WIDTH  = 0.18;//0.27;   // m

// ============================================================
// Learned skill-transition models (deterministic MEAN values)
//   Delta vector order: [delta_x_body (m), delta_y_body (m), delta_theta (rad)]
//   Translation is expressed in the robot body frame at the START of the skill.
//   (Std-devs / covariances intentionally not used yet.)
// ============================================================

// A. Stand -> HandStand (65 trials)
static constexpr double SKILL_S2H_DELTA[3] = {
    +0.097632147996,   // delta_x_body
    -0.062812786308,   // delta_y_body
    -0.208528304000    // delta_theta
};

// B. HandStand -> Stand (65 trials)
static constexpr double SKILL_H2S_DELTA[3] = {
    -0.084099647866,   // delta_x_body
    -0.048083618562,   // delta_y_body
    -0.032417929738    // delta_theta
};

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

// HandStand+Move command limits (same as go2_navg_exec.cc)
//   Deadbands: |v| below these is set to 0 (applied AFTER clamping)
static constexpr double HSMOVE_MIN_V_LINEAR = 0.08;   // m/s   (vx, vy)
static constexpr double HSMOVE_MIN_V_YAW    = 0.1;   // rad/s (vyaw)
//   Safety clamps: vx in [0, +MAX] (no reverse); vy, vyaw in [-MAX, +MAX]
static constexpr double HSMOVE_MAX_VX   = 0.6;        // m/s
static constexpr double HSMOVE_MAX_VY   = 0.4;        // m/s
static constexpr double HSMOVE_MAX_VYAW = 0.5;        // rad/s


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

double Configuration::weight_theta = 1.0;   // overwritten once in main() from cfg.w_theta (planner.cfg)

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
    double x_min = -0.25;
    double x_max = 1.5;
    double y_min = -0.5;
    double y_max = 2.4;

    // Search settings
    int blossom_number = 10;
    int seed = -1;
    double w_theta = 1;
    bool smooth_path = false;
    bool allow_reverse = false;
    bool allow_lateral = false;

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
    std::string waypoints_out = "skill_chain_se2_waypoints.txt";
    std::string velocity_commands_out = "skill_chain_velocity_commands.txt";
    std::string query_file = "query.cfg";

    // Poses
    Configuration start;
    Configuration goal;
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

// ------------------------------------------------------------
// Mode-dependent robot footprint
// ------------------------------------------------------------
struct Footprint
{
    double length;
    double width;
};

static inline Footprint footprint_for_mode(RobotMode mode, const PlannerConfig &cfg)
{
    switch (mode)
    {
    case RobotMode::HANDSTAND:
        return {HANDSTAND_LENGTH, HANDSTAND_WIDTH};
    case RobotMode::STAND:
    default:
        return {cfg.robot_length, cfg.robot_width};
    }
}

// ------------------------------------------------------------
// Generic body-frame transition: q_next = q (+) Delta_body
// Shared later by Stand->HandStand, HandStand+Move, HandStand->Stand.
// ------------------------------------------------------------
static inline Configuration apply_body_frame_transition(const Configuration &q,
                                                        double delta_x_body,
                                                        double delta_y_body,
                                                        double delta_theta)
{
    const double c = std::cos(q.theta);
    const double s = std::sin(q.theta);

    const double dx_world = c * delta_x_body - s * delta_y_body;
    const double dy_world = s * delta_x_body + c * delta_y_body;
    
    //-- getting the handstand pose in world frame (x, y, theta) after applying the body-frame delta//
    Configuration q_next;
    q_next.x = q.x + dx_world;
    q_next.y = q.y + dy_world;
    q_next.theta = wrap_angle(q.theta + delta_theta);
    return q_next;
}

// ============================================================
// STEP 2: Learned-skill math (pure functions, no tree / no collision)
// ============================================================

// HandStand+Move command  V = [vx, vy, vyaw]^T  (no limits attached)
struct HSMoveCommand
{
    double vx = 0.0;
    double vy = 0.0;
    double vyaw = 0.0;
};

// Relative displacement expressed in a robot BODY frame
// = [delta_x_body (m), delta_y_body (m), delta_theta (rad)]
struct BodyDelta
{
    double dx_body = 0.0;
    double dy_body = 0.0;
    double dtheta = 0.0;
};

// Output of the composed HandStand+Move prediction pipeline
struct HSMovePrediction
{
    HSMoveCommand raw_cmd;          // inverse-model output before clamping
    HSMoveCommand cmd;              // command actually used (after clamp + deadband)
    Configuration q_pred;           // forward-model landing pose
    bool was_clamped = false;       // true if any component was clamped
    double distance_before = 0.0;   // current.distance(q_rand)
    double distance_after = 0.0;    // q_pred.distance(q_rand)
    bool makes_progress = false;    // distance_after < distance_before (informational only)
};

// ------------------------------------------------------------
// Displacement from `current` to `target`, expressed in the CURRENT body frame
// ------------------------------------------------------------
static inline BodyDelta body_frame_delta_to_target(const Configuration &current,
                                                   const Configuration &target)
{
    const double c = std::cos(current.theta);
    const double s = std::sin(current.theta);

    const double dx_world = target.x - current.x;
    const double dy_world = target.y - current.y;

    BodyDelta d;
    d.dx_body =  c * dx_world + s * dy_world;
    d.dy_body = -s * dx_world + c * dy_world;
    d.dtheta  = wrap_angle(target.theta - current.theta);
    return d;
}

// ------------------------------------------------------------
// Stand -> HandStand: fixed learned mean transition (not steerable)
// ------------------------------------------------------------
static inline Configuration predict_stand_to_handstand(const Configuration &q_stand)
{
    return apply_body_frame_transition(q_stand,
                                       SKILL_S2H_DELTA[0],
                                       SKILL_S2H_DELTA[1],
                                       SKILL_S2H_DELTA[2]);
}

// ------------------------------------------------------------
// HandStand -> Stand: fixed learned mean transition (not steerable)
// ------------------------------------------------------------
static inline Configuration predict_handstand_to_stand(const Configuration &q_handstand)
{
    return apply_body_frame_transition(q_handstand,
                                       SKILL_H2S_DELTA[0],
                                       SKILL_H2S_DELTA[1],
                                       SKILL_H2S_DELTA[2]);
}

// ------------------------------------------------------------
// HandStand+Move RAW inverse model:  V = B_inv * (Delta_desired - C)
// No clamping here. MAX_V is NOT used.
// ------------------------------------------------------------
static inline HSMoveCommand hsmove_inverse_raw(const BodyDelta &desired)
{
    const double a0 = desired.dx_body - SKILL_HSMOVE_C[0];
    const double a1 = desired.dy_body - SKILL_HSMOVE_C[1];
    const double a2 = desired.dtheta  - SKILL_HSMOVE_C[2];

    HSMoveCommand v;
    v.vx   = SKILL_HSMOVE_B_INV[0][0] * a0 + SKILL_HSMOVE_B_INV[0][1] * a1 + SKILL_HSMOVE_B_INV[0][2] * a2;
    v.vy   = SKILL_HSMOVE_B_INV[1][0] * a0 + SKILL_HSMOVE_B_INV[1][1] * a1 + SKILL_HSMOVE_B_INV[1][2] * a2;
    v.vyaw = SKILL_HSMOVE_B_INV[2][0] * a0 + SKILL_HSMOVE_B_INV[2][1] * a1 + SKILL_HSMOVE_B_INV[2][2] * a2;
    return v;
}

// ------------------------------------------------------------
// HandStand+Move admissible control domain:
//   vx   in [0, HSMOVE_MAX_VX]                  (no reverse: negative vx -> 0)
//   vy   in [-HSMOVE_MAX_VY,   +HSMOVE_MAX_VY]   (signed, lateral allowed)
//   vyaw in [-HSMOVE_MAX_VYAW, +HSMOVE_MAX_VYAW] (signed, both yaw directions)
// Any modification sets is_clamped. Deadband is applied afterwards, separately.
// ------------------------------------------------------------
static inline HSMoveCommand clamp_hsmove_command(const HSMoveCommand &raw, bool &is_clamped)
{
    HSMoveCommand v = raw;
    is_clamped = false;

    if (v.vx < 0.0) {
        v.vx = 0.0;                 // reverse is outside the HSMove domain
        is_clamped = true;
    } else if (v.vx > HSMOVE_MAX_VX) {
        v.vx = HSMOVE_MAX_VX;
        is_clamped = true;
    }
    if (std::fabs(v.vy) > HSMOVE_MAX_VY) {
        v.vy = std::copysign(HSMOVE_MAX_VY, v.vy);
        is_clamped = true;
    }
    if (std::fabs(v.vyaw) > HSMOVE_MAX_VYAW) {
        v.vyaw = std::copysign(HSMOVE_MAX_VYAW, v.vyaw);
        is_clamped = true;
    }
    return v;
}

// ------------------------------------------------------------
// Deadband (applied AFTER clamping): small components are zeroed
// ------------------------------------------------------------
static inline HSMoveCommand deadband_hsmove_command(const HSMoveCommand &in)
{
    HSMoveCommand v = in;
    if (std::fabs(v.vx)   < HSMOVE_MIN_V_LINEAR) v.vx   = 0.0;
    if (std::fabs(v.vy)   < HSMOVE_MIN_V_LINEAR) v.vy   = 0.0;
    if (std::fabs(v.vyaw) < HSMOVE_MIN_V_YAW)    v.vyaw = 0.0;
    return v;
}

// ------------------------------------------------------------
// HandStand+Move forward model in the body frame:  Delta = B * V + C
// Note: a zero command still yields Delta = C (learned bias).
// ------------------------------------------------------------
static inline BodyDelta hsmove_forward_body_delta(const HSMoveCommand &v)
{
    BodyDelta d;
    d.dx_body = SKILL_HSMOVE_B[0][0] * v.vx + SKILL_HSMOVE_B[0][1] * v.vy + SKILL_HSMOVE_B[0][2] * v.vyaw + SKILL_HSMOVE_C[0];
    d.dy_body = SKILL_HSMOVE_B[1][0] * v.vx + SKILL_HSMOVE_B[1][1] * v.vy + SKILL_HSMOVE_B[1][2] * v.vyaw + SKILL_HSMOVE_C[1];
    d.dtheta  = SKILL_HSMOVE_B[2][0] * v.vx + SKILL_HSMOVE_B[2][1] * v.vy + SKILL_HSMOVE_B[2][2] * v.vyaw + SKILL_HSMOVE_C[2];
    return d;
}

// ------------------------------------------------------------
// HandStand+Move forward prediction in the world frame
// ------------------------------------------------------------
static inline Configuration predict_handstand_move(const Configuration &q,
                                                   const HSMoveCommand &v)
{
    const BodyDelta d = hsmove_forward_body_delta(v);
    return apply_body_frame_transition(q, d.dx_body, d.dy_body, d.dtheta);
}

// ------------------------------------------------------------
// Composed pipeline:
//   target -> body delta -> raw inverse V -> clamp -> deadband -> forward model -> landing pose
// No tree insertion, no goal check, no collision check, no acceptance decision.
// Must stay LAST in Block P1 (it calls the helpers above).
// ------------------------------------------------------------
static inline HSMovePrediction predict_handstand_move_toward(const Configuration &q_current,
                                                             const Configuration &q_rand)
{
    HSMovePrediction out;

    const BodyDelta desired = body_frame_delta_to_target(q_current, q_rand);
    out.raw_cmd = hsmove_inverse_raw(desired);
    const HSMoveCommand clamped = clamp_hsmove_command(out.raw_cmd, out.was_clamped);
    out.cmd = deadband_hsmove_command(clamped);

    out.q_pred = predict_handstand_move(q_current, out.cmd);

    out.distance_before = q_current.distance(q_rand);
    out.distance_after  = out.q_pred.distance(q_rand);
    out.makes_progress  = (out.distance_after < out.distance_before);
    return out;
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
                                            RobotMode mode,
                                           const PlannerConfig &cfg)
{   
    const Footprint fp = footprint_for_mode(mode, cfg);
    for (const auto &ob : cfg.obstacles)
        if (rect_intersects_rect(px, py, pth, fp.length, fp.width,
                                 ob.pose.x, ob.pose.y, ob.pose.theta, ob.length, ob.width,
                                 cfg.safety_margin))
            return true;
    return false;
}

// ============================================================
// Collision checking
// ============================================================
static bool is_free(double px, double py, double pth, RobotMode mode, const PlannerConfig &cfg)
{
    // Check map bounds
    const Footprint fp = footprint_for_mode(mode, cfg);
    double RL = fp.length, RW = fp.width;
    double r = std::max(RL, RW) / 2.0; // Conservative radius for simple bounds check


    //Your code checks the negation of that:

    // (1) if left side crosses boundary
    // (2) or right side crosses boundary
    // (3) or bottom side crosses boundary
    // (4) or top side crosses boundary
    if (px - r < cfg.x_min || px + r > cfg.x_max ||
        py - r < cfg.y_min || py + r > cfg.y_max)
        return false;
    return !point_hits_any_obstacle(px, py, pth,mode, cfg);
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
            else if (key == "velocity_commands_out")
                cfg.velocity_commands_out = val;
            else if (key == "query_file")
                cfg.query_file = val;
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
// Collision checking
// ============================================================
// --- Collision Check (Robotic Footprint) ---
// ------------------------------------------------------------
// STEP 2: Collision check for ONE learned-skill edge.
// x, y interpolated linearly; theta interpolated along the shortest arc.
// Sampling: <= 0.01 m translation (as segment_free) and
//           <= 10 deg rotation (as rotation_free) between samples.
// Sampled approximation between start and predicted end pose.
// ------------------------------------------------------------
static inline bool skill_edge_free(const Configuration &q_start,
                                   const Configuration &q_end,
                                   RobotMode mode,
                                   const PlannerConfig &cfg)
{
    const double pos_res = 0.01;            // m   (same as segment_free)
    const double ang_res = M_PI / 18.0;     // rad (same as rotation_free)

    const double dx = q_end.x - q_start.x;
    const double dy = q_end.y - q_start.y;
    const double dtheta = wrap_angle(q_end.theta - q_start.theta);
    const double d = std::sqrt(dx * dx + dy * dy);

    const int n_pos = static_cast<int>(std::ceil(d / pos_res));
    const int n_ang = static_cast<int>(std::ceil(std::fabs(dtheta) / ang_res));
    const int n = std::max(2, std::max(n_pos, n_ang));

    for (int i = 0; i <= n; ++i)
    {
        const double t = static_cast<double>(i) / n;
        const double px  = q_start.x + t * dx;
        const double py  = q_start.y + t * dy;
        const double pth = wrap_angle(q_start.theta + t * dtheta);
        if (!is_free(px, py, pth, mode, cfg))
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
    RobotMode mode = RobotMode::STAND;   // NEW: discrete mode is part of the planner state
    SkillPrimitive incoming_skill = SkillPrimitive::START;  // skill on the edge parent -> this node
    HSMoveCommand incoming_cmd;          // final clamped + deadbanded HandStand+Move command (zero unless HANDSTAND_MOVE)
};

// Skill-chain goal: a STAND node whose XY position is within goal_tol of cfg.goal.
// Normally reached only via the fixed HandStand->Stand transition. Heading is free.
static bool is_goal_satisfied(const Configuration &p, RobotMode mode, const PlannerConfig &cfg)
{
    if (mode != RobotMode::STAND)
        return false;
    return p.distance(cfg.goal, {0, 1}) <= cfg.goal_tol;
}

static void update_best_goal(const std::vector<RRTNode> &tree, int idx,
                             const PlannerConfig &cfg, int &goal_idx,
                             double &best_goal_cost, int iter)
{
    if (is_goal_satisfied(tree[idx].point, tree[idx].mode, cfg))
    {
        if (tree[idx].cost < best_goal_cost)
        {
            best_goal_cost = tree[idx].cost;
            goal_idx = idx;
            std::cout << "New best goal cost found: " << best_goal_cost
                      << " (node " << idx << ") at iter " << iter << "\n";
        }
    }
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
                         RobotMode mode, int &best_idx, double &min_dist_sq, int depth = 0)
{
    if (root_idx == -1) return;

    const auto &kd = nodes[root_idx];
    const auto &rrt_node = tree[kd.idx];

    // 0. Mode filter: wrong-mode nodes are never candidates, but their KD subtrees
    //    are still traversed below (the KD layout is x,y only, mode-agnostic).
    // 1. Goal optimization check (optional for find_nearest, usually skip for duplicates)
    if (rrt_node.mode == mode &&
        (goal_idx == -2 || !(is_goal_satisfied(rrt_node.point, rrt_node.mode, cfg) && goal_idx != -1)))
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

    kdtree_query(nodes, near, q_sample, tree, cfg, goal_idx, mode, best_idx, min_dist_sq, depth + 1);

    double diff = std::fabs(q_val - kd_val);
    if (diff * diff < min_dist_sq)
    {
        kdtree_query(nodes, far, q_sample, tree, cfg, goal_idx, mode, best_idx, min_dist_sq, depth + 1);
    }
}

// Nearest node (SE(2) metric) among nodes whose mode == `mode`; -1 if none.
static int find_nearest(const std::vector<RRTNode> &tree, const std::vector<KDNode> &kd_nodes, int kd_root,
                        const Configuration &q_sample, const PlannerConfig &cfg, int goal_idx,
                        RobotMode mode)
{
    int best_idx = -1;
    double min_dist_sq = 1e36;
    kdtree_query(kd_nodes, kd_root, q_sample, tree, cfg, goal_idx, mode, best_idx, min_dist_sq);
    return best_idx;
}

static int find_duplicate(const std::vector<RRTNode> &tree, const std::vector<KDNode> &kd_nodes, int kd_root,
                          const Configuration &q, RobotMode mode)
{
    if (kd_root == -1) return -1;

    // Use KD-tree to find nearest SAME-MODE node, then check if it's within tolerance
    // (same pose + different mode is NOT a duplicate)
    int best_idx = -1;
    double min_dist_sq = 1e36;
    PlannerConfig dummy_cfg; // Goal optimization not needed for duplicate check
    kdtree_query(kd_nodes, kd_root, q, tree, dummy_cfg, -2, mode, best_idx, min_dist_sq);

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
                   const Configuration &point, RobotMode mode, int parent, double cost,
                   SkillPrimitive skill, const HSMoveCommand &cmd, long &rej_dup)
{
    int dup_idx = find_duplicate(tree, kd_nodes, kd_root, point, mode);
    if (dup_idx != -1) {
        if (cost < tree[dup_idx].cost) {
            // Rewire: the incoming-edge metadata moves together with parent/cost
            tree[dup_idx].cost = cost;
            tree[dup_idx].parent = parent;
            tree[dup_idx].incoming_skill = skill;
            tree[dup_idx].incoming_cmd = cmd;
        }
        rej_dup++;
        return dup_idx;
    }
    tree.push_back({point, parent, cost, mode, skill, cmd});
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
    std::vector<RRTNode> path_nodes;   // same states as `path`, plus mode / incoming skill / HSMove command
    std::vector<RRTNode> tree;
    unsigned int seed;
    long rej_dist = 0, rej_rot = 0, rej_col = 0, rej_dup = 0, rej_cost = 0;
};

// ============================================================
// RRT search with AO-RRT Cost-Bounding
// ============================================================
static RRTResult run_rrt(const PlannerConfig &cfg, unsigned int seed, double global_cost_bound = 1e18)
{   
    RRTResult result;
    result.seed = seed;
    std::srand(seed);

    // Configuration::weight_theta (= cfg.w_theta) is set once in main() after the configs load.

    //-- tree

    // --(a) Stores the actual RRT nodes.

    //-- (b)Each RRTNode likely contains:

    // -- (1) configuration point
    //-- (2) parent index
    //-- (3) accumulated path cost

    //-- (4) So each node is a state:

    // -- (a) qi=(xi,yi,θi)
    //-- (b) with parent:parent(i)


    //--This lets you reconstruct the final path by walking backward from goal to start.---//

    std::vector<RRTNode> tree;
    tree.reserve(MAX_ITERS);

    //-- kd_nodes--> This is probably the KD-tree structure used for fast nearest-neighbor lookup.--//
    std::vector<KDNode> kd_nodes;
    kd_nodes.reserve(MAX_ITERS);
    int kd_root = -1;

    //goal_idx = -1 means no goal-reaching node has been found yet
    // best_goal_cost stores the best path cost to goal found so far
    int goal_idx = -1;
    double best_goal_cost = global_cost_bound;

    long rej_dist = 0, rej_rot = 0, rej_col = 0, rej_dup = 0, rej_cost = 0;

    // (1) Root: STAND at cfg.start
    const int root_idx = add_node(tree, kd_nodes, kd_root, cfg.start, RobotMode::STAND, -1, 0.0,
                                  SkillPrimitive::START, HSMoveCommand{}, rej_dup);

    // (2) Fixed Stand -> HandStand, ONCE per run. Does NOT use q_rand.
    //     Edge checked conservatively with the larger HANDSTAND footprint;
    //     landing pose checked with the HANDSTAND footprint.
    // provides q_handstand point in world frame-----//
    const Configuration q_handstand = predict_stand_to_handstand(cfg.start);
    if (!skill_edge_free(cfg.start, q_handstand, RobotMode::HANDSTAND, cfg) ||
        !is_free(q_handstand.x, q_handstand.y, q_handstand.theta, RobotMode::HANDSTAND, cfg))
    {
        std::printf("  Stand->HandStand transition from start is in collision; run fails.\n");
        return result;
    }
    add_node(tree, kd_nodes, kd_root, q_handstand, RobotMode::HANDSTAND, root_idx,
             cfg.start.distance(q_handstand),
             SkillPrimitive::STAND_TO_HANDSTAND, HSMoveCommand{}, rej_dup);

    // (3) Random-sample loop: HandStand+Move is the ONLY q_rand-driven expansion
    for (int iter = 0; iter < MAX_ITERS; ++iter)
    {
        bool sampling_goal = false;
        const Configuration q_rand = sample(cfg, iter, GOAL_BIAS_PERCENT, sampling_goal);

        // Nearest HANDSTAND node only (the STAND root / STAND goal nodes are never steered)
        const int near_idx = find_nearest(tree, kd_nodes, kd_root, q_rand, cfg, goal_idx, RobotMode::HANDSTAND);
        if (near_idx == -1)
            continue;
        const RRTNode near = tree[near_idx];   // copy: add_node's push_back can invalidate references

        // q_rand -> body delta -> inverse model -> clamp -> deadband -> forward model -> q_pred
        const HSMovePrediction pred = predict_handstand_move_toward(near.point, q_rand);
        if (!pred.makes_progress) {
            rej_dist++;
            continue;
        }

        // AO-RRT Cost Pruning: Prune branches that exceed the best goal cost found so far
        const double h_cost = near.cost + near.point.distance(pred.q_pred);
        if (h_cost >= best_goal_cost) {
            rej_cost++;
            continue;
        }

        if (!skill_edge_free(near.point, pred.q_pred, RobotMode::HANDSTAND, cfg)) {
            rej_col++;
            continue;
        }

        // Insert the FORWARD-MODEL landing pose (not q_rand) together with the final (clamped + deadbanded) command
        const int h_idx = add_node(tree, kd_nodes, kd_root, pred.q_pred, RobotMode::HANDSTAND, near_idx, h_cost,
                                   SkillPrimitive::HANDSTAND_MOVE, pred.cmd, rej_dup);

        // (4) Fixed HandStand -> Stand recovery test from this HandStand node. Does NOT use q_rand.
        //     Edge checked conservatively with the larger STAND footprint.
        const Configuration q_h = tree[h_idx].point;
        const double q_h_cost = tree[h_idx].cost;
        const Configuration q_stand = predict_handstand_to_stand(q_h);

        if (!skill_edge_free(q_h, q_stand, RobotMode::STAND, cfg) ||
            !is_free(q_stand.x, q_stand.y, q_stand.theta, RobotMode::STAND, cfg))
            continue;   // no recovery here; q_h stays in the tree for further HandStand+Move

        // Goal decision uses the PREDICTED Stand landing pose (XY only), not q_h
        const double goal_error = q_stand.distance(cfg.goal, {0, 1});
        if (goal_error > cfg.goal_tol)
            continue;

        const double s_cost = q_h_cost + q_h.distance(q_stand);
        if (s_cost >= best_goal_cost) {
            rej_cost++;
            continue;
        }

        const int s_idx = add_node(tree, kd_nodes, kd_root, q_stand, RobotMode::STAND, h_idx, s_cost,
                                   SkillPrimitive::HANDSTAND_TO_STAND, HSMoveCommand{}, rej_dup);
        update_best_goal(tree, s_idx, cfg, goal_idx, best_goal_cost, iter);
    }

    result.rej_dist = rej_dist;
    result.rej_rot = rej_rot;
    result.rej_col = rej_col;
    result.rej_dup = rej_dup;
    result.rej_cost = rej_cost;

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
            result.path_nodes.push_back(tree[idx]);   // keeps mode, incoming_skill, incoming_cmd
        }
        result.tree = std::move(tree);
    }

    return result;
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
// Write HandStand+Move velocity commands to file (vx, vy, vyaw per line)
//   One row per HANDSTAND_MOVE edge, in final-path order. The command on
//   node i is the stored (clamped + deadbanded) command of edge (i-1) -> i.
//   START / STAND_TO_HANDSTAND / HANDSTAND_TO_STAND edges are not written.
// ============================================================
static bool write_velocity_commands_file(const std::vector<RRTNode> &path_nodes,
                                         const std::string &filepath)
{
    std::ofstream f(filepath);
    if (!f)
    {
        std::cerr << "ERROR: cannot open " << filepath << " for writing\n";
        return false;
    }

    const auto expected = std::count_if(path_nodes.begin(), path_nodes.end(),
                                        [](const RRTNode &n)
                                        { return n.incoming_skill == SkillPrimitive::HANDSTAND_MOVE; });

    long rows = 0;
    for (const auto &n : path_nodes)
    {
        if (n.incoming_skill != SkillPrimitive::HANDSTAND_MOVE)
            continue;
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%f, %f, %f",
                      n.incoming_cmd.vx, n.incoming_cmd.vy, n.incoming_cmd.vyaw);
        f << buf << "\n";
        ++rows;
    }

    if (!f)
    {
        std::cerr << "ERROR: failed while writing " << filepath << "\n";
        return false;
    }
    if (rows != expected)
    {
        std::cerr << "ERROR: wrote " << rows << " velocity rows but path has "
                  << expected << " HANDSTAND_MOVE edges\n";
        return false;
    }
    std::printf("Wrote %ld HandStand+Move velocity commands to %s\n", rows, filepath.c_str());
    return true;
}



// ============================================================
// Safety audit
// ============================================================
static const char *mode_name(RobotMode mode)
{
    return (mode == RobotMode::HANDSTAND) ? "HANDSTAND" : "STAND";
}

static const char *skill_name(SkillPrimitive skill)
{
    switch (skill)
    {
    case SkillPrimitive::STAND_TO_HANDSTAND: return "STAND_TO_HANDSTAND";
    case SkillPrimitive::HANDSTAND_MOVE:     return "HANDSTAND_MOVE";
    case SkillPrimitive::HANDSTAND_TO_STAND: return "HANDSTAND_TO_STAND";
    case SkillPrimitive::START:
    default:                                 return "START";
    }
}

// Each waypoint is checked with the footprint of its stored mode.
// (The edges were already checked with skill_edge_free during planning.)
static void collision_check_path(const std::vector<RRTNode> &path_nodes,
                                 const PlannerConfig &cfg)
{
    std::printf("\n=== Final Collision Check (Mode-dependent Rectangular Footprint) ===\n");
    std::printf("  %-4s  %-8s %-8s %-8s %-9s  %-18s  %8s %8s %8s  %-10s\n",
                "WP", "x", "y", "theta", "mode", "incoming skill", "vx", "vy", "vyaw", "Status");

    bool any_violation = false;
    for (std::size_t i = 0; i < path_nodes.size(); ++i)
    {
        const RRTNode &n = path_nodes[i];
        double px = n.point.x, py = n.point.y, pth = n.point.theta;
        bool free = is_free(px, py, pth, n.mode, cfg);
        std::printf("  %-4zu  %7.3f  %7.3f  %7.3f  %-9s  %-18s  %8.4f %8.4f %8.4f  %-10s\n",
                    i, px, py, pth, mode_name(n.mode), skill_name(n.incoming_skill),
                    n.incoming_cmd.vx, n.incoming_cmd.vy, n.incoming_cmd.vyaw,
                    free ? "Free" : "COLLISION");
        if (!free)
            any_violation = true;
    }
    if (any_violation)
        std::printf("  *** WARNING: path has safety violations! ***\n");
    else
        std::printf("  All waypoints are collision-free (Rectangular Footprint).\n");
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
    std::printf("  Goal Tol   : %.3f m (STAND mode, XY only)\n", cfg.goal_tol);
    std::printf("  w_theta    : %.3f\n\n", Configuration::weight_theta);
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
    return true;
}

int main(int argc, char **argv)
{
    PlannerConfig cfg;
    std::string config_file = "planner.cfg";

    // Parse CLI for the master config only
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc)
            config_file = argv[++i];
        else if (std::strcmp(argv[i], "-h") == 0 ||
                 std::strcmp(argv[i], "--help") == 0)
        {
            std::printf(
                "Usage: %s [options]\n"
                "\n"
                "Options:\n"
                "  --config <path>     Master config (default: planner.cfg)\n"
                "  -h, --help          This message\n",
                argv[0]);
            return 0;
        }
    }

    // Load static config first to get query_file path
    if (!load_configs(config_file, cfg))
        return 1;

    // Now load the query config (start/goal/obstacles)
    if (!load_configs(cfg.query_file, cfg))
        return 1;

    // Normalise start heading to [-pi, pi] once, after both configs are loaded
    // (start_theta comes from query.cfg)
    cfg.start.theta = wrap_angle(cfg.start.theta);

    //--(c)heading difference Δθ
    // --(i) Δθ is scaled by cfg.w_theta
    // --(ii) so orientation affects nearest-neighbor search and perhaps extension quality
    //--- (iii) tuning w_theta allows us to control how much the planner cares about heading similarity vs position similarity when finding nearest neighbors and evaluating extensions.
    // If large, the planner cares more about heading similarity.
    // If small, it mostly cares about position.
    Configuration::weight_theta = cfg.w_theta;

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
    std::remove(cfg.velocity_commands_out.c_str());
    std::remove("rrt_tree.txt");

    // ── Pre-search Safety Check ──
    if (!is_free(cfg.start.x, cfg.start.y, cfg.start.theta, RobotMode::STAND, cfg)) { 
        std::printf("ERROR: Start position is in collision!\n");
        return 0; // Return 0 so viz can show the setup
    }
    if (!is_free(cfg.goal.x, cfg.goal.y, 0.0, RobotMode::STAND, cfg)) {
        std::printf("ERROR: Goal position (%.3f, %.3f) is in collision at θ=0!\n",
                    cfg.goal.x, cfg.goal.y);
        return 0; // Return 0 so viz can show the setup
    }

    // ── Plan ──
    int num_runs = AOX_RUNS;
    RRTResult best_overall;
    best_overall.cost = 1e18;

    std::printf("\n=== Executing %d AO-RRT cost-bounded search attempts ===\n", num_runs);
    for (int i = 0; i < num_runs; ++i) {
        unsigned int run_seed = (cfg.seed == -1) ? (final_seed + i) : (unsigned int)cfg.seed;
        RRTResult res = run_rrt(cfg, run_seed, best_overall.cost);
        if (res.success) {
            std::printf("  Run %2d: cost = %10.4f (bound=%.4f, seed=%u)\n", i, res.cost, best_overall.cost, run_seed);
            if (res.cost < best_overall.cost) {
                best_overall = std::move(res);
            }
        } else {
            std::printf("  Run %2d: failed or bounded (seed=%u)\n", i, run_seed);
        }
        if (cfg.seed != -1) break; // Don't repeat if seed is fixed
    }

    if (!best_overall.success) {
        std::printf("RRT search failed to find a path in all attempts.\n");
        return 0; // Still return 0 so user can see the setup in viz
    }

    std::printf("\nGlobally best cost: %.4f (seed=%u)\n", best_overall.cost, best_overall.seed);
    std::printf("Best run rejections: dist=%ld, rot=%ld, col=%ld, dup=%ld, cost_pruned=%ld\n", 
                best_overall.rej_dist, best_overall.rej_rot, best_overall.rej_col, best_overall.rej_dup, best_overall.rej_cost);

    write_rrt_tree(best_overall.tree, best_overall.seed, "rrt_tree.txt");
    std::vector<Configuration> path = best_overall.path;

    
    collision_check_path(best_overall.path_nodes, cfg);


    if (!write_waypoints_file(path, cfg.waypoints_out))
        return 1;

    if (!write_velocity_commands_file(best_overall.path_nodes, cfg.velocity_commands_out))
        return 1;

    return 0;
}