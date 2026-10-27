// Move_HS_skill_chain_planner.cc — SE(2) x {STAND, HANDSTAND} Skill-Chain RRT (AO-RRT)
//
// Extends skill_chain_planner.cc (HandStand-only chain) with the ordinary four-legged MOVE skill.
// Planner state = (x, y, theta, mode). Four learned Unitree Go2 skill primitives:
//
//   skill                mode change             prediction model                 collision model
//   MOVE                 STAND     -> STAND      affine SKILL_MOVE_{B,C,B_INV}    STAND footprint
//   STAND_TO_HANDSTAND   STAND     -> HANDSTAND  fixed mean SKILL_S2H_DELTA       STAND + HANDSTAND (swept)
//   HANDSTAND_MOVE       HANDSTAND -> HANDSTAND  affine SKILL_HSMOVE_{B,C,B_INV}  HANDSTAND footprint
//   HANDSTAND_TO_STAND   HANDSTAND -> STAND      fixed mean SKILL_H2S_DELTA       STAND + HANDSTAND (swept)
//
// Configuration Files:
//   planner.cfg  — Master settings (map, robot geometry, tolerances, obstacle sizes)
//   query.cfg    — Dynamic problem settings (start/goal/obstacle poses)
//
// Build:      g++ -std=c++17 -O2 -Wall -Wextra -o move_hs_skill_chain_planner Move_HS_skill_chain_planner.cc
// Execution:  ./move_hs_skill_chain_planner --config planner.cfg
// Outputs (separate from the baseline skill_chain_* / rrt_tree.txt files):
//   move_hs_skill_chain_waypoints.txt   x, y, theta per final-path node
//   move_hs_skill_chain_steps.csv       typed skill sequence: mode, incoming skill, command per node
//   move_hs_rrt_tree.txt                search tree of the best run (for move_hs_skill_chain_visualize.py)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <ctime>


#define MAX_ITERS 2000
#define AOX_RUNS 100
#define GOAL_BIAS_PERCENT 10   // default only; planner.cfg key goal_bias_percent overrides it
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
    MOVE,                // STAND walking: learned affine model steered toward q_rand
    STAND_TO_HANDSTAND,  // fixed learned transition (no q_rand)
    HANDSTAND_MOVE,      // learned affine model steered toward q_rand
    HANDSTAND_TO_STAND   // fixed learned transition (no q_rand)
};

// HANDSTAND footprint (STAND footprint still comes from planner.cfg)
static constexpr double HANDSTAND_LENGTH = 0.135;//0.135;0.385;   // m
static constexpr double HANDSTAND_WIDTH  = 0.18;//0.18;//0.27;   // m

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


// D. Move (STAND, four-legged walking) affine model:  Delta_p_body = B * V + C,   V = [vx, vy, vyaw]^T
//    Source: Go2_Walk_Base_Data_Sensor/Move_affine_model_parameters.txt
//    Separate from the HandStand+Move model (C); the two are never mixed.
static constexpr double SKILL_MOVE_B[3][3] = {
    { 0.375530102549, -0.061046249791, -0.006211056048},
    {-0.072228507265,  0.487359283063,  0.021297848558},
    {-0.065697504153,  0.010927555476,  0.564123039835}
};

static constexpr double SKILL_MOVE_C[3] = {
     0.005285933291,
     0.045387586548,
    -0.098820307162
};

static constexpr double SKILL_MOVE_B_INV[3][3] = {
    { 2.731643394643,  0.341778520379,  0.017172216656},
    { 0.391268936744,  2.102567639117, -0.075072228707},
    { 0.310546685527, -0.000925203758,  1.776117011656}
};

// HandStand+Move command limits (same as go2_navg_exec.cc)
//   Deadbands: |v| below these is set to 0 (applied AFTER clamping)
static constexpr double HSMOVE_MIN_V_LINEAR = 0.08;   // m/s   (vx, vy)
static constexpr double HSMOVE_MIN_V_YAW    = 0.1;   // rad/s (vyaw)
//   Safety clamps: vx in [0, +MAX] (no reverse); vy, vyaw in [-MAX, +MAX]
static constexpr double HSMOVE_MAX_VX   = 0.2;        // m/s
static constexpr double HSMOVE_MAX_VY   = 0.4;        // m/s
static constexpr double HSMOVE_MAX_VYAW = 0.5;        // rad/s

// MOVE command limits — independent of the HandStand+Move limits above.
//   Provisional offline-planning domain, NOT a validated envelope of the real robot;
//   to be adjusted from the MOVE demonstration data.
//   Safety clamps: vx in [0, +MAX] (no reverse); vy, vyaw in [-MAX, +MAX]
static constexpr double MOVE_MAX_VX   =  0.2;         // m/s
static constexpr double MOVE_MAX_VY   =  0.4;         // m/s
static constexpr double MOVE_MAX_VYAW =  0.5;         // rad/s
//   Deadbands: |v| below these is set to 0 (applied AFTER clamping)
static constexpr double MOVE_MIN_V_LINEAR = 0.08;     // m/s   (vx, vy)
static constexpr double MOVE_MIN_V_YAW    = 0.10;     // rad/s (vyaw)


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

    // Extra cost on every STAND_TO_HANDSTAND / HANDSTAND_TO_STAND edge, on top of its SE(2)
    // displacement, so that solutions do not switch modes without need. Must be >= 0.
    // Optional planner.cfg key: mode_switch_cost
    double mode_switch_cost = 0.1;

    // Mode-dependent motion-cost weights (provisional experimental values, must be >= 0):
    //   MOVE edge cost           = stand_motion_weight     * SE(2) distance
    //   HANDSTAND_MOVE edge cost = handstand_motion_weight * SE(2) distance
    // A larger HandStand weight makes walking preferable wherever walking is feasible.
    // Optional planner.cfg keys: stand_motion_weight, handstand_motion_weight
    double stand_motion_weight = 1.0;
    double handstand_motion_weight = 4.0;

    // Percentage of samples drawn at the goal, from the first iteration (0..100).
    // Optional planner.cfg key: goal_bias_percent
    int goal_bias_percent = GOAL_BIAS_PERCENT;

    // Optional strict skill-chain benchmark (off by default): only accept a goal reached by a
    // path with MOVE before the first STAND_TO_HANDSTAND, >= 1 HANDSTAND_MOVE, a
    // HANDSTAND_TO_STAND and MOVE after the last HANDSTAND_TO_STAND.
    // Optional planner.cfg key: require_move_before_and_after_handstand
    bool require_move_before_and_after_handstand = false;

    // Goal tolerance (Euclidean, metres)
    double goal_tol = 0.2;

    // Robot geometry
    double robot_radius = 0.5;
    double robot_length = 0.4;
    double robot_width = 0.35;

    // Obstacles
    std::vector<Obstacle> obstacles;
    double safety_margin = 0.1;

    // File paths. Own keys (move_hs_*), so the baseline planner's waypoints_out /
    // velocity_commands_out settings in the shared planner.cfg are never picked up here.
    std::string waypoints_out = "move_hs_skill_chain_waypoints.txt";
    std::string steps_out = "move_hs_skill_chain_steps.csv";
    std::string tree_out = "move_hs_rrt_tree.txt";
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
// Shared by Move, Stand->HandStand, HandStand+Move, HandStand->Stand.
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

// Velocity command  V = [vx, vy, vyaw]^T  (no limits attached).
// Shared by MOVE and HANDSTAND_MOVE: on the robot both are SportClient Move(vx, vy, vyaw); they differ
// in the learned model, the command limits and the footprint, all selected by the current mode.
struct VelocityCommand
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

// Output of the composed MOVE / HandStand+Move prediction pipeline
struct MovePrediction
{
    VelocityCommand raw_cmd;        // inverse-model output before clamping
    VelocityCommand cmd;            // command actually used (after clamp + deadband)
    Configuration q_pred;           // forward-model landing pose
    bool was_clamped = false;       // true if any component was clamped
    double distance_before = 0.0;   // current.distance(q_rand)
    double distance_after = 0.0;    // q_pred.distance(q_rand)
    bool makes_progress = false;    // distance_after < distance_before
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
static inline VelocityCommand hsmove_inverse_raw(const BodyDelta &desired)
{
    const double a0 = desired.dx_body - SKILL_HSMOVE_C[0];
    const double a1 = desired.dy_body - SKILL_HSMOVE_C[1];
    const double a2 = desired.dtheta  - SKILL_HSMOVE_C[2];

    VelocityCommand v;
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
static inline VelocityCommand clamp_hsmove_command(const VelocityCommand &raw, bool &is_clamped)
{
    VelocityCommand v = raw;
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
static inline VelocityCommand deadband_hsmove_command(const VelocityCommand &in)
{
    VelocityCommand v = in;
    if (std::fabs(v.vx)   < HSMOVE_MIN_V_LINEAR) v.vx   = 0.0;
    if (std::fabs(v.vy)   < HSMOVE_MIN_V_LINEAR) v.vy   = 0.0;
    if (std::fabs(v.vyaw) < HSMOVE_MIN_V_YAW)    v.vyaw = 0.0;
    return v;
}

// ------------------------------------------------------------
// HandStand+Move forward model in the body frame:  Delta = B * V + C
// Note: a zero command still yields Delta = C (learned bias).
// ------------------------------------------------------------
static inline BodyDelta hsmove_forward_body_delta(const VelocityCommand &v)
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
                                                   const VelocityCommand &v)
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
static inline MovePrediction predict_handstand_move_toward(const Configuration &q_current,
                                                           const Configuration &q_rand)
{
    MovePrediction out;

    const BodyDelta desired = body_frame_delta_to_target(q_current, q_rand);
    out.raw_cmd = hsmove_inverse_raw(desired);
    const VelocityCommand clamped = clamp_hsmove_command(out.raw_cmd, out.was_clamped);
    out.cmd = deadband_hsmove_command(clamped);

    out.q_pred = predict_handstand_move(q_current, out.cmd);

    out.distance_before = q_current.distance(q_rand);
    out.distance_after  = out.q_pred.distance(q_rand);
    out.makes_progress  = (out.distance_after < out.distance_before);
    return out;
}

// ============================================================
// Block P2: MOVE (STAND walking) learned-skill math.
// Same structure as the HandStand+Move pipeline (Block P1), but ONLY the
// SKILL_MOVE_* model and the MOVE_* limits are used here.
// ============================================================

// ------------------------------------------------------------
// MOVE RAW inverse model:  V = MOVE_B_inv * (Delta_desired - MOVE_C)
// No clamping here.
// ------------------------------------------------------------
static inline VelocityCommand move_inverse_raw(const BodyDelta &desired)
{
    const double a0 = desired.dx_body - SKILL_MOVE_C[0];
    const double a1 = desired.dy_body - SKILL_MOVE_C[1];
    const double a2 = desired.dtheta  - SKILL_MOVE_C[2];

    VelocityCommand v;
    v.vx   = SKILL_MOVE_B_INV[0][0] * a0 + SKILL_MOVE_B_INV[0][1] * a1 + SKILL_MOVE_B_INV[0][2] * a2;
    v.vy   = SKILL_MOVE_B_INV[1][0] * a0 + SKILL_MOVE_B_INV[1][1] * a1 + SKILL_MOVE_B_INV[1][2] * a2;
    v.vyaw = SKILL_MOVE_B_INV[2][0] * a0 + SKILL_MOVE_B_INV[2][1] * a1 + SKILL_MOVE_B_INV[2][2] * a2;
    return v;
}

// ------------------------------------------------------------
// MOVE admissible control domain:
//   vx   in [0, MOVE_MAX_VX]                (no reverse: negative vx -> 0)
//   vy   in [-MOVE_MAX_VY,   +MOVE_MAX_VY]
//   vyaw in [-MOVE_MAX_VYAW, +MOVE_MAX_VYAW]
// Any modification sets is_clamped. Deadband is applied afterwards, separately.
// ------------------------------------------------------------
static inline VelocityCommand clamp_move_command(const VelocityCommand &raw, bool &is_clamped)
{
    VelocityCommand v = raw;
    is_clamped = false;

    if (v.vx < 0.0) {
        v.vx = 0.0;                 // reverse is outside the MOVE domain
        is_clamped = true;
    } else if (v.vx > MOVE_MAX_VX) {
        v.vx = MOVE_MAX_VX;
        is_clamped = true;
    }
    if (std::fabs(v.vy) > MOVE_MAX_VY) {
        v.vy = std::copysign(MOVE_MAX_VY, v.vy);
        is_clamped = true;
    }
    if (std::fabs(v.vyaw) > MOVE_MAX_VYAW) {
        v.vyaw = std::copysign(MOVE_MAX_VYAW, v.vyaw);
        is_clamped = true;
    }
    return v;
}

// ------------------------------------------------------------
// MOVE deadband (applied AFTER clamping): small components are zeroed
// ------------------------------------------------------------
static inline VelocityCommand deadband_move_command(const VelocityCommand &in)
{
    VelocityCommand v = in;
    if (std::fabs(v.vx)   < MOVE_MIN_V_LINEAR) v.vx   = 0.0;
    if (std::fabs(v.vy)   < MOVE_MIN_V_LINEAR) v.vy   = 0.0;
    if (std::fabs(v.vyaw) < MOVE_MIN_V_YAW)    v.vyaw = 0.0;
    return v;
}

// ------------------------------------------------------------
// MOVE forward model in the body frame:  Delta = MOVE_B * V + MOVE_C
// ------------------------------------------------------------
static inline BodyDelta move_forward_body_delta(const VelocityCommand &v)
{
    BodyDelta d;
    d.dx_body = SKILL_MOVE_B[0][0] * v.vx + SKILL_MOVE_B[0][1] * v.vy + SKILL_MOVE_B[0][2] * v.vyaw + SKILL_MOVE_C[0];
    d.dy_body = SKILL_MOVE_B[1][0] * v.vx + SKILL_MOVE_B[1][1] * v.vy + SKILL_MOVE_B[1][2] * v.vyaw + SKILL_MOVE_C[1];
    d.dtheta  = SKILL_MOVE_B[2][0] * v.vx + SKILL_MOVE_B[2][1] * v.vy + SKILL_MOVE_B[2][2] * v.vyaw + SKILL_MOVE_C[2];
    return d;
}

// ------------------------------------------------------------
// MOVE forward prediction in the world frame
// ------------------------------------------------------------
static inline Configuration predict_move(const Configuration &q,
                                         const VelocityCommand &v)
{
    const BodyDelta d = move_forward_body_delta(v);
    return apply_body_frame_transition(q, d.dx_body, d.dy_body, d.dtheta);
}

// ------------------------------------------------------------
// Composed MOVE pipeline (same steps as predict_handstand_move_toward):
//   target -> body delta -> raw inverse V -> clamp -> deadband -> forward model -> landing pose
// ------------------------------------------------------------
static inline MovePrediction predict_move_toward(const Configuration &q_current,
                                                 const Configuration &q_rand)
{
    MovePrediction out;

    const BodyDelta desired = body_frame_delta_to_target(q_current, q_rand);
    out.raw_cmd = move_inverse_raw(desired);
    const VelocityCommand clamped = clamp_move_command(out.raw_cmd, out.was_clamped);
    out.cmd = deadband_move_command(clamped);

    out.q_pred = predict_move(q_current, out.cmd);

    out.distance_before = q_current.distance(q_rand);
    out.distance_after  = out.q_pred.distance(q_rand);
    out.makes_progress  = (out.distance_after < out.distance_before);
    return out;
}

// ------------------------------------------------------------
// Command / pose validity for velocity-commanded skills
// ------------------------------------------------------------
static inline bool is_finite_command(const VelocityCommand &v)
{
    return std::isfinite(v.vx) && std::isfinite(v.vy) && std::isfinite(v.vyaw);
}

static inline bool is_finite_pose(const Configuration &q)
{
    return std::isfinite(q.x) && std::isfinite(q.y) && std::isfinite(q.theta);
}

// All components zeroed by the deadband: no motion is commanded, and the forward
// model would only return its intercept C.
static inline bool is_zero_command(const VelocityCommand &v)
{
    return v.vx == 0.0 && v.vy == 0.0 && v.vyaw == 0.0;
}

// ------------------------------------------------------------
// Sanity check of a hard-coded learned model: max |B * B_inv - I| over all entries
// ------------------------------------------------------------
static double affine_inverse_residual(const double B[3][3], const double B_inv[3][3])
{
    double worst = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            double s = 0.0;
            for (int k = 0; k < 3; ++k)
                s += B[i][k] * B_inv[k][j];
            worst = std::max(worst, std::fabs(s - (i == j ? 1.0 : 0.0)));
        }
    return worst;
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
            else if (key == "mode_switch_cost")
                cfg.mode_switch_cost = std::stod(val);
            else if (key == "stand_motion_weight")
                cfg.stand_motion_weight = std::stod(val);
            else if (key == "handstand_motion_weight")
                cfg.handstand_motion_weight = std::stod(val);
            else if (key == "goal_bias_percent")
                cfg.goal_bias_percent = std::stoi(val);
            else if (key == "require_move_before_and_after_handstand")
                cfg.require_move_before_and_after_handstand = (val == "true" || val == "1");
            else if (key == "move_hs_waypoints_out")
                cfg.waypoints_out = val;
            else if (key == "move_hs_steps_out")
                cfg.steps_out = val;
            else if (key == "move_hs_tree_out")
                cfg.tree_out = val;
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

// ------------------------------------------------------------
// Collision check for a FIXED mode switch (STAND_TO_HANDSTAND / HANDSTAND_TO_STAND).
// The posture change between the two learned endpoint poses is not modelled, so every
// sample of the swept edge (both endpoints included) must be free under BOTH the STAND
// and the HANDSTAND footprint. This covers the start pose with the start-mode footprint
// and the landing pose with the landing-mode footprint, and never assumes that the
// smaller HANDSTAND rectangle is valid during the posture change.
// ------------------------------------------------------------
static inline bool mode_switch_edge_free(const Configuration &q_from,
                                         const Configuration &q_to,
                                         const PlannerConfig &cfg)
{
    return skill_edge_free(q_from, q_to, RobotMode::STAND, cfg) &&
           skill_edge_free(q_from, q_to, RobotMode::HANDSTAND, cfg);
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
    RobotMode mode = RobotMode::STAND;   // discrete mode is part of the planner state
    SkillPrimitive incoming_skill = SkillPrimitive::START;  // skill on the edge parent -> this node
    VelocityCommand incoming_cmd;        // final clamped + deadbanded MOVE / HandStand+Move command (zero for START and mode switches)
    bool switch_tried = false;           // the node's single fixed mode switch has already been attempted
    unsigned char progress = 0;          // strict-benchmark path-history flags (PROGRESS_*); always 0 when disabled
};

static const char *mode_name(RobotMode mode)
{
    return (mode == RobotMode::HANDSTAND) ? "HANDSTAND" : "STAND";
}

static const char *skill_name(SkillPrimitive skill)
{
    switch (skill)
    {
    case SkillPrimitive::MOVE:               return "MOVE";
    case SkillPrimitive::STAND_TO_HANDSTAND: return "STAND_TO_HANDSTAND";
    case SkillPrimitive::HANDSTAND_MOVE:     return "HANDSTAND_MOVE";
    case SkillPrimitive::HANDSTAND_TO_STAND: return "HANDSTAND_TO_STAND";
    case SkillPrimitive::START:
    default:                                 return "START";
    }
}

static inline bool is_mode_switch(SkillPrimitive skill)
{
    return skill == SkillPrimitive::STAND_TO_HANDSTAND ||
           skill == SkillPrimitive::HANDSTAND_TO_STAND;
}

// MOVE / HANDSTAND_MOVE edges carry a real Move(vx, vy, vyaw) command;
// START and the fixed mode switches carry a zero placeholder only.
static inline bool skill_has_velocity_command(SkillPrimitive skill)
{
    return skill == SkillPrimitive::MOVE || skill == SkillPrimitive::HANDSTAND_MOVE;
}

// A node may take its single fixed mode switch once, and not directly after the opposite
// switch: STAND -> HANDSTAND -> STAND (or the reverse) with no motion in between is only a
// fixed displacement, never useful navigation.
static inline bool can_switch_mode(const RRTNode &n)
{
    return !n.switch_tried && !is_mode_switch(n.incoming_skill);
}

// ------------------------------------------------------------
// Edge cost parent -> child. The ONLY definition of edge cost: tree insertion, AO-RRT
// pruning, best-goal updates, final validation and the CSV all use it.
//   MOVE               : stand_motion_weight     * SE2(parent, child)
//   HANDSTAND_MOVE     : handstand_motion_weight * SE2(parent, child)
//   STAND_TO_HANDSTAND,
//   HANDSTAND_TO_STAND : SE2(parent, child) + mode_switch_cost
// ------------------------------------------------------------
static double edge_cost(SkillPrimitive skill, const Configuration &from, const Configuration &to,
                        const PlannerConfig &cfg)
{
    const double d = from.distance(to);
    switch (skill)
    {
    case SkillPrimitive::MOVE:               return cfg.stand_motion_weight * d;
    case SkillPrimitive::HANDSTAND_MOVE:     return cfg.handstand_motion_weight * d;
    case SkillPrimitive::STAND_TO_HANDSTAND:
    case SkillPrimitive::HANDSTAND_TO_STAND: return d + cfg.mode_switch_cost;
    case SkillPrimitive::START:
    default:                                 return 0.0;
    }
}

// ------------------------------------------------------------
// Path-history flags for the optional strict benchmark (require_move_before_and_after_handstand).
// Each node carries the flags of its path root -> node, so two nodes with the same pose and
// mode but different task progress are different planner states (never merged as duplicates).
// ------------------------------------------------------------
static constexpr unsigned char PROGRESS_WALKED_BEFORE_HS = 1u << 0;   // MOVE before the first STAND_TO_HANDSTAND
static constexpr unsigned char PROGRESS_ENTERED_HS       = 1u << 1;   // >= 1 STAND_TO_HANDSTAND
static constexpr unsigned char PROGRESS_HS_MOVED         = 1u << 2;   // >= 1 HANDSTAND_MOVE
static constexpr unsigned char PROGRESS_STOOD_UP         = 1u << 3;   // >= 1 HANDSTAND_TO_STAND
static constexpr unsigned char PROGRESS_WALKED_AFTER_HS  = 1u << 4;   // MOVE after the LAST HANDSTAND_TO_STAND
static constexpr unsigned char PROGRESS_CHAIN_COMPLETE =
    PROGRESS_WALKED_BEFORE_HS | PROGRESS_ENTERED_HS | PROGRESS_HS_MOVED | PROGRESS_STOOD_UP | PROGRESS_WALKED_AFTER_HS;

// Flags after taking `skill` from a node with flags p
static inline unsigned char next_progress(unsigned char p, SkillPrimitive skill)
{
    switch (skill)
    {
    case SkillPrimitive::MOVE:   // a STAND node after STAND_TO_HANDSTAND has always stood up again
        return static_cast<unsigned char>(p | ((p & PROGRESS_ENTERED_HS) ? PROGRESS_WALKED_AFTER_HS
                                                                          : PROGRESS_WALKED_BEFORE_HS));
    case SkillPrimitive::STAND_TO_HANDSTAND:
        return static_cast<unsigned char>(p | PROGRESS_ENTERED_HS);
    case SkillPrimitive::HANDSTAND_MOVE:
        return static_cast<unsigned char>(p | PROGRESS_HS_MOVED);
    case SkillPrimitive::HANDSTAND_TO_STAND:   // a new stand-up needs its own MOVE afterwards
        return static_cast<unsigned char>((p | PROGRESS_STOOD_UP) & ~PROGRESS_WALKED_AFTER_HS);
    case SkillPrimitive::START:
    default:
        return p;
    }
}

// Strict benchmark only: entering HandStand for the first time without an earlier MOVE can
// never satisfy "MOVE before the first STAND_TO_HANDSTAND", so that branch is not generated.
static inline bool strict_switch_allowed(unsigned char p, SkillPrimitive skill)
{
    return skill != SkillPrimitive::STAND_TO_HANDSTAND || (p & PROGRESS_WALKED_BEFORE_HS);
}

// Skill-chain goal: a STAND node whose XY position is within goal_tol of cfg.goal (heading free).
// HANDSTAND nodes never satisfy it, so a plan may end with MOVE or with HANDSTAND_TO_STAND.
// With the strict benchmark enabled, the path to the node must also have completed the
// MOVE -> HandStand -> MOVE chain (path-history flags).
static bool is_goal_satisfied(const RRTNode &n, const PlannerConfig &cfg)
{
    if (n.mode != RobotMode::STAND)
        return false;
    if (cfg.require_move_before_and_after_handstand && n.progress != PROGRESS_CHAIN_COMPLETE)
        return false;
    return n.point.distance(cfg.goal, {0, 1}) <= cfg.goal_tol;
}

static void update_best_goal(const std::vector<RRTNode> &tree, int idx,
                             const PlannerConfig &cfg, int &goal_idx,
                             double &best_goal_cost, int iter)
{
    if (is_goal_satisfied(tree[idx], cfg))
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

// Iterative insert. The child link is written by INDEX after push_back, so no reference
// into `nodes` is held across a possible reallocation (the tree may outgrow its reserve).
static void kdtree_insert(std::vector<KDNode> &nodes, int &root_idx, int new_node_idx, const Configuration &point)
{
    const int new_kd = static_cast<int>(nodes.size());
    nodes.push_back({new_node_idx, point.x, point.y, -1, -1});
    if (root_idx == -1)
    {
        root_idx = new_kd;
        return;
    }

    int cur = root_idx;
    for (int depth = 0;; ++depth)
    {
        bool split_x = (depth % 2 == 0);
        double val = split_x ? point.x : point.y;
        double cur_val = split_x ? nodes[cur].x : nodes[cur].y;
        const bool go_left = (val < cur_val);

        const int next = go_left ? nodes[cur].left : nodes[cur].right;
        if (next == -1)
        {
            if (go_left)
                nodes[cur].left = new_kd;
            else
                nodes[cur].right = new_kd;
            return;
        }
        cur = next;
    }
}

static void kdtree_query(const std::vector<KDNode> &nodes, int root_idx, const Configuration &q_sample,
                         const std::vector<RRTNode> &tree, const PlannerConfig &cfg, int goal_idx,
                         RobotMode mode, int progress, int &best_idx, double &min_dist_sq, int depth = 0)
{
    if (root_idx == -1) return;

    const auto &kd = nodes[root_idx];
    const auto &rrt_node = tree[kd.idx];

    // 0. Mode filter (+ task-progress filter for duplicate checks; progress < 0 = any):
    //    filtered-out nodes are never candidates, but their KD subtrees are still
    //    traversed below (the KD layout is x,y only, mode-agnostic).
    // 1. Goal optimization check (optional for find_nearest, usually skip for duplicates)
    if (rrt_node.mode == mode && (progress < 0 || rrt_node.progress == progress) &&
        (goal_idx == -2 || !(is_goal_satisfied(rrt_node, cfg) && goal_idx != -1)))
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

    kdtree_query(nodes, near, q_sample, tree, cfg, goal_idx, mode, progress, best_idx, min_dist_sq, depth + 1);

    double diff = std::fabs(q_val - kd_val);
    if (diff * diff < min_dist_sq)
    {
        kdtree_query(nodes, far, q_sample, tree, cfg, goal_idx, mode, progress, best_idx, min_dist_sq, depth + 1);
    }
}

// Nearest node (SE(2) metric) among nodes whose mode == `mode` (any task progress); -1 if none.
static int find_nearest(const std::vector<RRTNode> &tree, const std::vector<KDNode> &kd_nodes, int kd_root,
                        const Configuration &q_sample, const PlannerConfig &cfg, int goal_idx,
                        RobotMode mode)
{
    int best_idx = -1;
    double min_dist_sq = 1e36;
    kdtree_query(kd_nodes, kd_root, q_sample, tree, cfg, goal_idx, mode, -1, best_idx, min_dist_sq);
    return best_idx;
}

static int find_duplicate(const std::vector<RRTNode> &tree, const std::vector<KDNode> &kd_nodes, int kd_root,
                          const Configuration &q, RobotMode mode, unsigned char progress)
{
    if (kd_root == -1) return -1;

    // Use KD-tree to find nearest node with the SAME mode and task progress, then check if it's
    // within tolerance (same pose + different mode or different progress is NOT a duplicate)
    int best_idx = -1;
    double min_dist_sq = 1e36;
    PlannerConfig dummy_cfg; // Goal optimization not needed for duplicate check
    kdtree_query(kd_nodes, kd_root, q, tree, dummy_cfg, -2, mode, progress, best_idx, min_dist_sq);

    if (best_idx != -1) {
        const double eps_pos = 1e-3;
        const double eps_th = 1e-2;
        if (tree[best_idx].point.distance(q, {0, 1}) < eps_pos &&
            tree[best_idx].point.distance(q, 2) < eps_th)
            return best_idx;
    }
    return -1;
}

// Plain RRT insertion. Returns the index of the new node, or -1 if a node with the SAME
// mode, the same task progress and approximately the same pose already exists (the new node
// is then rejected; existing nodes are never re-parented and their costs never change).
static int add_node(std::vector<RRTNode> &tree, std::vector<KDNode> &kd_nodes, int &kd_root,
                    const Configuration &point, RobotMode mode, unsigned char progress, int parent,
                    double cost, SkillPrimitive skill, const VelocityCommand &cmd, long &rej_dup)
{
    if (find_duplicate(tree, kd_nodes, kd_root, point, mode, progress) != -1) {
        rej_dup++;
        return -1;
    }
    RRTNode n;
    n.point = point;            // copied before push_back: `point` may not alias `tree`
    n.parent = parent;
    n.cost = cost;
    n.mode = mode;
    n.incoming_skill = skill;
    n.incoming_cmd = cmd;
    n.progress = progress;
    tree.push_back(n);
    int new_idx = static_cast<int>(tree.size() - 1);
    kdtree_insert(kd_nodes, kd_root, new_idx, n.point);
    return new_idx;
}

// ============================================================
// Random Sampler
// ============================================================
static Configuration sample(const PlannerConfig &cfg, bool &sampling_goal)
{
    Configuration q;
    // Goal bias from the first iteration. rand() is only drawn for it when the bias is enabled,
    // so a seeded run stays reproducible, and goal_bias_percent = 0 gives the unbiased sequence.
    sampling_goal = (cfg.goal_bias_percent > 0) && ((rand() % 100) < cfg.goal_bias_percent);
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
    std::vector<RRTNode> path_nodes;   // same states as `path`, plus mode / incoming skill / command
    std::vector<RRTNode> tree;
    unsigned int seed;
    long rej_dist = 0, rej_rot = 0, rej_col = 0, rej_dup = 0, rej_cost = 0, rej_cmd = 0, rej_phase = 0;
};

// ============================================================
// State of ONE RRT run
// ============================================================
struct SearchState
{
    const PlannerConfig &cfg;

    //-- tree

    // --(a) Stores the actual RRT nodes.

    //-- (b)Each RRTNode likely contains:

    // -- (1) configuration point
    //-- (2) parent index
    //-- (3) accumulated path cost

    //-- (4) So each node is a state:

    // -- (a) qi=(xi,yi,θi, mode_i)
    //-- (b) with parent:parent(i)


    //--This lets you reconstruct the final path by walking backward from goal to start.---//
    std::vector<RRTNode> tree;

    //-- kd_nodes--> This is probably the KD-tree structure used for fast nearest-neighbor lookup.--//
    std::vector<KDNode> kd_nodes;
    int kd_root = -1;

    //goal_idx = -1 means no goal-reaching node has been found yet
    // best_goal_cost stores the best path cost to goal found so far
    int goal_idx = -1;
    double best_goal_cost = 1e18;

    long rej_dist = 0, rej_rot = 0, rej_col = 0, rej_dup = 0, rej_cost = 0, rej_cmd = 0, rej_phase = 0;

    explicit SearchState(const PlannerConfig &c) : cfg(c) {}
};

// ------------------------------------------------------------
// Insert an already validated (cost-bounded, collision-free) edge and run the goal check.
// Returns true only if a NEW node was created (a duplicate returns false).
// ------------------------------------------------------------
static bool insert_candidate(SearchState &s, const Configuration &q, RobotMode mode, int parent,
                             double cost, SkillPrimitive skill, const VelocityCommand &cmd, int iter)
{
    // Path history is only tracked for the strict benchmark; otherwise every node has progress 0
    const unsigned char progress = s.cfg.require_move_before_and_after_handstand
                                       ? next_progress(s.tree[parent].progress, skill) : 0;
    const int idx = add_node(s.tree, s.kd_nodes, s.kd_root, q, mode, progress, parent, cost, skill, cmd,
                             s.rej_dup);
    if (idx == -1)
        return false;
    update_best_goal(s.tree, idx, s.cfg, s.goal_idx, s.best_goal_cost, iter);
    return true;
}

// ------------------------------------------------------------
// Fixed mode switch from node idx using the learned MEAN transition:
//   STAND node     -> STAND_TO_HANDSTAND -> HANDSTAND node
//   HANDSTAND node -> HANDSTAND_TO_STAND -> STAND node
// Not steered toward q_rand, so each node has exactly one possible switch child;
// it is attempted at most once per node (switch_tried).
// ------------------------------------------------------------
static bool try_mode_switch(SearchState &s, int idx, int iter)
{
    if (!can_switch_mode(s.tree[idx]))
        return false;
    s.tree[idx].switch_tried = true;
    const RRTNode from = s.tree[idx];   // copy: insertion can reallocate the tree

    const bool to_handstand = (from.mode == RobotMode::STAND);
    const Configuration q_to = to_handstand ? predict_stand_to_handstand(from.point)
                                            : predict_handstand_to_stand(from.point);
    const RobotMode mode_to = to_handstand ? RobotMode::HANDSTAND : RobotMode::STAND;
    const SkillPrimitive skill = to_handstand ? SkillPrimitive::STAND_TO_HANDSTAND
                                              : SkillPrimitive::HANDSTAND_TO_STAND;

    if (s.cfg.require_move_before_and_after_handstand && !strict_switch_allowed(from.progress, skill)) {
        s.rej_phase++;
        return false;
    }

    const double cost = from.cost + edge_cost(skill, from.point, q_to, s.cfg);
    if (cost >= s.best_goal_cost) {
        s.rej_cost++;
        return false;
    }
    if (!mode_switch_edge_free(from.point, q_to, s.cfg)) {
        s.rej_col++;
        return false;
    }
    // Zero command = output placeholder; the primitive is the discrete posture change.
    return insert_candidate(s, q_to, mode_to, idx, cost, skill, VelocityCommand{}, iter);
}

// ------------------------------------------------------------
// Velocity-commanded skill from node idx toward q_rand; the node's mode selects the skill:
//   STAND     -> MOVE            (SKILL_MOVE_* model,   MOVE_* limits,   STAND footprint)
//   HANDSTAND -> HANDSTAND_MOVE  (SKILL_HSMOVE_* model, HSMOVE_* limits, HANDSTAND footprint)
// The mode is unchanged. The FORWARD-MODEL landing pose (never q_rand) is inserted
// together with the final (clamped + deadbanded) command used to predict it.
// ------------------------------------------------------------
static bool try_velocity_skill(SearchState &s, int idx, const Configuration &q_rand, int iter)
{
    const RRTNode near = s.tree[idx];   // copy: insertion can reallocate the tree
    const bool standing = (near.mode == RobotMode::STAND);

    const MovePrediction pred = standing ? predict_move_toward(near.point, q_rand)
                                         : predict_handstand_move_toward(near.point, q_rand);

    // Reject non-finite and all-zero commands for BOTH skills: Move(0, 0, 0) is not used as a
    // navigation command, even though the affine intercept C predicts a non-zero displacement.
    if (!is_finite_command(pred.cmd) || !is_finite_pose(pred.q_pred) || is_zero_command(pred.cmd)) {
        s.rej_cmd++;
        return false;
    }
    if (!pred.makes_progress) {
        s.rej_dist++;
        return false;
    }

    // AO-RRT Cost Pruning: Prune branches that exceed the best goal cost found so far
    const SkillPrimitive skill = standing ? SkillPrimitive::MOVE : SkillPrimitive::HANDSTAND_MOVE;
    const double cost = near.cost + edge_cost(skill, near.point, pred.q_pred, s.cfg);
    if (cost >= s.best_goal_cost) {
        s.rej_cost++;
        return false;
    }

    // Map bounds + obstacles along the whole edge, with the footprint of the node's own mode
    if (!skill_edge_free(near.point, pred.q_pred, near.mode, s.cfg)) {
        s.rej_col++;
        return false;
    }

    return insert_candidate(s, pred.q_pred, near.mode, idx, cost, skill, pred.cmd, iter);
}

// ------------------------------------------------------------
// Expansion of one node:
//   STAND:     MOVE toward q_rand and STAND_TO_HANDSTAND are evaluated independently;
//              every feasible successor is inserted (up to two new nodes).
//   HANDSTAND: HANDSTAND_TO_STAND first; HANDSTAND_MOVE toward q_rand only if the switch
//              added no new node. The switch is tried at most once per node (switch_tried),
//              so any later selection of the same node goes straight to HANDSTAND_MOVE.
// Each candidate has its own command, collision and AO-RRT cost checks; duplicates are
// rejected. A mode switch is never generated directly after the opposite switch.
// ------------------------------------------------------------
static void expand_node(SearchState &s, int idx, const Configuration &q_rand, int iter)
{
    if (s.tree[idx].mode == RobotMode::STAND)
    {
        try_velocity_skill(s, idx, q_rand, iter);   // MOVE
        try_mode_switch(s, idx, iter);              // STAND_TO_HANDSTAND
    }
    else
    {
        try_mode_switch(s, idx, iter);       // HANDSTAND_TO_STAND
        try_velocity_skill(s, idx, q_rand, iter);   // HANDSTAND_MOVE
        
    }
}

// Walk parents from the goal node back to the root (guarded against a corrupted parent chain)
static std::vector<RRTNode> extract_path(const std::vector<RRTNode> &tree, int goal_idx)
{
    std::vector<RRTNode> path;
    for (int curr = goal_idx; curr != -1; curr = tree[curr].parent)
    {
        if (path.size() > tree.size())
        {
            std::cerr << "ERROR: cycle in RRT parent structure\n";
            return {};
        }
        path.push_back(tree[curr]);
    }
    std::reverse(path.begin(), path.end());
    return path;
}

// ============================================================
// RRT search with AO-RRT Cost-Bounding
// ============================================================
static RRTResult run_rrt(const PlannerConfig &cfg, unsigned int seed, double global_cost_bound = 1e18)
{
    RRTResult result;
    result.seed = seed;
    std::srand(seed);

    // Configuration::weight_theta (= cfg.w_theta) is set once in main() after the configs load.

    SearchState s(cfg);
    s.best_goal_cost = global_cost_bound;

    // Two nodes expanded per iteration, at most two successors each (+ root). This is only a
    // reservation: both vectors may grow past it, since nodes are always referenced by index.
    const std::size_t expected_nodes = 1 + 4 * static_cast<std::size_t>(MAX_ITERS);
    s.tree.reserve(expected_nodes);
    s.kd_nodes.reserve(expected_nodes);

    // (1) Root: STAND at cfg.start. No automatic HandStand: walking is considered first.
    add_node(s.tree, s.kd_nodes, s.kd_root, cfg.start, RobotMode::STAND, 0, -1, 0.0,
             SkillPrimitive::START, VelocityCommand{}, s.rej_dup);

    // (2) Random-sample loop. For each q_rand the nearest node of EACH mode is expanded
    //     (see expand_node), so STAND and HANDSTAND branches grow side by side.
    for (int iter = 0; iter < MAX_ITERS; ++iter)
    {
        bool sampling_goal = false;
        const Configuration q_rand = sample(cfg, sampling_goal);

        for (RobotMode mode : {RobotMode::STAND, RobotMode::HANDSTAND})
        {
            const int near_idx = find_nearest(s.tree, s.kd_nodes, s.kd_root, q_rand, cfg, s.goal_idx, mode);
            if (near_idx != -1)
                expand_node(s, near_idx, q_rand, iter);
        }
    }

    result.rej_dist = s.rej_dist;
    result.rej_rot = s.rej_rot;
    result.rej_col = s.rej_col;
    result.rej_dup = s.rej_dup;
    result.rej_cost = s.rej_cost;
    result.rej_cmd = s.rej_cmd;
    result.rej_phase = s.rej_phase;

    if (s.goal_idx != -1)
    {
        result.path_nodes = extract_path(s.tree, s.goal_idx);   // keeps mode, incoming_skill, incoming_cmd
        if (result.path_nodes.empty())
            return result;
        result.success = true;
        result.cost = s.best_goal_cost;
        for (const auto &n : result.path_nodes)
            result.path.push_back(n.point);
        result.tree = std::move(s.tree);
    }

    return result;
}

// Tree file: "x y theta parent cost mode incoming_skill" per node
// (first five columns as in the baseline rrt_tree.txt)
static bool write_rrt_tree(const std::vector<RRTNode> &tree, unsigned int seed, const std::string &filename)
{
    std::ofstream tree_file(filename);
    if (!tree_file) return false;
    tree_file << "Seed: " << seed << "\n";
    for (const auto &node : tree) {
        tree_file << node.point.x << " " << node.point.y << " " << node.point.theta << " " << node.parent << " " << node.cost
                  << " " << mode_name(node.mode) << " " << skill_name(node.incoming_skill) << "\n";
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
        char buf[1024];   // fits the worst case (a %f can be 317 chars), so never truncated
        std::snprintf(buf, sizeof(buf), "%f, %f, %f", p.x, p.y, p.theta);
        f << buf << "\n";
    }
    std::printf("Wrote %zu waypoints to %s\n", path.size(), filepath.c_str());
    return true;
}

// ============================================================
// Write the typed skill sequence (CSV, one row per final-path node, in path order)
//   wp,x,y,theta,mode,incoming_skill,vx,vy,vyaw,has_velocity_cmd
//   mode             : the node's own mode (= mode AFTER its incoming skill)
//   incoming_skill   : START (row 0), MOVE, STAND_TO_HANDSTAND, HANDSTAND_MOVE, HANDSTAND_TO_STAND
//   vx, vy, vyaw     : MOVE / HANDSTAND_MOVE: the final clamped + deadbanded command (the one used
//                      by the forward model). START / fixed mode switches: 0,0,0 placeholder.
//   has_velocity_cmd : 1 = the edge is a Move(vx, vy, vyaw) call; 0 = the zeros are a placeholder,
//                      NOT a Move(0, 0, 0) call.
//   edge_cost, cost  : edge_cost() of the incoming edge, and the accumulated path cost
//                      (appended columns; the first ten are unchanged)
// ============================================================
static bool write_skill_steps_csv(const std::vector<RRTNode> &path_nodes, const PlannerConfig &cfg,
                                  const std::string &filepath)
{
    std::ofstream f(filepath);
    if (!f)
    {
        std::cerr << "ERROR: cannot open " << filepath << " for writing\n";
        return false;
    }

    f << "wp,x,y,theta,mode,incoming_skill,vx,vy,vyaw,has_velocity_cmd,edge_cost,cost\n";
    for (std::size_t i = 0; i < path_nodes.size(); ++i)
    {
        const RRTNode &n = path_nodes[i];
        const double ec = (i == 0) ? 0.0 : edge_cost(n.incoming_skill, path_nodes[i - 1].point, n.point, cfg);
        char buf[4096];   // fits the worst case (a %f can be 317 chars), so never truncated
        std::snprintf(buf, sizeof(buf), "%zu,%f,%f,%f,%s,%s,%f,%f,%f,%d,%f,%f",
                      i, n.point.x, n.point.y, n.point.theta,
                      mode_name(n.mode), skill_name(n.incoming_skill),
                      n.incoming_cmd.vx, n.incoming_cmd.vy, n.incoming_cmd.vyaw,
                      skill_has_velocity_command(n.incoming_skill) ? 1 : 0, ec, n.cost);
        f << buf << "\n";
    }

    if (!f)
    {
        std::cerr << "ERROR: failed while writing " << filepath << "\n";
        return false;
    }
    std::printf("Wrote %zu skill steps to %s\n", path_nodes.size(), filepath.c_str());
    return true;
}



// ============================================================
// Safety audit
// ============================================================

// Modes each skill may connect (parent mode -> child mode)
static bool skill_modes_valid(SkillPrimitive skill, RobotMode from, RobotMode to)
{
    switch (skill)
    {
    case SkillPrimitive::MOVE:               return from == RobotMode::STAND     && to == RobotMode::STAND;
    case SkillPrimitive::STAND_TO_HANDSTAND: return from == RobotMode::STAND     && to == RobotMode::HANDSTAND;
    case SkillPrimitive::HANDSTAND_MOVE:     return from == RobotMode::HANDSTAND && to == RobotMode::HANDSTAND;
    case SkillPrimitive::HANDSTAND_TO_STAND: return from == RobotMode::HANDSTAND && to == RobotMode::STAND;
    case SkillPrimitive::START:
    default:                                 return false;
    }
}

// Re-apply the learned model of the stored edge (skill + command) to the parent pose
static Configuration repredict_edge(const RRTNode &parent, const RRTNode &child)
{
    switch (child.incoming_skill)
    {
    case SkillPrimitive::MOVE:               return predict_move(parent.point, child.incoming_cmd);
    case SkillPrimitive::HANDSTAND_MOVE:     return predict_handstand_move(parent.point, child.incoming_cmd);
    case SkillPrimitive::STAND_TO_HANDSTAND: return predict_stand_to_handstand(parent.point);
    case SkillPrimitive::HANDSTAND_TO_STAND: return predict_handstand_to_stand(parent.point);
    case SkillPrimitive::START:
    default:                                 return child.point;
    }
}

// Edge collision check with the same collision model the planner used for that skill
static bool edge_free_for_skill(const RRTNode &parent, const RRTNode &child, const PlannerConfig &cfg)
{
    switch (child.incoming_skill)
    {
    case SkillPrimitive::MOVE:
        return skill_edge_free(parent.point, child.point, RobotMode::STAND, cfg);
    case SkillPrimitive::HANDSTAND_MOVE:
        return skill_edge_free(parent.point, child.point, RobotMode::HANDSTAND, cfg);
    case SkillPrimitive::STAND_TO_HANDSTAND:
    case SkillPrimitive::HANDSTAND_TO_STAND:
        return mode_switch_edge_free(parent.point, child.point, cfg);
    case SkillPrimitive::START:
    default:
        return false;
    }
}

// MOVE / HANDSTAND_MOVE edges need a finite, non-zero command; START and the fixed mode
// switches must carry the (0, 0, 0) placeholder
static bool edge_command_valid(const RRTNode &child)
{
    if (skill_has_velocity_command(child.incoming_skill))
        return is_finite_command(child.incoming_cmd) && !is_zero_command(child.incoming_cmd);
    return is_zero_command(child.incoming_cmd);
}

// Without rewiring, every stored pose IS the model prediction from its parent, so the
// re-prediction error is expected to be exactly zero
static constexpr double REPREDICT_TOL = 1e-9;

static const char *footprint_label(RobotMode mode)
{
    return (mode == RobotMode::HANDSTAND) ? "HANDSTAND fp" : "STAND fp";
}

static const char *edge_model_label(SkillPrimitive skill)
{
    switch (skill)
    {
    case SkillPrimitive::MOVE:           return "STAND fp";
    case SkillPrimitive::HANDSTAND_MOVE: return "HANDSTAND fp";
    case SkillPrimitive::STAND_TO_HANDSTAND:
    case SkillPrimitive::HANDSTAND_TO_STAND: return "STAND+HANDSTAND fp";
    case SkillPrimitive::START:
    default:                             return "-";
    }
}

// Each waypoint is checked with the footprint of its stored mode. Each edge is checked for
// a valid mode change and command, with the collision model of its skill, and by
// re-applying its stored skill + command to the parent pose to confirm the stored landing
// pose. Returns false on any violation.
static bool collision_check_path(const std::vector<RRTNode> &path_nodes,
                                 const PlannerConfig &cfg)
{
    const Footprint fs = footprint_for_mode(RobotMode::STAND, cfg);
    const Footprint fh = footprint_for_mode(RobotMode::HANDSTAND, cfg);

    std::printf("\n=== Final Collision Check (Mode-dependent Rectangular Footprint) ===\n");
    std::printf("  STAND fp     : L=%.3f x W=%.3f m (planner.cfg robot_length x robot_width)\n", fs.length, fs.width);
    std::printf("  HANDSTAND fp : L=%.3f x W=%.3f m (HANDSTAND_LENGTH x HANDSTAND_WIDTH)\n", fh.length, fh.width);
    std::printf("  Obstacles inflated by safety_margin = %.3f m\n\n", cfg.safety_margin);
    std::printf("  %-4s  %-8s %-8s %-8s %-9s  %-12s  %-18s  %8s %8s %8s  %-10s\n",
                "WP", "x", "y", "theta", "mode", "footprint", "incoming skill", "vx", "vy", "vyaw", "Status");

    bool ok = true;
    for (std::size_t i = 0; i < path_nodes.size(); ++i)
    {
        const RRTNode &n = path_nodes[i];
        double px = n.point.x, py = n.point.y, pth = n.point.theta;
        bool free = is_free(px, py, pth, n.mode, cfg);
        std::printf("  %-4zu  %7.3f  %7.3f  %7.3f  %-9s  %-12s  %-18s  %8.4f %8.4f %8.4f  %-10s\n",
                    i, px, py, pth, mode_name(n.mode), footprint_label(n.mode), skill_name(n.incoming_skill),
                    n.incoming_cmd.vx, n.incoming_cmd.vy, n.incoming_cmd.vyaw,
                    free ? "Free" : "COLLISION");
        if (!free)
            ok = false;
    }

    std::printf("\n  Edges (predicted transition, collision model of its skill):\n");
    std::printf("  %-8s  %-18s  %-20s  %-18s  %-10s  %s\n",
                "Edge", "skill", "mode change", "collision model", "Status", "re-predict err");
    double worst_err = 0.0;
    for (std::size_t i = 1; i < path_nodes.size(); ++i)
    {
        const RRTNode &p = path_nodes[i - 1];
        const RRTNode &c = path_nodes[i];
        const bool modes_ok = skill_modes_valid(c.incoming_skill, p.mode, c.mode);
        const bool cmd_ok = edge_command_valid(c);
        const double err = repredict_edge(p, c).distance(c.point);
        const bool model_ok = (err <= REPREDICT_TOL);
        const bool free = edge_free_for_skill(p, c, cfg);
        worst_err = std::max(worst_err, err);

        const char *status = !modes_ok ? "BAD-MODE" : !cmd_ok ? "BAD-CMD" : !model_ok ? "MODEL-ERR"
                                                    : free ? "Free" : "COLLISION";
        char edge[48], modes[32];   // sized for two 20-digit %zu, so never truncated
        std::snprintf(edge, sizeof(edge), "%zu->%zu", i - 1, i);
        std::snprintf(modes, sizeof(modes), "%s->%s", mode_name(p.mode), mode_name(c.mode));
        std::printf("  %-8s  %-18s  %-20s  %-18s  %-10s  %.2e\n",
                    edge, skill_name(c.incoming_skill), modes, edge_model_label(c.incoming_skill),
                    status, err);
        if (!modes_ok || !cmd_ok || !model_ok || !free)
            ok = false;
    }

    std::printf("\n  Mode-transition checks (fixed learned mean transitions):\n");
    bool any_switch = false;
    for (std::size_t i = 1; i < path_nodes.size(); ++i)
    {
        const RRTNode &p = path_nodes[i - 1];
        const RRTNode &c = path_nodes[i];
        if (!is_mode_switch(c.incoming_skill))
            continue;
        any_switch = true;
        const bool start_free = is_free(p.point.x, p.point.y, p.point.theta, p.mode, cfg);
        const bool land_free  = is_free(c.point.x, c.point.y, c.point.theta, c.mode, cfg);
        const bool swept_free = mode_switch_edge_free(p.point, c.point, cfg);
        std::printf("    %zu->%zu  %-18s  start [%s] %s | landing [%s] %s | swept [STAND+HANDSTAND fp] %s\n",
                    i - 1, i, skill_name(c.incoming_skill),
                    footprint_label(p.mode), start_free ? "Free" : "COLLISION",
                    footprint_label(c.mode), land_free ? "Free" : "COLLISION",
                    swept_free ? "Free" : "COLLISION");
        if (!start_free || !land_free || !swept_free)
            ok = false;
    }
    if (!any_switch)
        std::printf("    (none: the path contains no mode switch)\n");

    std::printf("\n  Max re-prediction error (SE(2) metric): %.2e (tolerance %.0e)\n", worst_err, REPREDICT_TOL);
    if (!ok)
        std::printf("  *** WARNING: path has safety or consistency violations! ***\n");
    else
        std::printf("  All waypoints and edges are collision-free (mode-dependent Rectangular Footprint).\n");
    return ok;
}

// Path cost re-summed edge by edge with edge_cost() (must equal the goal node's stored cost)
static double path_cost_from_edges(const std::vector<RRTNode> &path_nodes, const PlannerConfig &cfg)
{
    double cost = 0.0;
    for (std::size_t i = 1; i < path_nodes.size(); ++i)
        cost += edge_cost(path_nodes[i].incoming_skill, path_nodes[i - 1].point, path_nodes[i].point, cfg);
    return cost;
}

// Strict benchmark checked directly on the skill sequence (independent of the stored progress
// flags): MOVE before the first STAND_TO_HANDSTAND, >= 1 HANDSTAND_MOVE, a HANDSTAND_TO_STAND,
// and MOVE after the last HANDSTAND_TO_STAND.
static bool strict_chain_satisfied(const std::vector<RRTNode> &path_nodes)
{
    std::size_t first_s2h = 0, last_h2s = 0;   // 0 = not found (index 0 is the root)
    bool hs_moved = false;
    for (std::size_t i = 1; i < path_nodes.size(); ++i)
    {
        const SkillPrimitive k = path_nodes[i].incoming_skill;
        if (k == SkillPrimitive::STAND_TO_HANDSTAND && first_s2h == 0) first_s2h = i;
        if (k == SkillPrimitive::HANDSTAND_TO_STAND) last_h2s = i;
        if (k == SkillPrimitive::HANDSTAND_MOVE) hs_moved = true;
    }
    if (first_s2h == 0 || last_h2s == 0 || !hs_moved)
        return false;

    bool walked_before = false, walked_after = false;
    for (std::size_t i = 1; i < first_s2h; ++i)
        walked_before = walked_before || path_nodes[i].incoming_skill == SkillPrimitive::MOVE;
    for (std::size_t i = last_h2s + 1; i < path_nodes.size(); ++i)
        walked_after = walked_after || path_nodes[i].incoming_skill == SkillPrimitive::MOVE;
    return walked_before && walked_after;
}

// ============================================================
// Final-path validation: only a path that passes every check is exported
// ============================================================
static bool validate_final_path(const std::vector<RRTNode> &path_nodes, const PlannerConfig &cfg,
                                double path_cost)
{
    if (path_nodes.empty())
    {
        std::printf("\n*** Final path is empty ***\n");
        return false;
    }
    const bool edges_ok = collision_check_path(path_nodes, cfg);

    const RRTNode &first = path_nodes.front();
    const RRTNode &last = path_nodes.back();
    const bool start_ok = first.incoming_skill == SkillPrimitive::START && first.mode == RobotMode::STAND &&
                          first.point.distance(cfg.start) <= REPREDICT_TOL;
    const bool goal_ok = is_goal_satisfied(last, cfg);   // incl. path-history flags in strict mode
    const double resummed = path_cost_from_edges(path_nodes, cfg);
    const bool cost_ok = std::fabs(resummed - path_cost) <= 1e-9;
    const bool strict = cfg.require_move_before_and_after_handstand;
    const bool chain_ok = !strict || strict_chain_satisfied(path_nodes);

    std::printf("\n=== Final Path Validation ===\n");
    std::printf("  Waypoint / edge checks above        : %s\n", edges_ok ? "OK" : "FAIL");
    std::printf("  Starts at cfg.start in STAND        : %s\n", start_ok ? "OK" : "FAIL");
    std::printf("  Goal condition (STAND, goal_tol%s) : %s (mode %s, XY error %.4f m, goal_tol %.3f)\n",
                strict ? ", chain" : "", goal_ok ? "OK" : "FAIL", mode_name(last.mode),
                last.point.distance(cfg.goal, {0, 1}), cfg.goal_tol);
    std::printf("  Stored cost == re-summed edge costs : %s (%.6f vs %.6f)\n",
                cost_ok ? "OK" : "FAIL", path_cost, resummed);
    std::printf("  Strict MOVE/HandStand/MOVE chain    : %s\n",
                !strict ? "(benchmark disabled)" : (chain_ok ? "OK" : "FAIL"));

    const bool ok = edges_ok && start_ok && goal_ok && cost_ok && chain_ok;
    std::printf("  => %s\n", ok ? "VALID" : "INVALID");
    return ok;
}

// ============================================================
// Skill-chain summary of the final path
// ============================================================

// Consecutive edges with the same skill grouped as (skill, count), in path order
static std::vector<std::pair<SkillPrimitive, int>> skill_groups(const std::vector<RRTNode> &path_nodes)
{
    std::vector<std::pair<SkillPrimitive, int>> groups;
    for (std::size_t i = 1; i < path_nodes.size(); ++i)
    {
        const SkillPrimitive k = path_nodes[i].incoming_skill;
        if (!groups.empty() && groups.back().first == k)
            groups.back().second++;
        else
            groups.push_back({k, 1});
    }
    return groups;
}

static void print_skill_summary(const std::vector<RRTNode> &path_nodes, const PlannerConfig &cfg,
                                double path_cost)
{
    long n_move = 0, n_hsmove = 0, n_s2h = 0, n_h2s = 0;
    double cost_move = 0.0, cost_hsmove = 0.0, cost_switch = 0.0;
    for (std::size_t i = 1; i < path_nodes.size(); ++i)
    {
        const SkillPrimitive k = path_nodes[i].incoming_skill;
        const double ec = edge_cost(k, path_nodes[i - 1].point, path_nodes[i].point, cfg);
        if (k == SkillPrimitive::MOVE) { n_move++; cost_move += ec; }
        else if (k == SkillPrimitive::HANDSTAND_MOVE) { n_hsmove++; cost_hsmove += ec; }
        else if (k == SkillPrimitive::STAND_TO_HANDSTAND) { n_s2h++; cost_switch += ec; }
        else if (k == SkillPrimitive::HANDSTAND_TO_STAND) { n_h2s++; cost_switch += ec; }
    }
    const double resummed_cost = path_cost_from_edges(path_nodes, cfg);

    const auto groups = skill_groups(path_nodes);
    std::string sequence;
    for (const auto &g : groups)
    {
        if (!sequence.empty())
            sequence += " -> ";
        sequence += skill_name(g.first);
        if (g.second > 1)
            sequence += " x" + std::to_string(g.second);
    }
    if (sequence.empty())
        sequence = "(none: the start already satisfies the goal)";

    // Is MOVE -> STAND_TO_HANDSTAND -> HANDSTAND_MOVE -> HANDSTAND_TO_STAND -> MOVE in the chain?
    const SkillPrimitive pattern[5] = {SkillPrimitive::MOVE, SkillPrimitive::STAND_TO_HANDSTAND,
                                       SkillPrimitive::HANDSTAND_MOVE, SkillPrimitive::HANDSTAND_TO_STAND,
                                       SkillPrimitive::MOVE};
    bool has_pattern = false;
    for (std::size_t i = 0; i + 5 <= groups.size() && !has_pattern; ++i)
    {
        has_pattern = true;
        for (std::size_t j = 0; j < 5; ++j)
            if (groups[i + j].first != pattern[j])
                has_pattern = false;
    }

    std::string missing;
    if (n_move == 0)   missing += " MOVE";
    if (n_s2h == 0)    missing += " STAND_TO_HANDSTAND";
    if (n_hsmove == 0) missing += " HANDSTAND_MOVE";
    if (n_h2s == 0)    missing += " HANDSTAND_TO_STAND";

    const RRTNode &last = path_nodes.back();
    std::printf("\n=== Skill-Chain Summary (final path) ===\n");
    std::printf("  Skill sequence           : %s\n", sequence.c_str());
    std::printf("  MOVE steps               : %ld\n", n_move);
    std::printf("  HANDSTAND_MOVE steps     : %ld\n", n_hsmove);
    std::printf("  STAND_TO_HANDSTAND       : %ld\n", n_s2h);
    std::printf("  HANDSTAND_TO_STAND       : %ld\n", n_h2s);
    std::printf("  Mode switches (total)    : %ld\n", n_s2h + n_h2s);
    std::printf("  Final mode               : %s\n", mode_name(last.mode));
    std::printf("  Final goal error (XY)    : %.4f m (goal_tol %.3f)\n",
                last.point.distance(cfg.goal, {0, 1}), cfg.goal_tol);
    std::printf("  Final path cost          : %.4f (re-summed from edges: %.4f)\n", path_cost, resummed_cost);
    std::printf("  Cost breakdown           : MOVE %.4f (x%.2f) | HANDSTAND_MOVE %.4f (x%.2f) | switches %.4f (+%.3f each)\n",
                cost_move, cfg.stand_motion_weight, cost_hsmove, cfg.handstand_motion_weight,
                cost_switch, cfg.mode_switch_cost);
    std::printf("  Strict chain benchmark   : %s\n",
                cfg.require_move_before_and_after_handstand ? "enabled" : "disabled");
    if (missing.empty())
        std::printf("  Primitives used          : all four\n");
    else
        std::printf("  Primitives used          : NOT all four (missing:%s)\n", missing.c_str());
    if (n_s2h + n_h2s == 0)
        std::printf("  -> This solution uses MOVE only; no HandStand was needed.\n");
    std::printf("  MOVE -> STAND_TO_HANDSTAND -> HANDSTAND_MOVE -> HANDSTAND_TO_STAND -> MOVE : %s\n",
                has_pattern ? "present" : "NOT present");
}

// ============================================================
// Print configuration summary
// ============================================================
static void print_config(const PlannerConfig &cfg)
{
    std::printf("\n=== SE(2) x {STAND, HANDSTAND} Skill-Chain RRT (MOVE + HandStand) ===\n");
    std::printf("  Start      : (%.3f, %.3f, %.3f) STAND\n",
                cfg.start.x, cfg.start.y, cfg.start.theta);
    std::printf("  Goal       : (%.3f, %.3f, [Any]) STAND\n",
                cfg.goal.x, cfg.goal.y);
    std::printf("  Robot R    : %.3f m\n", cfg.robot_radius);
    std::printf("  STAND fp   : L=%.3f x W=%.3f m\n", cfg.robot_length, cfg.robot_width);
    std::printf("  HANDST. fp : L=%.3f x W=%.3f m\n", HANDSTAND_LENGTH, HANDSTAND_WIDTH);
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
    std::printf("  Edge cost  : MOVE %.2f x SE2 | HANDSTAND_MOVE %.2f x SE2 | switch SE2 + %.3f\n",
                cfg.stand_motion_weight, cfg.handstand_motion_weight, cfg.mode_switch_cost);
    std::printf("  Goal bias  : %d %%\n", cfg.goal_bias_percent);
    std::printf("  Strict     : require_move_before_and_after_handstand = %s\n",
                cfg.require_move_before_and_after_handstand ? "true" : "false");
    std::printf("  w_theta    : %.3f\n", Configuration::weight_theta);
    std::printf("  Model check: max|B*B_inv - I|  MOVE %.2e   HANDSTAND_MOVE %.2e\n\n",
                affine_inverse_residual(SKILL_MOVE_B, SKILL_MOVE_B_INV),
                affine_inverse_residual(SKILL_HSMOVE_B, SKILL_HSMOVE_B_INV));
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

    // Edge costs must stay non-negative (AO-RRT pruning assumes cost never decreases along a branch)
    if (!(cfg.mode_switch_cost >= 0.0) || !(cfg.stand_motion_weight >= 0.0) || !(cfg.handstand_motion_weight >= 0.0))
    {
        std::fprintf(stderr, "ERROR: mode_switch_cost, stand_motion_weight and handstand_motion_weight must be >= 0 "
                             "(got %f, %f, %f)\n",
                     cfg.mode_switch_cost, cfg.stand_motion_weight, cfg.handstand_motion_weight);
        return 1;
    }
    if (cfg.goal_bias_percent < 0 || cfg.goal_bias_percent > 100)
    {
        std::fprintf(stderr, "ERROR: goal_bias_percent must be in [0, 100] (got %d)\n", cfg.goal_bias_percent);
        return 1;
    }

    // The hard-coded learned models must be consistent: B * B_inv == I
    if (affine_inverse_residual(SKILL_MOVE_B, SKILL_MOVE_B_INV) > 1e-6 ||
        affine_inverse_residual(SKILL_HSMOVE_B, SKILL_HSMOVE_B_INV) > 1e-6)
    {
        std::fprintf(stderr, "ERROR: learned affine model B * B_inv deviates from identity\n");
        return 1;
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

    // ── Cleanup old files (this planner's own outputs only) ──
    std::remove(cfg.waypoints_out.c_str());
    std::remove(cfg.steps_out.c_str());
    std::remove(cfg.tree_out.c_str());

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
        if (cfg.require_move_before_and_after_handstand)
            std::printf("(Strict benchmark enabled: no path with MOVE -> STAND_TO_HANDSTAND -> HANDSTAND_MOVE -> "
                        "HANDSTAND_TO_STAND -> MOVE ending in the goal was found.)\n");
        return 0; // Still return 0 so user can see the setup in viz
    }

    std::printf("\nGlobally best cost: %.4f (seed=%u)\n", best_overall.cost, best_overall.seed);
    std::printf("Best run rejections: dist=%ld, rot=%ld, col=%ld, dup=%ld, cost_pruned=%ld, bad_cmd=%ld, strict_phase=%ld\n",
                best_overall.rej_dist, best_overall.rej_rot, best_overall.rej_col, best_overall.rej_dup, best_overall.rej_cost,
                best_overall.rej_cmd, best_overall.rej_phase);

    long n_stand_nodes = 0, n_switch_edges = 0;
    for (const auto &n : best_overall.tree) {
        if (n.mode == RobotMode::STAND) n_stand_nodes++;
        if (is_mode_switch(n.incoming_skill)) n_switch_edges++;
    }
    std::printf("Best run tree: %zu nodes (STAND %ld, HANDSTAND %ld), %ld mode-switch edges\n",
                best_overall.tree.size(), n_stand_nodes,
                static_cast<long>(best_overall.tree.size()) - n_stand_nodes, n_switch_edges);

    write_rrt_tree(best_overall.tree, best_overall.seed, cfg.tree_out);   // written even if invalid, for diagnosis
    std::vector<Configuration> path = best_overall.path;


    const bool path_valid = validate_final_path(best_overall.path_nodes, cfg, best_overall.cost);
    print_skill_summary(best_overall.path_nodes, cfg, best_overall.cost);

    if (!path_valid) {
        std::printf("\nERROR: final path failed validation; %s and %s were NOT written.\n",
                    cfg.waypoints_out.c_str(), cfg.steps_out.c_str());
        return 1;
    }

    if (!write_waypoints_file(path, cfg.waypoints_out))
        return 1;

    if (!write_skill_steps_csv(best_overall.path_nodes, cfg, cfg.steps_out))
        return 1;

    return 0;
}
