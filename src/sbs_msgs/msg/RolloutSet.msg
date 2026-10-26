# A set of candidate futures produced by one agent's rollout generator.
#
# Flat parallel arrays rather than an array of Rollout messages: a few hundred
# rollouts x 30 steps is already ~200 KB per publication, and per-element
# message overhead would make it materially worse.
#
# Trajectories are laid out rollout-major: rollout r step k is at index
# r * n_steps + k. Commands are 4 per rollout, [v_a, w_a, v_b, w_b], being the
# two piecewise-constant segments.
#
# Nothing here says what the agent intends to do -- it is the whole candidate
# set, and the same message type carries an agent's own rollouts and its
# prediction of its partner's. Neither is ever sent between vehicles.

std_msgs/Header header

uint16 n_rollouts
uint16 n_steps
float32 dt
float32 segment_split          # seconds before the command switches

float32[] commands             # 4 * n_rollouts
float32[] x                    # n_rollouts * n_steps, all in header.frame_id
float32[] y
float32[] yaw
float32[] v                    # achieved, after the actuation lag
float32[] w

# State the set was generated from, so a consumer can reproduce or check it.
float32 origin_x
float32 origin_y
float32 origin_yaw
float32 origin_v
float32 origin_w
