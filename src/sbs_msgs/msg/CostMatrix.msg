# Utility of every (own rollout, partner rollout) pair, for both agents.
#
# The joint problem over sampled rollouts is a small bimatrix game, and every
# solution concept -- independent, anticipated partner, social optimum,
# gamma-weighted, optimistic -- is a different reduction of these same two
# matrices. Publishing them rather than a decision keeps the cost function and
# the coordination model independently swappable, and it is exactly the
# quantity offline inverse RL needs, so recording this topic is enough to fit
# the weights later.
#
# Matrices are row-major over own rollouts: pair (i, j) is at i * n_partner + j.
#
# Size warning: at ~360 rollouts each these are ~1 MB per publication, so 10 Hz
# is ~10 MB/s. Cut n_v/n_w in the generator (the count falls quadratically) or
# compose evaluator and selector into one container for zero-copy if it bites.

std_msgs/Header header

uint16 n_own
uint16 n_partner

float32[] u_own                # n_own * n_partner, utility to this agent
float32[] u_partner            # n_own * n_partner, utility to the partner
bool[] feasible_own            # n_own; false means it clips a wall

string[] term_names
float32[] term_weights

# Context the utilities were computed under, so a consumer or an offline fit
# can reproduce them.
float32 curvature
float32 object_distance
float32 target_separation      # d_R*(object_distance)
float32 preferred_speed        # v*(curvature)
