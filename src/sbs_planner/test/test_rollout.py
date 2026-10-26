"""Tests for control-space rollout generation. No ROS needed."""

import numpy as np
import pytest

from sbs_planner.rollout import (RolloutConfig, VehicleModel, command_grid,
                                 generate, integrate, sample_commands)


def test_config_step_counts():
    cfg = RolloutConfig(horizon=3.0, dt=0.1, segment_split=1.5)
    assert cfg.n_steps == 30 and cfg.split_step == 15


def test_lag_reproduces_first_order_step_response():
    """The whole point of putting the lag inside the integrator."""
    cfg = RolloutConfig(n_v=1, n_w=1)
    commands = np.array([[1.0, 0.0, 1.0, 0.0]])
    out = integrate((0, 0, 0, 0, 0), commands, cfg)
    t = np.arange(1, cfg.n_steps + 1) * cfg.dt
    expected = 1.0 * (1.0 - np.exp(-t / cfg.model.tau_v))
    assert np.abs(out["v"][0] - expected).max() < 1e-9


def test_lag_means_the_vehicle_undershoots_a_naive_plan():
    cfg = RolloutConfig(n_v=1, n_w=1)
    commands = np.array([[1.2, 0.0, 1.2, 0.0]])
    out = integrate((0, 0, 0, 0, 0), commands, cfg)
    naive = 1.2 * cfg.horizon                      # if it reached speed at once
    assert out["x"][0, -1] < naive * 0.95          # measurably short


def test_command_grid_clips_to_envelope_and_dedupes():
    grid = command_grid(0.1, 0.6, 5, 0.0, 1.3)
    assert grid.min() >= 0.0 and grid.max() <= 1.3
    assert len(np.unique(grid)) == len(grid)


def test_samples_are_centred_on_the_current_state():
    cfg = RolloutConfig()
    commands = sample_commands(0.9, 0.0, cfg)
    # The reachable set brackets the current speed.
    assert commands[:, 0].min() < 0.9 < commands[:, 0].max()


def test_lateral_acceleration_limit_is_enforced():
    model = VehicleModel(lat_acc_max=0.3)
    cfg = RolloutConfig(model=model)
    commands = sample_commands(1.2, 0.6, cfg)
    assert np.all(np.abs(commands[:, 0] * commands[:, 1]) <= 0.3 + 1e-9)
    assert np.all(np.abs(commands[:, 2] * commands[:, 3]) <= 0.3 + 1e-9)


def test_second_segment_actually_changes_the_command():
    """A single arc cannot turn then straighten; two segments must."""
    cfg = RolloutConfig()
    commands = sample_commands(1.0, 0.5, cfg)
    assert np.any(np.abs(commands[:, 3] - commands[:, 1]) > 1e-6)


def test_trajectory_shapes_and_finiteness():
    cfg = RolloutConfig()
    commands, traj = generate((1.0, 2.0, 0.3, 1.0, 0.1), cfg)
    for name in ("x", "y", "yaw", "v", "w"):
        assert traj[name].shape == (len(commands), cfg.n_steps)
        assert np.all(np.isfinite(traj[name]))


def test_rollouts_start_from_the_given_pose():
    cfg = RolloutConfig()
    commands, traj = generate((5.0, -2.0, 0.0, 0.0, 0.0), cfg)
    # First step has barely moved from the origin state.
    assert np.all(np.hypot(traj["x"][:, 0] - 5.0, traj["y"][:, 0] + 2.0) < 0.05)


def test_straight_command_travels_along_heading():
    cfg = RolloutConfig(n_v=1, n_w=1)
    commands = np.array([[1.0, 0.0, 1.0, 0.0]])
    out = integrate((0.0, 0.0, np.pi / 2, 1.0, 0.0), commands, cfg)
    assert out["x"][0, -1] == pytest.approx(0.0, abs=1e-6)
    assert out["y"][0, -1] > 2.5


def test_candidates_are_distinct():
    cfg = RolloutConfig()
    commands, traj = generate((0.0, 0.0, 0.0, 1.1, 0.0), cfg)
    ends = np.column_stack([traj["x"][:, -1], traj["y"][:, -1]])
    assert len(np.unique(np.round(ends, 3), axis=0)) > 0.8 * len(commands)


def test_set_size_is_tractable_for_a_pairwise_matrix():
    cfg = RolloutConfig()
    for state in ((0, 0, 0, 1.1, 0.0), (0, 0, 0, 0.85, 0.45), (0, 0, 0, 0.0, 0.0)):
        commands, _ = generate(state, cfg)
        assert 30 < len(commands) < 600      # matrix stays ~10^5 pairs


def test_pair_matrices_are_transposes():
    """The property every cooperative concept rests on.

    Both vehicles must solve the same bimatrix -- each one's own-utility being
    the transpose of the other's partner-utility -- because that is what lets
    two agents choosing independently, with nothing shared between them, land
    on the two halves of one joint optimum. Coarsening one axis of the pair
    breaks it silently: the terms stay correct and the formation still falls
    apart.
    """
    from sbs_planner.cost import PairContext, Weights, evaluate

    cfg = RolloutConfig()
    _, a = generate((0.0, 0.0, 0.1, 1.0, 0.05), cfg)
    _, b = generate((0.3, 1.4, 0.1, 0.9, 0.05), cfg)
    # The two counts need not match -- `generate` prunes against each
    # vehicle's own state -- but each vehicle must build the other's set from
    # the other's state, which is what makes the matrices transposes.

    shared = dict(dt=cfg.dt, kappa_own=0.2, kappa_partner=0.2,
                  object_distance=5.0)
    path_a = np.linspace(0.0, 1.0, a["x"].shape[0])
    path_b = np.linspace(0.0, 1.0, b["x"].shape[0])
    weights = Weights()

    own_of_a = evaluate(a, b, PairContext(
        path_dist_own=path_a, path_dist_partner=path_b,
        side_sign_own=1.0, side_sign_partner=-1.0, **shared), weights)[0]
    partner_of_b = evaluate(b, a, PairContext(
        path_dist_own=path_b, path_dist_partner=path_a,
        side_sign_own=-1.0, side_sign_partner=1.0, **shared), weights)[1]

    assert np.abs(own_of_a - partner_of_b.T).max() < 1e-3


def test_standstill_is_not_admissible_on_a_clear_route():
    """Every formation term is zero at zero motion, so parking is an optimum
    of the objective. It has to be excluded by constraint, not by weight."""
    from sbs_planner.cost import admissible, moving

    cfg = RolloutConfig()
    _, from_rest = generate((0.0, 0.0, 0.0, 0.0, 0.0), cfg)
    ok = moving(from_rest, 0.30)
    assert ok.any()
    assert from_rest["v"][ok, :5].mean(axis=1).min() > 0.0

    # With no walls anywhere, the speed floor is what prunes the candidates.
    kept = admissible(from_rest, None, 0.30)
    assert kept.sum() < len(kept)


def test_limiter_preserves_the_path_it_slows():
    """A clamp must not reshape the trajectory the planner checked.

    Clipping omega without v changes omega/v -- the curvature -- so the
    vehicle would follow a different path from the one tested against the
    walls. Scaling both leaves the geometry alone and only slows the
    traversal.
    """
    from sbs_planner.velocity_limiter_node import scale_to_limits

    # inside every limit: a no-op
    assert scale_to_limits(0.9, 0.1, 1.05, 0.80, 1.50) == 1.0

    for v, w in ((1.30, 0.20), (2.00, 0.20), (1.00, 1.20), (1.50, 1.50)):
        k = scale_to_limits(v, w, 1.05, 0.80, 1.50)
        assert 0.0 < k < 1.0
        assert abs(v * k) <= 1.05 + 1e-9
        assert abs(w * k) <= 0.80 + 1e-9
        assert abs(v * k * w * k) <= 1.50 + 1e-9
        assert abs((w * k) / (v * k) - w / v) < 1e-9      # curvature held
