"""Tests for Episode, EpisodeStep, and termination helpers."""

import pytest
from choreo_mini.core.episode import (
    Episode,
    EpisodeStep,
    nash_convergence_detector,
    max_rounds_terminator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_episode(
    n_agents=2,
    reward_val=1.0,
    max_rounds=10,
    termination_fn=None,
    env_update_fn=None,
):
    """Build a minimal Episode with deterministic stub agents."""
    agents = {f"agent_{i}": (lambda env, r, i=i: f"action_{i}") for i in range(n_agents)}
    env = {"round": 0}

    def reward_fn(env_state, actions, round_):
        return {name: reward_val for name in actions}

    return Episode(
        agents=agents,
        env=env,
        reward_fn=reward_fn,
        env_update_fn=env_update_fn,
        termination_fn=termination_fn,
        max_rounds=max_rounds,
    )


# ---------------------------------------------------------------------------
# EpisodeStep
# ---------------------------------------------------------------------------

class TestEpisodeStep:
    def test_fields(self):
        step = EpisodeStep(round=1, env_state={"x": 1}, actions={"a": "act"}, rewards={"a": 0.5})
        assert step.round == 1
        assert step.env_state == {"x": 1}
        assert step.actions == {"a": "act"}
        assert step.rewards == {"a": 0.5}

    def test_default_rewards_empty(self):
        step = EpisodeStep(round=1, env_state={}, actions={})
        assert step.rewards == {}


# ---------------------------------------------------------------------------
# Episode construction
# ---------------------------------------------------------------------------

class TestEpisodeInit:
    def test_requires_at_least_one_agent(self):
        with pytest.raises(ValueError, match="at least one agent"):
            Episode(agents={}, env={}, reward_fn=lambda e, a, r: {})

    def test_requires_max_rounds_ge_1(self):
        with pytest.raises(ValueError, match="max_rounds"):
            Episode(
                agents={"a": lambda e, r: "x"},
                env={},
                reward_fn=lambda e, a, r: {"a": 0.0},
                max_rounds=0,
            )

    def test_initial_state(self):
        ep = _make_episode()
        assert ep.round == 0
        assert ep.done is False
        assert ep.trajectory == []


# ---------------------------------------------------------------------------
# Episode.step()
# ---------------------------------------------------------------------------

class TestEpisodeStep_:
    def test_step_increments_round(self):
        ep = _make_episode()
        ep.step()
        assert ep.round == 1

    def test_step_returns_episode_step(self):
        ep = _make_episode()
        s = ep.step()
        assert isinstance(s, EpisodeStep)
        assert s.round == 1

    def test_step_records_actions(self):
        ep = _make_episode(n_agents=2)
        s = ep.step()
        assert "agent_0" in s.actions
        assert "agent_1" in s.actions
        assert s.actions["agent_0"] == "action_0"

    def test_step_records_rewards(self):
        ep = _make_episode(reward_val=2.5)
        s = ep.step()
        assert s.rewards["agent_0"] == pytest.approx(2.5)

    def test_step_appends_to_trajectory(self):
        ep = _make_episode()
        ep.step()
        ep.step()
        assert len(ep.trajectory) == 2

    def test_step_raises_when_done(self):
        ep = _make_episode(max_rounds=1)
        ep.step()
        assert ep.done
        with pytest.raises(RuntimeError, match="already done"):
            ep.step()

    def test_env_snapshot_is_copy(self):
        """Mutations to env after step should not affect the recorded snapshot."""
        ep = _make_episode()
        ep._env["round"] = 0
        s = ep.step()
        ep._env["round"] = 999          # mutate live env
        assert s.env_state["round"] == 0

    def test_env_update_fn_called(self):
        called = []

        def updater(env, actions, round_):
            called.append(round_)
            new = dict(env)
            new["updated"] = True
            return new

        ep = _make_episode(env_update_fn=updater)
        ep.step()
        assert called == [1]
        assert ep.env["updated"] is True

    def test_no_env_update_fn_leaves_env_unchanged(self):
        ep = _make_episode()
        original_env = dict(ep.env)
        ep.step()
        assert ep.env == original_env


# ---------------------------------------------------------------------------
# Episode.run()
# ---------------------------------------------------------------------------

class TestEpisodeRun:
    def test_run_returns_full_trajectory(self):
        ep = _make_episode(max_rounds=5)
        traj = ep.run()
        assert len(traj) == 5
        assert ep.done

    def test_run_stops_at_termination(self):
        term = max_rounds_terminator(3)
        ep = _make_episode(max_rounds=10, termination_fn=term)
        traj = ep.run()
        assert len(traj) == 3

    def test_run_when_already_done_returns_empty_additional(self):
        ep = _make_episode(max_rounds=2)
        ep.run()
        assert ep.done
        # run again should return same trajectory (while loop exits immediately)
        traj2 = ep.run()
        assert len(traj2) == 2          # trajectory unchanged


# ---------------------------------------------------------------------------
# Episode.reset()
# ---------------------------------------------------------------------------

class TestEpisodeReset:
    def test_reset_clears_state(self):
        ep = _make_episode(max_rounds=3)
        ep.run()
        ep.reset()
        assert ep.round == 0
        assert ep.done is False
        assert ep.trajectory == []

    def test_reset_with_new_env(self):
        ep = _make_episode()
        ep.reset(env={"round": 99})
        assert ep.env["round"] == 99

    def test_reset_allows_rerun(self):
        ep = _make_episode(max_rounds=2)
        ep.run()
        ep.reset()
        traj = ep.run()
        assert len(traj) == 2


# ---------------------------------------------------------------------------
# Episode.summary()
# ---------------------------------------------------------------------------

class TestEpisodeSummary:
    def test_summary_before_run(self):
        ep = _make_episode()
        s = ep.summary()
        assert s["rounds"] == 0
        assert s["cumulative_rewards"] == {}

    def test_summary_after_run(self):
        ep = _make_episode(n_agents=2, reward_val=1.0, max_rounds=4)
        ep.run()
        s = ep.summary()
        assert s["rounds"] == 4
        assert s["done"] is True
        # each agent earns 1.0 per round × 4 rounds = 4.0
        assert s["cumulative_rewards"]["agent_0"] == pytest.approx(4.0)
        assert s["cumulative_rewards"]["agent_1"] == pytest.approx(4.0)

    def test_summary_final_actions(self):
        ep = _make_episode(n_agents=1, max_rounds=3)
        ep.run()
        s = ep.summary()
        assert "agent_0" in s["final_actions"]


# ---------------------------------------------------------------------------
# nash_convergence_detector
# ---------------------------------------------------------------------------

class TestNashConvergenceDetector:
    def test_requires_window_ge_2(self):
        with pytest.raises(ValueError, match="window"):
            nash_convergence_detector(window=1)

    def test_does_not_fire_before_window(self):
        detect = nash_convergence_detector(window=3, reward_threshold=0.01)
        traj = [
            EpisodeStep(round=i, env_state={}, actions={}, rewards={"a": 1.0})
            for i in range(1, 3)        # only 2 steps, window=3
        ]
        assert detect(traj[-1], traj) is False

    def test_fires_when_rewards_stable(self):
        detect = nash_convergence_detector(window=3, reward_threshold=0.01)
        traj = [
            EpisodeStep(round=i, env_state={}, actions={}, rewards={"a": 1.0, "b": 2.0})
            for i in range(1, 4)
        ]
        assert detect(traj[-1], traj) is True

    def test_does_not_fire_when_rewards_vary(self):
        detect = nash_convergence_detector(window=3, reward_threshold=0.01)
        rewards_sequence = [0.0, 0.5, 1.0]
        traj = [
            EpisodeStep(round=i, env_state={}, actions={}, rewards={"a": r})
            for i, r in enumerate(rewards_sequence, start=1)
        ]
        assert detect(traj[-1], traj) is False

    def test_integration_episode_converges(self):
        """Episode should converge once rewards stabilise across window rounds."""
        detect = nash_convergence_detector(window=3, reward_threshold=0.01)
        ep = Episode(
            agents={"a": lambda e, r: "act"},
            env={},
            reward_fn=lambda e, a, r: {"a": 1.0},   # constant — always stable
            termination_fn=detect,
            max_rounds=20,
        )
        traj = ep.run()
        assert len(traj) == 3          # window=3, fires as soon as window fills
        assert ep.done


# ---------------------------------------------------------------------------
# max_rounds_terminator
# ---------------------------------------------------------------------------

class TestMaxRoundsTerminator:
    def test_fires_at_n(self):
        term = max_rounds_terminator(4)
        traj = [EpisodeStep(round=i, env_state={}, actions={}) for i in range(1, 5)]
        assert term(traj[-1], traj) is True

    def test_does_not_fire_before_n(self):
        term = max_rounds_terminator(4)
        traj = [EpisodeStep(round=i, env_state={}, actions={}) for i in range(1, 4)]
        assert term(traj[-1], traj) is False


# ---------------------------------------------------------------------------
# Package-level import
# ---------------------------------------------------------------------------

class TestPackageExports:
    def test_episode_importable_from_top_level(self):
        import choreo_mini
        assert hasattr(choreo_mini, "Episode")
        assert hasattr(choreo_mini, "EpisodeStep")
        assert hasattr(choreo_mini, "nash_convergence_detector")
        assert hasattr(choreo_mini, "max_rounds_terminator")
