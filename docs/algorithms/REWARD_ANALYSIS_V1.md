## Reward Function Engineering & Empirical Analysis (V1)

This document describes **reward design V1** and the observations from the
corresponding experiments. It is a historical baseline, not a final reward
specification. When the reward calculation changes, create a new versioned
analysis document rather than replacing this one.

The reward function combines a distance-progress shaping term with terminal
outcomes. The DQN training method is documented separately in
[`training.md`](../training.md).

---

### 1. Mathematical Formulation

The reward calculation is split across `AgentCar::compute_reward()` and the
main simulation system. `compute_reward()` handles collision, parking, and
ordinary progress rewards. The simulation system applies the timeout reward
when the episode reaches 3,000 steps.

$$
r_t =
\begin{cases}
-5.0 & \text{if the car has crashed} \\
+5.0 & \text{if the car is parked} \\
-2.0 & \text{if the episode reaches 3,000 steps} \\
\mathrm{clamp}(\Delta d - 0.005,\,-0.05,\,0.05)
  & \text{otherwise}
\end{cases}
$$

Each of the first three cases sets $\text{done}_t=\text{true}$; the ordinary
progress case sets $\text{done}_t=\text{false}$.

Where:

- $\Delta d = d_{t-1} - d_t$. It is positive when the car moves closer to
  the goal and negative when it moves away.
- `STEP_PENALTY` ($0.005$) is subtracted on every non-terminal step. It
  penalizes taking additional steps, including idle steps.
- `DISTANCE_REWARD_SCALE` is currently `1.0`, so the progress term is
  `DISTANCE_REWARD_SCALE * Δd`.
- The ordinary step reward is hard-clamped to $[-0.05, 0.05]$. Terminal
  rewards are not passed through this clamp.

The previous distance is initialized when the car is reset and is updated
after the transition is recorded. This makes $\Delta d$ represent progress
between consecutive simulation steps.

#### Parking Condition:
An episode is marked successful (`is_parked = true`) when both conditions hold:
$$\text{Distance to Target} \le 25.0 \text{ px} \quad \land \quad \vert{}\text{Speed}\vert{} \le 10.0 \text{ units/s}$$

---

### 2. Reward Constants and Episode Rules

| Constant | Value | Description |
| :--- | :---: | :--- |
| `CRASH_PENALTY` | `-5.0` | Penalty for wall or obstacle collision |
| `PARKING_REWARD` | `+5.0` | Reward for reaching target below speed threshold |
| `TIMEOUT_PENALTY` | `-2.0` | Penalty for failing to park before episode step limit |
| `STEP_PENALTY` | `0.005` | Constant step cost to incentivize shortest path |
| `PARKING_RADIUS` | `25.0` | Target arrival proximity threshold |
| `PARKING_SPEED_THRESHOLD` | `10.0` | Maximum velocity allowed for successful parking |
| `DISTANCE_REWARD_SCALE` | `1.0` | Multiplier for distance delta ($\Delta d$) |
| `MAX_EPISODE_STEPS` | `3,000` | Timeout limit applied by the simulation system |

Terminal conditions are checked in this order:

1. A crash ends the episode with `CRASH_PENALTY`.
2. Otherwise, a valid parked state ends the episode with `PARKING_REWARD`.
3. Otherwise, reaching `MAX_EPISODE_STEPS` ends the episode with
   `TIMEOUT_PENALTY`.
4. If none applies, the progress reward is returned and the episode
   continues.

The timeout branch is implemented in
[`system.rs`](../../src/system.rs), while the reward constants and parking
check are implemented in
[`agent_car.rs`](../../src/agent/agent_car.rs).

---

### 3. Empirical Observations & Behavior Analysis

During the experiments associated with this V1 baseline, several distinct
agent behaviors and failure modes emerged:

* **Low Top Speed & Hesitation**: Even though the physics engine allows higher speeds, the agent defaults to low-velocity movements to maximize control and avoid overshooting distance rewards.
* **Obstacle Local Minima & Detour Paralysis**:
  * When faced with a wall blocking the direct path to the goal, the agent struggles to execute wide detours.
  * Because moving around a wall temporarily increases distance to goal ($\Delta d < 0$), the step reward becomes negative, trapping the car in a local minimum in front of the wall.
* **LiDAR Sensing Anomalies**:
  * **Wall Lockup**: If 2 or 3 LiDAR rays register a wall in close proximity, the agent often freezes completely for several frames.
  * **Open Space Inaction**: Conversely, when no LiDAR rays register obstacles (all clear), the agent sometimes exhibits passive or zero-action behaviors due to a lack of immediate spatial gradients.
* **Straight-Line vs. Corridor Navigation**:
  * **Line-of-Sight Scenarios**: Performs reliably when there are no obstacles between the agent and the goal spot.
  * **Narrow Gaps**: Over time, the agent successfully learned to squeeze through tight gaps between two obstacles, provided the trajectory reduced overall distance to the goal.

---

### 4. Performance Metrics (V1 Benchmark)

Training was conducted over **3,000 episodes** with epsilon decayed to
$\varepsilon = 0.05$:

The results below were obtained with the
[`model_V1.bin`](../../checkpoints/model_V1.bin) checkpoint, which is the
current V1 model used for evaluation.

These values are empirical observations, not guaranteed performance targets.
They depend on the model checkpoint, random seeds, map generation, simulator
timing, and evaluation protocol. Record those details for future comparisons.

* **Average Steps per Episode**: $\approx 405$ steps
* **Convergence & Loss**: The loss stabilized at a constant plateau with maximum $Q$-values oscillating predictably between **$4.0$ and $6.0$**.
* **Success Rate**:
  * **Inference Mode ($\varepsilon = 0.0$)**: $\mathbf{\approx 30\%}$ success rate over 100 evaluation episodes.
  * **Training Mode ($\varepsilon = 0.05$)**: $\mathbf{\approx 64\%}$ success rate (slight random action noise helped the agent break free from local freeze states).

The difference between training and inference success rates suggests that the
V1 policy can benefit from occasional random actions, but it does not by
itself establish that exploration improves the learned policy. A deterministic
evaluation protocol should be used when comparing future reward versions.