# CBF-QP safety filter: formulation

Design note for the Week 2 filter (MP). Status: formulation only; no solver code yet. Numbers marked
*proxy* come from the public `lerobot/svla_so101_pickplace` demos (another SO-101) and must be
remeasured on our arm.

The filter takes one SmolVLA action chunk and returns the closest chunk (in a tool-motion metric)
that respects joint position and velocity limits and keeps every collision sphere clear of the
distance field and the table along the whole commanded path, between ticks included. It is the
identity when the chunk is already safe. It runs on the CPU, once per chunk.

## 1. Notation

| Symbol | Meaning |
| --- | --- |
| $n = 5$ | Arm joints, order `ARM_JOINTS` (pan, lift, elbow, wrist flex, wrist roll). The gripper is not a decision variable |
| $\Delta t = 1/30$ s, $H = 50$ | Control period and chunk length (LeRobot defaults; one chunk = 1.67 s) |
| $d \in \{0,\dots,H-1\}$ | Committed actions: already sent and not changeable (0 in synchronous execution; > 0 with real-time chunking) |
| $a_k \in \mathbb{R}^6$ | VLA action for tick $k = 1..H$ (LeRobot degrees, gripper 0–100) |
| $\varphi$ | `kinematics.lerobot_to_urdf`: per joint $\varphi_j(a) = \varsigma_j \frac{\pi}{180}(a_j + o_j)$, affine and diagonal (signs $\varsigma_j$, offsets $o_j$) |
| $r_k \in \mathbb{R}^n$ | Reference: arm part of $\varphi(a_k)$, radians. $\theta_k$: gripper part, passed through unchanged |
| $\bar q_0$ | Measured arm position at chunk start; $\bar q_k$ for $1 \le k \le d$ the committed commands |
| $q_k,\ k \in \mathcal{K} = \{d+1,\dots,H\}$ | Decision: commanded arm position at tick $k$ |
| $\mathcal{I}$ | Collision spheres of all links except `base_link` ($\lvert\mathcal{I}\rvert = 30$) |
| $c_i, \rho_i, \ell(i)$ | Sphere centre in its link frame, radius (1 mm margin included), link |
| $p_i(q) \in \mathbb{R}^3$ | Sphere centre in the base frame at arm position $q$ (gripper at $\theta_k$) |
| $D : \mathbb{R}^3 \to \mathbb{R}$ | `DistanceField`: trilinear interpolant of the signed EDT, clamped outside the box |
| $z_T = -0.0024$ m | Table height in the base frame (base mesh bottom) |
| $q_{\min}, q_{\max}$ | `joint_limits()` (URDF ∩ calibrated range), shrunk by $\epsilon_q = 1°$ |
| $\bar v \in \mathbb{R}^n_{>0}$ | Planning velocity limit per joint (rad/s), $\bar v \le$ 4.7 rad/s servo maximum |

Inequalities and $\lvert\cdot\rvert$ on vectors are elementwise.

## 2. Assumptions

Each assumption is a modelling claim with a number we measure; the margins in §3 absorb them.

- **A1 Tracking.** The real arm stays within a per-joint distance $\epsilon_{\text{off}} \in \mathbb{R}^n$ of the
  commanded joint-space polyline $\bar q_0, q_1, \dots$. It may lag along the polyline (the
  servos lag the command by up to 26° at p99, *proxy*); only the off-path part matters.
  *Proxy:* distance to the last 0.5 s of commands, p50 2.2°, p99 4.3°, max 5.1° (Euclidean over the
  five joints). Includes leader/follower calibration mismatch, absent at policy execution.
- **A2 Monotone joints.** Between two ticks each joint moves monotonically from $q_{k-1,j}$ to
  $q_{k,j}$ and stays inside $[q_{\min}, q_{\max}]$ (true for a per-joint position servo with no overshoot;
  overshoot goes into $\epsilon_{\text{off}}$).
- **A3 Static scene.** Obstacles do not move during the chunk. Week 4 replaces $D$ by a forecast
  $D_k$ per tick; every formula below holds with $D \to D_k$.
- **A4 Conservative field.** $D(p) \le$ true distance to observed obstacles for points outside them (true by
  construction up to depth noise $\epsilon_{\text{sense}}$: the field reads 0–1 voxel closer). Unobserved space counts as free.
- **A5 Kinematic model.** $\lVert p_i^{\text{true}} - p_i(q)\rVert \le \epsilon_{\text{FK}}$; after the touch fit,
  $\epsilon_{\text{FK}}$ = its maximum residual. Until then FK is off by 2–3 cm, too much for the filter to mean anything.

## 3. Barriers

**Lipschitz constant of the field.** Along each axis, neighbouring EDT values differ by at most one
voxel, so each component of $\nabla D$ is in $[-1, 1]$. The gradient norm of a trilinear interpolant is
largest at cell corners, so the exact constant of a built field is

```math
L_D = \max_{\text{cells},\ \text{corners}} \big\lVert (\Delta_x, \Delta_y, \Delta_z)/v \big\rVert_2 \;\le\; \sqrt{3},
```

with $\Delta$ the edge differences meeting at the corner. On the test fields $L_D = \sqrt{3}$
exactly (at obstacle corners), so we use $L_D = \sqrt3$. Clamping outside the box is a projection
and keeps the bound.

**Lever arms.** For joint $j$ upstream of $\ell(i)$, $\partial p_i / \partial q_j = z_j \times (p_i - o_j)$ (unit axis $z_j$,
origin $o_j$, base frame); zero otherwise. Its norm is at most $\lVert p_i - o_j \rVert$, which by the
triangle inequality along the chain is at most

```math
\beta_{ij} = \sum_{\text{joints } j' \text{ after } j \text{ up to } \ell(i)} \lVert t_{j'} \rVert + \lVert c_i \rVert ,
```

with $t_{j'}$ the URDF joint origin offsets. This holds for every configuration. Per-link
maxima (m), joints pan…roll: gripper 0.47 / 0.40 / 0.29 / 0.15 / 0.09; moving jaw 0.49 / 0.43 / 0.31 /
0.17 / 0.11; shoulder 0.05 / 0 / 0 / 0 / 0. Integrating along an axis-aligned path inside the joint box gives

```math
\lVert p_i(q) - p_i(q') \rVert \le \beta_i^\top \lvert q - q' \rvert \qquad \forall q, q' \in [q_{\min}, q_{\max}] .
```

**Obstacle barrier** per sphere, with every margin folded in:

```math
h_i(q) = D(p_i(q)) - \rho_i - \eta_i, \qquad
\eta_i = c_{\text{user}} + L_D\,\big(\beta_i^\top \epsilon_{\text{off}} + \epsilon_{\text{FK}}\big) + \epsilon_{\text{sense}} .
```

**Floor barrier** (exact plane, Lipschitz constant 1, no voxel error):

```math
f_i(q) = e_z^\top p_i(q) - z_T - \rho_i - \eta_i^{f}, \qquad \eta_i^f = c^f_{\text{user}} + \beta_i^\top \epsilon_{\text{off}} + \epsilon_{\text{FK}} .
```

The table is removed from the obstacle cloud (`geometry.remove_table`), so it enters only here.

**Lemma 1 (no collision between ticks).** Under A2, for any path $q(s)$ from $q_{k-1}$ to $q_k$,

```math
\min_s h_i(q(s)) \;\ge\; \sigma_{ik} := \tfrac12\big(h_i(q_{k-1}) + h_i(q_k)\big) - \kappa\,\beta_i^\top \lvert q_k - q_{k-1}\rvert,
\qquad \kappa = L_D/2 .
```

*Proof.* With $a(s) = \beta_i^\top\lvert q(s) - q_{k-1}\rvert$ and $b(s) = \beta_i^\top\lvert q(s) - q_k\rvert$, monotonicity gives
$a + b = B := \beta_i^\top\lvert q_k - q_{k-1}\rvert$. Since $h_i$ is $L_D$-Lipschitz in $p_i$, $h_i(q(s)) \ge \max(h_{k-1} - L_D a,\ h_k - L_D (B - a))$.
The minimum over $a \in [0, B]$ of this maximum of two lines is the value where they cross,
$\tfrac12(h_{k-1} + h_k - L_D B)$. The crossing lies in $[0,B]$ because $\lvert h_k - h_{k-1}\rvert \le L_D B$. ∎

For moving-jaw spheres the gripper joint moves too: add $\kappa\,\beta_{i,\text{grip}}\lvert\theta_k - \theta_{k-1}\rvert$, a known constant,
to the motion term. The same holds for $f_i$ with $\kappa^f = 1/2$. So $\sigma_{ik} \ge 0$ for all $i, k$ means the commanded path
keeps every sphere at least $\eta_i$ from observed obstacles, and the real arm at least $c_{\text{user}}$ under A1–A5. The motion term is a
speed limit that tightens near obstacles: at p50 demo speed it is ~3 mm for the gripper; with all
joints at their p99 speed simultaneously it is ~11 cm (*proxy*), forcing the arm to slow down near obstacles.

**Discrete-time CBF.** For a rate $\gamma \in (0, 1]$:

```math
\sigma_{i,d+1} \ge (1-\gamma)\, h_i(\bar q_d), \qquad
\sigma_{ik} \ge (1-\gamma)\, \sigma_{i,k-1}, \quad k = d+2,\dots,H .
```

If $h_i(\bar q_d) \ge 0$, induction gives $\sigma_{ik} \ge (1-\gamma)^{k-d} h_i(\bar q_d) \ge 0$ for every $k$: safe on every segment,
approaching the boundary at most geometrically. $\gamma = 1$ keeps only $\sigma \ge 0$. If $h_i(\bar q_d) < 0$ (an obstacle
appeared inside the margin), the same rows demand recovery at rate $\gamma$; the slack in §4 keeps that feasible.

## 4. The QP (one sequential-QP iteration)

**Linearisation point** $\hat q_k$, $k \in \mathcal{K}$: first the rate-limited reference $\Pi(r)$ (Lemma 3), then $\Pi$ of the previous
solution. With $\hat p_i = p_i(\hat q_k)$:

```math
\hat h_{ik} = h_i(\hat q_k), \qquad \hat g_{ik} = J_i(\hat q_k)^\top \nabla D(\hat p_i), \qquad
J_i(q)_{:,j} = \mathbb{1}[j \prec \ell(i)]\; z_j \times (p_i - o_j) ,
```

```math
\tilde h_{ik}(q) = \hat h_{ik} + \hat g_{ik}^\top (q - \hat q_k) \;\; (k \in \mathcal{K}), \qquad \tilde h_{ik} = h_i(\bar q_k) \;\; (k \le d,\ \text{exact}) .
```

$\nabla D$ is continuous inside a voxel and jumps across faces. It is evaluated exactly (`DistanceField.query`); the
Jacobian is analytic (one FK per tick gives every $z_j, o_j$), not finite differences.

**Variables** $x = (q_k)_{k\in\mathcal{K}} \in \mathbb{R}^{n(H-d)}$; motion bounds $m = (m_k) \in \mathbb{R}^{n(H-d)}$;
slacks $s^o, s^f \in \mathbb{R}^{H-d}_{\ge 0}$, one per tick and barrier family.

**Objective.** With $\delta_k = q_k - r_k$; $\delta_k = \bar q_k - r_k$ for $1 \le k \le d$; $\delta_0 = \delta_{-1} = 0$:

```math
\min_{x,\,m,\,s}\;\; \sum_{k\in\mathcal{K}} \delta_k^\top W_k \delta_k
\;+\; \mu \sum_{k\in\mathcal{K}} (\delta_k - 2\delta_{k-1} + \delta_{k-2})^\top W_k (\delta_k - 2\delta_{k-1} + \delta_{k-2})
\;+\; \rho_s \sum_{k\in\mathcal{K}} (s^o_k + s^f_k)
```

```math
W_k = J_T(\tilde r_k)^\top \operatorname{diag}(1,1,1,\ell_r^2,\ell_r^2,\ell_r^2)\, J_T(\tilde r_k) + \varepsilon I_n ,
```

$J_T \in \mathbb{R}^{6\times n}$ the geometric Jacobian of `gripper_frame_link` (linear rows first), $\tilde r_k$ the reference
clipped to the joint box. $\delta^\top W \delta$ is the squared tool displacement (m²) plus $\ell_r^2 \times$ squared rotation;
$\varepsilon I$ makes $W_k \succ 0$. Smoothness acts on the *deviation*, not on $q$, so a safe chunk is not smoothed.

**Constraints**, for $k \in \mathcal{K}$ with $q_d := \bar q_d$:

1. Position box and trust region: $\max(l_k, \hat q_k - \tau) \le q_k \le \min(u_k, \hat q_k + \tau)$, where
   $l_k = \min(q_{\min},\ \bar q_d + (k-d)\bar v\Delta t)$ and $u_k = \max(q_{\max},\ \bar q_d - (k-d)\bar v\Delta t)$. These equal the joint box
   unless the arm starts outside it, in which case they let it return at full speed.
2. Motion: $-m_k \le q_k - q_{k-1} \le m_k$ and $0 \le m_k \le \bar v \Delta t$. Together these are the velocity limit.
3. Obstacle CBF, all $i \in \mathcal{I}$ not pruned:
   ```math
   \tilde\sigma_{ik} = \tfrac12\big(\tilde h_{i,k-1}(q_{k-1}) + \tilde h_{ik}(q_k)\big) - \kappa\,\beta_i^\top m_k ,
   ```
   ```math
   \tilde\sigma_{i,d+1} + s^o_{d+1} \ge (1-\gamma)\,h_i(\bar q_d), \qquad
   \tilde\sigma_{ik} - (1-\gamma)\,\tilde\sigma_{i,k-1} + s^o_k \ge 0 \quad (k \ge d+2) .
   ```
4. Floor CBF: the same rows with $f_i$, $\kappa^f = 1/2$, $s^f$.
5. $s^o, s^f \ge 0$.

Every row is linear in $(x, m, s)$ and the objective is a convex quadratic, so this is a QP. Using
$m_k \ge \lvert q_k - q_{k-1}\rvert$ in place of $\lvert q_k - q_{k-1}\rvert$ only lowers $\tilde\sigma$, so it is conservative. Bounding $m$ above keeps the
pruning finite.

**Lemma 3 (always feasible).** For a sequence $y$, let $\Pi(y)$ be the forward rate limiter $\hat q_d = \bar q_d$,
$\hat q_k = \operatorname{clip}\big(y_k;\ \max(l_k, \hat q_{k-1} - \bar v\Delta t),\ \min(u_k, \hat q_{k-1} + \bar v\Delta t)\big)$. Each interval is non-empty because $l_k \le u_k$,
$l_k - l_{k-1} \le \bar v \Delta t$ and $u_{k-1} - u_k \le \bar v\Delta t$ (with $l_d = u_d = \bar q_d$). So $\hat q = \Pi(y)$, for any $y$, satisfies constraints 1–2 (with $m_k = \lvert \hat q_k - \hat q_{k-1}\rvert$), and with
large enough slacks, 3–5. Every later linearisation point is $\Pi$ of a previous solution, so the same
holds. Hence every subproblem is feasible, including a start inside an obstacle margin.

**Exact penalty.** If the hard-constrained problem ($s = 0$) is feasible and
$\rho_s > \sum_{i} \lambda^\star_{ik}$ (the multipliers of the tick-$k$ rows) for every $k$, the solution has $s = 0$ and equals the
hard-constrained solution. Rough estimate: multipliers ~0.1 m² of cost per metre of violation
(deviation ~0.1 rad, $W$ ~0.2 m²/rad², $\lVert \hat g\rVert$ ~0.4 m/rad), so $\rho_s = 10$. We log
$\lambda^\star$ to check.

**Proposition (minimal intervention).** If $\delta_k = 0$ for $k \le d$ and the reference $r$ satisfies 1–5 with $s = 0$, the
optimal $x$ is unique and equals $r$. *Proof:* then $\Pi(r) = r$ and $\tilde h = h$ at $r$, so $r$ is feasible. The objective is
$\ge 0$, equals 0 at $r$, and is strictly convex in $x$ because $W_k \succ 0$. ∎

**Pruning** (exact for the linearised problem). Over the trust region,
$\tilde h_{ik} \in [\underline h_{ik}, \overline h_{ik}] = [\hat h_{ik} - \lvert\hat g_{ik}\rvert^\top\tau,\ \hat h_{ik} + \lvert \hat g_{ik}\rvert^\top \tau]$ and $0 \le m_k \le \bar v \Delta t$. Row $(i, k \ge d+2)$ holds for every feasible point,
and can be dropped, when

```math
\tfrac12\big(\underline h_{i,k-1} + \underline h_{ik}\big) - \kappa\,\beta_i^\top \bar v \Delta t
\;\ge\; (1-\gamma)\,\tfrac12\big(\overline h_{i,k-2} + \overline h_{i,k-1}\big) .
```

The first row ($k = d+1$) is handled the same way. Away from obstacles nearly every row drops.

## 5. Per-chunk algorithm

1. $r_k = \varphi(a_k)$; read $\bar q_0$ (and $\bar q_{1..d}$); $\hat q^{(0)} = \Pi(r)$.
2. For $t = 0, 1, \dots, T_{\max}-1$ ($T_{\max} = 3$): FK and analytic Jacobians at $\hat q^{(t)}$ (all ticks, batched), field
   queries, pruning, build and solve the QP (OSQP, warm-started, `polish=True`). Set $\hat q^{(t+1)} = \Pi(x^\star)$. Stop
   early if $\lVert \hat q^{(t+1)} - \hat q^{(t)} \rVert_\infty < 10^{-3}$ rad.
3. Take the last $\hat q = \Pi(x^\star)$, so the limits hold exactly despite solver tolerance.
4. **Exact check**, on the result and over all spheres (no pruning, no linearisation): $\sigma_{ik} \ge 0$ and the
   floor analogue, computed with the true $h$, $f$ and the actual $\lvert q_k - q_{k-1}\rvert$; and
   `self_collisions(q_k)` empty at every tick.
5. Pass: send $\varphi^{-1}(q_k)$ with the gripper $\theta_k$. Fail, or over the time budget (§6): fall back
   - (a) to the rest of the previous accepted chunk, if it passes step 4 against the newest field;
   - (b) otherwise hold the measured position (safe under A3 when $h(\bar q_0) \ge 0$), stop, and flag the operator.
6. Log: iterations, rows kept, $\max s$, $\lambda^\star$, build and solve times, intervention
   $\sum_k \lVert J_T \delta_k \rVert$, and which check failed.

The linearisation has no error bound ($\nabla D$ is discontinuous), and sequential QP has no convergence
guarantee here. Neither is needed: step 4 is the certificate, and step 5 covers failure.

## 6. Size and latency budget

With $d = 0$: 250 position + 250 motion variables + 100 slacks. Rows: 250 + 250 bounds, 500 motion,
at most $30 \times 50$ obstacle and as many floor rows before pruning, each with at most $3n + 2n + 1 = 26$
non-zeros. $P$ is block-banded with bandwidth two blocks.

SmolVLA chunk inference on the RTX 3060 Laptop, as LeRobot loads it (VLM and action expert in
bfloat16), two 640×480 cameras, 10 flow steps (`scripts/time_smolvla_inference.py`,
`outputs/smolvla_latency.json`):

| Batch (samples) | Chunk median | Chunk p95 | VLM prefix | Per flow step |
| --- | --- | --- | --- | --- |
| 1 | 236 ms (213 ms in a second run) | 301 ms | 57 ms | 18 ms |
| 4 | 360 ms | 373 ms | 195 ms | 17 ms |
| 8 | 650 ms | 655 ms | 385 ms | 26 ms |

Upcasting every weight to float32 is slower (325 ms at batch 1). At batch 1 the ten flow steps are about
75% of the time, which is where Week 3 guidance adds its per-step cost.

In synchronous execution the arm pauses for inference plus filter, so the filter target is ≤ 10% of a
chunk:

| Step | Target (p95) |
| --- | --- |
| FK + analytic Jacobians, 50 ticks × 30 spheres | ≤ 5 ms |
| Field queries + pruning | ≤ 2 ms |
| OSQP solve, warm start, per iteration | ≤ 5 ms |
| Whole filter, ≤ 3 iterations + exact check | **≤ 25 ms**; over 60 ms counts as a failure (fallback) |

The finite-difference `clearance_jacobian` (≈ 4 ms per tick, 200 ms per chunk) is too slow for this; the analytic
Jacobian above replaces it. OSQP with a fixed sparsity pattern (all rows present, pruned rows zeroed with
$l = -\infty$) versus rebuilding with only the kept rows: benchmark both.

## 7. Parameters

| Parameter | Default | Source |
| --- | --- | --- |
| $\gamma$ | 0.2 | Ablation 0.1–1; at $h$ = 5 cm it allows ≤ 1 cm closer per tick |
| $\tau$ | 0.1 rad | Trust region; larger = fewer iterations, worse linearisation |
| $\mu$ | 1 | Smoothness of the deviation |
| $\ell_r$, $\varepsilon$ | 0.05 m, $10^{-4}$ m² | Rotation weight (1 rad ≈ 5 cm); regulariser |
| $\rho_s$ | 10 | Exact penalty, ≫ logged multipliers |
| $c_{\text{user}}$, $c^f_{\text{user}}$ | 1 cm, 0.5 cm | Clearance we promise |
| $\bar v$ | p99 of our demo speeds | *Proxy* 1.4–3.5 rad/s per joint; servo max 4.7 |
| $\epsilon_{\text{off}}$ | measure | *Proxy* ≤ 5.1° (all joints together) |
| $\epsilon_{\text{FK}}$ | touch-fit max residual | Pending the touch session |
| $\epsilon_{\text{sense}}$ | ~3 mm | D435 depth noise at ~0.6 m; check on the static scene |

## 8. What it does not cover

- Moving obstacles (A3) until the Week 4 forecast replaces $D$ by $D_k$.
- Occluded or unobserved space: treated as free.
- The grasp target and the arm itself: removed from the cloud by masking (CV), so not obstacles.
- Self-collision: checked at ticks in step 4, not constrained in the QP.
- Dynamics, torque and the servo's own trajectory beyond A1–A2.

## 9. To measure or decide before coding

1. $\epsilon_{\text{off}}$ per joint on our follower. From our demo logs it is confounded with leader/follower
   mismatch; a clean value needs a commanded test trajectory (motor commands: explicit approval first).
2. $\bar v$ from our demos (the PLAN item on teleop speed percentiles).
3. **Jaw spheres versus the table at grasp poses.** With 5–13 mm sphere bulge and $c^f_{\text{user}}$, the floor
   barrier may forbid the low grasps in our own demos. Check the minimum $f_i$ over our demo grasps first. If they
   violate it, use a per-link floor margin (jaw spheres 0) or a tighter jaw model.
4. $\epsilon_{\text{sense}}$ from repeated static frames once the camera is on its stand.
5. OSQP fixed pattern versus rebuild (§6); cvxpy as the reference solution in tests.
