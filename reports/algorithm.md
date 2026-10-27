# SmartTrack AI: Technical Algorithm & Mathematical Specification

## Problem Statement 2 (PS-2)
**Identity-Preserving Multi-Object Tracking through Dense Occlusion**

---

## 1. Mathematical Formulation of Multi-Object Tracking

The Multi-Object Tracking (MOT) problem is formulated under the **tracking-by-detection** paradigm as an online, causal data association problem over discrete time steps $t \in \{1, 2, \dots, T\}$.

At time step $t$, the input consists of:
1. An incoming video frame $I_t \in \mathbb{R}^{H \times W \times 3}$.
2. A set of $M_t$ bounding box detections produced by an object detector:
   $$D_t = \{d_t^j = (b_t^j, c_t^j)\}_{j=1}^{M_t}$$
   where $b_t^j = [x_1, y_1, x_2, y_2] \in \mathbb{R}^4$ is the bounding box in image coordinates, and $c_t^j \in [0, 1]$ is the detection confidence score.

The tracker maintains a dynamic set of $N_t$ active track identities:
$$\mathcal{T}_t = \{T_t^i\}_{i=1}^{N_t}$$

Each track $T^i$ possesses:
- A unique integer identifier $\text{ID}(T^i) \in \mathbb{N}^+$.
- A kinematic state distribution parameterized by a mean vector $\hat{\mathbf{x}}_t^i$ and covariance matrix $\mathbf{P}_t^i$.
- An appearance representation vector $\mathbf{f}_t^i \in \mathbb{R}^D$ ($D=80$ for spatial-pyramid HSV, or $D=1280$ for deep MobileNetV2 embeddings).
- A track state $S_t^i \in \{\text{TENTATIVE}, \text{CONFIRMED}, \text{LOST}, \text{REMOVED}\}$.
- Tracking history counters: total hits, consecutive misses ($\text{age}_{lost}$), and trajectory centroid coordinates.

The tracking objective is to compute a causal bijective matching between the active track set $\mathcal{T}_{t-1}$ and the current detection set $D_t$, minimizing identity switches ($\text{IDSW}$) and bounding box localization errors while recovering tracks through occlusion without future-frame hindsight.

---

## 2. Motion Modeling via 8-State Kalman Filtering

To model bounding box kinematics between consecutive frames, each track incorporates an 8-dimensional continuous-time constant velocity linear dynamical system discretized at frame intervals $\Delta t = 1$.

### 2.1 State Vector Representation
The bounding box state $\mathbf{x} \in \mathbb{R}^8$ is parameterized by the bounding box center $(u, v)$, aspect ratio $a = w / h$, bounding box height $h$, and their first-order temporal derivatives:
$$\mathbf{x} = \begin{bmatrix} u & v & a & h & \dot{u} & \dot{v} & \dot{a} & \dot{h} \end{bmatrix}^T$$

The observation vector $\mathbf{z} \in \mathbb{R}^4$ matches the detector output converted to $(u, v, a, h)$:
$$\mathbf{z} = \begin{bmatrix} u & v & a & h \end{bmatrix}^T$$

### 2.2 State Transition and Observation Models
$$\mathbf{x}_t = \mathbf{F} \mathbf{x}_{t-1} + \mathbf{w}_t, \quad \mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})$$
$$\mathbf{z}_t = \mathbf{H} \mathbf{x}_t + \mathbf{v}_t, \quad \mathbf{v}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{R})$$

Where the transition matrix $\mathbf{F} \in \mathbb{R}^{8 \times 8}$ and measurement matrix $\mathbf{H} \in \mathbb{R}^{4 \times 8}$ are:
$$\mathbf{F} = \begin{bmatrix}
\mathbf{I}_{4 \times 4} & \mathbf{I}_{4 \times 4} \Delta t \\
\mathbf{0}_{4 \times 4} & \mathbf{I}_{4 \times 4}
\end{bmatrix}, \qquad
\mathbf{H} = \begin{bmatrix} \mathbf{I}_{4 \times 4} & \mathbf{0}_{4 \times 4} \end{bmatrix}$$

### 2.3 Time Update (Prediction)
Before association, each active track predicts its expected spatial state:
$$\hat{\mathbf{x}}_{t|t-1} = \mathbf{F} \hat{\mathbf{x}}_{t-1|t-1}$$
$$\mathbf{P}_{t|t-1} = \mathbf{F} \mathbf{P}_{t-1|t-1} \mathbf{F}^T + \mathbf{Q}$$

Process noise $\mathbf{Q}$ is scaled proportionally to expected pedestrian acceleration:
$$\mathbf{Q} = \text{diag}\left( (\sigma_p h)^2, (\sigma_p h)^2, (\sigma_a)^2, (\sigma_p h)^2, (\sigma_v h)^2, (\sigma_v h)^2, (\sigma_{va})^2, (\sigma_v h)^2 \right)$$
where $\sigma_p = 0.05$ (position noise factor) and $\sigma_v = 0.00625$ (velocity noise factor).

### 2.4 Measurement Update (Correction)
When detection $d_t^j$ is associated with track $T^i$, the measurement residual $\mathbf{y}_t$ and Kalman gain $\mathbf{K}_t$ are evaluated:
$$\mathbf{y}_t = \mathbf{z}_t^j - \mathbf{H} \hat{\mathbf{x}}_{t|t-1}$$
$$\mathbf{S}_t = \mathbf{H} \mathbf{P}_{t|t-1} \mathbf{H}^T + \mathbf{R}$$
$$\mathbf{K}_t = \mathbf{P}_{t|t-1} \mathbf{H}^T \mathbf{S}_t^{-1}$$
$$\hat{\mathbf{x}}_{t|t} = \hat{\mathbf{x}}_{t|t-1} + \mathbf{K}_t \mathbf{y}_t$$
$$\mathbf{P}_{t|t} = (\mathbf{I} - \mathbf{K}_t \mathbf{H}) \mathbf{P}_{t|t-1}$$

Measurement noise covariance $\mathbf{R} = \text{diag}\left( (r_p h)^2, (r_p h)^2, (r_a)^2, (r_p h)^2 \right)$ reflects detector localization variance ($r_p = 0.1$).

---

## 3. Appearance Feature Representation

In dense crowds, purely geometric motion estimation degrades because pedestrians walking close together share near-identical spatial coordinates. SmartTrack AI incorporates visual appearance descriptors.

### 3.1 Spatial Pyramid HSV Histogram (Baseline Appearance)
To provide real-time feature extraction without requiring GPU hardware, SmartTrack AI computes a $2 \times 2$ spatial pyramid HSV histogram:
1. Bounding box image crop $I_{\text{crop}}$ is resized to $128 \times 64$ pixels.
2. The image is transformed to the HSV color space to decouple illumination ($V$) from chromaticity ($H, S$).
3. The crop is partitioned into 4 spatial quadrants (top-left, top-right, bottom-left, bottom-right) to capture vertical color stratification (e.g., upper shirt vs. lower pants).
4. In each quadrant, an 8-bin Hue and 4-bin Saturation histogram is constructed (16 bins per quadrant + 16-bin global summary = 80 dimensions).
5. The combined vector is $L_2$-normalized:
   $$\mathbf{f} = \frac{\mathbf{h}}{\|\mathbf{h}\|_2 + \epsilon}$$

### 3.2 Cosine Appearance Distance
The appearance cost between track appearance $\mathbf{f}_{\text{trk}}$ and detection appearance $\mathbf{f}_{\text{det}}$ is defined via normalized cosine distance:
$$C_{\text{app}}(T^i, d^j) = \frac{1 - \cos(\mathbf{f}_{\text{trk}}^i, \mathbf{f}_{\text{det}}^j)}{2} = \frac{1 - \frac{\mathbf{f}_{\text{trk}}^i \cdot \mathbf{f}_{\text{det}}^j}{\|\mathbf{f}_{\text{trk}}^i\| \|\mathbf{f}_{\text{det}}^j\|}}{2} \in [0, 1]$$

### 3.3 Exponential Moving Average (EMA) Feature Memory
To avoid appearance drift during mutual occlusion (where a track might absorb an occluder's colors), appearance vectors are updated cautiously using an exponential moving average only when detection confidence exceeds $\tau_{\text{conf}}$:
$$\mathbf{f}_t^i = (1 - \alpha_{\text{ema}}) \mathbf{f}_{t-1}^i + \alpha_{\text{ema}} \mathbf{f}_t^j, \quad \alpha_{\text{ema}} = 0.10$$
When occlusion is detected ($\text{IoU}_{\text{inter-track}} > 0.4$), the update is suppressed to safeguard track memory.

---

## 4. Data Association: Baseline vs. Enhanced Method

### 4.1 Baseline Tracker (IoU-Only Hungarian)
The baseline tracking method represents standard SORT-style tracking:
1. Predict next-frame bounding box positions using Kalman filters.
2. Construct cost matrix $C \in \mathbb{R}^{N \times M}$ where $C_{i, j} = 1 - \text{IoU}(\hat{b}^i, b^j)$.
3. Solve assignment using Kuhn-Munkres (Hungarian algorithm):
   $$\min_{\mathbf{X}} \sum_{i=1}^N \sum_{j=1}^M C_{i, j} X_{i, j} \quad \text{s.t.} \quad \sum_j X_{i, j} \leq 1, \quad \sum_i X_{i, j} \leq 1$$
4. Reject any match where $\text{IoU}(\hat{b}^i, b^j) < \tau_{\text{iou\_gate}}$ (default $0.3$).
5. Terminate tracks after 1-2 missed frames.

**Failure Modes in Crowds**:
- When two people cross, their predicted boxes overlap, resulting in frequent identity swaps.
- When a person is occluded for $>2$ frames, their track is immediately terminated, generating a brand-new ID upon reappearance.

### 4.2 Proposed Enhanced Tracker: Confidence-Aware Occlusion-Aware Two-Stage Association

SmartTrack AI introduces a two-stage hierarchical matching framework with track memory and adaptive cost weighting:

```
                      Frame Detections (D_t)
                                │
              ┌─────────────────┴─────────────────┐
              ▼                                   ▼
      High-Confidence                     Low-Confidence
    Detections (c >= 0.5)               Detections (0.2 <= c < 0.5)
              │                                   │
              │ [Stage 1]                         │
   Active Tracks (Confirmed)                      │
              │                                   │
      Combined Cost Matrix                        │
   C = w_mot*C_mot + w_app*C_app                  │
              │                                   │
      Hungarian Assignment                        │
              │                                   │
      Matched? ──► YES ──► Kalman Correction      │
              │            & EMA Appearance       │
              ▼ NO                                │
    Remaining Active Tracks                       ▼
              │                                [Stage 2]
              └───────────────► Hungarian Assignment (IoU-guided)
                                      │
                                      ▼ NO
                               Unmatched Detections
                                      │
                                      ▼ [Stage 3: Memory Recovery]
                               Lost Tracks Memory Pool
                               (age <= max_lost_frames)
                                      │
                               Spatial Radius & Appearance Gate
                                      │
                               Re-association (Preserves ID!)
```

#### Combined Cost Formula:
For track $T^i$ and detection $d^j$:
$$C_{\text{combined}}(i, j) = w_{\text{mot}} \cdot C_{\text{mot}}(i, j) + w_{\text{app}} \cdot C_{\text{app}}(i, j)$$
where $w_{\text{mot}} = 0.4$, $w_{\text{app}} = 0.6$, and $w_{\text{mot}} + w_{\text{app}} = 1.0$.

#### Spatial and Physical Gating:
Matches are invalidated prior to Hungarian solving ($C_{i, j} = \infty$) if:
1. $\text{IoU}(\hat{b}^i, b^j) < \tau_{\text{min\_iou}}$ AND Euclidean distance between centers $> 1.5 \times \max(w_i, h_i)$.
2. $C_{\text{app}}(i, j) > \tau_{\text{app\_max}}$ ($0.75$).

---

## 5. Track Lifecycle State Machine

```
              Detection Unmatched
                     │
                     ▼
             ┌───────────────┐
             │   TENTATIVE   │ (hit_count = 1)
             └───────┬───────┘
                     │ Matches >= min_hits (3 frames)
                     ▼
             ┌───────────────┐  Missed Frame (1 <= misses <= 3)
             │   CONFIRMED   │ ────────────────────────────────┐
             └───────┬───────┘                                 │
                     ▲                                         ▼
                     │ Re-associated                   ┌───────────────┐
                     └─────────────────────────────────┤     LOST      │
                                                       └───────┬───────┘
                                                               │ Misses > max_age (30 frames)
                                                               ▼
                                                       ┌───────────────┐
                                                       │    REMOVED    │ (Deleted from memory)
                                                       └───────────────┘
```

1. **Tentative**: Initialized on unmatched high-confidence detections. Not published to users until consistency is confirmed over `min_hits = 3` consecutive frames.
2. **Confirmed**: Active track with verified persistence. Bounding boxes are drawn and logged.
3. **Lost**: Track experienced detection dropout (occlusion or sensor noise). Retained in occlusion memory for `max_lost_time = 30` frames. Kalman velocity continues predicting trajectory with increasing uncertainty.
4. **Removed**: Expired tracks whose spatial covariance and time-to-reacquire exceeded safe limits. Prevents memory leaks.

---

## 6. End-to-End Online Causal Pipeline Pseudocode

```python
def process_video_frame(frame, tracks, detector, associator, config):
    # 1. Prediction (Prior Estimation)
    for track in tracks:
        track.predict_kalman()

    # 2. Causal Detection (Current frame only)
    raw_detections = detector.detect(frame)
    high_dets = [d for d in raw_detections if d.conf >= config.conf_threshold]
    low_dets = [d for d in raw_detections if 0.2 <= d.conf < config.conf_threshold]

    # 3. Extract Appearance
    for d in high_dets:
        d.feature = appearance_extractor.extract(frame, d.bbox)

    # 4. Stage 1: Primary Association (High-confidence detections + Active tracks)
    cost_matrix = associator.compute_combined_cost(
        active_tracks, high_dets, w_mot=0.4, w_app=0.6
    )
    matches_1, unmatched_tracks, unmatched_dets = hungarian_solve(
        cost_matrix, gate_threshold=0.7
    )

    for trk_idx, det_idx in matches_1:
        active_tracks[trk_idx].update(high_dets[det_idx])

    # 5. Stage 2: Secondary Recovery (Remaining active tracks + Low-confidence detections)
    rem_tracks = [active_tracks[i] for i in unmatched_tracks]
    matches_2, still_unmatched_tracks, _ = hungarian_iou_solve(rem_tracks, low_dets)

    for trk_idx, det_idx in matches_2:
        rem_tracks[trk_idx].update(low_dets[det_idx])

    # 6. Stage 3: Occlusion Re-association (Lost tracks + Remaining unmatched high-confidence detections)
    unmatched_high_dets = [high_dets[j] for j in unmatched_dets]
    lost_tracks = [t for t in tracks if t.state == TrackState.LOST]

    reassoc_matches = associator.reassociate_lost_tracks(
        lost_tracks, unmatched_high_dets, max_app_dist=0.45
    )
    for l_idx, d_idx in reassoc_matches:
        lost_tracks[l_idx].reacquire(unmatched_high_dets[d_idx])

    # 7. Track Lifecycle & Initialization
    for d in remaining_new_detections:
        tracks.append(Track.initialize(d))

    tracks.cleanup_expired(max_age=30)
    return tracks
```

---

## 7. Computational Complexity & Latency Bounds

- **Detection**: $\mathcal{O}(W \cdot H)$ via single forward pass of YOLOv8n ($~15-25$ ms on CPU).
- **Appearance Extraction**: $\mathcal{O}(N_{\text{det}} \cdot 128 \times 64)$ vectorized NumPy operations ($<1$ ms per person).
- **Kalman Prediction & Update**: $\mathcal{O}(N_{\text{trk}} \cdot 8^3) \approx \mathcal{O}(N_{\text{trk}})$ ($<0.05$ ms).
- **Hungarian Data Association**: $\mathcal{O}(\max(N, M)^3)$ via SciPy linear sum assignment ($<0.5$ ms for $N, M \leq 50$).
- **Total Frame Processing Time**: Bounded between $25-45$ ms/frame on standard Intel Core i5/i7 laptop CPUs ($22-40$ FPS). Real-time constraint is strictly satisfied.
