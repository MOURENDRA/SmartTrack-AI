# SmartTrack AI: Benchmark Evaluation Protocol & Metrics Specification

## Standard Evaluation Framework for Multi-Object Tracking

This document defines the mathematical foundations, standard benchmark formats, comparative experimental protocol, and occlusion stress-testing methodology for evaluating **SmartTrack AI** in academic research and faculty project evaluations.

---

## 1. Multi-Object Tracking Metrics Definition

### 1.1 Multiple Object Tracking Accuracy (MOTA)
MOTA is the classical CLEAR MOT metric measuring overall tracking accuracy considering false positives, false negatives, and identity switches:
$$\text{MOTA} = 1 - \frac{\sum_{t=1}^T (\text{FN}_t + \text{FP}_t + \text{IDSW}_t)}{\sum_{t=1}^T \text{GT}_t}$$

Where:
- $\text{GT}_t$: Number of ground-truth objects present in frame $t$.
- $\text{FN}_t$: False Negatives (ground-truth person undetected or untracked).
- $\text{FP}_t$: False Positives (spurious track hypothesis on background or duplicate detection).
- $\text{IDSW}_t$: Identity Switches (a ground-truth target is assigned a different track ID than in its preceding tracked frame).

*Range*: $(-\infty, 100\%]$. A score above $50\%$ on challenging MOT benchmarks denotes strong performance.

### 1.2 Identification F1 Score (IDF1)
While MOTA emphasizes detection quality, **IDF1** measures identity preservation consistency over the entire trajectory:
$$\text{IDF1} = \frac{2 \text{IDTP}}{2 \text{IDTP} + \text{IDFP} + \text{IDFN}}$$
Where:
- $\text{IDTP}$ (ID True Positives): Total overlap frames between ground truth and tracker trajectory assigned the same identity through global bipartite matching.
- $\text{IDFP}$ (ID False Positives): Frames where a tracker ID does not match ground truth identity.
- $\text{IDFN}$ (ID False Negatives): Frames where a ground truth ID is missed.

*Importance for PS-2*: In dense occlusion scenarios, a tracker may have good MOTA (detecting bounding boxes) but poor IDF1 (swapping IDs on every crossing). **IDF1 is the primary success criterion for SmartTrack AI.**

### 1.3 Higher Order Tracking Accuracy (HOTA)
HOTA explicitly balances detection accuracy ($\text{DetA}$) and association accuracy ($\text{AssA}$):
$$\text{HOTA}_\alpha = \sqrt{\text{DetA}_\alpha \cdot \text{AssA}_\alpha}$$
$$\text{HOTA} = \int_0^1 \text{HOTA}_\alpha \, d\alpha \approx \frac{1}{19} \sum_{\alpha \in \{0.05, 0.10, \dots, 0.95\}} \text{HOTA}_\alpha$$

Where:
$$\text{DetA}_\alpha = \frac{|\text{TP}_\alpha|}{|\text{TP}_\alpha| + |\text{FN}_\alpha| + |\text{FP}_\alpha|}$$
$$\text{AssA}_\alpha = \frac{1}{|\text{TP}_\alpha|} \sum_{c \in \text{TP}_\alpha} \frac{|\text{TPA}(c)|}{|\text{TPA}(c)| + |\text{FNA}(c)| + |\text{FPA}(c)|}$$
HOTA decouples detection errors from association errors, preventing high-recall detectors from artificially inflating tracking scores.

### 1.4 Identity Switches (IDSW)
An identity switch occurs when:
$$\text{ID}(T_t^i) \neq \text{ID}(T_{t-k}^i) \quad \text{for the same ground truth person } P^i$$
In dense crowds, mutual occlusion and bounding box ambiguities cause standard trackers to swap IDs or create a new ID when a person reappears. Lower IDSW is better.

### 1.5 Track Fragmentation (Frag)
The number of times a ground-truth trajectory is interrupted and later resumed.

### 1.6 Mostly Tracked (MT) & Mostly Lost (ML)
- **MT (Mostly Tracked)**: Percentage of ground truth trajectories covered by a single track for $>80\%$ of their lifespan.
- **ML (Mostly Lost)**: Percentage of ground truth trajectories covered for $<20\%$ of their lifespan.

---

## 2. Standard MOT Dataset Format

SmartTrack AI natively parses standard MOT16 / MOT17 / MOT20 challenge benchmarks.

### 2.1 Directory Structure
```
data/datasets/
└── MOT17-02-DPM/
    ├── seqinfo.ini       # Video metadata: name, imgWidth, imgHeight, seqLength, frameRate
    ├── img1/             # Sequential video frames (000001.jpg, 000002.jpg, ...)
    └── gt/
        └── gt.txt        # Ground truth annotations
```

### 2.2 MOT Ground Truth Line Format (`gt.txt`)
Comma-separated values per object bounding box:
```csv
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class_id>, <visibility>
1, 1, 912, 484, 143, 438, 1, 1, 0.85
1, 2, 838, 506, 117, 394, 1, 1, 0.60
2, 1, 915, 483, 142, 439, 1, 1, 0.90
```
- `<class_id> = 1`: Pedestrian.
- `<visibility>`: Fraction of pedestrian bounding box unoccluded $[0.0, 1.0]$. Used to benchmark performance across occlusion degrees.

---

## 3. Comparative Experimental Protocol

To prove that the enhanced tracking algorithm delivers genuine improvements over the baseline, experiments must follow strict controlled conditions:

| Parameter | Baseline Tracker | Enhanced Tracker |
|:---|:---|:---|
| **Detector** | YOLOv8n (conf $\geq 0.35$) | YOLOv8n (conf $\geq 0.35$) |
| **Motion Model** | 8-state Kalman Filter | 8-state Kalman Filter |
| **Association Cost** | $1.0 - \text{IoU}$ | $0.4 \cdot C_{\text{mot}} + 0.6 \cdot C_{\text{app}}$ |
| **Matching Strategy** | Single-stage Hungarian | Two-stage Confidence-Aware |
| **Track Retention** | $\text{max\_age} = 2$ frames | $\text{max\_lost\_time} = 30$ frames |
| **Appearance Memory**| None | 80-dim Spatial Pyramid HSV + EMA |
| **Spatial Gating** | IoU Gate ($0.3$) | IoU Gate ($0.25$) + Euclidean Search Radius |

### Execution Command:
```bash
python scripts/evaluate_mot.py \
    --data-dir data/datasets/MOT17-02-DPM \
    --compare \
    --output reports/benchmark_comparison.json
```

---

## 4. Stress-Testing Protocol for Dense Occlusion

To evaluate **Requirement 3 (Graceful degradation)** and **Requirement 11 (Stress analysis against increasing occlusion and crowd density)**, the system executes controlled stress simulations:

### 4.1 Occlusion Duration Stress Test
Artificially mask detections for selected targets for durations $k \in \{2, 5, 10, 15, 20, 30\}$ frames:
1. Ground truth target $P^*$ is tracked until frame $t_0$.
2. For frames $t_0 + 1$ to $t_0 + k$, detection for $P^*$ is withheld (simulating total occlusion behind another pedestrian or pillar).
3. At frame $t_0 + k + 1$, detection reappears.
4. Measure whether tracker retains original ID:
   $$\text{Recovery Rate}(k) = \frac{\text{Successfully Re-associated Tracks}}{\text{Total Occluded Tracks}}$$

### 4.2 Crowd Density Scaling Test
Simulate increasing pedestrian density by analyzing sequences partitioned by simultaneous person count:
- Sparse ($1 - 5$ persons/frame)
- Moderate ($6 - 15$ persons/frame)
- Dense ($16 - 30+$ persons/frame)

Measure frame latency and ID switches to verify:
1. Latency scales sub-quadratically and remains $< 50$ ms.
2. ID switches in the enhanced tracker do not explode exponentially with density.

---

## 5. Result Verification & Integrity Policy
- **No Fabricated Numbers**: Benchmark results must be computed directly by running `src/metrics.py` against ground truth annotations.
- **Proxy Labeling**: When running on unannotated user videos, UI indicators must be labelled as `"Estimated Track Discontinuities"` rather than claiming verified ground-truth ID switches.
