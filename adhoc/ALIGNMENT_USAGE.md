P## Counseling-Quality Assessment Pipeline

### 1. Raw Inputs & Pre-processing
| Modality | Source File(s) | Key Output CSV | Important Columns | Frame Rate |
|----------|----------------|----------------|-------------------|------------|
| **Facial Video** | `<session>_T.m2ts`, `<session>_C.m2ts` | `facial_landmarks.csv` | `x_* / y_*` (468 pts), `AUxx_intensity` | native FPS |
| **Body Pose** | same videos | `pose_landmarks.csv` | `<joint>_x / _y / _visibility` (33 joints) | native FPS |
| **Prosody (Audio)** | same videos | `prosody_features.csv` | `F0_sma`, `pcm_intensity_sma`, spectral centroid … | 100 Hz |

### 2. Feature Extraction Logic
#### Facial
* **AU Statistics** – mean intensity & activation-frequency (`>0.5`)
* **Face Movement** – Δdistance of face-centre per frame.
* **Landmark Stability** – std of x,y across time.

#### Pose
| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Movement / frame** | `Σ_j |p_t − p_{t-1}|` | Kinetic energy |
| **Key-joint activity** | mean(|x_t-x_{t-1}|) for nose & wrists & shoulders | Gesture emphasis |
| **Visibility reliability** | mean(visibility) | Tracking quality |

#### Prosody
* Energy variability (`pcm_intensity_sma` std)
* Pitch statistics (mean, range, std of `F0_sma`)
* Spectral centroid dynamics.

### 3. Therapist–Client Synchrony
| Modality | Paired Signals | Synchrony Score |
|----------|---------------|-----------------|
| Facial | common AU vectors | mean Pearson r |
| Pose | frame-wise movement vectors | Pearson r |
| Prosody | F0, energy, spectral | mean Pearson r |

`synchrony > 0.3` ⇒ in-sync frame (used for synchrony ratio).

### 4. Higher-level Indicators
* **Engagement Level** – mean activity (T & C).
* **Rapport** – mean modality synchrony.
* **Emotional Regulation** – inverse variability.
* **Non-verbal Communication** – AU diversity · gesture amplitude · vocal activity.

### 5. Dashboard Visuals (CounselingAssessmentVisualizer)
1. Progress bar overall score.
2. Radar chart (4 core competencies).
3. Detailed bar chart.
4. Counselor frame image (pose/facial).
5. **Face Emotion Flow**  
   • 100 uniformly-sampled frames (0-50 min)  
   • Therapist solid line (0–1)  
   • Client dashed line shifted +0.55 (0.55–1.55)  
   • Colour = dominant emotion per frame.
6. Additional sub-plots: Pose/AU/Prosody details, Engagement timeline, Recommendations.

### 6. Tool Chain
* **OpenFace 2.0** – AU & landmarks.
* **MediaPipe Pose** – 2-D body joints.
* **openSMILE / librosa** – prosody descriptors.
* **audalign** – audio fingerprint alignment.
* **pandas · numpy · matplotlib · seaborn** – analytics & plotting.
* **fire · tqdm** – CLI & progress bars.

> **Thresholds**  
> • Active AU = intensity > 0.5  
> • Visible joint = visibility > 0.5  
> • High movement = session 75-percentile.

Execution:
```bash
# 1) Align audio timelines + comparison plots
python adhoc/consolidated_analysis.py --mode alignment

# 2) Produce per-pair JSON + consolidated PNG
python adhoc/consolidated_analysis.py --mode consolidated

# 3) Build dashboards for all sessions
python adhoc/consolidated_analysis.py --mode visualize
``` 