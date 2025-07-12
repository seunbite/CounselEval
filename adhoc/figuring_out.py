import os
import time
import json
import pandas as pd
import numpy as np
from counseleval.src.load import logger

def analyze_nonverbal_cues_facial(landmarks_path, output_dir):
    """Analyze nonverbal cues from facial landmarks for psychological counseling assessment."""
    try:
        df = pd.read_csv(landmarks_path)
        
        # Initialize nonverbal cue metrics
        nonverbal_metrics = {}
        
        # 1. Emotional Expression Intensity
        au_cols = [col for col in df.columns if 'AU' in col and 'intensity' in col]
        if au_cols:
            # Key AUs for emotions
            emotion_aus = {
                'happiness': ['AU06_intensity', 'AU12_intensity'],  # Cheek raiser, Lip corner puller
                'sadness': ['AU01_intensity', 'AU04_intensity', 'AU15_intensity'],  # Inner brow raiser, Brow lowerer, Lip corner depressor
                'anger': ['AU04_intensity', 'AU05_intensity', 'AU07_intensity', 'AU23_intensity'],  # Brow lowerer, Upper lid raiser, Lid tightener, Lip tightener
                'fear': ['AU01_intensity', 'AU02_intensity', 'AU05_intensity', 'AU20_intensity'],  # Inner brow raiser, Outer brow raiser, Upper lid raiser, Lip stretcher
                'surprise': ['AU01_intensity', 'AU02_intensity', 'AU05_intensity', 'AU26_intensity'],  # Inner brow raiser, Outer brow raiser, Upper lid raiser, Jaw drop
                'disgust': ['AU09_intensity', 'AU15_intensity', 'AU16_intensity']  # Nose wrinkler, Lip corner depressor, Lower lip depressor
            }
            
            for emotion, aus in emotion_aus.items():
                available_aus = [au for au in aus if au in df.columns]
                if available_aus:
                    emotion_intensity = df[available_aus].mean(axis=1).mean()
                    emotion_variability = df[available_aus].mean(axis=1).std()
                    nonverbal_metrics[f'{emotion}_intensity'] = float(emotion_intensity)
                    nonverbal_metrics[f'{emotion}_variability'] = float(emotion_variability)
        
        # 2. Micro-expression Analysis (sudden changes)
        if au_cols:
            au_data = df[au_cols].fillna(0)
            # Calculate frame-to-frame differences
            au_diff = au_data.diff().abs()
            micro_expression_rate = (au_diff > au_diff.quantile(0.9)).sum().sum() / len(df)
            nonverbal_metrics['micro_expression_rate'] = float(micro_expression_rate)
            
            # Expression stability
            expression_stability = 1 / (au_diff.mean().mean() + 1e-8)
            nonverbal_metrics['expression_stability'] = float(expression_stability)
        
        # 3. Eye Contact Pattern Analysis
        eye_cols = [col for col in df.columns if any(eye in col.lower() for eye in ['eye', 'gaze'])]
        if eye_cols:
            # Estimate gaze direction stability
            eye_data = df[eye_cols].fillna(method='ffill').fillna(0)
            gaze_stability = 1 / (eye_data.std().mean() + 1e-8)
            nonverbal_metrics['gaze_stability'] = float(gaze_stability)
            
            # Eye contact consistency (low variability = more consistent)
            eye_variability = eye_data.var().mean()
            nonverbal_metrics['eye_contact_consistency'] = float(1 / (eye_variability + 1e-8))
        
        # 4. Overall Facial Expressiveness
        all_landmark_cols = [col for col in df.columns if col.startswith(('x_', 'y_'))]
        if all_landmark_cols:
            landmark_data = df[all_landmark_cols].fillna(method='ffill').fillna(0)
            
            # Calculate facial movement intensity
            movement_intensity = landmark_data.diff().abs().mean().mean()
            nonverbal_metrics['facial_movement_intensity'] = float(movement_intensity)
            
            # Facial tension (variation in landmark positions)
            facial_tension = landmark_data.std().mean()
            nonverbal_metrics['facial_tension'] = float(facial_tension)
        
        # 5. Emotional Regulation Indicators
        if au_cols:
            # Emotional volatility (how quickly emotions change)
            emotion_changes = df[au_cols].diff().abs().mean(axis=1)
            emotional_volatility = emotion_changes.mean()
            nonverbal_metrics['emotional_volatility'] = float(emotional_volatility)
            
            # Emotional suppression indicator (low AU activity with high tension)
            au_activity = df[au_cols].mean().mean()
            if 'facial_tension' in nonverbal_metrics:
                suppression_indicator = nonverbal_metrics['facial_tension'] / (au_activity + 1e-8)
                nonverbal_metrics['emotional_suppression_indicator'] = float(suppression_indicator)
        
        # Save results
        results_path = os.path.join(output_dir, 'facial_nonverbal_cues.json')
        with open(results_path, 'w') as f:
            json.dump(nonverbal_metrics, f, indent=2)
        
        logger.info(f"Facial nonverbal cues saved to: {results_path}")
        return results_path, nonverbal_metrics
        
    except Exception as e:
        logger.error(f"Could not analyze facial nonverbal cues: {e}")
        return None, {}

def analyze_nonverbal_cues_pose(pose_path, output_dir):
    """Analyze nonverbal cues from pose landmarks for psychological counseling assessment."""
    try:
        df = pd.read_csv(pose_path)
        
        nonverbal_metrics = {}
        
        # 1. Body Openness/Closure Analysis
        # Calculate distances between key body parts
        key_joints = ['left_shoulder', 'right_shoulder', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip']
        available_joints = [joint for joint in key_joints if f'{joint}_x' in df.columns and f'{joint}_y' in df.columns]
        
        if len(available_joints) >= 4:
            # Shoulder width (openness indicator)
            if 'left_shoulder_x' in df.columns and 'right_shoulder_x' in df.columns:
                shoulder_width = abs(df['right_shoulder_x'] - df['left_shoulder_x']).mean()
                nonverbal_metrics['shoulder_openness'] = float(shoulder_width)
            
            # Arm positioning (crossed vs open)
            if all(f'{joint}_x' in df.columns for joint in ['left_wrist', 'right_wrist', 'left_shoulder', 'right_shoulder']):
                # Check if arms are crossed (wrists closer to opposite sides)
                left_wrist_to_right_shoulder = np.sqrt(
                    (df['left_wrist_x'] - df['right_shoulder_x'])**2 + 
                    (df['left_wrist_y'] - df['right_shoulder_y'])**2
                ).mean()
                right_wrist_to_left_shoulder = np.sqrt(
                    (df['right_wrist_x'] - df['left_shoulder_x'])**2 + 
                    (df['right_wrist_y'] - df['left_shoulder_y'])**2
                ).mean()
                
                arm_crossing_tendency = 1 / (left_wrist_to_right_shoulder + right_wrist_to_left_shoulder + 1e-8)
                nonverbal_metrics['arm_crossing_tendency'] = float(arm_crossing_tendency)
        
        # 2. Postural Stability and Changes
        torso_joints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        available_torso = [joint for joint in torso_joints if f'{joint}_x' in df.columns and f'{joint}_y' in df.columns]
        
        if len(available_torso) >= 2:
            # Calculate center of torso
            torso_x_cols = [f'{joint}_x' for joint in available_torso]
            torso_y_cols = [f'{joint}_y' for joint in available_torso]
            
            center_x = df[torso_x_cols].mean(axis=1)
            center_y = df[torso_y_cols].mean(axis=1)
            
            # Postural sway (movement of body center)
            postural_sway = np.sqrt(center_x.diff()**2 + center_y.diff()**2).mean()
            nonverbal_metrics['postural_sway'] = float(postural_sway)
            
            # Postural stability
            postural_stability = 1 / (postural_sway + 1e-8)
            nonverbal_metrics['postural_stability'] = float(postural_stability)
        
        # 3. Hand Movement Analysis (self-soothing behaviors)
        hand_joints = ['left_wrist', 'right_wrist']
        available_hands = [joint for joint in hand_joints if f'{joint}_x' in df.columns and f'{joint}_y' in df.columns]
        
        if available_hands:
            # Hand movement intensity
            hand_movement_total = 0
            for joint in available_hands:
                movement = np.sqrt(df[f'{joint}_x'].diff()**2 + df[f'{joint}_y'].diff()**2)
                hand_movement_total += movement.mean()
            
            nonverbal_metrics['hand_movement_intensity'] = float(hand_movement_total / len(available_hands))
            
            # Hand fidgeting (high frequency small movements)
            hand_fidgeting = 0
            for joint in available_hands:
                movement = np.sqrt(df[f'{joint}_x'].diff()**2 + df[f'{joint}_y'].diff()**2)
                small_movements = (movement < movement.quantile(0.5)) & (movement > 0)
                hand_fidgeting += small_movements.sum()
            
            fidgeting_rate = hand_fidgeting / (len(df) * len(available_hands))
            nonverbal_metrics['hand_fidgeting_rate'] = float(fidgeting_rate)
        
        # 4. Body Tension Indicators
        all_joint_cols = [col for col in df.columns if col.endswith('_x') or col.endswith('_y')]
        if all_joint_cols:
            joint_data = df[all_joint_cols].fillna(method='ffill').fillna(0)
            
            # Overall body tension (variance in joint positions)
            body_tension = joint_data.var().mean()
            nonverbal_metrics['body_tension'] = float(body_tension)
            
            # Movement rigidity (low variability in movement patterns)
            movement_data = joint_data.diff().abs()
            movement_rigidity = 1 / (movement_data.std().mean() + 1e-8)
            nonverbal_metrics['movement_rigidity'] = float(movement_rigidity)
        
        # 5. Defensive vs Open Posture Scoring
        defensive_score = 0
        openness_score = 0
        
        if 'arm_crossing_tendency' in nonverbal_metrics:
            defensive_score += nonverbal_metrics['arm_crossing_tendency']
        if 'shoulder_openness' in nonverbal_metrics:
            openness_score += nonverbal_metrics['shoulder_openness']
        if 'postural_stability' in nonverbal_metrics:
            # High stability can indicate defensiveness or composure
            if nonverbal_metrics['postural_stability'] > 1.0:
                defensive_score += 0.5
            else:
                openness_score += 0.5
        
        nonverbal_metrics['defensive_posture_score'] = float(defensive_score)
        nonverbal_metrics['open_posture_score'] = float(openness_score)
        
        # Save results
        results_path = os.path.join(output_dir, 'pose_nonverbal_cues.json')
        with open(results_path, 'w') as f:
            json.dump(nonverbal_metrics, f, indent=2)
        
        logger.info(f"Pose nonverbal cues saved to: {results_path}")
        return results_path, nonverbal_metrics
        
    except Exception as e:
        logger.error(f"Could not analyze pose nonverbal cues: {e}")
        return None, {}

def analyze_nonverbal_cues_prosody(prosody_path, output_dir):
    """Analyze nonverbal cues from prosody features for psychological counseling assessment."""
    try:
        df = pd.read_csv(prosody_path)
        
        nonverbal_metrics = {}
        
        # 1. Voice Intensity and Energy Analysis
        intensity_cols = [col for col in df.columns if 'intensity' in col.lower() or 'energy' in col.lower()]
        if intensity_cols:
            for col in intensity_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    nonverbal_metrics[f'{col}_mean'] = float(col_data.mean())
                    nonverbal_metrics[f'{col}_variability'] = float(col_data.std())
                    
                    # Voice strain indicator (high intensity with high variability)
                    strain_indicator = col_data.mean() * col_data.std()
                    nonverbal_metrics[f'{col}_strain_indicator'] = float(strain_indicator)
        
        # 2. Pitch Analysis (F0)
        f0_cols = [col for col in df.columns if 'F0' in col or 'f0' in col or 'pitch' in col.lower()]
        if f0_cols:
            for col in f0_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    # Basic pitch statistics
                    nonverbal_metrics[f'{col}_mean'] = float(col_data.mean())
                    nonverbal_metrics[f'{col}_range'] = float(col_data.max() - col_data.min())
                    nonverbal_metrics[f'{col}_variability'] = float(col_data.std())
                    
                    # Emotional indicators
                    # High pitch variability can indicate emotional arousal
                    pitch_variability = col_data.std() / (col_data.mean() + 1e-8)
                    nonverbal_metrics[f'{col}_emotional_arousal'] = float(pitch_variability)
                    
                    # Pitch tremor (rapid fluctuations)
                    pitch_diff = col_data.diff().abs()
                    pitch_tremor = pitch_diff.mean()
                    nonverbal_metrics[f'{col}_tremor'] = float(pitch_tremor)
        
        # 3. Spectral Features Analysis
        spectral_cols = [col for col in df.columns if 'spectral' in col.lower() or 'mfcc' in col.lower()]
        if spectral_cols:
            # Voice quality indicators
            for col in spectral_cols[:5]:  # Limit to first 5 spectral features
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    nonverbal_metrics[f'{col}_mean'] = float(col_data.mean())
                    nonverbal_metrics[f'{col}_stability'] = float(1 / (col_data.std() + 1e-8))
        
        # 4. Speech Rate and Rhythm Analysis
        # Estimate speech rate from frame changes
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Overall speech activity (non-zero frames)
            speech_activity = (df[numeric_cols].abs() > df[numeric_cols].abs().quantile(0.1)).any(axis=1)
            speech_rate = speech_activity.sum() / len(df)
            nonverbal_metrics['speech_activity_rate'] = float(speech_rate)
            
            # Speech rhythm regularity
            speech_segments = []
            current_segment = 0
            for active in speech_activity:
                if active:
                    current_segment += 1
                else:
                    if current_segment > 0:
                        speech_segments.append(current_segment)
                        current_segment = 0
            
            if speech_segments:
                rhythm_regularity = 1 / (np.std(speech_segments) + 1e-8)
                nonverbal_metrics['speech_rhythm_regularity'] = float(rhythm_regularity)
        
        # 5. Stress and Anxiety Indicators
        # Combine multiple features to create composite scores
        stress_indicators = []
        
        # High pitch with high variability
        for col in f0_cols:
            if f'{col}_emotional_arousal' in nonverbal_metrics:
                stress_indicators.append(nonverbal_metrics[f'{col}_emotional_arousal'])
        
        # Voice tremor
        for col in f0_cols:
            if f'{col}_tremor' in nonverbal_metrics:
                stress_indicators.append(nonverbal_metrics[f'{col}_tremor'])
        
        # Voice strain
        for col in intensity_cols:
            if f'{col}_strain_indicator' in nonverbal_metrics:
                stress_indicators.append(nonverbal_metrics[f'{col}_strain_indicator'])
        
        if stress_indicators:
            voice_stress_score = np.mean(stress_indicators)
            nonverbal_metrics['voice_stress_composite_score'] = float(voice_stress_score)
        
        # 6. Confidence Indicators
        confidence_indicators = []
        
        # Stable intensity
        for col in intensity_cols:
            if f'{col}_variability' in nonverbal_metrics:
                stability = 1 / (nonverbal_metrics[f'{col}_variability'] + 1e-8)
                confidence_indicators.append(stability)
        
        # Stable pitch
        for col in f0_cols:
            if f'{col}_variability' in nonverbal_metrics:
                stability = 1 / (nonverbal_metrics[f'{col}_variability'] + 1e-8)
                confidence_indicators.append(stability)
        
        if confidence_indicators:
            voice_confidence_score = np.mean(confidence_indicators)
            nonverbal_metrics['voice_confidence_composite_score'] = float(voice_confidence_score)
        
        # Save results
        results_path = os.path.join(output_dir, 'prosody_nonverbal_cues.json')
        with open(results_path, 'w') as f:
            json.dump(nonverbal_metrics, f, indent=2)
        
        logger.info(f"Prosody nonverbal cues saved to: {results_path}")
        return results_path, nonverbal_metrics
        
    except Exception as e:
        logger.error(f"Could not analyze prosody nonverbal cues: {e}")
        return None, {}

def save_analysis_summary(analysis_file, video_info, output_dir, task_type):
    """Save a memory-efficient summary of analysis results."""
    try:
        df = pd.read_csv(analysis_file)
        
        # Create summary statistics
        summary = {
            'metadata': {
                'task_type': task_type,
                'video_info': video_info,
                'total_frames': len(df),
                'analysis_file': analysis_file,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'statistics': {},
            'key_frames': {},
            'trends': {}
        }
        
        # Get numeric columns for statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate summary statistics
        for col in numeric_cols:
            if df[col].notna().sum() > 0:  # Only if column has valid data
                summary['statistics'][col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'median': float(df[col].median()),
                    'q25': float(df[col].quantile(0.25)),
                    'q75': float(df[col].quantile(0.75))
                }
        
        # Sample key frames (every 10% of the video)
        frame_indices = np.linspace(0, len(df)-1, 11, dtype=int)
        for i, frame_idx in enumerate(frame_indices):
            frame_data = {}
            for col in numeric_cols[:10]:  # Limit to first 10 numeric columns to save space
                if frame_idx < len(df):
                    frame_data[col] = float(df.iloc[frame_idx][col]) if pd.notna(df.iloc[frame_idx][col]) else None
            summary['key_frames'][f'frame_{frame_idx}'] = frame_data
        
        # Calculate trends (moving averages for key features)
        window_size = max(1, len(df) // 20)  # 5% of total frames
        for col in numeric_cols[:5]:  # Top 5 features for trends
            if df[col].notna().sum() > window_size:
                moving_avg = df[col].rolling(window=window_size, center=True).mean()
                # Sample trend points (every 5% of video)
                trend_indices = np.linspace(0, len(moving_avg)-1, 21, dtype=int)
                trend_values = []
                for idx in trend_indices:
                    val = moving_avg.iloc[idx] if idx < len(moving_avg) and pd.notna(moving_avg.iloc[idx]) else None
                    trend_values.append(val)
                summary['trends'][col] = {
                    'values': trend_values,
                    'frame_indices': trend_indices.tolist()
                }
        
        # Save summary as JSON (human readable) and pickle (efficient)
        json_path = os.path.join(output_dir, f'{task_type}_summary.json')
        pickle_path = os.path.join(output_dir, f'{task_type}_summary.pkl')
        
        # Save JSON (smaller, human-readable)
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save pickle (more efficient for loading)
        import pickle
        with open(pickle_path, 'wb') as f:
            pickle.dump(summary, f)
        
        logger.info(f"Analysis summary saved to: {json_path} and {pickle_path}")
        return json_path, pickle_path
        
    except Exception as e:
        logger.warning(f"Could not save analysis summary: {e}")
        return None, None 