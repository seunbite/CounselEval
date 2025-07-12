import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import json
import pickle
from counseleval.src.load import logger

def create_prosody_plot(features_path, output_dir):
    """Create a data visualization plot for prosody analysis."""
    try:
        df = pd.read_csv(features_path)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Prosody Analysis Visualization', fontsize=16)
        
        # Plot energy/intensity if available
        if 'pcm_intensity_sma' in df.columns:
            axes[0, 0].plot(df['pcm_intensity_sma'])
            axes[0, 0].set_title('Energy/Intensity')
            axes[0, 0].set_ylabel('Intensity')
        
        # Plot pitch if available
        if 'F0_sma' in df.columns:
            axes[0, 1].plot(df['F0_sma'])
            axes[0, 1].set_title('Fundamental Frequency (F0)')
            axes[0, 1].set_ylabel('F0 (Hz)')
        
        # Plot spectral features if available
        spectral_cols = [col for col in df.columns if 'spectral' in col.lower()]
        if spectral_cols:
            axes[1, 0].plot(df[spectral_cols[0]])
            axes[1, 0].set_title(f'Spectral Feature: {spectral_cols[0]}')
            axes[1, 0].set_ylabel('Value')
        
        # Plot overall feature distribution
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            axes[1, 1].hist(df[numeric_cols[0]], bins=30, alpha=0.7)
            axes[1, 1].set_title(f'Distribution: {numeric_cols[0]}')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        viz_path = os.path.join(output_dir, 'prosody_plot.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Prosody plot saved to: {viz_path}")
        return viz_path
        
    except Exception as e:
        logger.warning(f"Could not create prosody plot: {e}")
        return None

def create_facial_plot(landmarks_path, output_dir):
    """Create a data visualization plot for facial expression analysis."""
    try:
        df = pd.read_csv(landmarks_path)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Facial Expression Analysis Visualization', fontsize=16)
        
        # Plot facial landmark positions over time
        x_cols = [col for col in df.columns if col.startswith('x_')]
        y_cols = [col for col in df.columns if col.startswith('y_')]
        
        if x_cols and y_cols:
            # Plot a few key landmarks
            key_landmarks = x_cols[:5]  # First 5 landmarks
            for i, col in enumerate(key_landmarks):
                axes[0, 0].plot(df[col], alpha=0.7, label=f'Landmark {i}')
            axes[0, 0].set_title('X-coordinates of Key Landmarks')
            axes[0, 0].set_ylabel('X Position')
            axes[0, 0].legend()
        
        # Plot action units if available
        au_cols = [col for col in df.columns if 'AU' in col and 'intensity' in col]
        if au_cols:
            # Plot first few action units
            key_aus = au_cols[:4]
            for au in key_aus:
                axes[0, 1].plot(df[au], alpha=0.7, label=au)
            axes[0, 1].set_title('Action Unit Intensities')
            axes[0, 1].set_ylabel('Intensity')
            axes[0, 1].legend()
        
        # Create a scatter plot of face center movement
        if x_cols and y_cols:
            center_x = df[x_cols].mean(axis=1)
            center_y = df[y_cols].mean(axis=1)
            axes[1, 0].scatter(center_x, center_y, alpha=0.6, s=1)
            axes[1, 0].set_title('Face Center Movement')
            axes[1, 0].set_xlabel('X Position')
            axes[1, 0].set_ylabel('Y Position')
        
        # Plot frame count
        axes[1, 1].plot(df.index)
        axes[1, 1].set_title('Frame Count')
        axes[1, 1].set_xlabel('Frame Index')
        axes[1, 1].set_ylabel('Frame Number')
        
        plt.tight_layout()
        viz_path = os.path.join(output_dir, 'facial_plot.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Facial plot saved to: {viz_path}")
        return viz_path
        
    except Exception as e:
        logger.warning(f"Could not create facial plot: {e}")
        return None

def create_pose_plot(pose_path, output_dir):
    """Create a data visualization plot for pose/gesture analysis."""
    try:
        df = pd.read_csv(pose_path)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Pose/Gesture Analysis Visualization', fontsize=16)
        
        # Plot key joint movements
        key_joints = ['left_shoulder', 'right_shoulder', 'left_wrist', 'right_wrist']
        available_joints = []
        
        for joint in key_joints:
            if f'{joint}_x' in df.columns:
                available_joints.append(joint)
        
        if available_joints:
            # Plot X coordinates of key joints
            for joint in available_joints[:4]:
                axes[0, 0].plot(df[f'{joint}_x'], alpha=0.7, label=joint)
            axes[0, 0].set_title('X-coordinates of Key Joints')
            axes[0, 0].set_ylabel('X Position')
            axes[0, 0].legend()
            
            # Plot Y coordinates of key joints
            for joint in available_joints[:4]:
                axes[0, 1].plot(df[f'{joint}_y'], alpha=0.7, label=joint)
            axes[0, 1].set_title('Y-coordinates of Key Joints')
            axes[0, 1].set_ylabel('Y Position')
            axes[0, 1].legend()
        
        # Plot visibility scores if available
        visibility_cols = [col for col in df.columns if 'visibility' in col]
        if visibility_cols:
            for col in visibility_cols[:4]:
                axes[1, 0].plot(df[col], alpha=0.7, label=col.replace('_visibility', ''))
            axes[1, 0].set_title('Joint Visibility Scores')
            axes[1, 0].set_ylabel('Visibility')
            axes[1, 0].legend()
        
        # Plot frame count
        axes[1, 1].plot(df.index)
        axes[1, 1].set_title('Frame Count')
        axes[1, 1].set_xlabel('Frame Index')
        axes[1, 1].set_ylabel('Frame Number')
        
        plt.tight_layout()
        viz_path = os.path.join(output_dir, 'pose_plot.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Pose plot saved to: {viz_path}")
        return viz_path
        
    except Exception as e:
        logger.warning(f"Could not create pose plot: {e}")
        return None

def create_facial_landmarks_frame(video_path, landmarks_path, output_dir, frame_number=None):
    """Create a video frame with facial landmarks overlaid."""
    try:
        # Read landmarks data
        df = pd.read_csv(landmarks_path)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Could not open video: {video_path}")
            return None
        
        # Get a frame (middle of video if frame_number not specified)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_number is None:
            frame_number = min(total_frames // 2, len(df) // 2)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            logger.warning("Could not read frame from video")
            return None
        
        # Get landmarks for this frame
        if frame_number >= len(df):
            frame_number = len(df) - 1
        
        frame_data = df.iloc[frame_number]
        
        # Draw facial landmarks
        x_cols = [col for col in df.columns if col.startswith('x_')]
        y_cols = [col for col in df.columns if col.startswith('y_')]
        
        if x_cols and y_cols:
            for i in range(min(len(x_cols), len(y_cols))):
                x = int(frame_data[x_cols[i]])
                y = int(frame_data[y_cols[i]])
                
                # Draw landmark point
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Green dots
                
                # Add landmark number for key points
                if i % 5 == 0:  # Show every 5th landmark number
                    cv2.putText(frame, str(i), (x+3, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Add frame info
        cv2.putText(frame, f'Frame: {frame_number}/{total_frames}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, 'Facial Landmarks', (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Save frame
        viz_path = os.path.join(output_dir, 'facial_landmarks_frame.png')
        cv2.imwrite(viz_path, frame)
        
        logger.info(f"Facial landmarks frame saved to: {viz_path}")
        return viz_path
        
    except Exception as e:
        logger.warning(f"Could not create facial landmarks frame: {e}")
        return None

def create_pose_landmarks_frame(video_path, pose_path, output_dir, frame_number=None):
    """Create a video frame with pose landmarks and skeleton overlaid."""
    try:
        # Read pose data
        df = pd.read_csv(pose_path)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Could not open video: {video_path}")
            return None
        
        # Get a frame (middle of video if frame_number not specified)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_number is None:
            frame_number = min(total_frames // 2, len(df) // 2)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            logger.warning("Could not read frame from video")
            return None
        
        # Get pose data for this frame
        if frame_number >= len(df):
            frame_number = len(df) - 1
        
        frame_data = df.iloc[frame_number]
        height, width = frame.shape[:2]
        
        # Define pose connections (MediaPipe pose skeleton)
        pose_connections = [
            # Face
            ('nose', 'left_eye_inner'), ('left_eye_inner', 'left_eye'),
            ('left_eye', 'left_eye_outer'), ('nose', 'right_eye_inner'),
            ('right_eye_inner', 'right_eye'), ('right_eye', 'right_eye_outer'),
            # Arms
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
            # Body
            ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            # Legs
            ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
        ]
        
        # Draw pose landmarks and connections
        landmarks = {}
        
        # Extract landmark positions
        for col in df.columns:
            if col.endswith('_x'):
                joint_name = col[:-2]
                if f'{joint_name}_y' in df.columns and f'{joint_name}_visibility' in df.columns:
                    x = frame_data[f'{joint_name}_x']
                    y = frame_data[f'{joint_name}_y']
                    visibility = frame_data[f'{joint_name}_visibility']
                    
                    if visibility > 0.5:  # Only draw visible landmarks
                        # Convert normalized coordinates to pixel coordinates
                        px = int(x * width) if x <= 1.0 else int(x)
                        py = int(y * height) if y <= 1.0 else int(y)
                        landmarks[joint_name] = (px, py)
        
        # Draw connections
        for start_joint, end_joint in pose_connections:
            if start_joint in landmarks and end_joint in landmarks:
                cv2.line(frame, landmarks[start_joint], landmarks[end_joint], (0, 255, 255), 2)  # Yellow lines
        
        # Draw landmark points
        for joint_name, (px, py) in landmarks.items():
            cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)  # Red dots
            # Label key joints
            if joint_name in ['nose', 'left_shoulder', 'right_shoulder', 'left_wrist', 'right_wrist']:
                cv2.putText(frame, joint_name.replace('_', ' '), (px+5, py-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add frame info
        cv2.putText(frame, f'Frame: {frame_number}/{total_frames}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, 'Pose Landmarks', (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Save frame
        viz_path = os.path.join(output_dir, 'pose_landmarks_frame.png')
        cv2.imwrite(viz_path, frame)
        
        logger.info(f"Pose landmarks frame saved to: {viz_path}")
        return viz_path
        
    except Exception as e:
        logger.warning(f"Could not create pose landmarks frame: {e}")
        return None

def create_nonverbal_cues_plot(cues_data, task_type, output_dir):
    """Create visualization plots for nonverbal cues analysis."""
    try:
        if not cues_data:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{task_type.title()} Nonverbal Cues Analysis', fontsize=16, fontweight='bold')
        
        # Convert data to pandas for easier plotting
        metrics_df = pd.DataFrame(list(cues_data.items()), columns=['Metric', 'Value'])
        
        # Plot 1: Top metrics by value
        top_metrics = metrics_df.nlargest(8, 'Value')
        axes[0, 0].barh(range(len(top_metrics)), top_metrics['Value'], color='skyblue')
        axes[0, 0].set_yticks(range(len(top_metrics)))
        axes[0, 0].set_yticklabels([m.replace('_', ' ').title()[:20] for m in top_metrics['Metric']])
        axes[0, 0].set_title('Top Nonverbal Cue Metrics')
        axes[0, 0].set_xlabel('Value')
        
        # Plot 2: Emotional indicators (if available)
        emotion_metrics = [k for k in cues_data.keys() if any(emotion in k.lower() for emotion in ['emotion', 'happiness', 'sadness', 'anger', 'fear', 'surprise'])]
        if emotion_metrics:
            emotion_values = [cues_data[k] for k in emotion_metrics]
            axes[0, 1].bar(range(len(emotion_metrics)), emotion_values, color='lightcoral')
            axes[0, 1].set_xticks(range(len(emotion_metrics)))
            axes[0, 1].set_xticklabels([m.replace('_', ' ').title()[:15] for m in emotion_metrics], rotation=45, ha='right')
            axes[0, 1].set_title('Emotional Indicators')
            axes[0, 1].set_ylabel('Intensity')
        else:
            axes[0, 1].text(0.5, 0.5, 'No emotional indicators\navailable', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Emotional Indicators')
        
        # Plot 3: Stress/tension indicators
        stress_metrics = [k for k in cues_data.keys() if any(stress in k.lower() for stress in ['stress', 'tension', 'anxiety', 'strain', 'rigidity'])]
        if stress_metrics:
            stress_values = [cues_data[k] for k in stress_metrics]
            axes[1, 0].bar(range(len(stress_metrics)), stress_values, color='orange')
            axes[1, 0].set_xticks(range(len(stress_metrics)))
            axes[1, 0].set_xticklabels([m.replace('_', ' ').title()[:15] for m in stress_metrics], rotation=45, ha='right')
            axes[1, 0].set_title('Stress/Tension Indicators')
            axes[1, 0].set_ylabel('Level')
        else:
            axes[1, 0].text(0.5, 0.5, 'No stress indicators\navailable', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Stress/Tension Indicators')
        
        # Plot 4: Composite scores and ratios
        composite_metrics = [k for k in cues_data.keys() if any(comp in k.lower() for comp in ['score', 'composite', 'ratio', 'indicator'])]
        if composite_metrics:
            composite_values = [cues_data[k] for k in composite_metrics]
            axes[1, 1].scatter(range(len(composite_metrics)), composite_values, s=100, c='green', alpha=0.7)
            axes[1, 1].set_xticks(range(len(composite_metrics)))
            axes[1, 1].set_xticklabels([m.replace('_', ' ').title()[:15] for m in composite_metrics], rotation=45, ha='right')
            axes[1, 1].set_title('Composite Scores')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No composite scores\navailable', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Composite Scores')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f'{task_type}_nonverbal_cues_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Nonverbal cues plot saved to: {plot_path}")
        return plot_path
        
    except Exception as e:
        logger.warning(f"Could not create nonverbal cues plot: {e}")
        return None

def load_and_plot_summary(summary_path, output_dir=None):
    """Load analysis summary and create comprehensive plots."""
    try:
        # Determine file type and load accordingly
        if summary_path.endswith('.json'):
            with open(summary_path, 'r') as f:
                summary = json.load(f)
        elif summary_path.endswith('.pkl'):
            with open(summary_path, 'rb') as f:
                summary = pickle.load(f)
        else:
            raise ValueError("Summary file must be either .json or .pkl")
        
        task_type = summary['metadata']['task_type']
        stats = summary['statistics']
        trends = summary['trends']
        
        if output_dir is None:
            output_dir = os.path.dirname(summary_path)
        
        # Create comprehensive visualization
        n_features = min(6, len(stats))
        fig = plt.figure(figsize=(16, 12))
        
        # Main title with metadata
        fig.suptitle(f'{task_type.title()} Analysis Summary\nTotal Frames: {summary["metadata"]["total_frames"]} | '
                    f'Duration: {summary["metadata"]["video_info"]["duration_seconds"]/60:.1f}min', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Statistics overview (box plots)
        ax1 = plt.subplot(3, 2, 1)
        feature_names = list(stats.keys())[:n_features]
        box_data = []
        for feature in feature_names:
            # Create approximate distribution from summary stats
            mean = stats[feature]['mean']
            std = stats[feature]['std']
            q25 = stats[feature]['q25']
            median = stats[feature]['median']
            q75 = stats[feature]['q75']
            
            # Approximate box plot data
            box_data.append([q25, median, q75, mean-std, mean+std])
        
        bp = ax1.boxplot(box_data, labels=[name.replace('_', '\n')[:20] for name in feature_names])
        ax1.set_title('Feature Statistics Overview')
        ax1.set_ylabel('Values')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Plot 2: Trends over time
        ax2 = plt.subplot(3, 2, 2)
        for feature_name, trend_data in list(trends.items())[:4]:
            frame_indices = trend_data['frame_indices']
            values = [v for v in trend_data['values'] if v is not None]
            valid_indices = [frame_indices[i] for i, v in enumerate(trend_data['values']) if v is not None]
            
            if len(values) > 1:
                ax2.plot(valid_indices, values, label=feature_name.replace('_', ' ')[:15], alpha=0.8, linewidth=2)
        
        ax2.set_title('Feature Trends Over Time')
        ax2.set_xlabel('Frame Index')
        ax2.set_ylabel('Feature Values')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Feature correlations (if multiple features)
        ax3 = plt.subplot(3, 2, 3)
        if len(stats) > 1:
            # Create correlation matrix from means
            features = list(stats.keys())[:8]  # Limit to 8 features
            means = [stats[f]['mean'] for f in features]
            stds = [stats[f]['std'] for f in features]
            
            # Simple correlation approximation (this is just for visualization)
            corr_data = np.outer(means, means) / (np.outer(stds, stds) + 1e-8)
            corr_data = np.clip(corr_data, -1, 1)
            
            im = ax3.imshow(corr_data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            ax3.set_xticks(range(len(features)))
            ax3.set_yticks(range(len(features)))
            ax3.set_xticklabels([f.replace('_', '\n')[:15] for f in features], rotation=45, ha='right')
            ax3.set_yticklabels([f.replace('_', ' ')[:15] for f in features])
            ax3.set_title('Feature Relationship Overview')
            plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        else:
            ax3.text(0.5, 0.5, 'Not enough features\nfor correlation analysis', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Feature Relationships')
        
        # Plot 4: Distribution comparison
        ax4 = plt.subplot(3, 2, 4)
        top_features = list(stats.keys())[:3]
        for i, feature in enumerate(top_features):
            # Create histogram from summary statistics
            mean = stats[feature]['mean']
            std = stats[feature]['std']
            
            # Generate sample distribution
            x = np.linspace(mean - 3*std, mean + 3*std, 100)
            y = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
            
            ax4.plot(x, y, label=feature.replace('_', ' ')[:15], alpha=0.7, linewidth=2)
        
        ax4.set_title('Feature Distributions')
        ax4.set_xlabel('Value')
        ax4.set_ylabel('Density')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Key frames summary
        ax5 = plt.subplot(3, 2, 5)
        key_frames = summary['key_frames']
        if key_frames and len(list(key_frames.values())[0]) > 0:
            # Get first feature from key frames
            first_feature = list(list(key_frames.values())[0].keys())[0]
            frame_nums = []
            values = []
            
            for frame_key, frame_data in key_frames.items():
                if first_feature in frame_data and frame_data[first_feature] is not None:
                    frame_num = int(frame_key.split('_')[1])
                    frame_nums.append(frame_num)
                    values.append(frame_data[first_feature])
            
            if len(frame_nums) > 1:
                ax5.plot(frame_nums, values, 'o-', linewidth=2, markersize=6, label=first_feature)
                ax5.set_title(f'Key Frame Analysis: {first_feature.replace("_", " ")}')
                ax5.set_xlabel('Frame Number')
                ax5.set_ylabel('Value')
                ax5.grid(True, alpha=0.3)
            else:
                ax5.text(0.5, 0.5, 'Insufficient key frame data', 
                        ha='center', va='center', transform=ax5.transAxes)
        else:
            ax5.text(0.5, 0.5, 'No key frame data available', 
                    ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Key Frames Analysis')
        
        # Plot 6: Summary metrics
        ax6 = plt.subplot(3, 2, 6)
        # Create a summary table
        summary_text = f"Analysis Summary:\n\n"
        summary_text += f"Task Type: {task_type.title()}\n"
        summary_text += f"Total Frames: {summary['metadata']['total_frames']:,}\n"
        summary_text += f"Duration: {summary['metadata']['video_info']['duration_seconds']/60:.1f} min\n"
        summary_text += f"Features Analyzed: {len(stats)}\n\n"
        
        # Top 3 most variable features
        if stats:
            variability = [(name, data.get('std', 0) / (abs(data.get('mean', 1)) + 1e-8)) 
                          for name, data in stats.items()]
            variability.sort(key=lambda x: x[1], reverse=True)
            
            summary_text += "Most Variable Features:\n"
            for i, (feature, cv) in enumerate(variability[:3]):
                summary_text += f"{i+1}. {feature.replace('_', ' ')}: {cv:.3f}\n"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
                verticalalignment='top', fontfamily='monospace', fontsize=10)
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        ax6.set_title('Analysis Metadata')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f'{task_type}_summary_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Summary plot saved to: {plot_path}")
        return plot_path
        
    except Exception as e:
        logger.error(f"Could not load and plot summary: {e}")
        return None 