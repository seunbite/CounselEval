import os
import time
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import numpy as np
import pandas as pd
import librosa
import cv2
import mediapipe as mp
from tqdm import tqdm
from counseleval.src.load import RobustVideoLoader, get_system_memory_info, logger

# Initialize MediaPipe modules
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

class MockOpenSmileTool:
    """Mock OpenSMILE tool for prosody analysis."""
    
    def __init__(self, config_name: str = 'IS09_emotion'):
        self.config_name = config_name
        logger.info(f"Initialized MockOpenSmileTool with config: {config_name}")
    
    def inference(self, audio_path: str, features_path: str):
        """Mock prosody feature extraction."""
        logger.info(f"Mock prosody analysis: {audio_path} -> {features_path}")
        
        # Load audio to get duration for realistic mock data
        try:
            y, sr = librosa.load(audio_path)
            duration = len(y) / sr
            n_frames = int(duration * 100)  # 100 Hz feature rate
        except:
            n_frames = 1000  # Fallback
        
        # Create mock prosodic features
        mock_data = {
            'timestamp': np.linspace(0, duration, n_frames),
            'F0': np.random.normal(150, 30, n_frames),  # Fundamental frequency
            'energy': np.random.normal(0.5, 0.2, n_frames),  # Energy
            'spectral_centroid': np.random.normal(2000, 500, n_frames),
            'mfcc_1': np.random.normal(0, 1, n_frames),
            'mfcc_2': np.random.normal(0, 1, n_frames),
        }
        
        df = pd.DataFrame(mock_data)
        df.to_csv(features_path, index=False)
        logger.info(f"Mock prosody features saved to: {features_path}")

class RealOpenFaceTool:
    """Real facial analysis tool using MediaPipe Face Mesh."""
    
    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir
        logger.info(f"Initialized RealOpenFaceTool with MediaPipe Face Mesh")
    
    def inference(self, video_path: str, landmarks_path: str, viz_params=None):
        """Real facial landmark extraction using MediaPipe."""
        logger.info(f"Real facial analysis: {video_path} -> {landmarks_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video file: {video_path}")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_results = []
        saved_viz_frames = []
        
        with mp_holistic.Holistic(
            static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5) as holistic:
            
            # Process every 2nd frame for performance
            frame_indices = range(0, total_frames, 2)
            
            for frame_idx in tqdm(frame_indices, desc="Processing facial landmarks"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, image = cap.read()
                if not success:
                    continue
                
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                
                # Process with MediaPipe
                results = holistic.process(image_rgb)
                
                # Extract facial landmarks (if detected)
                frame_data = {'frame': frame_idx}
                
                if results.face_landmarks:
                    # Extract 68 key facial landmarks (subset of MediaPipe's 468)
                    # Map MediaPipe face landmarks to OpenFace-like 68 landmarks
                    key_landmark_indices = [
                        # Jaw line (0-16)
                        172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454,
                        # Right eyebrow (17-21)
                        70, 63, 105, 66, 107,
                        # Left eyebrow (22-26)  
                        55, 65, 52, 53, 46,
                        # Nose (27-35)
                        168, 8, 9, 10, 151, 195, 197, 196, 3,
                        # Right eye (36-41)
                        33, 7, 163, 144, 145, 153,
                        # Left eye (42-47)
                        362, 382, 381, 380, 374, 373,
                        # Mouth outer (48-59)
                        61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
                        # Mouth inner (60-67)
                        78, 95, 88, 178, 87, 14, 317, 402
                    ]
                    
                    # Extract coordinates for 68 landmarks
                    for i, landmark_idx in enumerate(key_landmark_indices):
                        if landmark_idx < len(results.face_landmarks.landmark):
                            landmark = results.face_landmarks.landmark[landmark_idx]
                            # Convert normalized coordinates to pixel coordinates
                            h, w, _ = image.shape
                            frame_data[f'x_{i}'] = landmark.x * w
                            frame_data[f'y_{i}'] = landmark.y * h
                        else:
                            frame_data[f'x_{i}'] = 0
                            frame_data[f'y_{i}'] = 0
                    
                    # Add mock action units (MediaPipe doesn't provide AUs directly)
                    for i in range(1, 18):
                        frame_data[f'AU{i:02d}_intensity'] = np.random.uniform(0, 2)
                    
                    # Save visualization frame if needed
                    if viz_params and frame_idx in viz_params['viz_frames']:
                        viz_frame = self._create_facial_visualization_frame(image, results, frame_idx, total_frames)
                        viz_path = os.path.join(viz_params['output_dir'], f'facial_landmarks_frame_{frame_idx:04d}.png')
                        cv2.imwrite(viz_path, viz_frame)
                        saved_viz_frames.append(viz_path)
                        logger.info(f"Saved facial visualization frame: {viz_path}")
                else:
                    # No face detected - fill with zeros
                    for i in range(68):
                        frame_data[f'x_{i}'] = 0
                        frame_data[f'y_{i}'] = 0
                    for i in range(1, 18):
                        frame_data[f'AU{i:02d}_intensity'] = 0
                
                frame_results.append(frame_data)
        
        cap.release()
        
        # Save to CSV
        df = pd.DataFrame(frame_results)
        df.to_csv(landmarks_path, index=False)
        logger.info(f"Real facial landmarks saved to: {landmarks_path} ({len(frame_results)} frames)")
        
        if viz_params and saved_viz_frames:
            logger.info(f"Saved {len(saved_viz_frames)} facial visualization frames during analysis")
    
    def _create_facial_visualization_frame(self, image, results, frame_number, total_frames):
        """Create a visualization frame with facial landmarks overlaid."""
        viz_frame = image.copy()
        
        if results.face_landmarks:
            # Draw facial landmarks
            h, w, _ = image.shape
            
            # Key landmark indices for visualization (subset for clarity)
            key_landmark_indices = [
                172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454,  # Jaw
                70, 63, 105, 66, 107, 55, 65, 52, 53, 46,  # Eyebrows
                168, 8, 9, 10, 151, 195, 197, 196, 3,  # Nose
                33, 7, 163, 144, 145, 153, 362, 382, 381, 380, 374, 373,  # Eyes
                61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318  # Mouth
            ]
            
            # Draw landmark points
            for i, landmark_idx in enumerate(key_landmark_indices):
                if landmark_idx < len(results.face_landmarks.landmark):
                    landmark = results.face_landmarks.landmark[landmark_idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    
                    # Draw landmark point
                    cv2.circle(viz_frame, (x, y), 2, (0, 255, 0), -1)  # Green dots
                    
                    # Add landmark number for key points
                    if i % 10 == 0:  # Show every 10th landmark number
                        cv2.putText(viz_frame, str(i), (x+3, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Add frame info
        cv2.putText(viz_frame, f'Frame: {frame_number}/{total_frames}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(viz_frame, 'Facial Landmarks', (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return viz_frame

class RealMediaPipePoseTool:
    """Real pose analysis tool using MediaPipe Holistic."""
    
    def __init__(self, model_name: str = 'pose'):
        self.model_name = model_name
        logger.info(f"Initialized RealMediaPipePoseTool with MediaPipe Holistic")
    
    def inference(self, video_path: str, pose_path: str, viz_params=None):
        """Real pose landmark extraction using MediaPipe."""
        logger.info(f"Real pose analysis: {video_path} -> {pose_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video file: {video_path}")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # MediaPipe pose has 33 landmarks
        pose_landmarks = [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner',
            'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left',
            'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index',
            'right_index', 'left_thumb', 'right_thumb', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel',
            'right_heel', 'left_foot_index', 'right_foot_index'
        ]
        
        frame_results = []
        saved_viz_frames = []
        
        with mp_holistic.Holistic(
            static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5) as holistic:
            
            # Process every 2nd frame for performance
            frame_indices = range(0, total_frames, 2)
            
            for frame_idx in tqdm(frame_indices, desc="Processing pose landmarks"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, image = cap.read()
                if not success:
                    continue
                
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                
                # Process with MediaPipe
                results = holistic.process(image_rgb)
                
                # Extract pose landmarks
                frame_data = {'frame': frame_idx}
                
                if results.pose_landmarks:
                    pose_landmarks_data = results.pose_landmarks.landmark
                    
                    # Extract pose coordinates
                    for i, landmark_name in enumerate(pose_landmarks):
                        if i < len(pose_landmarks_data):
                            landmark = pose_landmarks_data[i]
                            frame_data[f'{landmark_name}_x'] = landmark.x
                            frame_data[f'{landmark_name}_y'] = landmark.y
                            frame_data[f'{landmark_name}_z'] = landmark.z
                            frame_data[f'{landmark_name}_visibility'] = landmark.visibility
                        else:
                            frame_data[f'{landmark_name}_x'] = 0
                            frame_data[f'{landmark_name}_y'] = 0
                            frame_data[f'{landmark_name}_z'] = 0
                            frame_data[f'{landmark_name}_visibility'] = 0
                    
                    # Save visualization frame if needed
                    if viz_params and frame_idx in viz_params['viz_frames']:
                        viz_frame = self._create_pose_visualization_frame(image, results, frame_idx, total_frames, pose_landmarks)
                        viz_path = os.path.join(viz_params['output_dir'], f'pose_landmarks_frame_{frame_idx:04d}.png')
                        cv2.imwrite(viz_path, viz_frame)
                        saved_viz_frames.append(viz_path)
                        logger.info(f"Saved pose visualization frame: {viz_path}")
                        
                else:
                    # No pose detected - fill with zeros
                    for landmark_name in pose_landmarks:
                        frame_data[f'{landmark_name}_x'] = 0
                        frame_data[f'{landmark_name}_y'] = 0
                        frame_data[f'{landmark_name}_z'] = 0
                        frame_data[f'{landmark_name}_visibility'] = 0
                
                # Add hand landmarks if available
                if results.left_hand_landmarks:
                    hand_landmarks = results.left_hand_landmarks.landmark
                    for i, landmark in enumerate(hand_landmarks):
                        frame_data[f'left_hand_{i}_x'] = landmark.x
                        frame_data[f'left_hand_{i}_y'] = landmark.y
                        frame_data[f'left_hand_{i}_z'] = landmark.z
                
                if results.right_hand_landmarks:
                    hand_landmarks = results.right_hand_landmarks.landmark
                    for i, landmark in enumerate(hand_landmarks):
                        frame_data[f'right_hand_{i}_x'] = landmark.x
                        frame_data[f'right_hand_{i}_y'] = landmark.y
                        frame_data[f'right_hand_{i}_z'] = landmark.z
                
                frame_results.append(frame_data)
        
        cap.release()
        
        # Save to CSV
        df = pd.DataFrame(frame_results)
        df.to_csv(pose_path, index=False)
        logger.info(f"Real pose landmarks saved to: {pose_path} ({len(frame_results)} frames)")
        
        if viz_params and saved_viz_frames:
            logger.info(f"Saved {len(saved_viz_frames)} pose visualization frames during analysis")
    
    def _create_pose_visualization_frame(self, image, results, frame_number, total_frames, pose_landmarks):
        """Create a visualization frame with pose landmarks and skeleton overlaid."""
        viz_frame = image.copy()
        height, width = image.shape[:2]
        
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
        
        if results.pose_landmarks:
            # Extract landmark positions
            landmarks_dict = {}
            pose_landmarks_data = results.pose_landmarks.landmark
            
            for i, landmark_name in enumerate(pose_landmarks):
                if i < len(pose_landmarks_data):
                    landmark = pose_landmarks_data[i]
                    if landmark.visibility > 0.5:  # Only draw visible landmarks
                        # Convert normalized coordinates to pixel coordinates
                        px = int(landmark.x * width)
                        py = int(landmark.y * height)
                        landmarks_dict[landmark_name] = (px, py)
            
            # Draw connections
            for start_joint, end_joint in pose_connections:
                if start_joint in landmarks_dict and end_joint in landmarks_dict:
                    cv2.line(viz_frame, landmarks_dict[start_joint], landmarks_dict[end_joint], (0, 255, 255), 2)  # Yellow lines
            
            # Draw landmark points
            for joint_name, (px, py) in landmarks_dict.items():
                cv2.circle(viz_frame, (px, py), 4, (0, 0, 255), -1)  # Red dots
                # Label key joints
                if joint_name in ['nose', 'left_shoulder', 'right_shoulder', 'left_wrist', 'right_wrist']:
                    cv2.putText(viz_frame, joint_name.replace('_', ' '), (px+5, py-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add frame info
        cv2.putText(viz_frame, f'Frame: {frame_number}/{total_frames}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(viz_frame, 'Pose Landmarks', (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return viz_frame

# Use real tools for pose/face and mock for audio (until we have real audio tools)
OpenSmileTool = MockOpenSmileTool
OpenFaceTool = RealOpenFaceTool
MediaPipePoseTool = RealMediaPipePoseTool

def analyze_prosody(video_path, output_dir, visualize=False, frame_range=(100, 200), num_viz_frames=10):
    """
    Extract prosodic features using OpenSMILE (e.g., pitch, energy, spectral).
    """
    try:
        # Use robust video loader
        loader = RobustVideoLoader(video_path)
        video_info = loader.get_video_info()
        
        logger.info(f"Starting prosody analysis for {video_info['duration_seconds']/60:.1f} minute video")
        
        # Extract audio using robust method
        audio_path = loader.extract_audio_robust(output_dir)
        
        # Initialize OpenSMILE tool
        tool = OpenSmileTool(config_name='IS09_emotion')
        features_path = os.path.join(output_dir, 'prosody_features.csv')
        
        logger.info("Extracting prosodic features with OpenSMILE...")
        start_time = time.time()
        
        tool.inference(audio_path=audio_path, features_path=features_path)
        
        analysis_time = time.time() - start_time
        logger.info(f"Prosody analysis completed in {analysis_time:.2f} seconds")
        
        # Note: Prosody analysis works on audio, so no frame visualization is applicable
        if visualize:
            logger.info("Note: Prosody analysis works on audio - no frame visualization available")
        
        # Cleanup
        loader.cleanup()
        
        return features_path
        
    except Exception as e:
        logger.error(f"Error in prosody analysis: {e}")
        raise

def analyze_facial_expressions(video_path, output_dir, visualize=False, frame_range=(100, 200), num_viz_frames=10):
    """
    Extract facial landmarks and action units using OpenFace.
    """
    try:
        # Use robust video loader
        loader = RobustVideoLoader(video_path)
        video_info = loader.get_video_info()
        
        logger.info(f"Starting facial expression analysis for {video_info['duration_seconds']/60:.1f} minute video")
        
        # Initialize OpenFace tool with visualization parameters
        tool = OpenFaceTool(model_dir=None)
        landmarks_path = os.path.join(output_dir, 'facial_landmarks.csv')
        
        # Set up visualization parameters
        viz_params = None
        if visualize:
            start_frame, end_frame = frame_range
            viz_params = {
                'output_dir': output_dir,
                'frame_range': frame_range,
                'num_viz_frames': num_viz_frames,
                'viz_frames': list(np.linspace(start_frame, end_frame, num_viz_frames, dtype=int))
            }
            logger.info(f"Will save {num_viz_frames} visualization frames between frames {start_frame}-{end_frame}")
        
        logger.info("Extracting facial landmarks with OpenFace...")
        start_time = time.time()
        
        tool.inference(video_path=str(video_path), landmarks_path=landmarks_path, viz_params=viz_params)
        
        analysis_time = time.time() - start_time
        logger.info(f"Facial expression analysis completed in {analysis_time:.2f} seconds")
        
        # Cleanup
        loader.cleanup()
        
        return landmarks_path
        
    except Exception as e:
        logger.error(f"Error in facial expression analysis: {e}")
        raise

def analyze_gestures_posture(video_path, output_dir, visualize=False, frame_range=(100, 200), num_viz_frames=10):
    """
    Extract body pose landmarks (hands & posture) using MediaPipe.
    """
    try:
        # Use robust video loader
        loader = RobustVideoLoader(video_path)
        video_info = loader.get_video_info()
        
        logger.info(f"Starting gesture/posture analysis for {video_info['duration_seconds']/60:.1f} minute video")
        
        # Initialize MediaPipe tool
        tool = MediaPipePoseTool(model_name='pose')
        pose_path = os.path.join(output_dir, 'pose_landmarks.csv')
        
        # Set up visualization parameters
        viz_params = None
        if visualize:
            start_frame, end_frame = frame_range
            viz_params = {
                'output_dir': output_dir,
                'frame_range': frame_range,
                'num_viz_frames': num_viz_frames,
                'viz_frames': list(np.linspace(start_frame, end_frame, num_viz_frames, dtype=int))
            }
            logger.info(f"Will save {num_viz_frames} visualization frames between frames {start_frame}-{end_frame}")
        
        logger.info("Extracting pose landmarks with MediaPipe...")
        start_time = time.time()
        
        tool.inference(video_path=str(video_path), pose_path=pose_path, viz_params=viz_params)
        
        analysis_time = time.time() - start_time
        logger.info(f"Gesture/posture analysis completed in {analysis_time:.2f} seconds")

        # Cleanup
        loader.cleanup()
        
        return pose_path
        
    except Exception as e:
        logger.error(f"Error in gesture/posture analysis: {e}")
        raise

def batch_process_videos(
    video_paths: List[Union[str, Path]], 
    output_dir: Union[str, Path],
    processes: Optional[List[str]] = None,
    num_workers: int = 1,
    batch_size: int = 1,
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Process multiple videos in batch with memory management.
    
    Args:
        video_paths: List of video paths to process
        output_dir: Output directory
        processes: List of processes to run (prosody, facial, pose). None for all.
    
    Returns:
        Dictionary with results for each video
    """
    if processes is None:
        processes = ['prosody', 'facial', 'pose']
    
    output_dir = Path(output_dir)
    results = {}
    
    for i, video_path in enumerate(video_paths, 1):
        video_path = Path(video_path)
        logger.info(f"Processing video {i}/{len(video_paths)}: {video_path.name}")
        
        # Create subdirectory for this video
        video_output_dir = output_dir / video_path.stem
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Check memory before processing
            memory_info = get_system_memory_info()
            logger.info(f"Memory usage before processing: {memory_info['percent_used']:.1f}%")
            
            if memory_info['percent_used'] > 90:
                logger.warning("High memory usage detected. Consider processing fewer videos simultaneously.")
            
            # Process video
            video_results = {}
            
            if 'prosody' in processes:
                video_results['prosody'] = analyze_prosody(str(video_path), str(video_output_dir))
            
            if 'facial' in processes:
                video_results['facial'] = analyze_facial_expressions(str(video_path), str(video_output_dir))
            
            if 'pose' in processes:
                video_results['pose'] = analyze_gestures_posture(str(video_path), str(video_output_dir))
            
            results[str(video_path)] = video_results
            
            logger.info(f"Successfully processed {video_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to process {video_path.name}: {e}")
            results[str(video_path)] = {'error': str(e)}
    
    return results

