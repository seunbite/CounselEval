import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import audalign
import pickle
from datetime import datetime
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CounselingQualityAnalyzer:
    """Analyzes counseling quality by comparing client and counselor nonverbal behaviors."""
    
    def __init__(self, base_output_dir: str):
        self.base_output_dir = base_output_dir
        self.task_types = ['facial', 'pose', 'prosody']
        self.alignment_cache_file = os.path.join(base_output_dir, 'audio_alignment_cache.json')
        
    def find_client_counselor_pairs(self) -> List[Tuple[str, str]]:
        """Find client-counselor pairs based on file naming convention."""
        pairs = []
        processed_files = set()
        
        # Scan for files ending with _T and _C
        for root, dirs, files in os.walk(self.base_output_dir):
            for dir_name in dirs:
                if dir_name.endswith('_T') or dir_name.endswith('_C'):
                    base_name = dir_name[:-2]  # Remove _T or _C
                    client_dir = os.path.join(root, f"{base_name}_T")
                    counselor_dir = os.path.join(root, f"{base_name}_C")
                    
                    if (os.path.exists(client_dir) and os.path.exists(counselor_dir) and
                        base_name not in processed_files):
                        pairs.append((client_dir, counselor_dir))
                        processed_files.add(base_name)
        
        logger.info(f"Found {len(pairs)} client-counselor pairs")
        return pairs
    
    def load_analysis_data(self, data_dir: str, task_type: str) -> Optional[Dict]:
        """Load analysis data for a specific task type."""
        try:
            # Define expected file names
            file_mapping = {
                'facial': 'facial_landmarks.csv',
                'pose': 'pose_landmarks.csv', 
                'prosody': 'prosody_features.csv'
            }
            
            expected_file = file_mapping.get(task_type)
            if not expected_file:
                return None
                
            file_path = os.path.join(data_dir, expected_file)
            if not os.path.exists(file_path):
                return None
                
            # Load data with error handling
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                logger.warning(f"Could not read CSV file {file_path}: {e}")
                return None
            
            # Load nonverbal cues if available
            cues_file = os.path.join(data_dir, f'{task_type}_nonverbal_cues.json')
            cues_data = {}
            if os.path.exists(cues_file):
                try:
                    with open(cues_file, 'r') as f:
                        cues_data = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load cues data from {cues_file}: {e}")
            
            return {
                'dataframe': df,
                'cues_data': cues_data,
                'file_path': file_path,
                'task_type': task_type
            }
            
        except Exception as e:
            logger.warning(f"Could not load {task_type} data from {data_dir}: {e}")
            return None
    
    def save_alignment_data(self, client_dir: str, counselor_dir: str, offset: float, 
                          alignment_results: Dict = None) -> None:
        """Save audio alignment data for a client-counselor pair."""
        alignment_cache = self.load_alignment_cache()
        
        # Create a unique key for this pair
        pair_key = f"{os.path.basename(client_dir)}_{os.path.basename(counselor_dir)}"
        
        alignment_info = {
            'client_dir': client_dir,
            'counselor_dir': counselor_dir,
            'offset': offset,
            'timestamp': datetime.now().isoformat(),
            'alignment_results': alignment_results
        }
        
        alignment_cache[pair_key] = alignment_info
        
        # Save to file
        with open(self.alignment_cache_file, 'w') as f:
            json.dump(alignment_cache, f, indent=2)
        
        logger.info(f"Saved alignment data for {pair_key} with offset {offset}")
    
    def load_alignment_cache(self) -> Dict:
        """Load the alignment cache from file."""
        if os.path.exists(self.alignment_cache_file):
            try:
                with open(self.alignment_cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load alignment cache: {e}")
                return {}
        return {}
    
    def get_saved_alignment(self, client_dir: str, counselor_dir: str) -> Optional[float]:
        """Get saved alignment offset for a client-counselor pair."""
        alignment_cache = self.load_alignment_cache()
        pair_key = f"{os.path.basename(client_dir)}_{os.path.basename(counselor_dir)}"
        
        if pair_key in alignment_cache:
            offset = alignment_cache[pair_key]['offset']
            logger.info(f"Using saved alignment offset {offset} for {pair_key}")
            return offset
        
        return None
    
    def apply_alignment_to_csv(self, client_df: pd.DataFrame, counselor_df: pd.DataFrame, 
                             offset: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply alignment offset to CSV dataframes."""
        # Apply offset to synchronize dataframes
        if offset > 0:
            client_sync = client_df.iloc[int(offset):].reset_index(drop=True)
            counselor_sync = counselor_df.reset_index(drop=True)
        else:
            client_sync = client_df.reset_index(drop=True)
            counselor_sync = counselor_df.iloc[int(-offset):].reset_index(drop=True)
            
        # Truncate to same length
        min_length = min(len(client_sync), len(counselor_sync))
        client_sync = client_sync.iloc[:min_length]
        counselor_sync = counselor_sync.iloc[:min_length]
        
        # Add synchronized frame indices
        client_sync['frame_index'] = range(min_length)
        counselor_sync['frame_index'] = range(min_length)
        
        return client_sync, counselor_sync
    
    def synchronize_timelines(self, client_data: Dict, counselor_data: Dict, 
                            use_cache: bool = True, save_to_cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Synchronize client and counselor data timelines using audio alignment.
        
        Uses audalign library to align the audio timelines between client and counselor data.
        Returns synchronized dataframes with aligned timestamps.
        """
        client_df = client_data['dataframe']
        counselor_df = counselor_data['dataframe']
        
        # Find audio files in the data directories
        client_dir = os.path.dirname(client_data['file_path'])
        counselor_dir = os.path.dirname(counselor_data['file_path'])
        
        # Check if we have cached alignment data
        if use_cache:
            cached_offset = self.get_saved_alignment(client_dir, counselor_dir)
            if cached_offset is not None:
                return self.apply_alignment_to_csv(client_df, counselor_df, cached_offset)
        
        # Look for audio files (common extensions)
        audio_extensions = ['.wav', '.mp3', '.flac', '.aac', '.mp4', '.m4a']
        client_audio_file = None
        counselor_audio_file = None
        
        for ext in audio_extensions:
            # Check for audio files in client directory
            client_audio_candidates = [f for f in os.listdir(client_dir) if f.endswith(ext)]
            if client_audio_candidates:
                client_audio_file = os.path.join(client_dir, client_audio_candidates[0])
                break
        
        for ext in audio_extensions:
            # Check for audio files in counselor directory
            counselor_audio_candidates = [f for f in os.listdir(counselor_dir) if f.endswith(ext)]
            if counselor_audio_candidates:
                counselor_audio_file = os.path.join(counselor_dir, counselor_audio_candidates[0])
                break
        
        # If no audio files found, fall back to simple timestamp-based synchronization
        if not client_audio_file or not counselor_audio_file:
            logger.warning("Audio files not found for alignment. Using simple timestamp-based synchronization.")
            return self._simple_timestamp_sync(client_df, counselor_df)
        
        try:
            # Initialize recognizer
            recognizer = audalign.FingerprintRecognizer()
            recognizer.config.set_accuracy(3)
            
            # Align the audio data
            results = audalign.align_files(
                client_audio_file,
                counselor_audio_file,
                recognizer=recognizer
            )
            
            # Get alignment offset - check different possible result structures
            offset = 0
            if isinstance(results, dict):
                # Try different possible keys that might contain the offset
                if 'shifts' in results:
                    offset = results['shifts'][0] if isinstance(results['shifts'], list) else results['shifts']
                elif 'offset' in results:
                    offset = results['offset']
                elif 'alignment_offset' in results:
                    offset = results['alignment_offset']
                elif 'time_offset' in results:
                    offset = results['time_offset']
                else:
                    # If no offset found, log the result structure for debugging
                    logger.warning(f"No offset found in results. Available keys: {list(results.keys())}")
                    offset = 0
            else:
                # If results is not a dict, it might be a tuple or single value
                logger.warning(f"Unexpected results type: {type(results)}")
                offset = 0
            
            # Save alignment data to cache if requested
            if save_to_cache:
                self.save_alignment_data(client_dir, counselor_dir, offset, results)
            
            # Apply offset to synchronize dataframes
            return self.apply_alignment_to_csv(client_df, counselor_df, offset)
            
        except Exception as e:
            logger.warning(f"Audio alignment failed: {e}. Using simple timestamp-based synchronization.")
            return self._simple_timestamp_sync(client_df, counselor_df)
    
    def _simple_timestamp_sync(self, client_df: pd.DataFrame, counselor_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Simple timestamp-based synchronization when audio alignment is not available."""
        # Truncate to same length
        min_length = min(len(client_df), len(counselor_df))
        client_sync = client_df.iloc[:min_length].reset_index(drop=True)
        counselor_sync = counselor_df.iloc[:min_length].reset_index(drop=True)
        
        # Add synchronized frame indices
        client_sync['frame_index'] = range(min_length)
        counselor_sync['frame_index'] = range(min_length)
        
        return client_sync, counselor_sync
    
    def calculate_emotional_synchrony(self, client_data: Dict, counselor_data: Dict) -> Dict:
        """Calculate emotional synchrony between client and counselor."""
        synchrony_metrics = {}
        
        # Synchronize timelines
        client_sync, counselor_sync = self.synchronize_timelines(client_data, counselor_data)
        
        # 1. Facial Expression Synchrony
        if client_data['task_type'] == 'facial' and counselor_data['task_type'] == 'facial':
            au_cols_client = [col for col in client_sync.columns if 'AU' in col and 'intensity' in col]
            au_cols_counselor = [col for col in counselor_sync.columns if 'AU' in col and 'intensity' in col]
            
            if au_cols_client and au_cols_counselor:
                # Calculate correlation for each AU
                au_correlations = {}
                for au in au_cols_client:
                    if au in au_cols_counselor:
                        try:
                            correlation = client_sync[au].corr(counselor_sync[au])
                            if not pd.isna(correlation):
                                au_correlations[au] = correlation
                        except Exception as e:
                            logger.warning(f"Could not calculate correlation for {au}: {e}")
                
                if au_correlations:
                    synchrony_metrics['facial_expression_correlation'] = np.mean(list(au_correlations.values()))
                    synchrony_metrics['facial_expression_synchrony'] = len([c for c in au_correlations.values() if c > 0.3]) / len(au_correlations)
        
        # 2. Movement Synchrony (for pose data)
        if client_data['task_type'] == 'pose' and counselor_data['task_type'] == 'pose':
            # Calculate movement patterns
            client_movement = self._calculate_movement_patterns(client_sync)
            counselor_movement = self._calculate_movement_patterns(counselor_sync)
            
            if client_movement is not None and counselor_movement is not None:
                try:
                    movement_correlation = client_movement.corr(counselor_movement)
                    if not pd.isna(movement_correlation):
                        synchrony_metrics['movement_synchrony'] = movement_correlation
                except Exception as e:
                    logger.warning(f"Could not calculate movement synchrony: {e}")
        
        # 3. Prosody Synchrony
        if client_data['task_type'] == 'prosody' and counselor_data['task_type'] == 'prosody':
            prosody_features = ['F0', 'energy', 'spectral_centroid']
            
            prosody_correlations = {}
            for feature in prosody_features:
                if feature in client_sync.columns and feature in counselor_sync.columns:
                    try:
                        correlation = client_sync[feature].corr(counselor_sync[feature])
                        if not pd.isna(correlation):
                            prosody_correlations[feature] = correlation
                    except Exception as e:
                        logger.warning(f"Could not calculate prosody correlation for {feature}: {e}")
            
            if prosody_correlations:
                synchrony_metrics['prosody_synchrony'] = np.mean(list(prosody_correlations.values()))
        
        # 4. Overall Synchrony Score
        synchrony_scores = [v for v in synchrony_metrics.values() if isinstance(v, (int, float)) and not pd.isna(v)]
        if synchrony_scores:
            synchrony_metrics['overall_synchrony_score'] = np.mean(synchrony_scores)
        
        return synchrony_metrics
    
    def _calculate_movement_patterns(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Calculate overall movement patterns from pose data."""
        try:
            # Get position columns
            pos_cols = [col for col in df.columns if col.endswith('_x') or col.endswith('_y')]
            if not pos_cols:
                return None
            
            # Calculate frame-to-frame movement
            movement = df[pos_cols].diff().abs().sum(axis=1)
            return movement
            
        except Exception as e:
            logger.warning(f"Could not calculate movement patterns: {e}")
            return None
    
    def analyze_counseling_quality(self, client_dir: str, counselor_dir: str, output_dir: str) -> Dict:
        """Analyze counseling quality for a client-counselor pair."""
        quality_analysis = {
            'client_dir': client_dir,
            'counselor_dir': counselor_dir,
            'synchrony_metrics': {},
            'quality_indicators': {},
            'recommendations': [],
            'individual_analysis': {
                'client': {},
                'counselor': {}
            }
        }
        
        # Load data for all task types
        client_data = {}
        counselor_data = {}
        
        for task_type in self.task_types:
            client_data[task_type] = self.load_analysis_data(client_dir, task_type)
            counselor_data[task_type] = self.load_analysis_data(counselor_dir, task_type)
        
        # Extract individual analysis results
        for task_type in self.task_types:
            if client_data.get(task_type):
                quality_analysis['individual_analysis']['client'][task_type] = self._extract_individual_analysis(
                    client_data[task_type], 'client'
                )
            
            if counselor_data.get(task_type):
                quality_analysis['individual_analysis']['counselor'][task_type] = self._extract_individual_analysis(
                    counselor_data[task_type], 'counselor'
                )
        
        # Calculate synchrony for each modality
        for task_type in self.task_types:
            if (client_data.get(task_type) and counselor_data.get(task_type)):
                synchrony = self.calculate_emotional_synchrony(
                    client_data[task_type], counselor_data[task_type]
                )
                quality_analysis['synchrony_metrics'][task_type] = synchrony
        
        # Calculate overall quality indicators
        quality_analysis['quality_indicators'] = self._calculate_quality_indicators(
            client_data, counselor_data, quality_analysis['synchrony_metrics']
        )
        
        # Generate recommendations
        quality_analysis['recommendations'] = self._generate_recommendations(
            quality_analysis['synchrony_metrics'], quality_analysis['quality_indicators']
        )
        
        # Save analysis
        analysis_path = os.path.join(output_dir, 'counseling_quality_analysis.json')
        with open(analysis_path, 'w') as f:
            json.dump(quality_analysis, f, indent=2)
        
        logger.info(f"Counseling quality analysis saved to: {analysis_path}")
        return quality_analysis
    
    def _extract_individual_analysis(self, data: Dict, role: str) -> Dict:
        """Extract individual analysis results for a specific role (client or counselor)."""
        if not data:
            return {}
        
        df = data['dataframe']
        task_type = data['task_type']
        cues_data = data.get('cues_data', {})
        
        analysis = {
            'role': role,
            'task_type': task_type,
            'total_frames': len(df),
            'basic_stats': {},
            'features': {},
            'nonverbal_cues': cues_data
        }
        
        try:
            # Extract basic statistics for all numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                analysis['basic_stats'] = {
                    'mean': df[numeric_cols].mean().to_dict(),
                    'std': df[numeric_cols].std().to_dict(),
                    'min': df[numeric_cols].min().to_dict(),
                    'max': df[numeric_cols].max().to_dict(),
                    'median': df[numeric_cols].median().to_dict()
                }
            
            # Task-specific feature extraction
            if task_type == 'prosody':
                analysis['features'] = self._extract_prosody_features(df)
            elif task_type == 'facial':
                analysis['features'] = self._extract_facial_features(df)
            elif task_type == 'pose':
                analysis['features'] = self._extract_pose_features(df)
                
        except Exception as e:
            logger.warning(f"Could not extract features for {role} {task_type}: {e}")
            
        return analysis
    
    def _extract_prosody_features(self, df: pd.DataFrame) -> Dict:
        """Extract prosody-specific features."""
        features = {}
        
        # Energy/Intensity features
        intensity_cols = [col for col in df.columns if 'intensity' in col.lower() or 'energy' in col.lower()]
        if intensity_cols:
            features['intensity_stats'] = {
                'mean': df[intensity_cols].mean().to_dict(),
                'variability': df[intensity_cols].std().to_dict()
            }
        
        # Pitch/F0 features
        pitch_cols = [col for col in df.columns if 'f0' in col.lower() or 'pitch' in col.lower()]
        if pitch_cols:
            features['pitch_stats'] = {
                'mean': df[pitch_cols].mean().to_dict(),
                'variability': df[pitch_cols].std().to_dict(),
                'range': (df[pitch_cols].max() - df[pitch_cols].min()).to_dict()
            }
        
        # Spectral features
        spectral_cols = [col for col in df.columns if 'spectral' in col.lower()]
        if spectral_cols:
            features['spectral_stats'] = {
                'mean': df[spectral_cols].mean().to_dict(),
                'variability': df[spectral_cols].std().to_dict()
            }
        
        # Overall prosody activity
        if len(df) > 1:
            features['overall_activity'] = df.select_dtypes(include=[np.number]).diff().abs().mean().mean()
        
        return features
    
    def _extract_facial_features(self, df: pd.DataFrame) -> Dict:
        """Extract facial-specific features."""
        features = {}
        
        # Action Units (AU) features
        au_cols = [col for col in df.columns if 'AU' in col and 'intensity' in col]
        if au_cols:
            features['action_units'] = {
                'mean_intensity': df[au_cols].mean().to_dict(),
                'max_intensity': df[au_cols].max().to_dict(),
                'activation_frequency': (df[au_cols] > 0.5).mean().to_dict()
            }
        
        # Facial landmarks
        x_cols = [col for col in df.columns if col.startswith('x_')]
        y_cols = [col for col in df.columns if col.startswith('y_')]
        
        if x_cols and y_cols:
            # Calculate face center movement
            if len(df) > 1:
                center_x = df[x_cols].mean(axis=1)
                center_y = df[y_cols].mean(axis=1)
                movement = np.sqrt(center_x.diff()**2 + center_y.diff()**2)
                features['face_movement'] = {
                    'mean_movement': movement.mean(),
                    'max_movement': movement.max(),
                    'movement_variability': movement.std()
                }
            
            # Landmark stability
            features['landmark_stability'] = {
                'x_variability': df[x_cols].std().mean(),
                'y_variability': df[y_cols].std().mean()
            }
        
        # Expression confidence scores
        confidence_cols = [col for col in df.columns if 'confidence' in col.lower()]
        if confidence_cols:
            features['expression_confidence'] = {
                'mean_confidence': df[confidence_cols].mean().to_dict(),
                'max_confidence': df[confidence_cols].max().to_dict()
            }
        
        return features
    
    def _extract_pose_features(self, df: pd.DataFrame) -> Dict:
        """Extract pose-specific features."""
        features = {}
        
        # Joint positions
        joint_cols = [col for col in df.columns if col.endswith('_x') or col.endswith('_y')]
        if joint_cols:
            features['joint_positions'] = {
                'mean_positions': df[joint_cols].mean().to_dict(),
                'position_variability': df[joint_cols].std().to_dict()
            }
        
        # Visibility scores
        visibility_cols = [col for col in df.columns if 'visibility' in col]
        if visibility_cols:
            features['visibility_scores'] = {
                'mean_visibility': df[visibility_cols].mean().to_dict(),
                'detection_reliability': (df[visibility_cols] > 0.5).mean().to_dict()
            }
        
        # Movement patterns
        if len(df) > 1:
            pos_cols = [col for col in df.columns if col.endswith('_x') or col.endswith('_y')]
            if pos_cols:
                movement = df[pos_cols].diff().abs().sum(axis=1)
                features['movement_patterns'] = {
                    'mean_movement': movement.mean(),
                    'max_movement': movement.max(),
                    'movement_variability': movement.std()
                }
        
        # Key joint analysis
        key_joints = ['left_shoulder', 'right_shoulder', 'left_wrist', 'right_wrist', 'nose']
        key_joint_data = {}
        for joint in key_joints:
            if f'{joint}_x' in df.columns and f'{joint}_y' in df.columns:
                if len(df) > 1:
                    x_movement = df[f'{joint}_x'].diff().abs().mean()
                    y_movement = df[f'{joint}_y'].diff().abs().mean()
                    key_joint_data[joint] = {
                        'x_movement': x_movement,
                        'y_movement': y_movement,
                        'total_movement': x_movement + y_movement
                    }
        
        if key_joint_data:
            features['key_joints'] = key_joint_data
        
        return features
    
    def _calculate_quality_indicators(self, client_data: Dict, counselor_data: Dict, synchrony_metrics: Dict) -> Dict:
        """Calculate overall counseling quality indicators."""
        indicators = {}
        
        # 1. Engagement Level
        engagement_scores = []
        for task_type in self.task_types:
            if client_data.get(task_type) and counselor_data.get(task_type):
                # Calculate engagement based on activity levels
                client_activity = self._calculate_activity_level(client_data[task_type])
                counselor_activity = self._calculate_activity_level(counselor_data[task_type])
                
                if client_activity and counselor_activity:
                    engagement = (client_activity + counselor_activity) / 2
                    engagement_scores.append(engagement)
        
        if engagement_scores:
            indicators['engagement_level'] = np.mean(engagement_scores)
        
        # 2. Rapport Building
        rapport_indicators = []
        for task_type in self.task_types:
            if task_type in synchrony_metrics:
                synchrony = synchrony_metrics[task_type].get('overall_synchrony_score', 0)
                rapport_indicators.append(synchrony)
        
        if rapport_indicators:
            indicators['rapport_building'] = np.mean(rapport_indicators)
        
        # 3. Emotional Regulation
        emotional_regulation = []
        for role_data in [client_data, counselor_data]:
            for task_type in self.task_types:
                if role_data.get(task_type):
                    regulation = self._calculate_emotional_regulation(role_data[task_type])
                    if regulation is not None:
                        emotional_regulation.append(regulation)
        
        if emotional_regulation:
            indicators['emotional_regulation'] = np.mean(emotional_regulation)
        
        return indicators
    
    def _calculate_activity_level(self, data: Dict) -> Optional[float]:
        """Calculate activity level from data."""
        try:
            df = data['dataframe']
            
            if data['task_type'] == 'facial':
                # Use AU intensities
                au_cols = [col for col in df.columns if 'AU' in col and 'intensity' in col]
                if au_cols:
                    return df[au_cols].mean().mean()
            
            elif data['task_type'] == 'pose':
                # Use movement patterns
                pos_cols = [col for col in df.columns if col.endswith('_x') or col.endswith('_y')]
                if pos_cols:
                    movement = df[pos_cols].diff().abs().sum(axis=1)
                    return movement.mean()
            
            elif data['task_type'] == 'prosody':
                # Use energy and pitch variation
                energy_cols = [col for col in df.columns if 'energy' in col.lower() or 'intensity' in col.lower()]
                if energy_cols:
                    return df[energy_cols].mean().mean()
            
            return None
            
        except Exception as e:
            logger.warning(f"Could not calculate activity level: {e}")
            return None
    
    def _calculate_emotional_regulation(self, data: Dict) -> Optional[float]:
        """Calculate emotional regulation score."""
        try:
            df = data['dataframe']
            
            if data['task_type'] == 'facial':
                # Calculate emotional stability
                au_cols = [col for col in df.columns if 'AU' in col and 'intensity' in col]
                if au_cols:
                    au_variability = df[au_cols].std().mean()
                    return 1 / (au_variability + 1e-8)
            
            elif data['task_type'] == 'pose':
                # Calculate movement stability
                pos_cols = [col for col in df.columns if col.endswith('_x') or col.endswith('_y')]
                if pos_cols:
                    movement_stability = df[pos_cols].diff().abs().std().mean()
                    return 1 / (movement_stability + 1e-8)
            
            elif data['task_type'] == 'prosody':
                # Calculate prosody stability
                prosody_cols = [col for col in df.columns if any(p in col.lower() for p in ['f0', 'energy', 'pitch'])]
                if prosody_cols:
                    prosody_stability = df[prosody_cols].std().mean()
                    return 1 / (prosody_stability + 1e-8)
            
            return None
            
        except Exception as e:
            logger.warning(f"Could not calculate emotional regulation: {e}")
            return None
    
    def _generate_recommendations(self, synchrony_metrics: Dict, quality_indicators: Dict) -> List[str]:
        """Generate counseling quality recommendations."""
        recommendations = []
        
        # Analyze synchrony
        overall_synchrony = 0
        synchrony_count = 0
        for task_type, metrics in synchrony_metrics.items():
            if 'overall_synchrony_score' in metrics:
                overall_synchrony += metrics['overall_synchrony_score']
                synchrony_count += 1
        
        if synchrony_count > 0:
            avg_synchrony = overall_synchrony / synchrony_count
            
            if avg_synchrony < 0.2:
                recommendations.append("Low emotional synchrony detected. Consider building more rapport through active listening and mirroring techniques.")
            elif avg_synchrony < 0.5:
                recommendations.append("Moderate synchrony observed. Focus on improving emotional attunement and response timing.")
            else:
                recommendations.append("Good emotional synchrony maintained. Continue current rapport-building strategies.")
        
        # Analyze engagement
        if 'engagement_level' in quality_indicators:
            engagement = quality_indicators['engagement_level']
            if engagement < 0.3:
                recommendations.append("Low engagement detected. Consider using more interactive techniques and checking client motivation.")
            elif engagement < 0.7:
                recommendations.append("Moderate engagement. Explore ways to increase client involvement and interest.")
            else:
                recommendations.append("Good engagement levels maintained. Continue current engagement strategies.")
        
        # Analyze emotional regulation
        if 'emotional_regulation' in quality_indicators:
            regulation = quality_indicators['emotional_regulation']
            if regulation < 0.5:
                recommendations.append("Emotional regulation challenges observed. Consider implementing grounding techniques and emotional regulation strategies.")
            else:
                recommendations.append("Good emotional regulation maintained. Continue supporting emotional stability.")
        
        return recommendations
    
    def create_consolidated_visualization(self, pair_analysis: Dict, output_dir: str) -> str:
        """Create comprehensive visualization for client-counselor pair analysis."""
        try:
            fig = plt.figure(figsize=(20, 16))
            fig.suptitle('Counseling Quality Analysis: Client-Counselor Pair', fontsize=20, fontweight='bold')
            
            # Plot 1: Synchrony Metrics
            ax1 = plt.subplot(3, 3, 1)
            synchrony_data = pair_analysis['synchrony_metrics']
            if synchrony_data:
                task_types = list(synchrony_data.keys())
                synchrony_scores = []
                for task_type in task_types:
                    score = synchrony_data[task_type].get('overall_synchrony_score', 0)
                    synchrony_scores.append(score)
                
                bars = ax1.bar(task_types, synchrony_scores, color=['skyblue', 'lightcoral', 'lightgreen'])
                ax1.set_title('Emotional Synchrony by Modality')
                ax1.set_ylabel('Synchrony Score')
                ax1.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, score in zip(bars, synchrony_scores):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{score:.3f}', ha='center', va='bottom')
            
            # Plot 2: Quality Indicators
            ax2 = plt.subplot(3, 3, 2)
            quality_data = pair_analysis['quality_indicators']
            if quality_data:
                indicators = list(quality_data.keys())
                values = list(quality_data.values())
                
                bars = ax2.bar(indicators, values, color='gold')
                ax2.set_title('Counseling Quality Indicators')
                ax2.set_ylabel('Score')
                ax2.set_ylim(0, 1)
                
                # Rotate x-axis labels
                plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
                
                # Add value labels
                for bar, value in zip(bars, values):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom')
            
            # Plot 3: Timeline Comparison (placeholder)
            ax3 = plt.subplot(3, 3, 3)
            ax3.text(0.5, 0.5, 'Timeline Comparison\n(Feature to be implemented)', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=self.fontsize)
            ax3.set_title('Behavioral Timeline')
            
            # Plot 4-6: Detailed modality analysis (placeholders)
            for i, task_type in enumerate(['facial', 'pose', 'prosody']):
                ax = plt.subplot(3, 3, 4 + i)
                if task_type in synchrony_data:
                    metrics = synchrony_data[task_type]
                    if metrics:
                        metric_names = list(metrics.keys())
                        metric_values = list(metrics.values())
                        
                        # Filter out non-numeric values
                        numeric_data = [(name, value) for name, value in zip(metric_names, metric_values) 
                                      if isinstance(value, (int, float)) and not pd.isna(value)]
                        
                        if numeric_data:
                            names, values = zip(*numeric_data)
                            bars = ax.bar(range(len(names)), values, color='lightblue')
                            ax.set_title(f'{task_type.title()} Synchrony Details')
                            ax.set_xticks(range(len(names)))
                            ax.set_xticklabels([name.replace('_', '\n')[:15] for name in names], rotation=45, ha='right')
                            ax.set_ylabel('Score')
                else:
                    ax.text(0.5, 0.5, f'No {task_type} data\navailable', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{task_type.title()} Analysis')
            
            # Plot 7: Recommendations
            ax7 = plt.subplot(3, 3, 7)
            recommendations = pair_analysis.get('recommendations', [])
            if recommendations:
                # Create a text box with recommendations
                rec_text = '\n'.join([f'• {rec}' for rec in recommendations[:5]])  # Limit to 5 recommendations
                ax7.text(0.05, 0.95, rec_text, transform=ax7.transAxes, 
                        verticalalignment='top', fontsize=self.fontsize, 
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            else:
                ax7.text(0.5, 0.5, 'No specific\nrecommendations', 
                        ha='center', va='center', transform=ax7.transAxes)
            ax7.set_title('Recommendations')
            ax7.axis('off')
            
            # Plot 8: Summary Statistics
            ax8 = plt.subplot(3, 3, 8)
            summary_text = f"Analysis Summary:\n\n"
            summary_text += f"Client: {os.path.basename(pair_analysis['client_dir'])}\n"
            summary_text += f"Counselor: {os.path.basename(pair_analysis['counselor_dir'])}\n\n"
            
            if quality_data:
                summary_text += f"Overall Quality: {quality_data.get('engagement_level', 0):.3f}\n"
                summary_text += f"Rapport Building: {quality_data.get('rapport_building', 0):.3f}\n"
                summary_text += f"Emotional Regulation: {quality_data.get('emotional_regulation', 0):.3f}\n"
            
            ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, 
                    verticalalignment='top', fontsize=self.fontsize, fontfamily='monospace')
            ax8.set_title('Summary')
            ax8.axis('off')
            
            # Plot 9: Quality Score Distribution
            ax9 = plt.subplot(3, 3, 9)
            if quality_data:
                scores = list(quality_data.values())
                ax9.hist(scores, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
                ax9.set_title('Quality Score Distribution')
                ax9.set_xlabel('Quality Score')
                ax9.set_ylabel('Frequency')
                ax9.axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.3f}')
                ax9.legend()
            else:
                ax9.text(0.5, 0.5, 'No quality data\navailable', 
                        ha='center', va='center', transform=ax9.transAxes)
                ax9.set_title('Quality Distribution')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(output_dir, 'counseling_quality_consolidated.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Consolidated visualization saved to: {plot_path}")
            return plot_path
            
        except Exception as e:
            logger.error(f"Could not create consolidated visualization: {e}")
            return None
    
    def visualize_separate_timelines(self, client_data: Dict, counselor_data: Dict, 
                                   output_dir: str, pair_name: str = "pair") -> str:
        """Create separate visualizations for client and counselor timelines."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Timeline Analysis - {pair_name}', fontsize=self.fontsize, fontweight='bold')
            
            client_df = client_data['dataframe']
            counselor_df = counselor_data['dataframe']
            task_type = client_data['task_type']
            
            # Client timeline (top row)
            ax1 = axes[0, 0]
            ax2 = axes[0, 1]
            
            # Counselor timeline (bottom row)
            ax3 = axes[1, 0]
            ax4 = axes[1, 1]
            
            # Plot client data
            if task_type == 'prosody':
                # Plot prosody features for client
                if 'pitch_mean' in client_df.columns:
                    ax1.plot(client_df.index, client_df['pitch_mean'], label='Pitch Mean', alpha=0.7)
                if 'energy_mean' in client_df.columns:
                    ax1.plot(client_df.index, client_df['energy_mean'], label='Energy Mean', alpha=0.7)
                ax1.set_title('Client Prosody Features')
                ax1.set_xlabel('Time (frames)')
                ax1.set_ylabel('Value')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Plot counselor data
                if 'pitch_mean' in counselor_df.columns:
                    ax3.plot(counselor_df.index, counselor_df['pitch_mean'], label='Pitch Mean', alpha=0.7)
                if 'energy_mean' in counselor_df.columns:
                    ax3.plot(counselor_df.index, counselor_df['energy_mean'], label='Energy Mean', alpha=0.7)
                ax3.set_title('Counselor Prosody Features')
                ax3.set_xlabel('Time (frames)')
                ax3.set_ylabel('Value')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
            elif task_type == 'facial':
                # Plot facial expression confidence for client
                expression_cols = [col for col in client_df.columns if 'confidence' in col.lower()]
                for col in expression_cols[:3]:  # Limit to first 3 expressions
                    ax1.plot(client_df.index, client_df[col], label=col.replace('_confidence', ''), alpha=0.7)
                ax1.set_title('Client Facial Expressions')
                ax1.set_xlabel('Time (frames)')
                ax1.set_ylabel('Confidence')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Plot counselor facial expressions
                for col in expression_cols[:3]:
                    if col in counselor_df.columns:
                        ax3.plot(counselor_df.index, counselor_df[col], label=col.replace('_confidence', ''), alpha=0.7)
                ax3.set_title('Counselor Facial Expressions')
                ax3.set_xlabel('Time (frames)')
                ax3.set_ylabel('Confidence')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
            elif task_type == 'pose':
                # Plot pose landmarks for client
                landmark_cols = [col for col in client_df.columns if 'landmark' in col.lower()][:5]
                for col in landmark_cols:
                    ax1.plot(client_df.index, client_df[col], label=col, alpha=0.7)
                ax1.set_title('Client Pose Landmarks')
                ax1.set_xlabel('Time (frames)')
                ax1.set_ylabel('Position')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Plot counselor pose landmarks
                for col in landmark_cols:
                    if col in counselor_df.columns:
                        ax3.plot(counselor_df.index, counselor_df[col], label=col, alpha=0.7)
                ax3.set_title('Counselor Pose Landmarks')
                ax3.set_xlabel('Time (frames)')
                ax3.set_ylabel('Position')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # Activity levels comparison
            client_activity = client_df.iloc[:, 1:].std(axis=1)  # Skip first column (usually index)
            counselor_activity = counselor_df.iloc[:, 1:].std(axis=1)
            
            ax2.plot(client_activity.index, client_activity, label='Client Activity', alpha=0.7)
            ax2.set_title('Client Activity Level')
            ax2.set_xlabel('Time (frames)')
            ax2.set_ylabel('Activity (std)')
            ax2.grid(True, alpha=0.3)
            
            ax4.plot(counselor_activity.index, counselor_activity, label='Counselor Activity', alpha=0.7)
            ax4.set_title('Counselor Activity Level')
            ax4.set_xlabel('Time (frames)')
            ax4.set_ylabel('Activity (std)')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(output_dir, f'{pair_name}_separate_timelines_{task_type}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Separate timeline visualization saved to: {plot_path}")
            return plot_path
            
        except Exception as e:
            logger.error(f"Could not create separate timeline visualization: {e}")
            return None
    
    def visualize_alignment_comparison(self, client_data: Dict, counselor_data: Dict, 
                                     output_dir: str, pair_name: str = "pair") -> str:
        """Create visualization comparing aligned vs unaligned data."""
        try:
            # Get original data
            client_df_orig = client_data['dataframe']
            counselor_df_orig = counselor_data['dataframe']
            
            # Get aligned data
            client_df_aligned, counselor_df_aligned = self.synchronize_timelines(client_data, counselor_data)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Alignment Comparison - {pair_name}', fontsize=self.fontsize, fontweight='bold')
            
            # Choose a representative column for comparison
            task_type = client_data['task_type']
            if task_type == 'prosody':
                col_name = 'pitch_mean' if 'pitch_mean' in client_df_orig.columns else client_df_orig.columns[1]
            elif task_type == 'facial':
                col_name = [col for col in client_df_orig.columns if 'confidence' in col.lower()][0] if [col for col in client_df_orig.columns if 'confidence' in col.lower()] else client_df_orig.columns[1]
            elif task_type == 'pose':
                col_name = [col for col in client_df_orig.columns if 'landmark' in col.lower()][0] if [col for col in client_df_orig.columns if 'landmark' in col.lower()] else client_df_orig.columns[1]
            else:
                col_name = client_df_orig.columns[1]
            
            # Plot original data
            ax1 = axes[0, 0]
            ax1.plot(client_df_orig.index, client_df_orig[col_name], label='Client', alpha=0.7)
            ax1.plot(counselor_df_orig.index, counselor_df_orig[col_name], label='Counselor', alpha=0.7)
            ax1.set_title('Original (Unaligned) Data')
            ax1.set_xlabel('Time (frames)')
            ax1.set_ylabel(col_name)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot aligned data
            ax2 = axes[0, 1]
            ax2.plot(client_df_aligned.index, client_df_aligned[col_name], label='Client', alpha=0.7)
            ax2.plot(counselor_df_aligned.index, counselor_df_aligned[col_name], label='Counselor', alpha=0.7)
            ax2.set_title('Aligned Data')
            ax2.set_xlabel('Time (frames)')
            ax2.set_ylabel(col_name)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Cross-correlation before alignment
            ax3 = axes[1, 0]
            if len(client_df_orig) > 0 and len(counselor_df_orig) > 0:
                min_len = min(len(client_df_orig), len(counselor_df_orig))
                corr_orig = np.correlate(client_df_orig[col_name].iloc[:min_len], 
                                       counselor_df_orig[col_name].iloc[:min_len], mode='full')
                ax3.plot(corr_orig)
                ax3.set_title('Cross-correlation (Original)')
                ax3.set_xlabel('Lag')
                ax3.set_ylabel('Correlation')
                ax3.grid(True, alpha=0.3)
            
            # Cross-correlation after alignment
            ax4 = axes[1, 1]
            if len(client_df_aligned) > 0 and len(counselor_df_aligned) > 0:
                corr_aligned = np.correlate(client_df_aligned[col_name], 
                                          counselor_df_aligned[col_name], mode='full')
                ax4.plot(corr_aligned)
                ax4.set_title('Cross-correlation (Aligned)')
                ax4.set_xlabel('Lag')
                ax4.set_ylabel('Correlation')
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(output_dir, f'{pair_name}_alignment_comparison_{task_type}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Alignment comparison visualization saved to: {plot_path}")
            return plot_path
            
        except Exception as e:
            logger.error(f"Could not create alignment comparison visualization: {e}")
            return None

class CounselingAssessmentVisualizer:
    """Creates comprehensive counseling assessment visualizations from analysis results."""
    
    def __init__(self, analysis_dir: str):
        self.analysis_dir = analysis_dir
        self.color_palette = {
            'excellent': '#2E8B57',    # Sea green
            'good': '#32CD32',         # Lime green  
            'average': '#FFD700',      # Gold
            'poor': '#FF6347',         # Tomato
            'very_poor': '#DC143C'     # Crimson
        }
        self.fontsize = 12
        
    def load_analysis_results(self) -> Dict[str, Dict]:
        """Load all counseling quality analysis JSON files."""
        results = {}
        
        # Search for counseling_quality_analysis.json files
        for root, dirs, files in os.walk(self.analysis_dir):
            if 'counseling_quality_analysis.json' in files:
                json_path = os.path.join(root, 'counseling_quality_analysis.json')
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    
                    # Extract session name from directory
                    session_name = os.path.basename(root)
                    results[session_name] = data
                    logger.info(f"Loaded analysis for session: {session_name}")
                    
                except Exception as e:
                    logger.warning(f"Could not load {json_path}: {e}")
        
        logger.info(f"Total loaded sessions: {len(results)}")
        return results
    
    def _normalize_score(self, score: float, mean: float = 0.5, std: float = 0.2) -> float:
        """Normalize score using statistical scaling to make values more meaningful."""
        if score == 0:
            return 0
        
        # Apply z-score normalization then scale to 0-100
        z_score = (score - mean) / std
        # Use sigmoid function to map to 0-100 range
        normalized = 100 / (1 + np.exp(-z_score))
        return max(0, min(100, normalized))
    
    def calculate_comprehensive_scores(self, data: Dict) -> Dict:
        """Calculate comprehensive assessment scores from analysis data."""
        scores = {}
        
        # 1. Synchrony Skills (0-100)
        synchrony_metrics = data.get('synchrony_metrics', {})
        synchrony_scores = []
        
        for task_type, metrics in synchrony_metrics.items():
            if 'overall_synchrony_score' in metrics:
                synchrony_scores.append(metrics['overall_synchrony_score'])
        
        raw_synchrony = np.mean(synchrony_scores) if synchrony_scores else 0
        scores['synchrony_skills'] = self._normalize_score(raw_synchrony, mean=0.3, std=0.2) if raw_synchrony > 0 else 0
        
        # 2. Engagement Management (0-100)  
        quality_indicators = data.get('quality_indicators', {})
        engagement_level = quality_indicators.get('engagement_level', 0)
        rapport_building = quality_indicators.get('rapport_building', 0)
        
        raw_engagement = (engagement_level + rapport_building) / 2
        scores['engagement_management'] = self._normalize_score(raw_engagement, mean=0.4, std=0.25) if raw_engagement > 0 else 0
        
        # 3. Emotional Regulation (0-100, inverted - lower is better)
        emotional_regulation = quality_indicators.get('emotional_regulation', 1)
        # Apply inverse normalization (higher regulation value means better stability)
        if emotional_regulation > 0:
            # Cap extremely high values and normalize
            capped_regulation = min(emotional_regulation, 10)  # Cap at 10 for normalization
            scores['emotional_regulation'] = self._normalize_score(capped_regulation, mean=2, std=1.5)
        else:
            scores['emotional_regulation'] = 0
        
        # 4. Nonverbal Communication (0-100)
        nonverbal_score = self._calculate_nonverbal_communication_score(data)
        scores['nonverbal_communication'] = nonverbal_score
        
        # 5. Overall Performance
        score_weights = {
            'synchrony_skills': 0.25,
            'engagement_management': 0.25, 
            'emotional_regulation': 0.25,
            'nonverbal_communication': 0.25
        }
        
        scores['overall_performance'] = sum(
            scores[key] * weight for key, weight in score_weights.items()
        )
        
        # Add detailed breakdown with normalization
        scores['detailed_breakdown'] = {
            'facial_synchrony': self._normalize_score(synchrony_metrics.get('facial', {}).get('overall_synchrony_score', 0), mean=0.3, std=0.2),
            'pose_synchrony': self._normalize_score(synchrony_metrics.get('pose', {}).get('overall_synchrony_score', 0), mean=0.3, std=0.2),
            'prosody_synchrony': self._normalize_score(synchrony_metrics.get('prosody', {}).get('overall_synchrony_score', 0), mean=0.3, std=0.2),
            'engagement_level': self._normalize_score(engagement_level, mean=0.4, std=0.25),
            'rapport_building': self._normalize_score(rapport_building, mean=0.3, std=0.2)
        }
        
        return scores
    
    def _calculate_nonverbal_communication_score(self, data: Dict) -> float:
        """Calculate nonverbal communication effectiveness score."""
        individual_analysis = data.get('individual_analysis', {})
        counselor_data = individual_analysis.get('counselor', {})
        
        scores = []
        
        # Facial expression utilization
        facial_data = counselor_data.get('facial', {})
        if facial_data:
            features = facial_data.get('features', {})
            if 'action_units' in features:
                au_diversity = len(features['action_units'].get('mean_intensity', {}))
                au_activation = np.mean(list(features['action_units'].get('activation_frequency', {}).values()))
                # Better scaling for facial score
                diversity_score = self._normalize_score(au_diversity / 20, mean=0.5, std=0.3)  # Normalize AU diversity
                activation_score = self._normalize_score(au_activation, mean=0.3, std=0.2)  # Normalize activation
                facial_score = (diversity_score + activation_score) / 2
                scores.append(facial_score)
        
        # Gesture utilization
        pose_data = counselor_data.get('pose', {})
        if pose_data:
            features = pose_data.get('features', {})
            if 'movement_patterns' in features:
                movement_value = features['movement_patterns'].get('mean_movement', 0)
                movement_score = self._normalize_score(movement_value, mean=0.1, std=0.05)  # Better scaling for movement
                scores.append(movement_score)
        
        # Vocal utilization
        prosody_data = counselor_data.get('prosody', {})
        if prosody_data:
            features = prosody_data.get('features', {})
            if 'overall_activity' in features:
                vocal_value = features['overall_activity']
                vocal_score = self._normalize_score(vocal_value, mean=0.2, std=0.1)  # Better scaling for vocal activity
                scores.append(vocal_score)
        
        return np.mean(scores) if scores else 50  # Default to middle score
    
    def get_performance_level(self, score: float) -> Tuple[str, str]:
        """Get performance level and color for a given score."""
        if score >= 90:
            return "Excellent", self.color_palette['excellent']
        elif score >= 75:
            return "Good", self.color_palette['good']
        elif score >= 60:
            return "Average", self.color_palette['average']
        elif score >= 40:
            return "Poor", self.color_palette['poor']
        else:
            return "Very Poor", self.color_palette['very_poor']
    
    def create_assessment_dashboard(self, session_name: str, data: Dict, output_dir: str) -> str:
        """Create a comprehensive assessment dashboard for a single session."""
        scores = self.calculate_comprehensive_scores(data)
        
        # Create figure with custom layout - more space between rows
        fig = plt.figure(figsize=(20, 26))
        gs = GridSpec(6, 4, figure=fig, height_ratios=[1.2, 1.5, 1.5, 1.5, 1.2, 1], hspace=0.4, wspace=0.3)
        
        # Set font
        plt.rcParams['font.family'] = 'DejaVu Sans'
        
        # Main title
        fig.suptitle(f'Counseling Assessment Report - {session_name}', fontsize=24, fontweight='bold', y=0.88)
        
        # 1. Overall Performance Score (Top)
        ax_overall = fig.add_subplot(gs[0, :])
        self._create_overall_score_display(ax_overall, scores)
        
        # 2. Radar Chart (Core Competencies)
        ax_radar = fig.add_subplot(gs[1, 0])
        self._create_radar_chart(ax_radar, scores)
        
        # 3. Detailed Breakdown Bar Chart
        ax_breakdown = fig.add_subplot(gs[1, 1])
        self._create_detailed_breakdown_chart(ax_breakdown, scores)
        
        # 4. Synchrony Analysis
        ax_sync = fig.add_subplot(gs[1, 2])
        self._create_synchrony_analysis(ax_sync, data)
        
        # 5. Client vs Counselor Comparison
        ax_comparison = fig.add_subplot(gs[1, 3])
        self._create_client_counselor_comparison(ax_comparison, data)
        
        # 6. Facial Expression Analysis
        ax_facial = fig.add_subplot(gs[2, :2])
        self._create_facial_analysis(ax_facial, data)
        
        # 7. Face Emotion Flow Chart
        ax_emotion = fig.add_subplot(gs[2, 2:])
        self._create_face_emotion_flowchart(ax_emotion, data)
        
        # 8. Pose/Gesture Analysis
        ax_pose = fig.add_subplot(gs[3, :2])
        self._create_pose_analysis(ax_pose, data)
        
        # 9. Engagement Timeline
        ax_timeline = fig.add_subplot(gs[3, 2:])
        self._create_engagement_timeline(ax_timeline, data)
        
        # 10. Detailed Metrics Table
        ax_table = fig.add_subplot(gs[4, :])
        self._create_detailed_metrics_table(ax_table, scores, data)
        
        # Save the dashboard
        output_path = os.path.join(output_dir, f'{session_name}_assessment_dashboard.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Assessment dashboard saved to: {output_path}")
        return output_path
    
    def _create_overall_score_display(self, ax, scores):
        """Create overall performance score display."""
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.8, 0.8)
        
        overall_score = scores['overall_performance']
        level, color = self.get_performance_level(overall_score)
        
        # Progress bar
        bar_width = overall_score
        ax.barh(0, bar_width, height=0.4, color=color, alpha=0.9)
        ax.barh(0, 100, height=0.4, color='lightgray', alpha=0.3)
        
        # Score text
        ax.text(50, 0, f'{overall_score:.1f}/100', ha='center', va='center', 
                fontsize=22, fontweight='bold', color='white')
        
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(['Very Poor', 'Poor', 'Average', 'Good', 'Excellent'], fontsize=self.fontsize)
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    def _create_radar_chart(self, ax, scores):
        """Create radar chart for core competencies."""
        categories = ['Synchrony\nSkills', 'Engagement\nManagement', 'Emotional\nRegulation', 'Nonverbal\nCommunication']
        values = [
            scores['synchrony_skills'],
            scores['engagement_management'], 
            scores['emotional_regulation'],
            scores['nonverbal_communication']
        ]
        
        # Number of variables
        N = len(categories)
        
        # Angles for each category
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Values for plotting
        values += values[:1]  # Complete the circle
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, color='#1f77b4')
        ax.fill(angles, values, alpha=0.25, color='#1f77b4')
        
        # Add category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=self.fontsize)
        
        # Set y-axis limits and labels
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=self.fontsize)
        ax.grid(True)
        
        ax.set_title('Core Competency Analysis', fontsize=self.fontsize, fontweight='bold', pad=20)
    
    def _create_detailed_breakdown_chart(self, ax, scores):
        """Create detailed breakdown bar chart."""
        breakdown = scores['detailed_breakdown']
        
        categories = ['Facial Synchrony', 'Pose Synchrony', 'Voice Synchrony', 'Engagement', 'Rapport Building']
        values = [
            breakdown['facial_synchrony'],
            breakdown['pose_synchrony'],
            breakdown['prosody_synchrony'],
            breakdown['engagement_level'],
            breakdown['rapport_building']
        ]
        
        colors = [self.get_performance_level(v)[1] for v in values]
        
        bars = ax.barh(categories, values, color=colors, alpha=0.8)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(value + 1, i, f'{value:.1f}', va='center', fontsize=self.fontsize, fontweight='bold')
        
        ax.set_xlim(0, 100)
        ax.set_xlabel('Score', fontsize=self.fontsize)
        ax.set_title('Detailed Metrics Analysis', fontsize=self.fontsize, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    def _create_synchrony_analysis(self, ax, data):
        """Create synchrony analysis with counselor pose/facial frame images."""
        import matplotlib.image as mpimg
        import glob
        
        # Get counselor directory from data (should be _T directory)
        counselor_dir = data.get('counselor_dir', '')
        if not counselor_dir:
            ax.text(0.5, 0.5, 'No counselor data\navailable', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Counselor Frame Analysis', fontsize=12, fontweight='bold')
            return
        
        # Ensure we're looking at the therapist directory (_T)
        if counselor_dir.endswith('_C'):
            # Replace _C with _T to get therapist directory
            therapist_dir = counselor_dir.replace('_C', '_T')
        else:
            therapist_dir = counselor_dir
        
        # Look for pose or facial frame images in therapist directory
        pose_frame_path = os.path.join(therapist_dir, 'pose_landmarks_frame.png')
        facial_frame_patterns = [
            os.path.join(therapist_dir, 'facial_landmarks_frame_*.png'),
            os.path.join(therapist_dir, 'facial_landmarks_frame.png')
        ]
        
        image_path = None
        image_type = None
        
        # Check for pose frame first
        if os.path.exists(pose_frame_path):
            image_path = pose_frame_path
            image_type = 'Pose'
        else:
            # Check for facial frame patterns
            for pattern in facial_frame_patterns:
                facial_files = glob.glob(pattern)
                if facial_files:
                    image_path = facial_files[0]  # Take the first match
                    image_type = 'Facial'
                    break
        
        if image_path and os.path.exists(image_path):
            try:
                # Load and display the image
                img = mpimg.imread(image_path)
                ax.imshow(img)
                ax.set_title(f'Counselor {image_type} Landmarks', fontsize=12, fontweight='bold')
                ax.axis('off')
                
                # Add image info text
                ax.text(0.02, 0.98, f'{image_type} Frame Analysis', 
                       transform=ax.transAxes, fontsize=10, fontweight='bold',
                       verticalalignment='top', color='white',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
                
            except Exception as e:
                logger.warning(f"Could not load image {image_path}: {e}")
                ax.text(0.5, 0.5, f'Could not load\n{image_type.lower()} frame image', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title('Counselor Frame Analysis', fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No pose or facial\nframe images available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Counselor Frame Analysis', fontsize=12, fontweight='bold')
    
    def _create_client_counselor_comparison(self, ax, data):
        """Create client vs counselor comparison chart."""
        individual_analysis = data.get('individual_analysis', {})
        client_data = individual_analysis.get('client', {})
        counselor_data = individual_analysis.get('counselor', {})
        
        # Extract activity levels
        activities = {
            'Client': [],
            'Counselor': []
        }
        
        categories = []
        
        # Facial activity
        if 'facial' in client_data and 'facial' in counselor_data:
            client_facial = client_data['facial'].get('features', {}).get('face_movement', {})
            counselor_facial = counselor_data['facial'].get('features', {}).get('face_movement', {})
            
            if client_facial and counselor_facial:
                activities['Client'].append(client_facial.get('mean_movement', 0))
                activities['Counselor'].append(counselor_facial.get('mean_movement', 0))
                categories.append('Facial Activity')
        
        # Pose activity
        if 'pose' in client_data and 'pose' in counselor_data:
            client_pose = client_data['pose'].get('features', {}).get('movement_patterns', {})
            counselor_pose = counselor_data['pose'].get('features', {}).get('movement_patterns', {})
            
            if client_pose and counselor_pose:
                activities['Client'].append(client_pose.get('mean_movement', 0))
                activities['Counselor'].append(counselor_pose.get('mean_movement', 0))
                categories.append('Pose Activity')
        
        # Prosody activity
        if 'prosody' in client_data and 'prosody' in counselor_data:
            client_prosody = client_data['prosody'].get('features', {})
            counselor_prosody = counselor_data['prosody'].get('features', {})
            
            if client_prosody and counselor_prosody:
                activities['Client'].append(client_prosody.get('overall_activity', 0))
                activities['Counselor'].append(counselor_prosody.get('overall_activity', 0))
                categories.append('Voice Activity')
        
        if not categories:
            ax.text(0.5, 0.5, 'No comparison data\navailable', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=self.fontsize)
            ax.set_title('Counselor vs Client Comparison', fontsize=self.fontsize, fontweight='bold')
            return
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, activities['Client'], width, label='Client', color='#ff7f0e', alpha=0.8)
        bars2 = ax.bar(x + width/2, activities['Counselor'], width, label='Counselor', color='#1f77b4', alpha=0.8)
        
        ax.set_xlabel('Activity Type', fontsize=self.fontsize)
        ax.set_ylabel('Activity Level', fontsize=self.fontsize)
        ax.set_title('Counselor vs Client Comparison', fontsize=self.fontsize, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    def _create_facial_analysis(self, ax, data):
        """Create facial expression analysis."""
        counselor_data = data.get('individual_analysis', {}).get('counselor', {}).get('facial', {})
        
        if not counselor_data:
            ax.text(0.5, 0.5, 'No facial data available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=self.fontsize)
            ax.set_title('Facial Expression Analysis', fontsize=self.fontsize, fontweight='bold')
            return
        
        features = counselor_data.get('features', {})
        au_data = features.get('action_units', {})
        
        if not au_data:
            ax.text(0.5, 0.5, 'No action unit data available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=self.fontsize)
            ax.set_title('Facial Expression Analysis', fontsize=self.fontsize, fontweight='bold')
            return
        
        # Plot AU activation frequency
        au_freq = au_data.get('activation_frequency', {})
        if au_freq:
            aus = list(au_freq.keys())[:10]  # Top 10 AUs
            frequencies = [au_freq[au] for au in aus]
            
            bars = ax.bar(aus, frequencies, color='lightblue', alpha=0.8)
            
            # Add value labels
            for bar, freq in zip(bars, frequencies):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{freq:.2f}', ha='center', va='bottom', fontsize=self.fontsize)
            
            ax.set_ylabel('Activation Frequency', fontsize=self.fontsize)
            ax.set_title('Action Unit Activation Analysis', fontsize=self.fontsize, fontweight='bold')
            ax.tick_params(axis='x', rotation=30)
            ax.grid(axis='y', alpha=0.3)
    
    def _create_face_emotion_flowchart(self, ax, data):
        """Create face emotion flow chart using frame-level AU intensities.
        
        • Loads therapist (_T) facial_landmarks.csv directly.
        • Uniformly samples ≤100 frames across the session (mapped to 0–50 min).
        • For every sampled frame, computes emotion scores from AU intensities, chooses
          the dominant emotion, and plots a scatter/line whose colour encodes that
          emotion. The y-axis shows the dominant emotion intensity (0–1)."""
        import os
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        
        fontsize = getattr(self, "fontsize", 12)

        # Locate therapist directory and CSV
        counselor_dir = data.get('counselor_dir', '')
        if not counselor_dir:
            ax.text(0.5, 0.5, 'No therapist directory found', ha='center', va='center',
                    transform=ax.transAxes, fontsize=fontsize)
            ax.set_title('Face Emotion Flow Analysis', fontsize=fontsize, fontweight='bold')
            return
        therapist_dir = counselor_dir.replace('_C', '_T') if counselor_dir.endswith('_C') else counselor_dir
        csv_path = os.path.join(therapist_dir, 'facial_landmarks.csv')
        if not os.path.exists(csv_path):
            ax.text(0.5, 0.5, 'facial_landmarks.csv not found', ha='center', va='center',
                    transform=ax.transAxes, fontsize=fontsize)
            ax.set_title('Face Emotion Flow Analysis', fontsize=fontsize, fontweight='bold')
            return

        # Load AU intensities per frame
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            ax.text(0.5, 0.5, f'CSV load error: {exc}', ha='center', va='center',
                    transform=ax.transAxes, fontsize=fontsize)
            ax.set_title('Face Emotion Flow Analysis', fontsize=fontsize, fontweight='bold')
            return

        total_frames = len(df)
        if total_frames == 0:
            ax.text(0.5, 0.5, 'No frame data available', ha='center', va='center',
                    transform=ax.transAxes, fontsize=fontsize)
            ax.set_title('Face Emotion Flow Analysis', fontsize=fontsize, fontweight='bold')
            return

        # Emotion → AU mapping
        emotion_aus = {
            'Joy': ['AU06', 'AU12', 'AU25'],
            'Sadness': ['AU01', 'AU04', 'AU15'],
            'Surprise': ['AU01', 'AU02', 'AU05', 'AU26'],
            'Fear': ['AU01', 'AU02', 'AU04', 'AU20'],
            'Anger': ['AU04', 'AU07', 'AU23'],
            'Disgust': ['AU09', 'AU10', 'AU16'],
            'Contempt': ['AU12', 'AU14']
        }
        # Colour palette
        emotion_colors = {
            'Joy': '#FFD700',      # gold
            'Sadness': '#4682B4',  # steel blue
            'Surprise': '#FF6347', # tomato
            'Fear': '#9370DB',     # medium purple
            'Anger': '#DC143C',    # crimson
            'Disgust': '#32CD32',  # lime green
            'Contempt': '#FF69B4'  # hot pink
        }

        # Uniformly sample ≤100 frames
        num_samples = min(100, total_frames)
        sample_idx = np.linspace(0, total_frames - 1, num_samples, dtype=int)
        times_min = sample_idx / (total_frames - 1) * 50  # map to 0–50 minutes

        dom_emotions = []
        dom_scores = []
        for idx in sample_idx:
            frame_scores = {}
            for emotion, aus in emotion_aus.items():
                au_vals = []
                for au in aus:
                    col = f"{au}_intensity"
                    if col in df.columns:
                        au_vals.append(df[col].iloc[idx])
                if au_vals:
                    # Normalize intensity (OpenFace AU intensities are 0–5)
                    frame_scores[emotion] = np.clip(np.mean(au_vals) / 5.0, 0.0, 1.0)
                else:
                    frame_scores[emotion] = 0.0
            # Pick dominant emotion for this frame
            dominant = max(frame_scores.items(), key=lambda x: x[1])
            dom_emotions.append(dominant[0])
            dom_scores.append(dominant[1])

        # Plot emotion trajectory for counselor
        for i in range(len(times_min) - 1):
            colour = emotion_colors.get(dom_emotions[i], '#808080')
            ax.plot(times_min[i:i + 2], dom_scores[i:i + 2], color=colour, linewidth=2)
            ax.scatter(times_min[i], dom_scores[i], color=colour, s=40, edgecolors='black', linewidth=0.5)
        # Last point
        ax.scatter(times_min[-1], dom_scores[-1], color=emotion_colors.get(dom_emotions[-1], '#808080'),
                   s=40, edgecolors='black', linewidth=0.5)

        # --- Add client emotion trajectory (shifted upward) ---
        client_dir = data.get('client_dir', '')
        if client_dir:
            client_csv = os.path.join(client_dir, 'facial_landmarks.csv')
            if os.path.exists(client_csv):
                try:
                    df_c = pd.read_csv(client_csv)
                    total_frames_c = len(df_c)
                    if total_frames_c > 0:
                        num_samples_c = min(100, total_frames_c)
                        idx_c = np.linspace(0, total_frames_c - 1, num_samples_c, dtype=int)
                        t_c = idx_c / (total_frames_c - 1) * 50  # minutes
                        d_emotions_c = []
                        d_scores_c = []
                        for idx in idx_c:
                            fs = {}
                            for emo, aus in emotion_aus.items():
                                vals = []
                                for au in aus:
                                    col = f"{au}_intensity"
                                    if col in df_c.columns:
                                        vals.append(df_c[col].iloc[idx])
                                fs[emo] = np.clip(np.mean(vals) / 5.0, 0.0, 1.0) if vals else 0.0
                            dom_c = max(fs.items(), key=lambda x: x[1])
                            d_emotions_c.append(dom_c[0])
                            d_scores_c.append(dom_c[1] + 0.55)  # shift up for visibility
                        for i in range(len(t_c) - 1):
                            col_c = emotion_colors.get(d_emotions_c[i], '#808080')
                            ax.plot(t_c[i:i + 2], d_scores_c[i:i + 2], color=col_c, linewidth=2, linestyle='--', alpha=0.8)
                            ax.scatter(t_c[i], d_scores_c[i], color=col_c, s=40, edgecolors='black', linewidth=0.5)
                        ax.scatter(t_c[-1], d_scores_c[-1], color=emotion_colors.get(d_emotions_c[-1], '#808080'),
                                   s=40, edgecolors='black', linewidth=0.5)
                except Exception as e:
                    logger.warning(f"Client emotion flow error: {e}")

        # Legend (emotion colours)
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=emo,
                                      markerfacecolor=clr, markeredgecolor='black', markersize=6)
                          for emo, clr in emotion_colors.items()]
        ax.legend(handles=legend_handles, fontsize=8, loc='upper right', ncol=2)

        # Annotate dominant overall emotion for counselor
        overall = max(set(dom_emotions), key=dom_emotions.count)
        ax.text(0.02, 0.95, f'Dominant overall (T): {overall}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        ax.set_xlabel('Time (minutes)', fontsize=fontsize)
        ax.set_ylabel('Dominant emotion intensity', fontsize=fontsize)
        ax.set_title('Face Emotion Flow Analysis', fontsize=fontsize, fontweight='bold')
        ax.set_ylim(0, 1.6)
        ax.grid(True, alpha=0.3)
    
    def _create_pose_analysis(self, ax, data):
        """Create pose/gesture analysis."""
        counselor_data = data.get('individual_analysis', {}).get('counselor', {}).get('pose', {})
        
        if not counselor_data:
            ax.text(0.5, 0.5, 'No pose data available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=self.fontsize)
            ax.set_title('Pose/Gesture Analysis', fontsize=self.fontsize, fontweight='bold')
            return
        
        features = counselor_data.get('features', {})
        key_joints = features.get('key_joints', {})
        
        if not key_joints:
            ax.text(0.5, 0.5, 'No key joint data available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=self.fontsize)
            ax.set_title('Pose/Gesture Analysis', fontsize=self.fontsize, fontweight='bold')
            return
        
        joints = list(key_joints.keys())
        movements = [key_joints[joint].get('total_movement', 0) for joint in joints]
        
        bars = ax.bar(joints, movements, color='orange', alpha=0.8)
        
        # Add value labels
        for bar, movement in zip(bars, movements):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(movements) * 0.01,
                   f'{movement:.3f}', ha='center', va='bottom', fontsize=self.fontsize)
        
        ax.set_ylabel('Movement Level', fontsize=self.fontsize)
        ax.set_title('Key Joint Movement Analysis', fontsize=self.fontsize, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    def _create_engagement_timeline(self, ax, data):
        """Create engagement timeline analysis."""
        # Extract engagement-related data from both client and counselor
        client_data = data.get('individual_analysis', {}).get('client', {})
        counselor_data = data.get('individual_analysis', {}).get('counselor', {})
        
        # Simulate engagement timeline based on available data
        engagement_data = []
        time_points = []
        
        # Extract activity levels from different modalities
        for role, role_data in [('Client', client_data), ('Counselor', counselor_data)]:
            role_engagement = []
            
            # Get facial engagement (AU activity)
            facial_data = role_data.get('facial', {})
            if facial_data and 'features' in facial_data:
                face_features = facial_data['features']
                if 'action_units' in face_features:
                    au_activity = face_features['action_units'].get('activation_frequency', {})
                    if au_activity:
                        facial_engagement = np.mean(list(au_activity.values()))
                        role_engagement.append(facial_engagement)
            
            # Get pose engagement (movement activity)
            pose_data = role_data.get('pose', {})
            if pose_data and 'features' in pose_data:
                pose_features = pose_data['features']
                if 'movement_patterns' in pose_features:
                    movement_activity = pose_features['movement_patterns'].get('mean_movement', 0)
                    # Normalize movement for engagement scale
                    movement_engagement = min(1.0, movement_activity * 2)
                    role_engagement.append(movement_engagement)
            
            # Get prosody engagement (voice activity)
            prosody_data = role_data.get('prosody', {})
            if prosody_data and 'features' in prosody_data:
                prosody_features = prosody_data['features']
                if 'overall_activity' in prosody_features:
                    voice_activity = prosody_features['overall_activity']
                    # Normalize voice activity for engagement scale
                    voice_engagement = min(1.0, voice_activity * 5)
                    role_engagement.append(voice_engagement)
            
            # Calculate average engagement for this role
            if role_engagement:
                avg_engagement = np.mean(role_engagement)
                engagement_data.append((role, avg_engagement))
        
        if not engagement_data:
            ax.text(0.5, 0.5, 'No engagement data available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=self.fontsize)
            ax.set_title('Engagement Timeline Analysis', fontsize=self.fontsize, fontweight='bold')
            return
        
        # Create simulated timeline (10-minute session with 5 time points)
        time_points = np.linspace(0, 10, 6)  # 0, 2, 4, 6, 8, 10 minutes
        
        # Create engagement curves for each role
        colors = ['#ff7f0e', '#1f77b4']
        
        for i, (role, base_engagement) in enumerate(engagement_data):
            # Simulate engagement variation over time
            # Add some realistic fluctuation
            np.random.seed(42 + i)  # For reproducible results
            variation = np.random.normal(0, 0.1, len(time_points))
            
            # Create engagement curve with natural flow
            engagement_curve = []
            for j, t in enumerate(time_points):
                # Typical engagement pattern: high start, dip in middle, recovery
                time_factor = 1 - 0.3 * np.sin(np.pi * t / 10) + 0.1 * np.cos(2 * np.pi * t / 10)
                engagement_point = base_engagement * time_factor + variation[j]
                engagement_point = max(0, min(1, engagement_point))  # Clamp to [0, 1]
                engagement_curve.append(engagement_point)
            
            # Plot engagement curve
            ax.plot(time_points, engagement_curve, 'o-', linewidth=3, 
                   color=colors[i], label=role, markersize=8)
            
            # Add trend line
            z = np.polyfit(time_points, engagement_curve, 2)
            p = np.poly1d(z)
            ax.plot(time_points, p(time_points), '--', color=colors[i], alpha=0.5, linewidth=2)
        
        # Add engagement zones
        ax.axhspan(0.8, 1.0, alpha=0.1, color='green', label='High Engagement')
        ax.axhspan(0.5, 0.8, alpha=0.1, color='yellow', label='Moderate Engagement')
        ax.axhspan(0.0, 0.5, alpha=0.1, color='red', label='Low Engagement')
        
        # Calculate overall engagement trend
        if len(engagement_data) >= 2:
            client_engagement = engagement_data[0][1] if engagement_data[0][0] == 'Client' else engagement_data[1][1]
            counselor_engagement = engagement_data[1][1] if engagement_data[1][0] == 'Counselor' else engagement_data[0][1]
            
            # Add engagement analysis text
            if client_engagement > 0.7 and counselor_engagement > 0.7:
                trend_text = "High mutual engagement"
            elif client_engagement > 0.5 and counselor_engagement > 0.5:
                trend_text = "Moderate engagement levels"
            else:
                trend_text = "Low engagement detected"
            
            ax.text(0.02, 0.98, f"Trend: {trend_text}", transform=ax.transAxes, 
                   fontsize=self.fontsize, verticalalignment='top', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
        
        ax.set_xlabel('Time (minutes)', fontsize=self.fontsize)
        ax.set_ylabel('Engagement Score', fontsize=self.fontsize)
        ax.set_title('Engagement Timeline Analysis', fontsize=self.fontsize, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=self.fontsize)
    
    def _create_recommendations_panel(self, ax, data, scores):
        """Create comprehensive recommendations panel."""
        recommendations = data.get('recommendations', [])
        
        # Generate detailed score-based recommendations
        additional_recs = []
        
        # Synchrony recommendations
        if scores['synchrony_skills'] < 40:
            additional_recs.append("🔄 Critical: Practice basic mirroring techniques and emotional synchrony")
        elif scores['synchrony_skills'] < 60:
            additional_recs.append("🔄 Improve: Focus on mirroring client's nonverbal cues and emotional states")
        elif scores['synchrony_skills'] < 75:
            additional_recs.append("🔄 Enhance: Advanced synchrony techniques for better rapport building")
        
        # Engagement recommendations
        if scores['engagement_management'] < 40:
            additional_recs.append("💡 Critical: Implement active engagement strategies and participation techniques")
        elif scores['engagement_management'] < 60:
            additional_recs.append("💡 Improve: Use more interactive techniques and motivational interviewing")
        elif scores['engagement_management'] < 75:
            additional_recs.append("💡 Enhance: Advanced engagement techniques and client involvement strategies")
        
        # Emotional regulation recommendations
        if scores['emotional_regulation'] < 40:
            additional_recs.append("🧘 Critical: Learn grounding techniques and emotional stability practices")
        elif scores['emotional_regulation'] < 60:
            additional_recs.append("🧘 Improve: Practice emotional regulation and stress management techniques")
        elif scores['emotional_regulation'] < 75:
            additional_recs.append("🧘 Enhance: Advanced emotional regulation and mindfulness practices")
        
        # Nonverbal communication recommendations
        if scores['nonverbal_communication'] < 40:
            additional_recs.append("🎭 Critical: Basic nonverbal communication training needed")
        elif scores['nonverbal_communication'] < 60:
            additional_recs.append("🎭 Improve: Enhance facial expression and gesture diversity")
        elif scores['nonverbal_communication'] < 75:
            additional_recs.append("🎭 Enhance: Advanced nonverbal communication techniques")
        
        # Overall performance recommendations
        overall_score = scores['overall_performance']
        if overall_score < 40:
            additional_recs.append("⚠️ Priority: Comprehensive counseling skills development program recommended")
        elif overall_score < 60:
            additional_recs.append("📈 Focus: Targeted skill development in key areas")
        elif overall_score >= 85:
            additional_recs.append("⭐ Excellent: Maintain current high standards and consider peer mentoring")
        
        # Combine all recommendations
        all_recommendations = recommendations + additional_recs
        
        if not all_recommendations:
            rec_text = "⭐ Excellent counseling performance demonstrated across all areas.\n🎯 Continue current approaches and consider advanced specialization.\n📚 Explore peer mentoring or supervision opportunities."
        else:
            # Prioritize and format recommendations
            rec_text = '\n'.join(all_recommendations[:8])  # Top 8 recommendations
        
        # Create a more visually appealing recommendation box
        ax.text(0.05, 0.95, rec_text, transform=ax.transAxes, fontsize=15,
               verticalalignment='top', fontweight='normal',
               bbox=dict(boxstyle='round,pad=0.6', facecolor='lightblue', 
                        alpha=0.8, edgecolor='navy', linewidth=1))
        
        # Add a priority indicator
        priority_colors = {
            'Critical': '#FF4444',
            'Improve': '#FF8C00', 
            'Enhance': '#32CD32',
            'Excellent': '#4169E1'
        }
        
        # Count recommendation types
        critical_count = sum(1 for rec in all_recommendations if 'Critical' in rec)
        improve_count = sum(1 for rec in all_recommendations if 'Improve' in rec)
        enhance_count = sum(1 for rec in all_recommendations if 'Enhance' in rec)
        
        # Add priority summary
        priority_text = f"Priority Summary: {critical_count} Critical | {improve_count} Improve | {enhance_count} Enhance"
        ax.text(0.05, 0.05, priority_text, transform=ax.transAxes, fontsize=self.fontsize,
               verticalalignment='bottom', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
        
        ax.set_title('Improvement Recommendations', fontsize=self.fontsize, fontweight='bold', pad=15)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def _create_detailed_metrics_table(self, ax, scores, data):
        """Create detailed metrics table."""
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = [
            ['Area', 'Score', 'Rating', 'Detailed Metrics'],
            ['Synchrony Skills', f"{scores['synchrony_skills']:.1f}", 
             self.get_performance_level(scores['synchrony_skills'])[0],
             f"Facial: {scores['detailed_breakdown']['facial_synchrony']:.1f}pts"],
            ['Engagement Management', f"{scores['engagement_management']:.1f}", 
             self.get_performance_level(scores['engagement_management'])[0],
             f"Engagement: {scores['detailed_breakdown']['engagement_level']:.1f}pts"],
            ['Emotional Regulation', f"{scores['emotional_regulation']:.1f}", 
             self.get_performance_level(scores['emotional_regulation'])[0],
             f"Stability: {data.get('quality_indicators', {}).get('emotional_regulation', 0):.1f}"],
            ['Nonverbal Communication', f"{scores['nonverbal_communication']:.1f}", 
             self.get_performance_level(scores['nonverbal_communication'])[0],
             'Expression, Gesture, Voice combined'],
            ['Overall Assessment', f"{scores['overall_performance']:.1f}", 
             self.get_performance_level(scores['overall_performance'])[0],
             'Weighted average of all areas']
        ]
        
        # Create table
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.2, 0.15, 0.15, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style data rows with colors based on performance
        for i in range(1, len(table_data)):
            score = float(table_data[i][1])
            _, color = self.get_performance_level(score)
            table[(i, 1)].set_facecolor(color)
            table[(i, 1)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Detailed Assessment Results', fontsize=self.fontsize, fontweight='bold')

def run_visualize_analysis(
    analysis_dir: str = '/scratch2/iyy1112/consolidated_analysis',
    output_dir: str = '/scratch2/iyy1112/assessment_dashboards'
) -> Dict:
    """Run visualization analysis to create assessment dashboards."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize visualizer
    visualizer = CounselingAssessmentVisualizer(analysis_dir)
    
    # Load all analysis results
    analysis_results = visualizer.load_analysis_results()
    
    if not analysis_results:
        logger.warning("No analysis results found!")
        return {}
    
    # Create dashboard for each session
    dashboard_results = {}
    
    for session_name, data in analysis_results.items():
        logger.info(f"Creating assessment dashboard for: {session_name}")
        
        try:
            dashboard_path = visualizer.create_assessment_dashboard(
                session_name, data, output_dir
            )
            dashboard_results[session_name] = {
                'dashboard_path': dashboard_path,
                'scores': visualizer.calculate_comprehensive_scores(data)
            }
            
        except Exception as e:
            logger.error(f"Failed to create dashboard for {session_name}: {e}")
            continue
    
    # Create summary report
    summary_path = os.path.join(output_dir, 'assessment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(dashboard_results, f, indent=2)
    
    logger.info(f"Assessment visualization complete. Results saved to: {output_dir}")
    logger.info(f"Summary report: {summary_path}")
    
    return dashboard_results

def run_alignment_analysis(
    base_output_dir: str = '/scratch2/iyy1112/outputs',
    output_dir: str = '/scratch2/iyy1112/alignment_analysis',
    force_realign: bool = False
) -> Dict:
    """Run audio alignment analysis and save alignment data for all pairs."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = CounselingQualityAnalyzer(base_output_dir)
    
    # Find client-counselor pairs
    pairs = analyzer.find_client_counselor_pairs()
    
    if not pairs:
        logger.warning("No client-counselor pairs found!")
        return {}
    
    alignment_results = {}
    
    for i, (client_dir, counselor_dir) in enumerate(pairs):
        logger.info(f"Processing alignment for pair {i+1}/{len(pairs)}: {os.path.basename(client_dir)} - {os.path.basename(counselor_dir)}")
        
        base_name = os.path.basename(client_dir).replace('_T', '')
        pair_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(pair_output_dir, exist_ok=True)
        
        try:
            # Load data for each task type
            for task_type in analyzer.task_types:
                client_data = analyzer.load_analysis_data(client_dir, task_type)
                counselor_data = analyzer.load_analysis_data(counselor_dir, task_type)
                
                if client_data and counselor_data:
                    # Synchronize timelines (this will save alignment data)
                    client_sync, counselor_sync = analyzer.synchronize_timelines(
                        client_data, counselor_data, 
                        use_cache=not force_realign, 
                        save_to_cache=True
                    )
                    
                    # Create separate timeline visualizations
                    client_data['dataframe'] = client_sync
                    counselor_data['dataframe'] = counselor_sync
                    
                    viz_path = analyzer.visualize_separate_timelines(
                        client_data, counselor_data, pair_output_dir, base_name
                    )
                    
                    # Create alignment comparison visualization
                    comparison_path = analyzer.visualize_alignment_comparison(
                        client_data, counselor_data, pair_output_dir, base_name
                    )
                    
                    if base_name not in alignment_results:
                        alignment_results[base_name] = {}
                    
                    alignment_results[base_name][task_type] = {
                        'client_sync_length': len(client_sync),
                        'counselor_sync_length': len(counselor_sync),
                        'timeline_viz_path': viz_path,
                        'comparison_viz_path': comparison_path
                    }
                    
                    logger.info(f"Processed {task_type} alignment for {base_name}")
                    
        except Exception as e:
            logger.error(f"Failed to process alignment for pair {base_name}: {e}")
    
    # Save alignment results summary
    results_file = os.path.join(output_dir, 'alignment_analysis_summary.json')
    with open(results_file, 'w') as f:
        json.dump(alignment_results, f, indent=2)
    
    logger.info(f"Alignment analysis complete. Results saved to: {results_file}")
    return alignment_results

def run_synchronized_analysis(
    base_output_dir: str = '/scratch2/iyy1112/outputs',
    output_dir: str = '/scratch2/iyy1112/synchronized_analysis'
) -> Dict:
    """Run analysis using pre-computed alignment data."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = CounselingQualityAnalyzer(base_output_dir)
    
    # Check if alignment cache exists
    if not os.path.exists(analyzer.alignment_cache_file):
        logger.warning("No alignment cache found. Run alignment analysis first.")
        return {}
    
    # Load alignment cache
    alignment_cache = analyzer.load_alignment_cache()
    
    # Find client-counselor pairs
    pairs = analyzer.find_client_counselor_pairs()
    
    if not pairs:
        logger.warning("No client-counselor pairs found!")
        return {}
    
    sync_results = {}
    
    for i, (client_dir, counselor_dir) in enumerate(pairs):
        base_name = os.path.basename(client_dir).replace('_T', '')
        pair_key = f"{os.path.basename(client_dir)}_{os.path.basename(counselor_dir)}"
        
        logger.info(f"Processing synchronized analysis for pair {i+1}/{len(pairs)}: {base_name}")
        
        pair_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(pair_output_dir, exist_ok=True)
        
        if pair_key not in alignment_cache:
            logger.warning(f"No alignment data found for {base_name}")
            continue
            
        try:
            offset = alignment_cache[pair_key]['offset']
            
            # Process each task type using cached alignment
            for task_type in analyzer.task_types:
                client_data = analyzer.load_analysis_data(client_dir, task_type)
                counselor_data = analyzer.load_analysis_data(counselor_dir, task_type)
                
                if client_data and counselor_data:
                    # Apply alignment using cached offset
                    client_sync, counselor_sync = analyzer.apply_alignment_to_csv(
                        client_data['dataframe'], counselor_data['dataframe'], offset
                    )
                    
                    # Save synchronized CSV files
                    client_sync_path = os.path.join(pair_output_dir, f'{task_type}_client_synchronized.csv')
                    counselor_sync_path = os.path.join(pair_output_dir, f'{task_type}_counselor_synchronized.csv')
                    
                    client_sync.to_csv(client_sync_path, index=False)
                    counselor_sync.to_csv(counselor_sync_path, index=False)
                    
                    # Create visualizations
                    client_data['dataframe'] = client_sync
                    counselor_data['dataframe'] = counselor_sync
                    
                    viz_path = analyzer.visualize_separate_timelines(
                        client_data, counselor_data, pair_output_dir, f"{base_name}_sync"
                    )
                    
                    if base_name not in sync_results:
                        sync_results[base_name] = {}
                    
                    sync_results[base_name][task_type] = {
                        'offset_used': offset,
                        'client_sync_path': client_sync_path,
                        'counselor_sync_path': counselor_sync_path,
                        'viz_path': viz_path,
                        'sync_length': len(client_sync)
                    }
                    
                    logger.info(f"Processed synchronized {task_type} for {base_name}")
                    
        except Exception as e:
            logger.error(f"Failed to process synchronized analysis for pair {base_name}: {e}")
    
    # Save synchronized results summary
    results_file = os.path.join(output_dir, 'synchronized_analysis_summary.json')
    with open(results_file, 'w') as f:
        json.dump(sync_results, f, indent=2)
    
    logger.info(f"Synchronized analysis complete. Results saved to: {results_file}")
    return sync_results

def run_consolidated_analysis(
    base_output_dir: str = '/scratch2/iyy1112/outputs',
    output_dir: str = '/scratch2/iyy1112/consolidated_analysis',
    create_visualizations: bool = True
) -> Dict:
    """Run consolidated analysis for all client-counselor pairs."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = CounselingQualityAnalyzer(base_output_dir)
    
    # Find client-counselor pairs
    pairs = analyzer.find_client_counselor_pairs()
    
    if not pairs:
        logger.warning("No client-counselor pairs found!")
        return {}
    
    # Process each pair
    all_results = {}
    
    for i, (client_dir, counselor_dir) in enumerate(pairs):
        logger.info(f"Processing pair {i+1}/{len(pairs)}: {os.path.basename(client_dir)} - {os.path.basename(counselor_dir)}")
        
        # Create pair-specific output directory
        base_name = os.path.basename(client_dir).replace('_T', '')
        pair_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(pair_output_dir, exist_ok=True)
        
        try:
            # Run counseling quality analysis
            quality_analysis = analyzer.analyze_counseling_quality(
                client_dir, counselor_dir, pair_output_dir
            )
            
            # Create consolidated visualization
            if create_visualizations:
                viz_path = analyzer.create_consolidated_visualization(
                    quality_analysis, pair_output_dir
                )
                quality_analysis['visualization_path'] = viz_path
            
            all_results[base_name] = quality_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze pair {base_name}: {e}")
            continue
    
    # Create summary report
    summary_path = os.path.join(output_dir, 'consolidated_analysis_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Consolidated analysis completed. Results saved to: {output_dir}")
    logger.info(f"Summary report: {summary_path}")
    
    return all_results

def main(
    mode: str = 'alignment',
    base_output_dir: str = '/scratch2/iyy1112/outputs',
    output_dir: str = None,
    force_realign: bool = False
):
    
    if mode == 'alignment':
        output_dir = output_dir or '/scratch2/iyy1112/alignment_analysis'
        logger.info("Running alignment analysis...")
        results = run_alignment_analysis(base_output_dir, output_dir, force_realign)
        logger.info(f"Alignment analysis complete. Results: {len(results)} pairs processed.")
        
        # Print alignment cache summary
        analyzer = CounselingQualityAnalyzer(base_output_dir)
        cache = analyzer.load_alignment_cache()
        logger.info("\nAlignment Cache Summary:")
        for key, info in cache.items():
            logger.info(f"  {key}: offset={info['offset']}, timestamp={info['timestamp']}")
            
    elif mode == 'synchronized':
        output_dir = output_dir or '/scratch2/iyy1112/synchronized_analysis'
        logger.info("Running synchronized analysis...")
        results = run_synchronized_analysis(base_output_dir, output_dir)
        logger.info(f"Synchronized analysis complete. Results: {len(results)} pairs processed.")
        
    elif mode == 'consolidated':
        output_dir = output_dir or '/scratch2/iyy1112/consolidated_analysis'
        logger.info("Running consolidated analysis...")
        results = run_consolidated_analysis(base_output_dir, output_dir)
        logger.info(f"Consolidated analysis complete. Results: {len(results)} pairs processed.")
    
    elif mode == 'visualize':
        analysis_dir = '/scratch2/iyy1112/alignment_analysis'
        viz_output_dir = '/scratch2/iyy1112/assessment_dashboards'
        logger.info("Running visualization analysis...")
        logger.info(f"Analysis directory: {analysis_dir}")
        logger.info(f"Output directory: {viz_output_dir}")
        results = run_visualize_analysis(analysis_dir, viz_output_dir)
        logger.info(f"Visualization analysis complete. Results: {len(results)} sessions processed.")
    
    else:
        logger.error(f"Unknown mode: {mode}. Available modes: alignment, synchronized, consolidated, visualize")
        return
    
    logger.info(f"All results saved to: {output_dir}")

if __name__ == '__main__':
    import fire
    fire.Fire(main)