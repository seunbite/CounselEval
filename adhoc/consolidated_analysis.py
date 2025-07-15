import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from counseleval.src.load import logger

# Import existing functions
from visualize import (
    create_prosody_plot, create_facial_plot, create_pose_plot,
    create_nonverbal_cues_plot, load_and_plot_summary
)
from figuring_out import (
    analyze_nonverbal_cues_facial, analyze_nonverbal_cues_pose, analyze_nonverbal_cues_prosody
)

class CounselingQualityAnalyzer:
    """Analyzes counseling quality by comparing client and counselor nonverbal behaviors."""
    
    def __init__(self, base_output_dir: str):
        self.base_output_dir = base_output_dir
        self.task_types = ['facial', 'pose', 'prosody']
        
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
                df = pd.read_csv(file_path, error_bad_lines=False, warn_bad_lines=False)
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
    
    def synchronize_timelines(self, client_data: Dict, counselor_data: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Synchronize client and counselor data timelines."""
        client_df = client_data['dataframe']
        counselor_df = counselor_data['dataframe']
        
        # Get minimum length
        min_length = min(len(client_df), len(counselor_df))
        
        # Truncate to same length
        client_sync = client_df.iloc[:min_length].reset_index(drop=True)
        counselor_sync = counselor_df.iloc[:min_length].reset_index(drop=True)
        
        # Add frame index for synchronization
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
            'recommendations': []
        }
        
        # Load data for all task types
        client_data = {}
        counselor_data = {}
        
        for task_type in self.task_types:
            client_data[task_type] = self.load_analysis_data(client_dir, task_type)
            counselor_data[task_type] = self.load_analysis_data(counselor_dir, task_type)
        
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
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
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
                        verticalalignment='top', fontsize=10, 
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
                    verticalalignment='top', fontsize=10, fontfamily='monospace')
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

if __name__ == '__main__':
    import fire
    fire.Fire(run_consolidated_analysis) 