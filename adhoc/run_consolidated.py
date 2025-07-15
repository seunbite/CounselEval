#!/usr/bin/env python3
"""
Simple script to run consolidated counseling analysis.
This script runs the complete pipeline including all three analyses and counseling quality assessment.
"""

import os
import sys
from pathlib import Path

print("DEBUG: Script started")

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("DEBUG: Added current directory to Python path")

try:
    from consolidated_main import run_counseling_quality_analysis
    print("DEBUG: Successfully imported run_counseling_quality_analysis")
except Exception as e:
    print(f"DEBUG: Error importing: {e}")
    sys.exit(1)

def main():
    """Run consolidated counseling analysis with default parameters."""
    
    print("DEBUG: Main function started")
    
    # Default paths
    base_output_dir = '/scratch2/iyy1112/outputs'
    consolidated_output_dir = '/scratch2/iyy1112/consolidated_analysis'
    
    print("Starting consolidated counseling analysis...")
    print(f"Base output directory: {base_output_dir}")
    print(f"Consolidated output directory: {consolidated_output_dir}")
    
    # Check if base output directory exists
    if not os.path.exists(base_output_dir):
        print(f"Error: Base output directory {base_output_dir} does not exist!")
        print("Please run the individual analyses first using main.py")
        return
    
    # Create consolidated output directory
    os.makedirs(consolidated_output_dir, exist_ok=True)
    
    try:
        print("DEBUG: About to call run_counseling_quality_analysis")
        # Run counseling quality analysis
        results = run_counseling_quality_analysis(
            base_output_dir=base_output_dir,
            consolidated_output_dir=consolidated_output_dir,
            create_visualizations=True
        )
        
        print(f"\nAnalysis completed successfully!")
        print(f"Found {len(results)} client-counselor pairs")
        print(f"Results saved to: {consolidated_output_dir}")
        
        # Print summary
        if results:
            print("\nSummary of findings:")
            for pair_name, pair_data in results.items():
                quality_indicators = pair_data.get('quality_indicators', {})
                synchrony_metrics = pair_data.get('synchrony_metrics', {})
                
                print(f"\n{pair_name}:")
                if quality_indicators:
                    print(f"  Engagement Level: {quality_indicators.get('engagement_level', 'N/A'):.3f}")
                    print(f"  Rapport Building: {quality_indicators.get('rapport_building', 'N/A'):.3f}")
                    print(f"  Emotional Regulation: {quality_indicators.get('emotional_regulation', 'N/A'):.3f}")
                
                if synchrony_metrics:
                    for modality, metrics in synchrony_metrics.items():
                        if 'overall_synchrony_score' in metrics:
                            print(f"  {modality.title()} Synchrony: {metrics['overall_synchrony_score']:.3f}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("DEBUG: Script entry point")
    main() 