#!/usr/bin/env python3
"""
Example script demonstrating how to load and plot analysis summaries
created by the main.py script.

Usage:
    python load_and_plot_example.py --summary_path path/to/summary.json
    python load_and_plot_example.py --summary_path path/to/summary.pkl
"""

import os
import sys
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import from main.py
sys.path.append(str(Path(__file__).parent))

from main import load_and_plot_summary

def main():
    parser = argparse.ArgumentParser(description='Load and plot analysis summary')
    parser.add_argument('--summary_path', type=str, required=True,
                       help='Path to the summary file (.json or .pkl)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (default: same as summary file)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.summary_path):
        print(f"Error: Summary file not found: {args.summary_path}")
        return 1
    
    print(f"Loading analysis summary from: {args.summary_path}")
    
    # Load and create plots
    plot_path = load_and_plot_summary(args.summary_path, args.output_dir)
    
    if plot_path:
        print(f"Summary plot created successfully: {plot_path}")
        return 0
    else:
        print("Error: Failed to create summary plot")
        return 1

if __name__ == '__main__':
    exit(main()) 