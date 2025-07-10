# Analysis Summary Features

## New Features

### 1. Multiple Frame Visualization
The main function now saves multiple visualization frames (default: 10 frames) between a specified frame range (default: frames 100-200).

### 2. Memory-Efficient Analysis Summaries
Instead of storing raw analysis data, the system now creates compact summaries containing:
- **Statistics**: Mean, std, min, max, median, quartiles for all numeric features
- **Key Frames**: Sample data points at regular intervals (every 10% of video)
- **Trends**: Moving averages for top features sampled at regular intervals
- **Metadata**: Video info, analysis type, timestamp, etc.

### 3. Flexible Loading and Plotting
Saved summaries can be loaded later to create comprehensive visualization plots without requiring the original large CSV files.

## Usage
### Installation
```bash
pip install -e .
pip intsall -r requirements.txt
```

### Command Line Usage

```bash
# Run analysis
python main.py --video_path video.m2ts --task_type facial --save_summary

# Load and plot summary later
python load_and_plot_example.py --summary_path outputs/video/facial_summary.json
```

## File Formats

### JSON Summary (.json)
- Human-readable format
- Easy to inspect and share
- Slightly larger file size
- Can be opened in any text editor

### Pickle Summary (.pkl)
- Binary format optimized for Python
- Faster loading
- Smaller file size
- Preserves exact data types

## Memory Efficiency

The summary format dramatically reduces storage requirements:

- **Original CSV**: Can be 100MB+ for long videos
- **JSON Summary**: Typically 10-100KB (1000x smaller)
- **Pickle Summary**: Typically 5-50KB (2000x smaller)

## Summary Plot Features

The generated summary plots include:

1. **Feature Statistics Overview**: Box plots of key statistics
2. **Feature Trends Over Time**: Moving averages of top features
3. **Feature Relationships**: Correlation-style visualization
4. **Feature Distributions**: Probability density approximations
5. **Key Frame Analysis**: Evolution of features at sample points
6. **Analysis Metadata**: Summary statistics and most variable features

## Example Output Structure

```
outputs/
└── video_name/
    ├── facial_analysis.csv           # Full analysis data
    ├── facial_landmarks_frame_0100.png  # Visualization frames (10 files)
    ├── facial_landmarks_frame_0111.png
    ├── ...
    ├── facial_landmarks_frame_0200.png
    ├── facial_summary.json           # Human-readable summary
    ├── facial_summary.pkl            # Efficient binary summary
    └── facial_summary_plot.png       # Comprehensive visualization
```

## Use Cases

1. **Quick Analysis Review**: Load summary plots to quickly understand analysis results
2. **Long-term Storage**: Store compact summaries instead of large CSV files
3. **Comparison Studies**: Compare summaries across multiple videos efficiently
4. **Remote Analysis**: Send small summary files instead of large datasets
5. **Progressive Analysis**: Analyze key frames first, then dive deeper if needed

## Tips

- Use JSON format for sharing and human inspection
- Use pickle format for programmatic processing and faster loading
- The summary retains enough information for most analysis purposes
- Original CSV files can still be kept for detailed analysis when needed
- Frame visualizations are saved with descriptive names including frame numbers 