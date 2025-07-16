# Audio Alignment and Synchronized Analysis

This document explains how to use the new audio alignment features added to the counseling quality analysis system.

## Features Added

1. **Audio Alignment Caching**: Save alignment time offsets for each client-counselor pair to avoid re-computing alignment every time.
2. **Synchronized CSV Export**: Apply saved alignment data to CSV files for separate analysis.
3. **Separate Timeline Visualizations**: Create visualizations comparing client and counselor timelines separately.
4. **Alignment Comparison**: Visualize the difference between aligned and unaligned data.

## Usage

### 1. First-time Audio Alignment Analysis

Run this to compute and save audio alignment data for all pairs:

```bash
python consolidated_analysis.py --mode alignment
```

This will:
- Perform audio alignment using the `audalign` library
- Save alignment offsets to `audio_alignment_cache.json` 
- Create timeline visualizations for each pair
- Generate alignment comparison plots

### 2. Synchronized Analysis (Using Cached Alignment)

After running alignment analysis, use the cached data:

```bash
python consolidated_analysis.py --mode synchronized
```

This will:
- Load previously computed alignment offsets from cache
- Apply alignment to CSV data files
- Save synchronized CSV files for client and counselor separately
- Create visualizations of the synchronized data

### 3. Force Re-alignment

If you want to recompute alignment (ignoring cache):

```bash
python consolidated_analysis.py --mode alignment --force_realign
```

### 4. Custom Output Directories

Specify custom output directories:

```bash
python consolidated_analysis.py --mode alignment --output_dir /path/to/custom/output
```

## Output Files

### Alignment Analysis Mode
- `audio_alignment_cache.json`: Contains offset data for each pair
- `{pair_name}_separate_timelines_{task_type}.png`: Timeline visualizations
- `{pair_name}_alignment_comparison_{task_type}.png`: Before/after alignment comparison
- `alignment_analysis_summary.json`: Summary of all results

### Synchronized Analysis Mode
- `{task_type}_client_synchronized.csv`: Synchronized client data
- `{task_type}_counselor_synchronized.csv`: Synchronized counselor data
- `{pair_name}_sync_separate_timelines_{task_type}.png`: Synchronized timeline visualizations
- `synchronized_analysis_summary.json`: Summary of all synchronized results

## API Usage

You can also use the functions directly in your code:

```python
from consolidated_analysis import CounselingQualityAnalyzer

# Initialize analyzer
analyzer = CounselingQualityAnalyzer('/path/to/outputs')

# Get saved alignment for a pair
offset = analyzer.get_saved_alignment(client_dir, counselor_dir)

# Apply alignment to CSV data
client_sync, counselor_sync = analyzer.apply_alignment_to_csv(
    client_df, counselor_df, offset
)

# Create visualizations
viz_path = analyzer.visualize_separate_timelines(
    client_data, counselor_data, output_dir, pair_name
)
```

## Benefits

1. **Efficiency**: Compute alignment once, reuse many times
2. **Consistency**: Same alignment applied across all analyses
3. **Flexibility**: Separate CSV files for independent analysis
4. **Visualization**: Clear comparison of aligned vs unaligned data
5. **Debugging**: Easy to inspect and verify alignment quality

## Task Types Supported

- `facial`: Facial expression analysis
- `pose`: Pose landmark analysis  
- `prosody`: Prosodic feature analysis

Each task type gets its own synchronized CSV files and visualizations. 