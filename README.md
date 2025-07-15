# Robust Video Analysis System

A comprehensive and robust video loading and analysis system designed to handle long videos in various formats (m2ts, mp4, avi, etc.) with advanced error handling, memory management, and progress tracking.

## Features

### 🎬 **Robust Video Loading**
- **Multiple Format Support**: Handles m2ts, mp4, avi, mov, mkv, flv, wmv, webm
- **Long Video Support**: Optimized for processing videos of any length
- **Memory Management**: Automatic chunking and memory monitoring
- **Error Handling**: Comprehensive error detection and recovery

### 📊 **Video Analysis**
- **Prosody Analysis**: Extract audio features using OpenSMILE
- **Facial Expression Analysis**: Extract facial landmarks using OpenFace
- **Gesture/Posture Analysis**: Extract body pose using MediaPipe

### 🔧 **Advanced Features**
- **Format Conversion**: Convert between video formats using FFmpeg
- **Batch Processing**: Process multiple videos efficiently
- **Progress Tracking**: Real-time progress and timing information
- **System Monitoring**: Memory usage tracking and warnings

## Installation

1. **Install Dependencies**:
```bash
pip install -e .
pip install -r requirements.txt
```

2. **Install FFmpeg** (optional, for format conversion):
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# CentOS/RHEL
sudo yum install ffmpeg

# macOS
brew install ffmpeg
```

## Quick Start

### Basic Video Loading
```python
from analysis import RobustVideoLoader

# Load a video
loader = RobustVideoLoader("path/to/video.m2ts")
video_info = loader.get_video_info()

print(f"Resolution: {video_info['width']}x{video_info['height']}")
print(f"Duration: {video_info['duration_seconds']/60:.1f} minutes")
print(f"File size: {video_info['file_size_gb']:.2f} GB")
```

### Full Analysis Pipeline
```python
from analysis import main

# Run complete analysis
results = main(
    video_path="path/to/video.m2ts",
    output_dir="outputs"
)

print(f"Processing time: {results['processing_time_minutes']:.1f} minutes")
```

### Batch Processing
```python
from analysis import batch_process_videos

video_paths = [
    "video1.m2ts",
    "video2.mp4",
    "video3.avi"
]

results = batch_process_videos(
    video_paths=video_paths,
    output_dir="batch_outputs",
    processes=['prosody', 'facial']  # Only run specific analyses
)
```

## API Reference

### RobustVideoLoader Class

#### Constructor
```python
RobustVideoLoader(video_path: Union[str, Path])
```

#### Methods
- `get_video_info() -> Dict[str, Any]`: Get comprehensive video information
- `load_video(chunk_size: Optional[int] = None) -> VideoReader`: Load video with memory management
- `extract_audio_robust(output_dir: Union[str, Path]) -> str`: Extract audio with error handling
- `cleanup()`: Clean up resources

### Utility Functions

#### Video Loading
- `load_video(video_path, chunk_size=None) -> VideoReader`: Simple video loading
- `load_video_with_info(video_path) -> tuple[VideoReader, Dict]`: Load video with info

#### Analysis Functions
- `analyze_prosody(video_path, output_dir) -> str`: Extract prosodic features
- `analyze_facial_expressions(video_path, output_dir) -> str`: Extract facial landmarks
- `analyze_gestures_posture(video_path, output_dir) -> str`: Extract pose landmarks

#### Utility Functions
- `convert_video_format(input_path, output_path, target_format='mp4', quality='medium') -> str`: Convert video format
- `batch_process_videos(video_paths, output_dir, processes=None) -> Dict`: Process multiple videos
- `get_system_memory_info() -> Dict[str, float]`: Get system memory usage

## Supported Video Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| MP4    | .mp4      | Most common, well supported |
| M2TS   | .m2ts     | Blu-ray format, large files |
| AVI    | .avi      | Legacy format |
| MOV    | .mov      | Apple QuickTime |
| MKV    | .mkv      | Matroska container |
| FLV    | .flv      | Flash video |
| WMV    | .wmv      | Windows Media |
| WebM   | .webm     | Web-optimized |

## Memory Management

The system automatically manages memory for long videos:

- **Chunking**: Long videos are processed in chunks (default: 1000 frames)
- **Memory Monitoring**: Real-time memory usage tracking
- **Warnings**: Alerts when memory usage exceeds 90%
- **Cleanup**: Automatic resource cleanup after processing

## Error Handling

The system provides robust error handling for:

- **File Not Found**: Clear error messages for missing files
- **Unsupported Formats**: Validation of video formats
- **Corrupted Files**: Graceful handling of corrupted video files
- **Memory Issues**: Warnings and suggestions for memory problems
- **Processing Failures**: Detailed error reporting for analysis failures

## Performance Optimization

### For Long Videos (>10 minutes)
- Use chunked processing
- Monitor memory usage
- Consider format conversion for better compatibility

### For Batch Processing
- Process videos sequentially to avoid memory issues
- Use specific analysis types to reduce processing time
- Monitor system resources

## Example Usage

See `example_usage.py` for comprehensive examples of all features.

### Running Examples
```bash
python example_usage.py
```

### Command Line Usage
```bash
# Basic analysis
python analysis.py --video_path="video.m2ts" --output_dir="outputs"

# With custom parameters
python analysis.py \
    --video_path="/path/to/long_video.m2ts" \
    --output_dir="analysis_results"
```

## Troubleshooting

### Common Issues

1. **"FFmpeg not found"**
   - Install FFmpeg: `sudo apt install ffmpeg` (Ubuntu) or `brew install ffmpeg` (macOS)

2. **"Memory usage high"**
   - Close other applications
   - Use chunked processing
   - Convert video to smaller format

3. **"Unsupported format"**
   - Convert video using `convert_video_format()`
   - Check if format is in `SUPPORTED_FORMATS`

4. **"Video loading failed"**
   - Check file path and permissions
   - Verify video file is not corrupted
   - Try converting to MP4 format

### Performance Tips

- **For very long videos**: Use format conversion to MP4 first
- **For batch processing**: Monitor memory and process sequentially
- **For memory-constrained systems**: Use smaller chunk sizes
- **For faster processing**: Use lower quality settings in format conversion

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 