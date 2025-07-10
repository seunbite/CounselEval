import os
import logging
import time
import psutil
import subprocess
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import cv2
import numpy as np
from moviepy import VideoFileClip

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_system_memory_info() -> Dict[str, float]:
    """Get current system memory usage information."""
    memory = psutil.virtual_memory()
    return {
        'total_gb': memory.total / (1024**3),
        'available_gb': memory.available / (1024**3),
        'used_gb': memory.used / (1024**3),
        'percent_used': memory.percent
    }

def convert_video_format(input_path: Union[str, Path], output_path: Union[str, Path], 
                        target_format: str = 'mp4', quality: str = 'medium') -> str:
    """
    Convert video to a different format using ffmpeg.
    
    Args:
        input_path: Input video path
        output_path: Output video path
        target_format: Target format (mp4, avi, etc.)
        quality: Quality preset (low, medium, high)
    
    Returns:
        Path to converted video
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Quality presets
    quality_presets = {
        'low': ['-crf', '28', '-preset', 'fast'],
        'medium': ['-crf', '23', '-preset', 'medium'],
        'high': ['-crf', '18', '-preset', 'slow']
    }
    
    if quality not in quality_presets:
        quality = 'medium'
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build ffmpeg command
    cmd = [
        'ffmpeg', '-i', str(input_path),
        '-c:v', 'libx264', '-c:a', 'aac',
        *quality_presets[quality],
        '-y', str(output_path)
    ]
    
    try:
        logger.info(f"Converting {input_path} to {output_path}...")
        start_time = time.time()
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        conversion_time = time.time() - start_time
        logger.info(f"Video conversion completed in {conversion_time:.2f} seconds")
        
        return str(output_path)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e.stderr}")
        raise RuntimeError(f"Video conversion failed: {e.stderr}")
    except FileNotFoundError:
        logger.error("FFmpeg not found. Please install ffmpeg.")
        raise RuntimeError("FFmpeg not found. Please install ffmpeg.")

class RobustVideoReader:
    """
    A robust video reader that replaces psifx.video.io.VideoReader with standard libraries.
    """
    
    def __init__(self, video_path: Union[str, Path]):
        self.video_path = Path(video_path)
        self.cap = None
        self.clip = None
        self._load_video()
    
    def _load_video(self):
        """Load video using OpenCV and MoviePy."""
        try:
            # Use OpenCV for video reading
            self.cap = cv2.VideoCapture(str(self.video_path))
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open video with OpenCV: {self.video_path}")
            
            # Use MoviePy for audio extraction
            self.clip = VideoFileClip(str(self.video_path))
            
        except Exception as e:
            logger.error(f"Error loading video: {e}")
            raise
    
    def extract_audio(self, output_dir: Union[str, Path], audio_format: str = 'wav') -> str:
        """
        Extract audio from video using MoviePy.
        
        Args:
            output_dir: Directory to save audio
            audio_format: Audio format (wav, mp3, etc.)
        
        Returns:
            Path to extracted audio file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        audio_filename = f"{self.video_path.stem}_audio.{audio_format}"
        audio_path = output_dir / audio_filename
        
        try:
            logger.info(f"Extracting audio to: {audio_path}")
            
            if self.clip.audio is None:
                raise RuntimeError("No audio track found in video")
            
            # Extract audio using MoviePy
            self.clip.audio.write_audiofile(
                str(audio_path),
                logger=None  # Suppress MoviePy logging
            )
            
            logger.info(f"Audio extracted successfully: {audio_path}")
            return str(audio_path)
            
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            raise RuntimeError(f"Failed to extract audio: {e}")
    
    def get_frame_count(self) -> int:
        """Get total number of frames."""
        if self.cap is None:
            raise RuntimeError("Video not loaded")
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def get_fps(self) -> float:
        """Get frames per second."""
        if self.cap is None:
            raise RuntimeError("Video not loaded")
        return self.cap.get(cv2.CAP_PROP_FPS)
    
    def get_duration(self) -> float:
        """Get video duration in seconds."""
        return self.get_frame_count() / self.get_fps()
    
    def get_resolution(self) -> tuple[int, int]:
        """Get video resolution as (width, height)."""
        if self.cap is None:
            raise RuntimeError("Video not loaded")
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read next frame."""
        if self.cap is None:
            raise RuntimeError("Video not loaded")
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def seek(self, frame_number: int):
        """Seek to specific frame."""
        if self.cap is None:
            raise RuntimeError("Video not loaded")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    def release(self):
        """Release resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.clip is not None:
            self.clip.close()
            self.clip = None

class RobustVideoLoader:
    """
    A robust video loader that handles long videos, multiple formats, and memory management.
    """
    
    SUPPORTED_FORMATS = {'.mp4', '.m2ts', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    MAX_VIDEO_SIZE_GB = 50  # Maximum video size in GB before warning
    CHUNK_SIZE_FRAMES = 1000  # Process video in chunks for memory efficiency
    
    def __init__(self, video_path: Union[str, Path]):
        self.video_path = Path(video_path)
        self.video_info = None
        self.reader = None
        self._validate_video()
    
    def _validate_video(self):
        """Validate video file exists and is supported."""
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        
        if self.video_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported video format: {self.video_path.suffix}. "
                           f"Supported formats: {self.SUPPORTED_FORMATS}")
        
        # Check file size
        file_size_gb = self.video_path.stat().st_size / (1024**3)
        if file_size_gb > self.MAX_VIDEO_SIZE_GB:
            logger.warning(f"Large video file detected: {file_size_gb:.2f} GB. "
                          f"This may take a long time to process.")
    
    def get_video_info(self) -> Dict[str, Any]:
        """Get comprehensive video information."""
        if self.video_info is None:
            try:
                # Use temporary reader to get info
                temp_reader = RobustVideoReader(self.video_path)
                width, height = temp_reader.get_resolution()
                fps = temp_reader.get_fps()
                frame_count = temp_reader.get_frame_count()
                duration = temp_reader.get_duration()
                
                self.video_info = {
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'frame_count': frame_count,
                    'duration_seconds': duration,
                    'format': self.video_path.suffix.lower(),
                    'file_size_gb': self.video_path.stat().st_size / (1024**3)
                }
                
                temp_reader.release()
                
                logger.info(f"Video info: {self.video_info['width']}x{self.video_info['height']}, "
                           f"{self.video_info['fps']:.2f} fps, "
                           f"{self.video_info['duration_seconds']/60:.1f} minutes")
                
            except Exception as e:
                logger.error(f"Error getting video info: {e}")
                raise
        
        return self.video_info
    
    def load_video(self, chunk_size: Optional[int] = None) -> RobustVideoReader:
        """
        Load video with robust error handling and memory management.
        
        Args:
            chunk_size: Number of frames to process at once (None for auto)
        
        Returns:
            RobustVideoReader instance
        """
        if chunk_size is None:
            video_info = self.get_video_info()
            # Adjust chunk size based on video length and memory considerations
            if video_info['frame_count'] > 10000:  # Long video
                chunk_size = self.CHUNK_SIZE_FRAMES
            else:
                chunk_size = video_info['frame_count']  # Process all at once
        
        try:
            logger.info(f"Loading video: {self.video_path}")
            start_time = time.time()
            
            # Initialize RobustVideoReader with error handling
            self.reader = RobustVideoReader(self.video_path)
            
            load_time = time.time() - start_time
            logger.info(f"Video loaded successfully in {load_time:.2f} seconds")
            
            return self.reader
            
        except Exception as e:
            logger.error(f"Error loading video: {e}")
            raise RuntimeError(f"Failed to load video {self.video_path}: {e}")
    
    def extract_audio_robust(self, output_dir: Union[str, Path]) -> str:
        """
        Extract audio with robust error handling for long videos.
        
        Args:
            output_dir: Directory to save extracted audio
            
        Returns:
            Path to extracted audio file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if self.reader is None:
                self.load_video()
            
            logger.info("Extracting audio from video...")
            start_time = time.time()
            
            audio_path = self.reader.extract_audio(output_dir=output_dir)
            
            extract_time = time.time() - start_time
            logger.info(f"Audio extraction completed in {extract_time:.2f} seconds")
            
            return audio_path
            
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            raise RuntimeError(f"Failed to extract audio: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        if self.reader is not None:
            try:
                self.reader.release()
                self.reader = None
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")

# Compatibility functions to replace psifx VideoReader
def load_video(video_path: Union[str, Path], chunk_size: Optional[int] = None) -> RobustVideoReader:
    """
    Robust video loading function with comprehensive error handling.
    
    Args:
        video_path: Path to video file
        chunk_size: Optional chunk size for memory management
        
    Returns:
        RobustVideoReader instance
    """
    loader = RobustVideoLoader(video_path)
    return loader.load_video(chunk_size=chunk_size)

def load_video_with_info(video_path: Union[str, Path]) -> tuple[RobustVideoReader, Dict[str, Any]]:
    """
    Load video and return both reader and video information.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Tuple of (RobustVideoReader, video_info_dict)
    """
    loader = RobustVideoLoader(video_path)
    reader = loader.load_video()
    info = loader.get_video_info()
    return reader, info
