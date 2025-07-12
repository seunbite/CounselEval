import os
import time
import glob
from counseleval.src.analysis import analyze_prosody, analyze_facial_expressions, analyze_gestures_posture
from counseleval.src.load import RobustVideoLoader, logger

# Import from separated modules
from visualize import (
    create_prosody_plot, create_facial_plot, create_pose_plot,
    create_facial_landmarks_frame, create_pose_landmarks_frame,
    create_nonverbal_cues_plot, load_and_plot_summary
)
from figuring_out import (
    analyze_nonverbal_cues_facial, analyze_nonverbal_cues_pose, analyze_nonverbal_cues_prosody,
    save_analysis_summary
)

# Task type mapping dictionary
TASK_CONFIG = {
    'prosody': {
        'analyze_func': analyze_prosody,
        'plot_func': create_prosody_plot,
        'cues_func': analyze_nonverbal_cues_prosody,
        'output_file': 'prosody_features.csv'
    },
    'facial': {
        'analyze_func': analyze_facial_expressions,
        'plot_func': create_facial_plot,
        'cues_func': analyze_nonverbal_cues_facial,
        'output_file': 'facial_landmarks.csv'
    },
    'pose': {
        'analyze_func': analyze_gestures_posture,
        'plot_func': create_pose_plot,
        'cues_func': analyze_nonverbal_cues_pose,
        'output_file': 'pose_landmarks.csv'
    }
}

def main(
    video_path='/scratch2/MIR_LAB/seungbeen/Study2/Study2_2_9_T.m2ts', 
    task_type='facial', # prosody, facial, pose
    visualize=True,
    plotting=True,
    save_summary=True,
    frame_range=(100, 200),
    num_viz_frames=10,
    output_dir='outputs'
    ):
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Validate task type
    if task_type not in TASK_CONFIG:
        raise ValueError(f"Invalid task type: {task_type}. Must be one of {list(TASK_CONFIG.keys())}")
    
    output_dir = os.path.join(output_dir, os.path.basename(video_path).split('.')[0])
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if data already exists
    config = TASK_CONFIG[task_type]
    existing_file = os.path.join(output_dir, config['output_file'])
    
    if os.path.exists(existing_file):
        logger.info(f"Data already exists for {video_path}, performing nonverbal cue analysis only")
        cues_path, cues_data = config['cues_func'](existing_file, output_dir)
        
        if plotting and cues_data:
            plot_path = create_nonverbal_cues_plot(cues_data, task_type, output_dir)
            logger.info(f"Nonverbal cues plot: {plot_path}")
        
        return {'nonverbal_cues_path': cues_path, 'nonverbal_cues_data': cues_data}
    
    # Load video and get info
    loader = RobustVideoLoader(video_path)
    video_info = loader.get_video_info()
    
    logger.info(f"{video_path}: {video_info['width']}x{video_info['height']}, "
                f"{video_info['fps']:.2f} fps, {video_info['duration_seconds']/60:.1f} minutes")
    
    total_start_time = time.time()
    results = {}
    
    # Get functions from config
    analyze_func = config['analyze_func']
    plot_func = config['plot_func']
    cues_func = config['cues_func']
    
    # Run analysis
    analysis_file = analyze_func(video_path, output_dir, visualize=visualize, frame_range=frame_range, num_viz_frames=num_viz_frames)
    logger.info(f"{task_type}: {analysis_file}")
    results[f'{task_type}_file'] = analysis_file
    
    # Perform nonverbal cue analysis
    cues_path, cues_data = cues_func(analysis_file, output_dir)
    results[f'{task_type}_nonverbal_cues'] = cues_path
    results[f'{task_type}_cues_data'] = cues_data
    
    # Create plots
    if plotting:
        plot_path = plot_func(analysis_file, output_dir)
        results[f'{task_type}_plot'] = plot_path
        
        if cues_data:
            cues_plot_path = create_nonverbal_cues_plot(cues_data, task_type, output_dir)
            results[f'{task_type}_cues_plot'] = cues_plot_path

    # Save summary
    if save_summary:
        json_path, pickle_path = save_analysis_summary(analysis_file, video_info, output_dir, task_type)
        results[f'{task_type}_summary_json'] = json_path
        results[f'{task_type}_summary_pkl'] = pickle_path
        
        if json_path:
            summary_plot_path = load_and_plot_summary(json_path, output_dir)
            results[f'{task_type}_summary_plot'] = summary_plot_path
    
    total_time = time.time() - total_start_time
    logger.info(f"Total processing time: {total_time/60:.1f} minutes")
    loader.cleanup()
    
    return results

def meta_run(
    video_path='/scratch2/MIR_LAB/seungbeen/Study2',
    start_idx=0,
    output_dir='/scratch2/iyy1112/outputs',
    chunk_size=2,
    just_number_check=False,
    task_type='pose'
):
    raw_video_paths = sorted(glob.glob(os.path.join(video_path, '*.m2ts')))
    todo_list = []
    done = 0
    
    # Get expected output file from config
    expected_file = TASK_CONFIG[task_type]['output_file']
    
    for video_file in raw_video_paths:
        video_name = os.path.basename(video_file).split('.')[0]
        if os.path.exists(os.path.join(output_dir, video_name, expected_file)):
            print(f"Skipping {video_file} because it already exists")
            done += 1
            continue
        else:
            print(f"Processing {video_file}...") 
            todo_list.append(video_file)
            
    print(f"Remaining {len(todo_list)} videos...")
    if just_number_check:
        return
    todo_list = todo_list[start_idx:start_idx+chunk_size if chunk_size else None]
    
    for video_file in todo_list:
        try:
            main(
                video_path=video_file, 
                task_type=task_type, 
                visualize=True,
                plotting=True,
                save_summary=True,
                frame_range=(0, 10),
                num_viz_frames=10,
                output_dir=output_dir
            )
        except Exception as e:
            logger.error(f"Failed to process {video_file}: {e}")
            continue

if __name__ == '__main__':
    import fire
    fire.Fire(meta_run)
    
    
