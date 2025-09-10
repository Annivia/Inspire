#!/usr/bin/env python3
"""
Debug Visual Concepts Visualizer

Creates a dynamic GIF visualization showing visual concept values over time,
synchronized with the reconstructed trajectory GIF for debugging and validation.

Run standalone without arguments - edit the configuration below to customize.
"""

import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Optional, Tuple
import sys

sys.path.append('/u/xzhang42/Inspire')
from vla_scripts.visual_concepts_extractor import extract_visual_concepts_from_state

# ===== CONFIGURATION - EDIT THESE SETTINGS =====
STATES_OUTPUT_DIR = "/work/nvme/bfbo/xzhang42/data/pilot_test/reconstructed_trajectory_data"
TASK_ID = 1
EPISODE_ID = 1
TASK_DESCRIPTION = ""  # Will be read from dataset
FRAME_DURATION = 100  # 100ms to match reconstruction script (10 FPS)
FRAME_SIZE = (1000, 800)  # (width, height) - Made taller and wider for better text spacing
# ===============================================


def load_reconstructed_states(states_h5_path: str) -> Dict[str, np.ndarray]:
    """
    Load reconstructed simulator states from HDF5 file.
    
    Args:
        states_h5_path: Path to states.h5 file
        
    Returns:
        Dict with state arrays [timesteps, ...]
    """
    states = {}
    
    with h5py.File(states_h5_path, 'r') as f:
        for key in f.keys():
            states[key] = f[key][:]
    
    return states


def extract_concepts_from_states(states: Dict[str, np.ndarray], task_description: str = "") -> List[Dict]:
    """
    Extract visual concepts from loaded states for each timestep.
    
    Args:
        states: States dict from load_reconstructed_states()
        task_description: Task description for context
        
    Returns:
        List of concept dicts, one per timestep
    """
    num_timesteps = len(states['robot_joint_pos'])
    concepts_timeline = []
    
    for timestep in range(num_timesteps):
        # Extract state for this timestep
        timestep_state = {}
        for key, values in states.items():
            if key == 'object_names':
                # String arrays don't need timestep indexing
                timestep_state[key] = values
            else:
                timestep_state[key] = values[timestep]
        
        # Extract visual concepts for this timestep
        concepts = extract_visual_concepts_from_state(timestep_state, task_description)
        concepts_timeline.append(concepts)
    
    return concepts_timeline


def create_concept_visualization_frame(concepts: Dict, timestep: int, frame_size: Tuple[int, int] = (1000, 800)) -> Image.Image:
    """
    Create a single frame showing visual concept values.
    
    Args:
        concepts: Visual concepts dict from extract_visual_concepts_from_state()
        timestep: Current timestep number
        frame_size: Output frame size (width, height)
        
    Returns:
        PIL Image with concept visualization
    """
    width, height = frame_size
    
    # Create white background
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to load fonts
    try:
        font_paths = [
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
            '/System/Library/Fonts/Arial.ttf', 
            '/Windows/Fonts/arial.ttf'
        ]
        title_font = small_font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                title_font = ImageFont.truetype(font_path, 20)
                small_font = ImageFont.truetype(font_path, 14)
                break
        
        if title_font is None:
            title_font = small_font = ImageFont.load_default()
    except:
        title_font = small_font = ImageFont.load_default()
    
    # Title
    title = f"Visual Concepts - Timestep {timestep}"
    draw.text((10, 10), title, fill='black', font=title_font)
    
    # Add LIBERO task prompt to top right
    task_prompt_x = width // 2 + 20
    task_prompt_y = 50
    task_prompt_width = width // 2 - 40
    
    # Draw task prompt box
    draw.rectangle([task_prompt_x - 10, task_prompt_y - 10, width - 10, task_prompt_y + 120], 
                   outline='blue', width=2, fill='lightblue')
    draw.text((task_prompt_x, task_prompt_y - 5), "LIBERO Task:", fill='darkblue', font=title_font)
    
    # Use the task description from configuration, or a default if empty
    if TASK_DESCRIPTION:
        task_text = TASK_DESCRIPTION
    else:
        # Extract task description from concepts if available
        task_text = "Task description not specified"
    
    # Word wrap the task description
    import textwrap
    wrapped_lines = textwrap.wrap(task_text, width=40)  # Adjust width as needed
    
    for i, line in enumerate(wrapped_lines[:4]):  # Show up to 4 lines
        draw.text((task_prompt_x, task_prompt_y + 20 + i*18), line, fill='darkblue', font=small_font)
    
    # Natural language description (left side)
    y_offset = 50
    key_objects = concepts.get('key_objects', {})
    relationships = concepts.get('relationships', {})
    
    # Generate human-readable natural language descriptions
    desc_lines = []
    T_found = key_objects.get('T') is not None
    R_found = key_objects.get('R') is not None
    P_found = key_objects.get('P') is not None
    G_found = key_objects.get('G') is not None
    
    # Describe target object placement
    if T_found and R_found:
        on_T_R = relationships.get('on_T_R', 0) == 1
        above_T_R = relationships.get('above_T_R', 0) == 1
        near_T_R = relationships.get('near_T_R', 0) == 1
        
        if on_T_R:
            desc_lines.append("• Target object is ON the target region")
        elif above_T_R:
            desc_lines.append("• Target object is ABOVE the target region")
        elif near_T_R:
            desc_lines.append("• Target object is NEAR the target region")
        else:
            desc_lines.append("• Target object is FAR from the target region")
    
    elif T_found and P_found:
        on_T_P = relationships.get('on_T_P', 0) == 1
        above_T_P = relationships.get('above_T_P', 0) == 1
        
        if on_T_P:
            desc_lines.append("• Target object is ON the table")
        elif above_T_P:
            desc_lines.append("• Target object is ABOVE the table")
        else:
            desc_lines.append("• Target object is NOT on the table")
    
    # Describe gripper position relative to target region
    if G_found and R_found:
        above_G_R = relationships.get('above_G_R', 0) == 1
        near_G_R = relationships.get('near_G_R', 0) == 1
        touching_G_R = relationships.get('touching_G_R', 0) == 1
        
        if touching_G_R:
            desc_lines.append("• Gripper is TOUCHING the target region")
        elif above_G_R:
            desc_lines.append("• Gripper is ABOVE the target region")
        elif near_G_R:
            desc_lines.append("• Gripper is NEAR the target region")
        else:
            desc_lines.append("• Gripper is FAR from the target region")
    
    # Describe gripper-target object interaction
    if G_found and T_found:
        touching_G_T = relationships.get('touching_G_T', 0) == 1
        above_G_T = relationships.get('above_G_T', 0) == 1
        near_G_T = relationships.get('near_G_T', 0) == 1
        aligned_G_T = relationships.get('aligned_G_T', 0) == 1
        
        if touching_G_T:
            desc_lines.append("• Gripper is GRASPING the target object")
        elif above_G_T and aligned_G_T:
            desc_lines.append("• Gripper is POSITIONED ABOVE the target object")
        elif near_G_T:
            desc_lines.append("• Gripper is APPROACHING the target object")
        else:
            desc_lines.append("• Gripper is MOVING AWAY from the target object")
    
    # Draw natural language description (left side, below task prompt)
    desc_start_y = task_prompt_y + 130  # Start below the task prompt box
    draw.text((10, desc_start_y), "Scene Description:", fill='purple', font=title_font)
    desc_y = desc_start_y + 30
    
    for line in desc_lines[:3]:  # Show top 3 descriptions
        draw.text((20, desc_y), line, fill='darkblue', font=small_font)
        desc_y += 25
    
    y_offset = desc_y + 20
    
    # Two-column layout for concepts
    col1_x = 20
    col2_x = width // 2 + 10
    col_width = width // 2 - 30
    
    # Binary relationships in left column
    binary_relationships = {}
    continuous_relationships = {}
    
    for rel_name, value in relationships.items():
        if isinstance(value, (int, float)):
            if value in [0.0, 1.0] and rel_name.startswith(('on_', 'above_', 'near_', 'aligned_', 'touching_')):
                binary_relationships[rel_name] = int(value)
            else:
                continuous_relationships[rel_name] = value
    
    # Left column: Binary relationships
    draw.text((col1_x, y_offset), "Binary Relations (✓/✗):", fill='green', font=title_font)
    y1_offset = y_offset + 30
    
    # Function to convert technical names to natural language descriptions
    def humanize_concept_name(rel_name):
        """Convert technical concept names to natural language descriptions"""
        name_map = {
            'on_T_P': 'Target on table (contact + vertical gap < 5 cm; target above table)',
            'on_T_R': 'Target on region (contact + vertical gap < 5 cm; target above region)',
            'above_T_P': 'Target above table without contact (Δz > 5 cm, lateral offset < 20 cm)',
            'above_T_R': 'Target above region without contact (Δz > 5 cm, lateral offset < 20 cm)',
            'above_G_T': 'Gripper above target without contact (Δz > 2 cm, lateral offset < 10 cm)',
            'above_G_R': 'Gripper above region without contact (Δz > 2 cm, lateral offset < 10 cm)',
            'near_G_T': 'Gripper near target (distance < max((size_gripper+size_target)/2, 0.10 m))',
            'near_T_R': 'Target near region (distance < max((size_target+size_region)/2, 0.10 m))',
            'near_G_R': 'Gripper near region (distance < max((size_gripper+size_region)/2, 0.10 m))',
            'aligned_G_T': 'Gripper aligned to target in XY (horizontal distance < 0.05 m)',
            'aligned_T_R': 'Target aligned to region in XY (horizontal distance < 0.05 m)',
            'touching_G_T': 'Gripper touching target (MuJoCo contact)',
            'touching_T_P': 'Target touching table (MuJoCo contact)',
            'touching_T_R': 'Target touching region (MuJoCo contact)',
            'touching_G_R': 'Gripper touching region (MuJoCo contact)',
            'dist_T_P': 'Euclidean 3D distance between target and table',
            'dist_T_R': 'Euclidean 3D distance between target and region',
            'height_T': 'World Z coordinate (height) of target',
            'height_P': 'World Z coordinate (height) of table',
            'height_R': 'World Z coordinate (height) of region',
            'gripper_x': 'World X of gripper',
            'gripper_y': 'World Y of gripper',
            'gripper_z': 'World Z of gripper',
            'normalized_dist_G_T': 'Distance(gripper,target) / near-threshold(gripper,target)'
        }
        return name_map.get(rel_name, rel_name.replace('_', ' ').title())
    
    # Show fixed set of binary relationships for stable viewing
    # Use sorted order to keep concepts in same position across frames
    binary_items = sorted(binary_relationships.items())
    # Show first 10 to avoid overcrowding, but keep consistent across frames
    if len(binary_items) > 10:
        binary_items = binary_items[:10]
    
    for rel_name, value in binary_items:
        color = 'darkgreen' if value == 1 else 'darkred'
        symbol = '✓' if value == 1 else '✗'
        # Convert to human-readable name
        human_name = humanize_concept_name(rel_name)
        text = f"{symbol} {human_name}"
        draw.text((col1_x + 10, y1_offset), text, fill=color, font=small_font)
        y1_offset += 22
    
    # Right column: Continuous values
    draw.text((col2_x, y_offset), "Continuous Values:", fill='orange', font=title_font)
    y2_offset = y_offset + 30
    
    # Show fixed set of continuous values for stable viewing
    # Use sorted order to keep concepts in same position across frames
    cont_items = sorted(continuous_relationships.items())
    # Show first 10 to avoid overcrowding, but keep consistent across frames
    if len(cont_items) > 10:
        cont_items = cont_items[:10]
    
    for rel_name, value in cont_items:
        # Convert to human-readable name
        human_name = humanize_concept_name(rel_name)
        if isinstance(value, float):
            text = f"{human_name}: {value:.3f}"
        else:
            text = f"{human_name}: {value}"
        draw.text((col2_x + 10, y2_offset), text, fill='black', font=small_font)
        y2_offset += 22
    
    # Summary at bottom
    summary_y = height - 100
    draw.rectangle([10, summary_y, width-10, height-10], outline='gray', width=2)
    draw.text((20, summary_y + 10), "Summary:", fill='purple', font=title_font)
    
    active_binary = sum(1 for v in binary_relationships.values() if v == 1)
    total_binary = len(binary_relationships)
    
    summary_text = [
        f"Active relations: {active_binary}/{total_binary} | Total features: {len(concepts.get('concept_vector', []))} | Objects: {len([v for v in key_objects.values() if v is not None])}/4"
    ]
    
    for i, line in enumerate(summary_text):
        draw.text((20, summary_y + 35 + i*20), line, fill='black', font=small_font)
    
    return img


def create_concepts_gif(
    states_h5_path: str,
    output_gif_path: str,
    task_description: str = "",
    frame_duration: int = 100,
    frame_size: Tuple[int, int] = (800, 600)
):
    """
    Create a GIF visualization of visual concepts over time.
    
    Args:
        states_h5_path: Path to reconstructed states.h5 file
        output_gif_path: Output path for concepts GIF
        task_description: Task description for context
        frame_duration: Duration per frame in milliseconds (100ms = 10 FPS)
        frame_size: Frame size (width, height)
    """
    print(f"[debug-viz] Loading states from {states_h5_path}")
    
    # Load reconstructed states
    states = load_reconstructed_states(states_h5_path)
    print(f"[debug-viz] Loaded {len(states['robot_joint_pos'])} timesteps")
    
    # Extract concepts for all timesteps
    print(f"[debug-viz] Extracting visual concepts...")
    concepts_timeline = extract_concepts_from_states(states, task_description)
    print(f"[debug-viz] Extracted concepts for {len(concepts_timeline)} timesteps")
    
    # Create visualization frames
    print(f"[debug-viz] Creating visualization frames...")
    frames = []
    
    for timestep, concepts in enumerate(concepts_timeline):
        frame = create_concept_visualization_frame(concepts, timestep, frame_size)
        frames.append(frame)
        
        if timestep % 10 == 0:
            print(f"[debug-viz] Created frame {timestep}/{len(concepts_timeline)}")
    
    # Save as GIF
    print(f"[debug-viz] Saving GIF to {output_gif_path}")
    
    if frames:
        frames[0].save(
            output_gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration,
            loop=0
        )
        
        print(f"[debug-viz] Created concepts GIF: {output_gif_path}")
        print(f"[debug-viz] GIF contains {len(frames)} frames at {1000/frame_duration:.1f} FPS")
    else:
        print(f"[debug-viz] ERROR: No frames created!")


def debug_single_episode(
    states_output_dir: str,
    task_id: int,
    episode_id: int,
    task_description: str = ""
):
    """
    Debug visual concepts for a single episode.
    
    Args:
        states_output_dir: Directory containing reconstructed states
        task_id: Task ID
        episode_id: Episode ID  
        task_description: Task description (will be read from dataset if empty)
    """
    states_dir = Path(states_output_dir) / f"task_{task_id}" / f"episode_{episode_id}"
    states_h5_path = states_dir / "states.h5"
    
    if not states_h5_path.exists():
        print(f"ERROR: States file not found: {states_h5_path}")
        return
    
    # Read task description from the original dataset that was used for reconstruction
    if not task_description:
        # Get task description from the optimized dataset's episode_index.h5
        dataset_dir = Path(states_output_dir).parent / "optimized_trajectory_data"
        if dataset_dir.exists():
            try:
                episode_index_path = dataset_dir / "episode_index.h5"
                if episode_index_path.exists():
                    with h5py.File(episode_index_path, 'r') as f:
                        # Load all episodes and find the matching one
                        task_ids = f['task_id'][:]
                        episode_ids = f['episode_id'][:]
                        task_descriptions = f['task_description'][:]
                        
                        # Find the episode that matches our task_id and episode_id
                        for i, (t_id, e_id) in enumerate(zip(task_ids, episode_ids)):
                            if int(t_id) == task_id and int(e_id) == episode_id:
                                raw_desc = task_descriptions[i]
                                if hasattr(raw_desc, 'decode'):
                                    task_description = raw_desc.decode('utf-8')
                                else:
                                    task_description = str(raw_desc)
                                print(f"[debug-viz] Found task description: '{task_description}'")
                                break
                        
                        if not task_description:
                            print(f"[debug-viz] WARNING: Could not find episode task_{task_id}/episode_{episode_id} in dataset")
                            task_description = f"Task {task_id} Episode {episode_id} (description not found)"
                            
                else:
                    print(f"[debug-viz] WARNING: Episode index file not found: {episode_index_path}")
                    task_description = f"Task {task_id} Episode {episode_id} (no episode index)"
                    
            except Exception as e:
                print(f"[debug-viz] ERROR reading task description: {e}")
                task_description = f"Task {task_id} Episode {episode_id} (read error)"
        else:
            print(f"[debug-viz] WARNING: Optimized dataset not found: {dataset_dir}")
            task_description = f"Task {task_id} Episode {episode_id} (dataset not found)"
    
    # Output paths
    output_gif_path = states_dir / "concepts_debug.gif"
    
    print(f"[debug-viz] Debugging episode task_{task_id}/episode_{episode_id}")
    print(f"[debug-viz] States: {states_h5_path}")
    print(f"[debug-viz] Output: {output_gif_path}")
    
    # Create concepts GIF
    create_concepts_gif(
        str(states_h5_path),
        str(output_gif_path),
        task_description=task_description,
        frame_duration=100,  # 10 FPS to match typical trajectory GIFs
        frame_size=(1000, 800)
    )
    
    # Print some debugging info
    print(f"\n[debug-viz] ===== EPISODE DEBUG COMPLETE =====")
    print(f"Generated files:")
    print(f"  - Concepts GIF: {output_gif_path}")
    
    # Check if trajectory GIF exists for comparison
    trajectory_gif = states_dir / "trajectory.gif"
    if trajectory_gif.exists():
        print(f"  - Trajectory GIF: {trajectory_gif}")
        print(f"\nCompare the two GIFs side by side to validate concept extraction!")
    else:
        print(f"  - No trajectory GIF found at: {trajectory_gif}")
        
    # Load and print first timestep concepts for inspection
    print(f"\n[debug-viz] ===== FIRST TIMESTEP CONCEPTS =====")
    states = load_reconstructed_states(str(states_h5_path))
    concepts_timeline = extract_concepts_from_states(states, task_description)
    
    if concepts_timeline:
        first_concepts = concepts_timeline[0]
        print(f"Key objects: {first_concepts['key_objects']}")
        print(f"Binary relationships:")
        for rel_name, value in sorted(first_concepts['relationships'].items()):
            if value in [0.0, 1.0] and rel_name.startswith(('on_', 'above_', 'near_', 'aligned_', 'touching_')):
                symbol = '✓' if value == 1.0 else '✗'
                print(f"  {symbol} {rel_name}: {int(value)}")


def main():
    """
    Main function - runs with settings from configuration section above.
    """
    print(f"[debug-viz] ===== VISUAL CONCEPTS DEBUG =====")
    print(f"[debug-viz] States directory: {STATES_OUTPUT_DIR}")
    print(f"[debug-viz] Task ID: {TASK_ID}")
    print(f"[debug-viz] Episode ID: {EPISODE_ID}")
    print(f"[debug-viz] Frame duration: {FRAME_DURATION}ms (matches reconstruction script)")
    print(f"[debug-viz] Frame size: {FRAME_SIZE}")
    print(f"")
    
    debug_single_episode(
        STATES_OUTPUT_DIR,
        TASK_ID,
        EPISODE_ID,
        TASK_DESCRIPTION
    )


if __name__ == "__main__":
    main()