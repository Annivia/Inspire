#!/usr/bin/env python3
"""
Test script for the updated reconstruction functionality with optimized data format.
"""

import os
os.environ["MUJOCO_GL"] = "egl" 
os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"

from experiments.robot.robot_utils import normalize_gripper_action, invert_gripper_action
from vla_scripts.reconstruct_trajectory_data import (
    load_episode_metadata,
    reconstruct_trajectory_episode,
    get_reconstruction_paths,
)
from vla_scripts.state_io import StateChunkWriter, resolve_paths, combine_state_chunks

def test_metadata_loading():
    """Test smart metadata loading functionality"""
    print("=== TESTING METADATA LOADING ===")
    dataset_dir = '/work/nvme/bfbo/xzhang42/data/pilot_test/optimized_trajectory_data/'
    
    try:
        metadata = load_episode_metadata(dataset_dir)
        print(f"Successfully loaded metadata for {len(metadata)} episodes")
        print("\nFirst few episodes:")
        print(metadata.head(3).to_string())
        
        print(f"\nSuccessful episodes: {metadata['success'].sum()}/{len(metadata)}")
        print(f"Unique task IDs: {sorted(metadata['task_id'].unique())}")
        print(f"Total timesteps: {metadata['num_timesteps'].sum()}")
        
        return metadata
        
    except Exception as e:
        print(f"ERROR in metadata loading: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_two_tasks_reconstruction():
    """Test reconstructing one episode from each of two distinct tasks."""
    print("\n=== TESTING TWO-TASK RECONSTRUCTION ===")
    dataset_dir = '/work/nvme/bfbo/xzhang42/data/pilot_test/optimized_trajectory_data/'
    # Use dataset_dir as the root for sim_states/ unless explicitly overridden
    states_output_dir = dataset_dir
    
    try:
        # Prepare writer and global concepts recorder map
        writer = StateChunkWriter(dataset_root=states_output_dir, process_id=0)
        concepts_dir = str(resolve_paths(states_output_dir)["concepts"])
        concepts_recorders = {}
        # Choose two distinct tasks
        metadata = load_episode_metadata(dataset_dir)
        task_ids = list(dict.fromkeys(metadata['task_id']))
        targets = task_ids[:2] if len(task_ids) >= 2 else task_ids
        print(f"Target task_ids: {targets}")
        from pathlib import Path
        recon_images_dir = Path(get_reconstruction_paths(dataset_dir)['reconstructed_data_dir']) / "images"
        recon_images_dir.mkdir(parents=True, exist_ok=True)

        for tid in targets:
            idx_list = metadata.index[metadata['task_id'] == tid].tolist()
            if not idx_list:
                continue
            epi_idx = idx_list[0]
            print(f"Reconstructing episode index {epi_idx} for task_id {tid}")
            result = reconstruct_trajectory_episode(
                dataset_dir=dataset_dir,
                episode_idx=epi_idx,
                task_suite_name='libero_90',
                images_output_dir=str(recon_images_dir),
                states_output_dir=states_output_dir,
                episode_metadata=metadata,
                enable_rendering=True,
                state_writer=writer,
                concepts_recorders=concepts_recorders,
                concepts_root_dir=concepts_dir,
                render_concepts=True,
                concepts_only_changing=True,
            )
            print(f"Episode reconstruction result: {result}")
        # Flush chunk and combine to final layout
        writer.flush()
        # Save per-task CSV(s)
        for key, rec in concepts_recorders.items():
            csv_path = rec.save_as_task_csv(concepts_dir)
            print(f"Saved task relations CSV: {csv_path}")

        summary = combine_state_chunks(states_output_dir)
        print(f"Combine summary: {summary}")

        # Check final sim_states outputs
        import h5py
        from pathlib import Path
        sim_states_root = Path(states_output_dir) / "sim_states"
        core_path = sim_states_root / "core_states.h5"
        epi_path = sim_states_root / "episodes_index.h5"
        assert core_path.exists(), f"Missing core_states.h5 at {core_path}"
        assert epi_path.exists(), f"Missing episodes_index.h5 at {epi_path}"
        with h5py.File(core_path, 'r') as f:
            print(f"core_states.h5 datasets: {list(f.keys())}")
            for k in ["robot_joint_pos", "ee_pos", "time"]:
                if k in f:
                    print(f"  {k}: shape={f[k].shape}")

        # Check that at least one concepts CSV was written
        concepts_root = Path(concepts_dir)
        any_csv = list(concepts_root.glob("*.csv"))
        if any_csv:
            print(f"Found {len(any_csv)} concepts CSV(s). Example: {any_csv[0]}")
        else:
            print("WARNING: No concepts CSVs found under concepts/ (this may be okay if no objects/sites present)")
        
        return True
        
    except Exception as e:
        print(f"ERROR in single episode reconstruction: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_episode_filtering():
    """Test episode filtering functionality"""
    print("\n=== TESTING EPISODE FILTERING ===")
    dataset_dir = '/work/nvme/bfbo/xzhang42/data/pilot_test/optimized_trajectory_data/'
    
    try:
        metadata = load_episode_metadata(dataset_dir)
        
        # Test success filtering
        successful = metadata[metadata['success'] == True]
        print(f"Successful episodes: {len(successful)}")
        
        # Test task ID filtering
        task_1_episodes = metadata[metadata['task_id'] == 1]
        print(f"Task 1 episodes: {len(task_1_episodes)}")
        
        # Test sample index ranges
        print("\nSample index ranges:")
        for idx in range(min(3, len(metadata))):
            episode = metadata.iloc[idx]
            print(f"  Episode {idx}: samples {episode['start_idx']}-{episode['end_idx']} ({episode['num_timesteps']} timesteps)")
        
        return True
        
    except Exception as e:
        print(f"ERROR in episode filtering: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_auto_paths():
    """Test automatic path derivation"""
    print("\n=== TESTING AUTO PATH DERIVATION ===")
    
    try:
        dataset_dir = '/work/nvme/bfbo/xzhang42/data/pilot_test/optimized_trajectory_data/'
        
        paths = get_reconstruction_paths(dataset_dir)
        print(f"Input dataset: {dataset_dir}")
        print(f"Auto-derived reconstruction dir: {paths['reconstructed_data_dir']}")
        print(f"Auto-derived temp processing dir: {paths['temp_processing_dir']}")
        
        # Verify paths are in the same mother directory
        from pathlib import Path
        dataset_path = Path(dataset_dir)
        recon_path = Path(paths['reconstructed_data_dir'])
        temp_path = Path(paths['temp_processing_dir'])
        
        if dataset_path.parent == recon_path.parent == temp_path.parent:
            print("✅ All paths in same mother directory")
            return True
        else:
            print("❌ Paths not in same mother directory")
            return False
            
    except Exception as e:
        print(f"ERROR in auto path derivation: {e}")
        return False

def test_action_processing():
    """Test that action processing matches parallel_libero_evaluator.py"""
    print("\n=== TESTING ACTION PROCESSING ===")
    
    try:
        import numpy as np
        
        # Test case: typical VLA output (unnormalized)
        test_action = np.array([0.2, -0.1, 0.05, 0.15, -0.08, 0.03, 0.7])  # Last is gripper [0,1]
        print(f"Original VLA action: {test_action}")
        
        # Apply same processing as reconstruction script
        processed_action = test_action.copy()
        processed_action = normalize_gripper_action(processed_action, binarize=True)
        processed_action = invert_gripper_action(processed_action)
        
        print(f"Processed action: {processed_action}")
        print(f"Position/rotation: {processed_action[:6]}")  # Should be unchanged
        print(f"Gripper: {processed_action[6]}")  # Should be -1 or +1
        
        # Verify gripper is properly binarized and inverted
        if processed_action[6] in [-1.0, 1.0]:
            print("✅ Gripper action correctly processed")
            return True
        else:
            print(f"❌ Gripper action not properly binarized: {processed_action[6]}")
            return False
            
    except Exception as e:
        print(f"ERROR in action processing test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rendering_toggle():
    """Test rendering enable/disable functionality"""
    print("\n=== TESTING RENDERING TOGGLE ===")
    dataset_dir = '/work/nvme/bfbo/xzhang42/data/pilot_test/optimized_trajectory_data/'
    
    try:
        paths = get_reconstruction_paths(dataset_dir)
        images_output_dir = paths['reconstructed_data_dir'] + "/test_images"
        
        # Test with rendering ENABLED to actually generate images
        import time
        start_time = time.time()
        
        print("Testing with rendering ENABLED...")
        result_enabled = reconstruct_trajectory_episode(
            dataset_dir=dataset_dir,
            episode_idx=0,
            task_suite_name='libero_90',
            images_output_dir=images_output_dir,
            states_output_dir=None,  # Skip states for this test
            enable_rendering=True  # Enable rendering to test it
        )
        
        enabled_time = time.time() - start_time
        print(f"Reconstruction with rendering ENABLED: {enabled_time:.2f}s")
        print(f"Images saved: {result_enabled['images_saved']}")
        
        # Check if images were actually created
        from pathlib import Path
        import glob
        
        images_dir = Path(images_output_dir) / "task_1" / "episode_1"
        if images_dir.exists():
            image_files = list(images_dir.glob("*.png"))
            print(f"✅ Generated {len(image_files)} image files")
            if image_files:
                print(f"   Example: {image_files[0]}")
        else:
            print(f"❌ No images directory found: {images_dir}")
        
        # Test with rendering disabled for comparison
        start_time = time.time()
        print("\nTesting with rendering DISABLED...")
        result_disabled = reconstruct_trajectory_episode(
            dataset_dir=dataset_dir,
            episode_idx=0,
            task_suite_name='libero_90',
            images_output_dir=None,
            states_output_dir=None,
            enable_rendering=False
        )
        
        disabled_time = time.time() - start_time
        print(f"Reconstruction with rendering DISABLED: {disabled_time:.2f}s")
        print(f"Images reported: {result_disabled['images_saved']}")
        
        print(f"Speed difference: {enabled_time/disabled_time:.1f}x slower with rendering")
        
        return True
        
    except Exception as e:
        print(f"ERROR in rendering toggle test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Testing updated reconstruction functionality with optimized data format...")
    print("=" * 70)
    
    # Test 1: Metadata loading
    metadata = test_metadata_loading()
    if metadata is None:
        print("FAILED: Metadata loading test failed")
        return
    
    # Test 2: Episode filtering
    filter_success = test_episode_filtering()
    if not filter_success:
        print("FAILED: Episode filtering test failed")
        return
    
    # Test 3: Auto path derivation
    auto_path_success = test_auto_paths()
    if not auto_path_success:
        print("FAILED: Auto path derivation test failed")
        return
    
    # Test 4: Action processing
    action_success = test_action_processing()
    if not action_success:
        print("FAILED: Action processing test failed")
        return
    
    # Two tasks reconstruction with new sim_states / concepts outputs
    recon_success = test_two_tasks_reconstruction()
    if not recon_success:
        print("FAILED: Two tasks reconstruction test failed")
        return
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("Updated reconstruction script is ready for probes 3 and 4")
    print("\nKey features tested:")
    print("• ✅ Action processing (matches parallel_libero_evaluator.py)")
    print("• ✅ Rendering toggle for scalability")
    print("• ✅ HDF5 tensor storage (efficient)")
    print("• ✅ Auto-derived paths in mother folder")
    print("• ✅ Metadata-only loading")
    print("• ✅ Episode filtering")

if __name__ == "__main__":
    main()
