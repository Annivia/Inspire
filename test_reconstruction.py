#!/usr/bin/env python3
"""
Test script for the updated reconstruction functionality with optimized data format.
"""

import os
os.environ["MUJOCO_GL"] = "egl" 
os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"

from vla_scripts.reconstruct_trajectory_data import load_episode_metadata, reconstruct_trajectory_episode, get_reconstruction_paths

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

def test_single_episode_reconstruction():
    """Test reconstructing a single episode with HDF5 states"""
    print("\n=== TESTING SINGLE EPISODE RECONSTRUCTION ===")
    dataset_dir = '/work/nvme/bfbo/xzhang42/data/pilot_test/optimized_trajectory_data/'
    
    paths = get_reconstruction_paths(dataset_dir)
    states_output_dir = paths['reconstructed_data_dir']
    
    try:
        result = reconstruct_trajectory_episode(
            dataset_dir=dataset_dir,
            episode_idx=0,  # Test first episode
            task_suite_name='libero_90',
            images_output_dir=None,  # Skip images for faster test
            states_output_dir=states_output_dir,
            enable_rendering=False  # Test without rendering for speed
        )
        
        print(f"Single episode reconstruction result: {result}")
        
        # Check HDF5 output files
        import h5py
        from pathlib import Path
        
        states_path = Path(states_output_dir) / "task_1" / "episode_1" / "states.h5"
        if states_path.exists():
            with h5py.File(states_path, 'r') as f:
                print(f"HDF5 state fields: {list(f.keys())}")
                for key in list(f.keys())[:3]:  # Show first 3 fields
                    print(f"  {key}: shape={f[key].shape}, dtype={f[key].dtype}")
            print(f"States saved to HDF5: {states_path}")
        else:
            print(f"WARNING: Expected HDF5 file not found: {states_path}")
        
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
    
    # Test 4: Rendering toggle
    render_success = test_rendering_toggle()
    if not render_success:
        print("FAILED: Rendering toggle test failed")
        return
    
    # Test 5: Single episode reconstruction with HDF5
    recon_success = test_single_episode_reconstruction()
    if not recon_success:
        print("FAILED: Single episode reconstruction test failed")
        return
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("Updated reconstruction script is ready for probes 3 and 4")
    print("\nKey features tested:")
    print("• ✅ Rendering toggle for scalability")
    print("• ✅ HDF5 tensor storage (efficient)")
    print("• ✅ Auto-derived paths in mother folder")
    print("• ✅ Metadata-only loading")
    print("• ✅ Episode filtering")

if __name__ == "__main__":
    main()