#!/usr/bin/env python3
"""
Test script for the complete image reconstruction pipeline.

This script tests:
1. Trajectory data collection with minimal image reconstruction clues
2. Image generation from the dataset
3. Cross-referenced data loading for linear probing
"""

import os
import sys
import subprocess
from pathlib import Path
import tempfile
import shutil

# Set up paths
sys.path.append('/u/xzhang42/Inspire')
sys.path.append('/u/xzhang42/Inspire/LIBERO')
sys.path.append('/u/xzhang42/Inspire/vq_bet_official')

def test_image_reconstruction_pipeline():
    """Test the complete image reconstruction pipeline."""
    print("=" * 60)
    print("TESTING IMAGE RECONSTRUCTION PIPELINE")
    print("=" * 60)
    
    # Define paths (use fast memory)
    base_dir = Path("/work/nvme/bfbo/xzhang42/Inspire")
    test_data_dir = base_dir / "test_image_pipeline"
    trajectory_data_path = test_data_dir / "trajectory_data_libero_90.h5"
    images_dir = test_data_dir / "generated_images"
    
    print(f"Test directory: {test_data_dir}")
    print(f"Trajectory data: {trajectory_data_path}")
    print(f"Images directory: {images_dir}")
    
    # Create test directory
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Check if we have existing trajectory data, if not run collection
        if not trajectory_data_path.exists():
            print("\n" + "="*40)
            print("STEP 1: COLLECTING TRAJECTORY DATA")
            print("="*40)
            
            print("No existing trajectory data found. Running data collection...")
            
            # Run trajectory data collection with image reconstruction clues
            collection_cmd = [
                "python", "/u/xzhang42/Inspire/vla_scripts/parallel_libero_evaluator.py",
                "--pretrained-checkpoint", "/work/nvme/bfbo/xzhang42/Inspire/runs/minivla-libero-90",
                "--task-suite-name", "libero_90",
                "--num-gpus", "1",
                "--num-processes", "1", 
                "--num-trails-per-task", "2",  # Just 2 episodes for testing
                "--steps", "50000",
                "--collect-trajectory-data",
                "--trajectory-data-save-path", str(test_data_dir),
                "--max-total-trajectories", "4",  # Very small for testing
                "--save-root", str(test_data_dir / "results")
            ]
            
            print("Running trajectory collection...")
            print(" ".join(collection_cmd))
            
            result = subprocess.run(collection_cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                print(f"❌ FAILED: Trajectory collection failed")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return False
            
            print("✅ Trajectory collection completed")
            
            # Combine trajectory files if multiple processes were used
            combine_cmd = [
                "python", "/u/xzhang42/Inspire/vla_scripts/combine_trajectory_files.py",
                "--data-dir", str(test_data_dir),
                "--task-suite-name", "libero_90"
            ]
            
            print("Combining trajectory files...")
            result = subprocess.run(combine_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ FAILED: File combination failed")
                print(f"STDERR: {result.stderr}")
                return False
            
            print("✅ Trajectory files combined")
        
        else:
            print("✅ Using existing trajectory data")
        
        # Verify trajectory data exists and has image reconstruction clues
        print(f"\n" + "="*40)
        print("STEP 2: VERIFYING TRAJECTORY DATA") 
        print("="*40)
        
        if not trajectory_data_path.exists():
            print(f"❌ FAILED: Trajectory data file not found: {trajectory_data_path}")
            return False
        
        # Check for image reconstruction clues
        try:
            import h5py
            with h5py.File(trajectory_data_path, 'r') as f:
                print(f"Dataset contains {len(f.keys())} task groups")
                
                # Check first episode for image reconstruction clues
                found_clues = False
                for task_name in f.keys():
                    if task_name.startswith('task_'):
                        task_group = f[task_name]
                        for episode_name in task_group.keys():
                            if episode_name.startswith('episode_'):
                                episode_group = task_group[episode_name]
                                metadata = episode_group['metadata']
                                
                                if 'img_task_id' in metadata.attrs:
                                    print(f"✅ Found image reconstruction clues in {task_name}/{episode_name}")
                                    print(f"   - img_task_id: {metadata.attrs['img_task_id']}")
                                    print(f"   - img_episode_id: {metadata.attrs['img_episode_id']}")
                                    print(f"   - img_env_seed: {metadata.attrs['img_env_seed']}")
                                    found_clues = True
                                    break
                        if found_clues:
                            break
                
                if not found_clues:
                    print(f"❌ FAILED: No image reconstruction clues found in dataset")
                    return False
                    
        except Exception as e:
            print(f"❌ FAILED: Error checking trajectory data: {e}")
            return False
        
        print("✅ Trajectory data verified with image reconstruction clues")
        
        # Step 3: Generate images from trajectory data
        print(f"\n" + "="*40)
        print("STEP 3: GENERATING IMAGES")
        print("="*40)
        
        generation_cmd = [
            "python", "/u/xzhang42/Inspire/vla_scripts/generate_images_from_trajectory_data.py",
            str(trajectory_data_path),
            "--output-dir", str(images_dir),
            "--task-suite-name", "libero_90"
        ]
        
        print("Generating images from trajectory data...")
        print(" ".join(generation_cmd))
        
        result = subprocess.run(generation_cmd, capture_output=True, text=True, timeout=1200)
        
        if result.returncode != 0:
            print(f"❌ FAILED: Image generation failed")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
        
        print("✅ Image generation completed")
        print(f"STDOUT: {result.stdout}")
        
        # Step 4: Verify generated images
        print(f"\n" + "="*40)
        print("STEP 4: VERIFYING GENERATED IMAGES")
        print("="*40)
        
        if not images_dir.exists():
            print(f"❌ FAILED: Images directory not found: {images_dir}")
            return False
        
        # Count generated images
        total_images = 0
        for task_dir in images_dir.iterdir():
            if task_dir.is_dir() and task_dir.name.startswith('task_'):
                for episode_dir in task_dir.iterdir():
                    if episode_dir.is_dir() and episode_dir.name.startswith('episode_'):
                        images_count = len(list(episode_dir.glob('*.png')))
                        total_images += images_count
                        print(f"✅ {task_dir.name}/{episode_dir.name}: {images_count} images")
        
        if total_images == 0:
            print(f"❌ FAILED: No images generated")
            return False
        
        print(f"✅ Total images generated: {total_images}")
        
        # Step 5: Test cross-referenced data loading
        print(f"\n" + "="*40) 
        print("STEP 5: TESTING CROSS-REFERENCED DATA LOADING")
        print("="*40)
        
        try:
            sys.path.append('/u/xzhang42/Inspire/vla_scripts')
            from load_trajectory_data_with_images import TrajectoryDataWithImages
            
            # Initialize data loader
            data_loader = TrajectoryDataWithImages(str(trajectory_data_path), str(images_dir))
            
            print("✅ TrajectoryDataWithImages initialized successfully")
            
            # Test matched data for probing
            hidden_states, vision_features, actions, images, metadata = data_loader.get_matched_data_for_probing(
                layer_idx=0,
                generation_step=0,
                include_images=True,
                successful_only=False  # Include all episodes for testing
            )
            
            print(f"✅ Matched data extracted:")
            print(f"   - Hidden states: {hidden_states.shape}")
            print(f"   - Vision features: {vision_features.shape}")  
            print(f"   - Actions: {actions.shape}")
            print(f"   - Images: {images.shape if images is not None else 'None'}")
            print(f"   - Metadata samples: {len(metadata)}")
            
            # Verify data consistency
            if images is not None:
                if (len(hidden_states) == len(vision_features) == len(actions) == len(images) == len(metadata)):
                    print("✅ Perfect data alignment confirmed")
                else:
                    print("❌ FAILED: Data alignment mismatch")
                    print(f"   Lengths: HS={len(hidden_states)}, VF={len(vision_features)}, A={len(actions)}, I={len(images)}, M={len(metadata)}")
                    return False
            
        except Exception as e:
            print(f"❌ FAILED: Cross-referenced data loading failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("✅ Cross-referenced data loading successful")
        
        # Final success
        print(f"\n" + "="*60)
        print("🎉 IMAGE RECONSTRUCTION PIPELINE TEST PASSED! 🎉")
        print("="*60)
        print("✅ Trajectory data collection with image reconstruction clues")
        print("✅ Image generation from minimal reconstruction clues") 
        print("✅ Cross-referenced data loading for linear probing")
        print(f"✅ Total images generated: {total_images}")
        print(f"✅ All data perfectly aligned and ready for analysis")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the complete pipeline test."""
    success = test_image_reconstruction_pipeline()
    
    if success:
        print("\n✅ All tests passed! Pipeline is ready for production use.")
        sys.exit(0)
    else:
        print("\n❌ Tests failed! Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()