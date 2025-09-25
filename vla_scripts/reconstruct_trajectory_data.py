#!/usr/bin/env python3
"""
Reconstruct trajectory data using stored actions from optimized trajectory dataset.

This script replays VLA trajectories using the exact actions stored in the optimized format,
allowing perfect reconstruction of both images and simulator states that correspond to the
collected hidden states and vision features for probes 3 and 4.

Supports smart metadata-only loading for efficient episode selection.

Defaults simplified for local runs:
- Dataset directory is set by DEFAULT_DATASET_DIR below (edit in-place as needed).
- Output paths are auto-derived; CLI path overrides are ignored.
- Concepts rendering is enabled by default.
"""

import os
os.environ["MUJOCO_GL"] = "egl" 
os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"

import argparse
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from PIL import ImageDraw, ImageFont
import sys
import json
from typing import Dict, List, Optional, Tuple
sys.path.append('/u/xzhang42/Inspire')
sys.path.append('/u/xzhang42/Inspire/LIBERO')
sys.path.append('/u/xzhang42/Inspire/vq_bet_official')

# Default dataset directory (edit by hand if needed)
DEFAULT_DATASET_DIR = '/work/nvme/bfbo/xzhang42/data/pilot_test/optimized_trajectory_data/'

from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env, get_libero_image
from experiments.robot.robot_utils import normalize_gripper_action, invert_gripper_action
import threading
import time
from collections import defaultdict

# Import shared visual concepts infrastructure
from vla_scripts.visual_concepts_extractor import (
    enumerate_concept_keys,
    CSVRelationsRecorder,
    evaluate_concepts,
    collect_scene_predicates,
    evaluate_site_methods,
    evaluate_site_geometry_methods,
    contact_obj_with_robot,
    build_contact_index,
    contact_between_bodies,
    get_env_inventory,
    get_site_parent_map,
    get_goal_predicates,
    derive_involved_from_goals,
    expand_overlap_objects,
    evaluate_concept_expressions,
    select_task_concepts,
)
from vla_scripts.state_io import StateChunkWriter, resolve_paths, combine_state_chunks


def _sanitize(s: str) -> str:
    import re
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]", "", s)
    return s or "task"

def _get_task_identifiers_from_env(control_or_bddl_env) -> Tuple[str, str]:
    """Robust task identifiers following LIBERO conventions.

    Priority:
    1) BDDL env class name (lowercased) — enforced to match parsed problem_name
    2) ControlEnv.problem_name / language_instruction if available
    3) parsed_problem fields as fallback
    """
    # Unwrap ControlEnv if needed
    bddl_env = control_or_bddl_env.env if hasattr(control_or_bddl_env, "env") else control_or_bddl_env
    # Preferred: class name of BDDL env (matched by _assert_problem_name)
    try:
        problem_name = bddl_env.__class__.__name__.lower()
    except Exception:
        problem_name = None

    # Language instruction
    language = None
    try:
        language = getattr(control_or_bddl_env, "language_instruction", None)
    except Exception:
        language = None

    # Fallbacks from parsed_problem if available
    try:
        parsed = getattr(bddl_env, "parsed_problem", {}) or {}
        if not problem_name:
            problem_name = str(parsed.get("problem_name", ""))
        if not language:
            li = parsed.get("language_instruction", [])
            language = " ".join(li) if isinstance(li, list) else str(li)
    except Exception:
        pass

    return str(problem_name or ""), str(language or "")


def load_episode_metadata(dataset_dir: str) -> pd.DataFrame:
    """
    Smart metadata-only loading from optimized trajectory dataset.
    
    Args:
        dataset_dir: Path to optimized trajectory data directory
        
    Returns:
        DataFrame with episode metadata including reconstruction clues
    """
    dataset_dir = Path(dataset_dir)
    episode_index_path = dataset_dir / "episode_index.h5"
    
    if not episode_index_path.exists():
        raise FileNotFoundError(f"Episode index not found: {episode_index_path}")
    
    print(f"[debug-metadata] Loading episode metadata from {episode_index_path}")
    
    # Load episode metadata efficiently
    episode_data = {}
    with h5py.File(episode_index_path, 'r') as f:
        for key in f.keys():
            dataset = f[key]
            if dataset.dtype.kind == 'S':  # String data
                episode_data[key] = [item.decode('utf-8') if hasattr(item, 'decode') else str(item) 
                                   for item in dataset[:]]
            else:
                episode_data[key] = dataset[:].tolist()
    
    episode_df = pd.DataFrame(episode_data)
    print(f"[debug-metadata] Loaded {len(episode_df)} episodes")
    
    return episode_df


class ParallelReconstructionCollector:
    """
    Parallel reconstruction collector for efficient state reconstruction.
    Follows same pattern as OptimizedTrajectoryDataCollector.
    """
    
    def __init__(self, 
                 save_dir: str,
                 process_id: int = 0,
                 temp_dir: Optional[str] = None):
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.process_id = process_id
        
        # Setup temporary processing directory
        if temp_dir:
            self.temp_dir = Path(temp_dir) / f"reconstruction_process_{process_id}"
        else:
            self.temp_dir = self.save_dir / "temp_reconstruction_processing" / f"reconstruction_process_{process_id}"
        
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory data accumulation for reconstruction states
        self.accumulated_data = {
            'states': defaultdict(list),  # state_field -> list of samples
            'images': [],  # For when rendering is enabled
            'episodes': []  # Episode metadata with indexing info
        }
        
        self.current_sample_count = 0
        self.lock = threading.Lock()
        
        print(f"[PARALLEL_RECONSTRUCTION] Process {process_id} initialized")
        print(f"[PARALLEL_RECONSTRUCTION] Save directory: {self.save_dir}")
        print(f"[PARALLEL_RECONSTRUCTION] Temp directory: {self.temp_dir}")
    
    def save_episode_reconstruction(self,
                                  episode_idx: int,
                                  task_id: int,
                                  episode_id: int,
                                  all_states: List[Dict],
                                  all_images: List[np.ndarray] = None,
                                  task_description: str = "",
                                  success: bool = True):
        """
        Accumulate episode reconstruction data in memory for later batch writing.
        """
        print(f"[PARALLEL_RECONSTRUCTION] Accumulating episode {episode_idx}: task_{task_id}/episode_{episode_id}")
        print(f"[PARALLEL_RECONSTRUCTION] States: {len(all_states)}, Images: {len(all_images) if all_images else 0}")
        
        if len(all_states) == 0:
            print(f"[PARALLEL_RECONSTRUCTION] WARNING: No state data to save!")
            return
        
        with self.lock:
            episode_start_idx = self.current_sample_count
            
            # Process each timestep's state data
            for timestep_idx, state_data in enumerate(all_states):
                
                # Accumulate state fields
                for state_field, state_value in state_data.items():
                    if state_field not in ['error']:  # Skip error strings
                        self.accumulated_data['states'][state_field].append(state_value)
                
                # Accumulate images if provided
                if all_images and timestep_idx < len(all_images):
                    self.accumulated_data['images'].append(all_images[timestep_idx])
                
                self.current_sample_count += 1
            
            episode_end_idx = self.current_sample_count - 1
            
            # Store episode metadata with indexing
            episode_metadata = {
                'episode_idx': episode_idx,
                'task_id': task_id,
                'episode_id': episode_id,
                'success': success,
                'task_description': task_description,
                'num_timesteps': len(all_states),
                'start_idx': episode_start_idx,
                'end_idx': episode_end_idx
            }
            
            self.accumulated_data['episodes'].append(episode_metadata)
            
            print(f"[PARALLEL_RECONSTRUCTION] Accumulated {len(all_states)} samples "
                  f"(total: {self.current_sample_count})")
    
    def save_chunk_to_temp(self):
        """
        Save accumulated reconstruction data to temporary chunk files with HDF5 compression.
        """
        print(f"[PARALLEL_RECONSTRUCTION] Saving accumulated data to temp files...")
        print(f"[PARALLEL_RECONSTRUCTION] Total samples: {self.current_sample_count}")
        
        if self.current_sample_count == 0:
            print(f"[PARALLEL_RECONSTRUCTION] No data to save!")
            return
        
        with self.lock:
            # HDF5 compression settings
            compression_kwargs = {
                'compression': 'gzip',
                'compression_opts': 6,
                'shuffle': True
            }
            
            # Save reconstructed states
            if self.accumulated_data['states']:
                states_path = self.temp_dir / "states_chunk.h5"
                with h5py.File(states_path, 'w') as f:
                    # Save each state field as stacked arrays
                    for state_field, field_data in self.accumulated_data['states'].items():
                        if field_data:
                            try:
                                # Stack to create [samples, ...] arrays
                                if state_field in ['object_names']:  # String arrays
                                    stacked_array = np.array(field_data, dtype='S64')
                                else:
                                    stacked_array = np.stack(field_data, axis=0)
                                
                                f.create_dataset(state_field, data=stacked_array, **compression_kwargs)
                                print(f"[PARALLEL_RECONSTRUCTION] Saved {state_field}: {stacked_array.shape}")
                            except Exception as e:
                                print(f"[RECONSTRUCTION_ERROR] Failed to stack {state_field}: {e}")
            
            # Save images if available
            if self.accumulated_data['images']:
                images_path = self.temp_dir / "images_chunk.h5"
                images_array = np.stack(self.accumulated_data['images'], axis=0)
                with h5py.File(images_path, 'w') as f:
                    f.create_dataset('images', data=images_array, **compression_kwargs)
                print(f"[PARALLEL_RECONSTRUCTION] Saved images: {images_array.shape}")
            
            # Save episode metadata
            episodes_path = self.temp_dir / "episodes_chunk.json"
            with open(episodes_path, 'w') as f:
                json.dump(self.accumulated_data['episodes'], f, indent=2)
            
            # Create processing manifest
            manifest = {
                'process_id': self.process_id,
                'total_samples': self.current_sample_count,
                'total_episodes': len(self.accumulated_data['episodes']),
                'state_fields': list(self.accumulated_data['states'].keys()),
                'has_images': len(self.accumulated_data['images']) > 0,
                'timestamp': time.time()
            }
            
            manifest_path = self.temp_dir / "reconstruction_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            print(f"[PARALLEL_RECONSTRUCTION] Chunk saving complete!")
            print(f"[PARALLEL_RECONSTRUCTION] Temp directory: {self.temp_dir}")


def reconstruct_trajectory_episode(
    dataset_dir: str,
    episode_idx: int,
    task_suite_name: str = "libero_90",
    images_output_dir: str = None,
    states_output_dir: str = None,
    episode_metadata: pd.DataFrame = None,
    enable_rendering: bool = True,
    state_writer: StateChunkWriter = None,
    concepts_recorders: Optional[Dict[str, CSVRelationsRecorder]] = None,
    concepts_root_dir: str = None,
    render_concepts: bool = False,
    concepts_only_changing: bool = True,
):
    """
    Reconstruct a single episode trajectory using stored actions from optimized format.
    
    Args:
        dataset_dir: Path to optimized trajectory data directory
        episode_idx: Episode index in the dataset (0-based)
        task_suite_name: LIBERO task suite name
        images_output_dir: Directory to save reconstructed images (if None, skip images)
        states_output_dir: Directory to save simulator states (if None, skip states)
        episode_metadata: Pre-loaded episode metadata (for efficiency)
        enable_rendering: Whether to enable rendering for image reconstruction (disable for scaling)
    """
    dataset_dir = Path(dataset_dir)
    
    # Load episode metadata if not provided
    if episode_metadata is None:
        episode_metadata = load_episode_metadata(dataset_dir)
    
    if episode_idx >= len(episode_metadata):
        raise ValueError(f"Episode index {episode_idx} out of range (max: {len(episode_metadata)-1})")
    
    episode_info = episode_metadata.iloc[episode_idx]
    
    # Extract episode information
    task_id = int(episode_info['task_id'])
    episode_id = int(episode_info['episode_id'])
    img_task_id = int(episode_info['img_task_id'])
    img_episode_id = int(episode_info['img_episode_id'])
    img_env_seed = int(episode_info['img_env_seed'])
    num_timesteps = int(episode_info['num_timesteps'])
    start_idx = int(episode_info['start_idx'])
    end_idx = int(episode_info['end_idx'])
    task_description = episode_info['task_description']
    
    # print(f"[debug-recon] Reconstructing episode {episode_idx}: task_{task_id}/episode_{episode_id}")
    # print(f"[debug-recon] Data range: samples {start_idx}-{end_idx} ({num_timesteps} timesteps)")
    
    # Load stored actions for this episode
    actions_path = dataset_dir / "actions.h5"
    with h5py.File(actions_path, 'r') as f:
        # Extract actions for this episode using index range
        stored_actions = f['actions'][start_idx:end_idx+1]  # Include end_idx
        # print(f"[debug-recon] Loaded stored actions: {stored_actions.shape}")
        
        # Fix: VQ-BET returns action horizons with shape (N, horizon, action_dim)
        # We only need the current action (first horizon element)

        ## Polina: This was premature filtering
        # if len(stored_actions.shape) == 3 and stored_actions.shape[1] > 1:
        #     print(f"[debug-recon] Action shape before horizon fix: {stored_actions.shape}")
        #     stored_actions = stored_actions[:, 0, :]  # Take first horizon element: (N, horizon, action_dim) -> (N, action_dim)
        #     print(f"[debug-recon] Action shape after horizon fix: {stored_actions.shape}")

        """For timestep > 0:
      - horizon_actions = stored_actions[timestep - 1]  # shape (horizon, 7)
      - For each sub_action in horizon_actions:
        - sub_action = normalize_gripper_action(sub_action, binarize=True)
        - sub_action = invert_gripper_action(sub_action)
        - obs, reward, done, info = env.step(sub_action.tolist())
        - Optionally save image/state for each substep (or just the last substep if you want the same frame count as before)
        """
        
        # # Verify we have the right number of actions
        # print(f"[debug-recon] Final actions shape: {stored_actions.shape}")
        # print(f"[debug-recon] Expected timesteps: {num_timesteps}")
        # print(f"[debug-recon] First few actions: {stored_actions[:3] if len(stored_actions) > 0 else 'None'}")
        
        if stored_actions.shape[0] != num_timesteps:
            print(f"[debug-recon] WARNING: Action count mismatch - expected {num_timesteps}, got {stored_actions.shape[0]}")
            # Adjust to match expected timesteps
            if stored_actions.shape[0] > num_timesteps:
                stored_actions = stored_actions[:num_timesteps]
            else:
                raise ValueError(f"Not enough actions for episode: expected {num_timesteps}, got {stored_actions.shape[0]}")
    
    # Initialize LIBERO environment
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(img_task_id)
    env, _ = get_libero_env(task, "prismatic", resolution=224)
    
    try:
        # CRITICAL: Set up environment exactly as during data collection
        # The order matters: seed AFTER getting env, then reset, then set initial state
        print(f"[debug-recon] Setting environment seed to {img_env_seed}")
        env.seed(img_env_seed)
        
        # Reset environment after seeding
        obs = env.reset()
        print(f"[debug-recon] Environment reset complete")
        
        # Set initial state if not libero_object (AFTER reset)
        if task_suite_name != 'libero_object':
            initial_states = task_suite.get_task_init_states(img_task_id)
            obs = env.set_init_state(initial_states[img_episode_id])
            print(f"[debug-recon] Set initial state for episode {img_episode_id}")
        else:
            print(f"[debug-recon] Using default initial state for libero_object")
        
        print(f"[debug-recon] Environment initialized - Task: {task_description}")
        print(f"[debug-recon] Reconstruction clues - task:{img_task_id}, episode:{img_episode_id}, seed:{img_env_seed}")

        # Contact setup introspection prints (no fallbacks)
        try:
            bddl_env = env.env if hasattr(env, "env") else env
            sim = getattr(bddl_env, "sim", None)
            obj_body_id = getattr(bddl_env, "obj_body_id", {})
            from vla_scripts.visual_concepts_extractor import get_env_inventory, get_site_parent_map, get_goal_predicates, derive_involved_from_goals
            inv = get_env_inventory(env)
            objects, sites = inv.get("objects", []), inv.get("sites", [])
            goals = get_goal_predicates(env)
            involved_objs, involved_sites = derive_involved_from_goals(goals, objects, sites)
            parent_map = get_site_parent_map(env)
            print("[contact-setup] involved_objects:", involved_objs)
            print("[contact-setup] involved_sites:", involved_sites)
            print("[contact-setup] site→parent:")
            for s2 in involved_sites:
                print(f"  - {s2} → {parent_map.get(s2)}")
            keys = sorted(list(obj_body_id.keys()))
            print("[contact-setup] obj_body_id keys (first 20):", keys[:20], ("…" if len(keys) > 20 else ""))
            robots = getattr(bddl_env, "robots", [])
            prefix = getattr(robots[0].robot_model, "naming_prefix", None) if robots else None
            print(f"[contact-setup] robot_prefix={prefix}")
            if sim is not None:
                nbody = int(sim.model.nbody)
                body_names = [sim.model.body_id2name(i) or "" for i in range(nbody)]
                print("[contact-setup] sample body_names (first 15):", body_names[:15])
        except Exception as e:
            print(f"[contact-setup] WARNING: {e}")
        
        # Create output directories
        if images_output_dir:
            episode_img_dir = Path(images_output_dir) / f"task_{task_id}" / f"episode_{episode_id}"
            episode_img_dir.mkdir(parents=True, exist_ok=True)

        # Prepare concepts recorder (per-task, persisted across episodes), using LIBERO identifiers
        task_name, language = _get_task_identifiers_from_env(env)
        concepts_recorder = None
        # Precompute contact index once per episode for fast contact queries
        contact_index = None
        if concepts_recorders is not None:
            # Key recorders by language instruction to avoid task-name collisions
            key_name = language if language else task_name
            key = _sanitize(key_name)
            if key not in concepts_recorders:
                chosen_concepts = select_task_concepts(env)
                rec = CSVRelationsRecorder(task_name=task_name, language=language)
                rec.initialize(chosen_concepts)
                concepts_recorders[key] = rec
            concepts_recorder = concepts_recorders[key]
            # If this episode introduces new concepts (e.g., new mj_contact pairs), merge them
            try:
                existing = set(concepts_recorder.concepts)
                desired = set(select_task_concepts(env))
                new_items = [c for c in desired if c not in existing]
                if new_items:
                    # Extend concept list and backfill prior snapshots with zeros
                    concepts_recorder.concepts.extend(new_items)
                    for i in range(len(concepts_recorder.ts)):
                        for c in new_items:
                            if c not in concepts_recorder.ts[i]:
                                concepts_recorder.ts[i][c] = 0
                    print(f"[debug-recon] concepts: added {len(new_items)} new items for this task")
            except Exception:
                pass
            # Build once; structure is static across timesteps
            try:
                contact_index = build_contact_index(env)
            except Exception:
                contact_index = None
        # Per-episode concept snapshots (for combined rendering)
        episode_concept_snapshots: List[Dict[str, int]] = []

        # Episode meta for state writer (aligned to existing indices)
        if state_writer is not None:
            ep_meta = {
                "episode_idx": int(episode_idx if "episode_idx" in episode_info else episode_id),
                "episode_id": int(episode_id),
                "task_name": str(task_name),
                "task_id": int(task_id),
                "language_instruction": language,
                "success": bool(episode_info.get("success", True)),
                "num_timesteps": int(num_timesteps),
                "orig_start_idx": int(start_idx),
                "orig_end_idx": int(end_idx),
            }
            state_writer.append_episode_meta(ep_meta)

        # Replay trajectory using stored actions
        all_images = []  # Store images for GIF generation
        images_saved = 0
        states_saved = 0

        # Helper: extract and append current simulator state to writer
        def _append_current_state_to_writer():
            if state_writer is None:
                return
            # Access underlying BDDL env
            bddl_env = env.env if hasattr(env, "env") else env
            sim = bddl_env.sim

            # Core robot / eef
            robot_qpos = np.asarray(sim.data.qpos[:7], dtype=np.float32)
            robot_qvel = np.asarray(sim.data.qvel[:7], dtype=np.float32)
            # EE position and quaternion (robust site/body lookup)
            ee_pos = None
            for site_name in ["gripper0_grip_site", "grip_site", "eef_site", "gripper_site"]:
                try:
                    sid = sim.model.site_name2id(site_name)
                    ee_pos = sim.data.site_xpos[sid].copy()
                    break
                except Exception:
                    continue
            if ee_pos is None:
                ee_pos = np.zeros(3, dtype=np.float32)
            ee_quat = None
            for body_name in ["gripper0_eef", "eef", "gripper_eef", "gripper"]:
                try:
                    ee_quat = sim.data.get_body_xquat(body_name).copy()
                    break
                except Exception:
                    continue
            if ee_quat is None:
                ee_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

            state_writer.append_core(robot_qpos, robot_qvel, ee_pos, ee_quat, float(sim.data.time))

            # Object states (use LIBERO object_states_dict order sorted by name)
            obj_states = getattr(bddl_env, "object_states_dict", {})
            names = sorted(list(obj_states.keys()))
            positions = []
            orientations = []
            body_ids = []
            extents = []
            for name in names:
                st = obj_states[name]
                geom = st.get_geom_state()
                positions.append(np.asarray(geom.get("pos", np.zeros(3)), dtype=np.float32))
                orientations.append(np.asarray(geom.get("quat", np.array([0, 0, 0, 1])), dtype=np.float32))
                # Body id if exists
                bid = -1
                try:
                    bid = int(bddl_env.obj_body_id.get(name, -1))
                except Exception:
                    bid = -1
                body_ids.append(bid)
                extents.append(0.0)  # placeholder; true extents not required for LIBERO-native checks
            positions = np.stack(positions, axis=0) if positions else np.zeros((0, 3), dtype=np.float32)
            orientations = np.stack(orientations, axis=0) if orientations else np.zeros((0, 4), dtype=np.float32)
            state_writer.set_task_objects_metadata(task_name, names, np.asarray(body_ids, dtype=np.int32), np.asarray(extents, dtype=np.float32))
            state_writer.append_task_objects(task_name, positions, orientations)

            # Contacts from MuJoCo
            ncon = int(sim.data.ncon)
            b1_ids = []
            b2_ids = []
            cpos = []
            cdist = []
            for i in range(ncon):
                c = sim.data.contact[i]
                try:
                    bid1 = int(sim.model.geom_bodyid[c.geom1])
                    bid2 = int(sim.model.geom_bodyid[c.geom2])
                    b1_ids.append(bid1)
                    b2_ids.append(bid2)
                    cpos.append(np.array([c.pos[0], c.pos[1], c.pos[2]], dtype=np.float32))
                    cdist.append(float(c.dist))
                except Exception:
                    continue
            if b1_ids:
                state_writer.append_contacts(task_name,
                                             np.asarray(b1_ids, dtype=np.int32),
                                             np.asarray(b2_ids, dtype=np.int32),
                                             np.stack(cpos, axis=0),
                                             np.asarray(cdist, dtype=np.float32))

        for timestep in range(num_timesteps):
            if timestep == 0:
                # First timestep - environment is already reset and initialized above
                print(f"[debug-recon] Starting trajectory replay from timestep 0")
            else:
                # Use the ACTUAL stored action from HDF5 file!
                action_idx = timestep - 1  # Actions are offset by 1 from timesteps
                if action_idx < len(stored_actions):
                    # Action horizon handling: All actions should be executed
                    horizon_actions = stored_actions[action_idx]  # shape (horizon, 7)
                    
                    # removed old SCALE-DEBUG
                    
                    # Execute each action in the horizon
                    if len(horizon_actions.shape) == 2:  # (horizon, 7)
                        for sub_action in horizon_actions:
                            # Step 1: normalize_gripper_action (same as line 388-390 in parallel_libero_evaluator.py)
                            action = normalize_gripper_action(sub_action.copy(), binarize=True)
                            
                            # Step 2: invert_gripper_action for prismatic (same as line 395-397 in parallel_libero_evaluator.py) 
                            action = invert_gripper_action(action)
                            
                            # removed old SCALE-DEBUG prints
                            
                            obs, reward, done, info = env.step(action.tolist())
                            
                            # if timestep <= 3:  # Debug first few steps
                                # print(f"[debug-recon] Sub-step result: reward={reward:.3f}, done={done}")
                    else:  # Single action (horizon=1)
                        action = normalize_gripper_action(horizon_actions.copy(), binarize=True)
                        action = invert_gripper_action(action)
                        
                        # removed old SCALE-DEBUG prints
                        
                        obs, reward, done, info = env.step(action.tolist())
                        
                        # if timestep <= 3:  # Debug first few steps
                        #     print(f"[debug-recon] Step result: reward={reward:.3f}, done={done}")
                else:
                    print(f"[debug-recon] ERROR: action_idx {action_idx} >= len(stored_actions) {len(stored_actions)}")
                    break
            
            # Save reconstructed image (only if rendering is enabled)
            if images_output_dir and enable_rendering:
                try:
                    img_array = get_libero_image(obs, 224, key="agentview_image")
                    img = Image.fromarray(img_array.astype(np.uint8))
                    
                    # Store image for GIF generation
                    all_images.append(img_array.astype(np.uint8))
                    
                    # Also save individual PNG if needed
                    img_path = episode_img_dir / f"timestep_{timestep:04d}.png"
                    img.save(img_path)
                    images_saved += 1
                except Exception as e:
                    print(f"[debug-recon] WARNING: Could not render image for timestep {timestep}: {e}")
                    if timestep == 0:
                        print(f"[debug-recon] Note: Rendering disabled for efficiency. Set enable_rendering=True if needed.")
            elif images_output_dir and not enable_rendering:
                # Skip rendering but count what would have been saved
                images_saved += 1
            
            # Append simulator state to chunk writer
            if state_writer is not None:
                _append_current_state_to_writer()
                states_saved += 1

            # Accumulate concepts
            if concepts_recorder is not None:
                # Evaluate only the pre-selected concepts for this task and append once per timestep
                concept_list = concepts_recorder.concepts if concepts_recorder.concepts else enumerate_concept_keys(env)
                # Recompute contact index at each timestep to reflect live contacts
                try:
                    _ci = build_contact_index(env)
                except Exception:
                    _ci = None

                # Helper: robust MuJoCo contact between named bodies (objects or fixtures)
                # Uses extractor's contact_between_bodies which supports body-subtrees
                def _mj_contact(a_name: str, b_name: str, ci) -> int:
                    try:
                        r = contact_between_bodies(env, a_name, b_name, ci)
                        return int(r) if (r is not None) else 0
                    except Exception:
                        return 0

                def _eval_expr(expr: str) -> int:
                    try:
                        base = expr.split(" ", 1)[0]  # strip annotations like " [parent_of=...]"
                        if not ("(" in base and base.endswith(")")):
                            return 0
                        head, rest = base.split("(", 1)
                        args = [a.strip() for a in rest[:-1].split(",") if a.strip()]
                        head_l = head.lower()
                        if head_l == "contact":
                            # contact(A,B): A may be object, B may be site or gripper
                            if len(args) != 2:
                                return 0
                            a, b = args
                            if b == "gripper":
                                (lambda r: 1 if r==1 else (_robot_contact_scan(_resolve_body_id(a), _ci)))(contact_obj_with_robot(env, a, _ci))
                            # If B is a site, use site methods; else fallback to MuJoCo contact
                            try:
                                sm = evaluate_site_methods(env, b, a)
                                if "check_contact" in sm:
                                    return 1 if sm["check_contact"] else 0
                            except Exception:
                                pass
                            return _mj_contact(a, b, _ci)
                        if head_l == "ontop":
                            if len(args) != 2:
                                return 0
                            a, site = args
                            sm = evaluate_site_methods(env, site, a)
                            return 1 if sm.get("check_ontop") else 0
                        if head_l == "contain":
                            if len(args) != 2:
                                return 0
                            a, site = args
                            sm = evaluate_site_methods(env, site, a)
                            return 1 if sm.get("check_contain") else 0
                        if head_l in ("in_box", "under", "on_top"):
                            if len(args) != 2:
                                return 0
                            a, site = args
                            gm = evaluate_site_geometry_methods(env, site, a)
                            return 1 if gm.get(head_l) else 0
                        if head_l == "is_open":
                            if len(args) != 1:
                                return 0
                            site = args[0]
                            um = evaluate_site_methods(env, site, None)
                            return 1 if um.get("is_open") else 0
                        if head_l == "is_close":
                            if len(args) != 1:
                                return 0
                            site = args[0]
                            um = evaluate_site_methods(env, site, None)
                            return 1 if um.get("is_close") else 0
                        if head_l == "mj_contact":
                            if len(args) != 2:
                                return 0
                            a, b = args
                            # Handle gripper using the robot-contact path directly
                            if a == "gripper" and b != "gripper":
                                r = contact_obj_with_robot(env, b, _ci)
                                return 1 if r == 1 else 0
                            if b == "gripper" and a != "gripper":
                                r = contact_obj_with_robot(env, a, _ci)
                                return 1 if r == 1 else 0
                            # Otherwise, use plain body contact (with subtree handling)
                            return _mj_contact(a, b, _ci)
                        if head_l in ("in", "on"):
                            # Fall back to generic predicate evaluator
                            return int(evaluate_concepts(env, [expr]).get(expr, 0))
                    except Exception:
                        return 0
                    return 0

                snapshot = evaluate_concept_expressions(env, concept_list, contact_index=_ci)
                # Contact debug (t=0, mid, last)
                if timestep in (0, max(0, num_timesteps//2), num_timesteps - 1):
                    try:
                        bddl_env = env.env if hasattr(env, 'env') else env
                        sim = getattr(bddl_env, 'sim', None)
                        ncon = int(getattr(sim.data, 'ncon', 0)) if sim is not None else -1
                        print(f"[contact-debug] t={timestep} ncon={ncon}")
                        if sim is not None and ncon > 0:
                            for i in range(min(20, ncon)):
                                c = sim.data.contact[i]
                                g1 = int(getattr(c, 'geom1', -1)); g2 = int(getattr(c, 'geom2', -1))
                                if g1 < 0 or g2 < 0:
                                    continue
                                b1 = int(sim.model.geom_bodyid[g1]); b2 = int(sim.model.geom_bodyid[g2])
                                n1 = sim.model.body_id2name(b1) or ''
                                n2 = sim.model.body_id2name(b2) or ''
                                print(f"  [pair] {b1}:{n1} <-> {b2}:{n2}")
                        # Targeted checks for first involved object and parent of first involved site
                        from vla_scripts.visual_concepts_extractor import get_env_inventory, get_site_parent_map, get_goal_predicates, derive_involved_from_goals
                        inv = get_env_inventory(env)
                        objects, sites = inv.get('objects', []), inv.get('sites', [])
                        goals = get_goal_predicates(env)
                        involved_objs, involved_sites = derive_involved_from_goals(goals, objects, sites)
                        parent_map = get_site_parent_map(env)
                        target_obj = involved_objs[0] if involved_objs else (objects[0] if objects else '')
                        parent = ''
                        if involved_sites:
                            pref = next((s for s in involved_sites if 'top' in s.lower()), involved_sites[0])
                            parent = parent_map.get(pref, '')
                        body_map = getattr(bddl_env, 'obj_body_id', {})
                        print(f"[contact-debug] target_obj={target_obj} parent_of_site={parent} obj_body_id[target]={body_map.get(target_obj)} obj_body_id[parent]={body_map.get(parent)}")
                        # Also report initial-site parent for the target object
                        init_parent=''
                        try:
                            from vla_scripts.visual_concepts_extractor import get_env_inventory, get_site_parent_map
                            inv2 = get_env_inventory(env)
                            sites2 = inv2.get('sites', [])
                            pm2 = get_site_parent_map(env)
                            base = "_".join(target_obj.split("_")[:-1]) or target_obj
                            for s2 in sites2:
                                if ('init' in s2.lower()) and (base in s2):
                                    init_parent = pm2.get(s2, '')
                                    break
                        except Exception:
                            init_parent=''
                        c_gr = contact_obj_with_robot(env, target_obj, _ci)
                        c_pa = contact_between_bodies(env, target_obj, parent, _ci) if parent else None
                        c_init = contact_between_bodies(env, target_obj, init_parent, _ci) if init_parent else None
                        print(f"[contact-debug] contact_obj_with_robot({target_obj})={c_gr}; contact_between_bodies({target_obj},{parent})={c_pa}; contact_between_bodies({target_obj},{init_parent})={c_init}")
                    except Exception as e:
                        print(f"[contact-debug] WARNING: {e}")
                concepts_recorder.append(snapshot)
                episode_concept_snapshots.append(snapshot)
                
            if timestep % 5 == 0:
                print(f"[debug-recon] Processed timestep {timestep}/{num_timesteps}")
        
        # Capture final post-action state and concepts once more (after last action)
        try:
            if state_writer is not None:
                _append_current_state_to_writer()
                states_saved += 1
            if concepts_recorder is not None:
                concept_list = concepts_recorder.concepts if concepts_recorder.concepts else enumerate_concept_keys(env)
                # Recompute contact index for final snapshot
                try:
                    _ci_final = build_contact_index(env)
                except Exception:
                    _ci_final = None
                def _mj_contact_final(a_name: str, b_name: str, ci) -> int:
                    try:
                        bddl_env = env.env if hasattr(env, 'env') else env
                        sim = getattr(bddl_env, 'sim', None)
                        if sim is None:
                            return 0
                        body_map = getattr(bddl_env, 'obj_body_id', {})
                        def get_bid(name: str):
                            bid = body_map.get(name)
                            if bid is not None:
                                return int(bid)
                            try:
                                return int(sim.model.body_name2id(name))
                            except Exception:
                                return None
                        ba = get_bid(a_name)
                        bb = get_bid(b_name)
                        if ba is None or bb is None:
                            return 0
                        if ci is not None:
                            pair = (ba, bb) if ba < bb else (bb, ba)
                            return 1 if pair in ci else 0
                        ncon = int(getattr(sim.data, 'ncon', 0))
                        for i in range(ncon):
                            c = sim.data.contact[i]
                            g1 = int(getattr(c, 'geom1', -1)); g2 = int(getattr(c, 'geom2', -1))
                            if g1 < 0 or g2 < 0:
                                continue
                            b1 = int(sim.model.geom_bodyid[g1]); b2 = int(sim.model.geom_bodyid[g2])
                            if (b1 == ba and b2 == bb) or (b1 == bb and b2 == ba):
                                return 1
                        return 0
                    except Exception:
                        return 0

                def _eval_expr_final(expr: str) -> int:
                    try:
                        base = expr.split(" ", 1)[0]
                        if not ("(" in base and base.endswith(")")):
                            return 0
                        head, rest = base.split("(", 1)
                        args = [a.strip() for a in rest[:-1].split(",") if a.strip()]
                        head_l = head.lower()
                        if head_l == "contact":
                            if len(args) != 2:
                                return 0
                            a, b = args
                            if b == "gripper":
                                return 1 if contact_obj_with_robot(env, a, _ci_final) else 0
                            try:
                                sm = evaluate_site_methods(env, b, a)
                                if "check_contact" in sm:
                                    return 1 if sm["check_contact"] else 0
                            except Exception:
                                pass
                            return _mj_contact_final(a, b, _ci_final)
                        if head_l == "ontop":
                            if len(args) != 2:
                                return 0
                            a, site = args
                            sm = evaluate_site_methods(env, site, a)
                            return 1 if sm.get("check_ontop") else 0
                        if head_l == "contain":
                            if len(args) != 2:
                                return 0
                            a, site = args
                            sm = evaluate_site_methods(env, site, a)
                            return 1 if sm.get("check_contain") else 0
                        if head_l in ("in_box", "under", "on_top"):
                            if len(args) != 2:
                                return 0
                            a, site = args
                            gm = evaluate_site_geometry_methods(env, site, a)
                            return 1 if gm.get(head_l) else 0
                        if head_l == "is_open":
                            if len(args) != 1:
                                return 0
                            site = args[0]
                            um = evaluate_site_methods(env, site, None)
                            return 1 if um.get("is_open") else 0
                        if head_l == "is_close":
                            if len(args) != 1:
                                return 0
                            site = args[0]
                            um = evaluate_site_methods(env, site, None)
                            return 1 if um.get("is_close") else 0
                        if head_l == "mj_contact":
                            if len(args) != 2:
                                return 0
                            a, b = args
                            if a == "gripper" and b != "gripper":
                                return 1 if contact_obj_with_robot(env, b, _ci_final) else 0
                            if b == "gripper" and a != "gripper":
                                return 1 if contact_obj_with_robot(env, a, _ci_final) else 0
                            return _mj_contact_final(a, b, _ci_final)
                        if head_l in ("in", "on"):
                            return int(evaluate_concepts(env, [expr]).get(expr, 0))
                    except Exception:
                        return 0
                    return 0
                final_snapshot = evaluate_concept_expressions(env, concept_list, contact_index=_ci_final)
                # Debug for final mj_contact
                try:
                    bddl_env = env.env if hasattr(env, 'env') else env
                    body_map = getattr(bddl_env, 'obj_body_id', {})
                    ncon = int(getattr(bddl_env.sim.data, 'ncon', 0)) if hasattr(bddl_env, 'sim') and hasattr(bddl_env.sim, 'data') else -1
                    mj_items = [(k, v) for k, v in final_snapshot.items() if k.startswith('mj_contact(')]
                    if mj_items:
                        print(f"[contact-debug] t=final ncon={ncon} mj_contact_count={len(mj_items)}")
                        for name, val in mj_items[:10]:
                            inner = name[len('mj_contact('):-1]
                            a, b = [s.strip() for s in inner.split(',')]
                            ba = _resolve_body_id(a)
                            bb = _resolve_body_id(b)
                            print(f"[contact-debug]  {name}: {val} (ba={ba}, bb={bb})")
                except Exception:
                    pass
                concepts_recorder.append(final_snapshot)
                episode_concept_snapshots.append(final_snapshot)
        except Exception as e:
            print(f"[warn] Failed to record final post-action snapshot: {e}")

        # Generate trajectory.gif only if not co-rendering concepts
        if images_output_dir and enable_rendering and all_images and not render_concepts:
            try:
                gif_path = episode_img_dir / "trajectory.gif"
                
                # Convert numpy arrays to PIL Images
                pil_images = [Image.fromarray(img_array) for img_array in all_images]
                
                # Create GIF with reasonable duration (100ms per frame = 10 FPS)
                pil_images[0].save(
                    gif_path,
                    save_all=True,
                    append_images=pil_images[1:],
                    duration=100,  # 100ms per frame
                    loop=0  # Loop forever
                )
                
                print(f"[debug-recon] Generated trajectory GIF: {gif_path}")
                print(f"[debug-recon] GIF contains {len(all_images)} frames at 10 FPS")
                
            except Exception as e:
                print(f"[debug-recon] WARNING: Could not generate GIF: {e}")
        
        # Concepts are saved per task at the end by the caller (after all episodes)
        
        # Optionally render a combined action+concepts GIF strictly aligned by timestep
        if images_output_dir and enable_rendering and render_concepts and all_images and episode_concept_snapshots:
            try:
                concept_names = concepts_recorder.concepts if concepts_recorder is not None else []
                # Strict alignment: both lists were appended within the same timestep loop
                T_img = len(all_images)
                T_con = len(episode_concept_snapshots)
                if T_img != T_con:
                    print(f"[warn] image frames ({T_img}) != concept snapshots ({T_con}); enforcing strict alignment by trimming to min")
                T = min(T_img, T_con)
                if concept_names:
                    mat = np.zeros((len(concept_names), T), dtype=np.int8)
                    for t in range(T):
                        snap = episode_concept_snapshots[t]
                        for i, cname in enumerate(concept_names):
                            mat[i, t] = int(snap.get(cname, 0))
                    # Optional smoothing to reduce flicker for mj_contact with gripper in visualization
                    try:
                        def _is_gripper_contact(name: str) -> bool:
                            n = name.lower()
                            return n.startswith('mj_contact(') and ('gripper' in n)
                        for i, cname in enumerate(concept_names):
                            if _is_gripper_contact(cname) and T >= 3:
                                row = mat[i, :].copy()
                                sm = row.copy()
                                for t in range(1, T - 1):
                                    if row[t-1] or row[t] or row[t+1]:
                                        sm[t] = 1
                                mat[i, :] = sm
                    except Exception:
                        pass
                    keep_idx = list(range(len(concept_names)))
                    if concepts_only_changing:
                        keep_idx = [i for i in range(len(concept_names)) if np.any(mat[i, :] != mat[i, 0])]
                        if not keep_idx:
                            keep_idx = list(range(len(concept_names)))
                    kept_names = [concept_names[i] for i in keep_idx]
                    kept_mat = mat[keep_idx, :]
                    # Render with external utils to avoid text distortion
                    from vla_scripts.concepts_render_utils import render_concept_frames, compose_action_concepts
                    # Fixed palette: 0=gray, 1=faint red
                    concept_frames = render_concept_frames(
                        kept_names,
                        kept_mat,
                        width=600,
                        bg_color=(20,20,20),
                        off_color=(150,150,150),
                        on_color=(220,80,80),
                    )
                    action_frames = [Image.fromarray(a) if isinstance(a, np.ndarray) else a for a in all_images[:T]]
                    combined = compose_action_concepts(action_frames, concept_frames, left_width=700, right_width=600)
                    episode_img_dir = Path(images_output_dir) / f"task_{task_id}" / f"episode_{episode_id}"
                    episode_img_dir.mkdir(parents=True, exist_ok=True)
                    out_path = episode_img_dir / "combined.gif"
                    combined[0].save(out_path, save_all=True, append_images=combined[1:], duration=100, loop=0)
                    print(f"[debug-recon] Saved combined action+concepts GIF: {out_path}")
            except Exception as e:
                print(f"[debug-recon] WARNING: Failed to render combined GIF: {e}")

        print(f"[debug-recon] Successfully reconstructed {task_id}/{episode_id}")
        print(f"[debug-recon] Images saved: {images_saved}, States saved: {states_saved}")
        
    finally:
        env.close()
    
    return {
        'images_saved': images_saved,
        'states_saved': states_saved,
        'task_description': task_description
    }


def _render_concept_frames_inline(concepts: List[str], values: np.ndarray, width: int = 500) -> List[Image.Image]:
    """Render text panels for concept values over time.

    Args:
        concepts: list of concept names, length N
        values: array of shape [N, T] with 0/1 ints
        width: panel width in pixels

    Returns:
        List of PIL Images, one per timestep (length T)
    """
    if values.size == 0:
        return []
    rows = len(concepts)
    row_h = 22
    pad = 12
    height = pad * 2 + rows * row_h
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    frames: List[Image.Image] = []
    T = values.shape[1]
    for t in range(T):
        img = Image.new("RGB", (width, height), color=(20, 20, 20))
        draw = ImageDraw.Draw(img)
        y = pad
        for i, name in enumerate(concepts):
            val = int(values[i, t])
            color = (0, 200, 0) if val == 1 else (200, 0, 0)
            draw.text((pad, y), name, fill=(220, 220, 220), font=font)
            draw.text((width - 60, y), "1" if val == 1 else "0", fill=color, font=font)
            y += row_h
        frames.append(img)
    return frames


def reconstruct_dataset(
    dataset_dir: str,
    images_output_dir: str = None,
    states_output_dir: str = None, 
    task_suite_name: str = "libero_90",
    max_episodes: int = None,
    episode_filter: Dict = None,
    enable_rendering: bool = True,
    combine_after: bool = True,
    render_concepts: bool = False,
    concepts_only_changing: bool = True,
    enable_state_io: bool = False,
):
    """
    Reconstruct all episodes in an optimized trajectory dataset.
    
    Args:
        dataset_dir: Path to optimized trajectory data directory
        images_output_dir: Directory to save images (None to skip)
        states_output_dir: Directory to save simulator states (None to skip)
        task_suite_name: LIBERO task suite name
        max_episodes: Maximum episodes to process (None for all)
        episode_filter: Dict with filtering criteria (e.g., {'success': True, 'task_id': [1,2,3]})
        enable_rendering: Whether to enable rendering for image reconstruction (disable for scaling)
    """
    dataset_dir = Path(dataset_dir)
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    print(f"[debug-recon] Loading optimized dataset: {dataset_dir}")
    print(f"[debug-recon] Images output: {images_output_dir}")
    print(f"[debug-recon] States output root: {states_output_dir or dataset_dir}")
    
    if images_output_dir:
        Path(images_output_dir).mkdir(parents=True, exist_ok=True)
    if states_output_dir:
        Path(states_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Prepare optional state chunk writer and concepts root (dataset_dir by default)
    state_root = str(states_output_dir) if states_output_dir else str(dataset_dir)
    writer = StateChunkWriter(dataset_root=state_root, process_id=0) if enable_state_io else None
    paths = resolve_paths(state_root)
    concepts_root = str(paths["concepts"])  # final save location for per-task CSVs
    concepts_recorders: Dict[str, CSVRelationsRecorder] = {}

    # Load episode metadata once
    episode_metadata = load_episode_metadata(dataset_dir)
    
    # Apply filtering
    if episode_filter:
        print(f"[debug-recon] Applying episode filter: {episode_filter}")
        for key, value in episode_filter.items():
            if key in episode_metadata.columns:
                if isinstance(value, list):
                    episode_metadata = episode_metadata[episode_metadata[key].isin(value)]
                else:
                    episode_metadata = episode_metadata[episode_metadata[key] == value]
        print(f"[debug-recon] Episodes after filtering: {len(episode_metadata)}")
    
    # Apply max_episodes limit
    if max_episodes:
        episode_metadata = episode_metadata.head(max_episodes)
        print(f"[debug-recon] Limited to {len(episode_metadata)} episodes")
    
    total_images = 0
    total_states = 0
    episodes_processed = 0
    
    for idx in range(len(episode_metadata)):
        episode_info = episode_metadata.iloc[idx]
        task_id = episode_info['task_id']
        episode_id = episode_info['episode_id']
        
        try:
            result = reconstruct_trajectory_episode(
                dataset_dir=dataset_dir,
                episode_idx=idx,
                task_suite_name=task_suite_name,
                images_output_dir=images_output_dir,
                states_output_dir=states_output_dir,
                episode_metadata=episode_metadata,
                enable_rendering=enable_rendering,
                state_writer=writer,
                concepts_recorders=concepts_recorders,
                concepts_root_dir=concepts_root,
                render_concepts=render_concepts,
                concepts_only_changing=concepts_only_changing,
            )
            
            total_images += result['images_saved']
            total_states += result['states_saved'] 
            episodes_processed += 1
            
        except Exception as e:
            print(f"[debug-recon] ERROR processing episode {idx} (task_{task_id}/episode_{episode_id}): {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"[debug-recon] ===== RECONSTRUCTION COMPLETE =====")
    print(f"[debug-recon] Episodes processed: {episodes_processed}")
    print(f"[debug-recon] Total images saved: {total_images}")
    print(f"[debug-recon] Total states saved: {total_states}")

    # Flush chunk files and optionally combine
    if enable_state_io and writer is not None:
        print(f"[debug-recon] Flushing chunk files...")
        writer.flush()
    # Save per-task CSVs named after the task
    for key, rec in concepts_recorders.items():
        csv_path = rec.save_as_task_csv(concepts_root)
        print(f"[debug-recon] Saved task relations CSV: {csv_path}")
    if enable_state_io and combine_after:
        print(f"[debug-recon] Combining state chunks into final sim_states/...")
        summary = combine_state_chunks(state_root)
        print(f"[debug-recon] Combine summary: {summary}")


def combine_reconstruction_chunks_to_optimized_format(
    temp_processing_dir: str,
    output_dir: str
):
    """
    Combine temporary reconstruction chunk files into final optimized format.
    Follows same pattern as combine_chunks_to_optimized_format for trajectory data.
    
    Args:
        temp_processing_dir: Directory containing reconstruction_process_* subdirectories
        output_dir: Final output directory for reconstructed data
    """
    temp_dir = Path(temp_processing_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[RECONSTRUCTION_COMBINER] Combining reconstruction chunks...")
    print(f"[RECONSTRUCTION_COMBINER] Temp dir: {temp_dir}")
    print(f"[RECONSTRUCTION_COMBINER] Output dir: {output_dir}")
    
    # Find all process directories
    process_dirs = [d for d in temp_dir.iterdir() if d.is_dir() and d.name.startswith('reconstruction_process_')]
    print(f"[RECONSTRUCTION_COMBINER] Found {len(process_dirs)} process directories")
    
    if not process_dirs:
        print(f"[RECONSTRUCTION_COMBINER] No process directories found!")
        return
    
    # Load all manifests to understand data structure
    manifests = []
    total_samples = 0
    all_state_fields = set()
    has_images = False
    
    for process_dir in process_dirs:
        manifest_path = process_dir / "reconstruction_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
                manifests.append({**manifest, 'process_dir': process_dir})
                total_samples += manifest['total_samples']
                all_state_fields.update(manifest['state_fields'])
                has_images = has_images or manifest['has_images']
    
    print(f"[RECONSTRUCTION_COMBINER] Total samples across all processes: {total_samples}")
    print(f"[RECONSTRUCTION_COMBINER] State fields: {sorted(all_state_fields)}")
    print(f"[RECONSTRUCTION_COMBINER] Has images: {has_images}")
    
    # HDF5 compression settings
    compression_kwargs = {
        'compression': 'gzip',
        'compression_opts': 6,
        'shuffle': True
    }
    
    # Combine states by field
    print(f"[RECONSTRUCTION_COMBINER] Combining state fields...")
    states_output_path = output_dir / "reconstructed_states.h5"
    
    with h5py.File(states_output_path, 'w') as output_f:
        for state_field in sorted(all_state_fields):
            field_chunks = []
            
            for manifest in manifests:
                states_path = manifest['process_dir'] / "states_chunk.h5"
                if states_path.exists():
                    with h5py.File(states_path, 'r') as f:
                        if state_field in f:
                            field_chunks.append(f[state_field][:])
            
            if field_chunks:
                combined_field = np.concatenate(field_chunks, axis=0)
                
                # Optimize chunking for sequential access
                chunk_size = min(10000, combined_field.shape[0])
                if len(combined_field.shape) == 1:
                    chunks = (chunk_size,)
                elif len(combined_field.shape) == 2:
                    chunks = (chunk_size, combined_field.shape[1])
                elif len(combined_field.shape) == 3:
                    chunks = (chunk_size, combined_field.shape[1], combined_field.shape[2])
                else:
                    chunks = True
                
                output_f.create_dataset(state_field,
                                      data=combined_field,
                                      chunks=chunks,
                                      **compression_kwargs)
                print(f"[RECONSTRUCTION_COMBINER] Combined {state_field}: {combined_field.shape}")
    
    # Combine images if available
    if has_images:
        print(f"[RECONSTRUCTION_COMBINER] Combining images...")
        image_chunks = []
        for manifest in manifests:
            images_path = manifest['process_dir'] / "images_chunk.h5"
            if images_path.exists():
                with h5py.File(images_path, 'r') as f:
                    image_chunks.append(f['images'][:])
        
        if image_chunks:
            combined_images = np.concatenate(image_chunks, axis=0)
            images_output_path = output_dir / "reconstructed_images.h5"
            with h5py.File(images_output_path, 'w') as f:
                f.create_dataset('images', data=combined_images, **compression_kwargs)
            print(f"[RECONSTRUCTION_COMBINER] Combined images: {combined_images.shape}")
    
    # Combine episode metadata and create index
    print(f"[RECONSTRUCTION_COMBINER] Creating episode index...")
    all_episodes = []
    sample_offset = 0
    
    for manifest in manifests:
        episodes_path = manifest['process_dir'] / "episodes_chunk.json"
        if episodes_path.exists():
            with open(episodes_path, 'r') as f:
                process_episodes = json.load(f)
                
                # Adjust start_idx and end_idx to account for global indexing
                for episode in process_episodes:
                    episode['start_idx'] += sample_offset
                    episode['end_idx'] += sample_offset
                
                all_episodes.extend(process_episodes)
                sample_offset += manifest['total_samples']
    
    # Create episode index DataFrame and save
    episode_df = pd.DataFrame(all_episodes)
    episode_index_path = output_dir / "reconstruction_episode_index.h5"
    
    with h5py.File(episode_index_path, 'w') as f:
        # Save each column separately for efficient access
        for col in episode_df.columns:
            if episode_df[col].dtype == 'object':
                # String columns need special handling
                f.create_dataset(col, data=episode_df[col].astype('S'))
            else:
                f.create_dataset(col, data=episode_df[col].values, **compression_kwargs)
    
    print(f"[RECONSTRUCTION_COMBINER] Saved episode index: {len(episode_df)} episodes")
    
    # Create summary metadata
    summary_path = output_dir / "reconstruction_summary.json"
    summary = {
        'total_samples': int(total_samples),
        'total_episodes': len(all_episodes),
        'state_fields': sorted(list(all_state_fields)),
        'has_images': has_images,
        'successful_episodes': int(episode_df['success'].sum() if 'success' in episode_df.columns else 0),
        'created_at': time.time(),
        'format_version': '1.0_reconstructed_states'
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"[RECONSTRUCTION_COMBINER] Combination complete!")
    print(f"[RECONSTRUCTION_COMBINER] Output directory: {output_dir}")
    print(f"[RECONSTRUCTION_COMBINER] Summary: {summary}")
    
    return summary


def get_reconstruction_paths(dataset_dir: str) -> Dict[str, str]:
    """
    Derive reconstruction paths from dataset directory following mother folder structure.
    
    Args:
        dataset_dir: Path to optimized trajectory data (e.g., /path/to/optimized_trajectory_data)
        
    Returns:
        Dict with reconstruction paths in same mother folder
        
    Example:
        Input:  /work/nvme/bfbo/xzhang42/data/pilot_test/optimized_trajectory_data
        Output: {
            'reconstructed_data_dir': /work/nvme/bfbo/xzhang42/data/pilot_test/reconstructed_trajectory_data,
            'temp_processing_dir': /work/nvme/bfbo/xzhang42/data/pilot_test/temp_reconstruction_processing
        }
    """
    dataset_path = Path(dataset_dir)
    mother_dir = dataset_path.parent  # e.g., /work/nvme/bfbo/xzhang42/data/pilot_test/
    
    return {
        'reconstructed_data_dir': str(mother_dir / 'reconstructed_trajectory_data'),
        'temp_processing_dir': str(mother_dir / 'temp_reconstruction_processing')
    }


def main():
    parser = argparse.ArgumentParser(description='Reconstruct trajectory data from optimized dataset (paths auto-derived)')
    parser.add_argument('dataset_dir', nargs='?', default=None, help='Path to optimized trajectory data directory (defaults to DEFAULT_DATASET_DIR)')
    parser.add_argument('--task-suite-name', default='libero_90', help='LIBERO task suite name')
    # Path overrides are ignored; outputs are auto-derived from dataset_dir
    parser.add_argument('--images-output-dir', help='(Ignored) Output directory for reconstructed images – auto-derived')
    parser.add_argument('--states-output-dir', help='(Ignored) Output directory for simulator states – auto-derived')
    parser.add_argument('--max-episodes', type=int, help='Maximum episodes to process')
    parser.add_argument('--episode-idx', type=int, help='Specific episode index to reconstruct (0-based)')
    parser.add_argument('--filter-success', action='store_true', help='Only process successful episodes')
    parser.add_argument('--filter-task-ids', nargs='+', type=int, help='Only process specific task IDs')
    parser.add_argument('--metadata-only', action='store_true', help='Only load and display metadata')
    parser.add_argument('--disable-rendering', action='store_true', help='Disable rendering for efficient state-only reconstruction')
    parser.add_argument('--auto-paths', action='store_true', default=True, help='Automatically derive output paths from dataset directory (always on)')
    parser.add_argument('--no-combine', action='store_true', help='Do not combine chunks after reconstruction')
    parser.add_argument('--enable-state-io', action='store_true', help='Enable saving simulator states to sim_states/')
    parser.add_argument('--render-concepts', action='store_true', default=True, help='Co-render concepts next to action frames and save combined.gif (default on)')
    parser.add_argument('--concepts-all', action='store_true', help='Render all concepts (default renders only changing concepts)')
    
    args = parser.parse_args()
    
    # Default dataset dir if not provided
    if not args.dataset_dir:
        args.dataset_dir = DEFAULT_DATASET_DIR

    # Handle metadata-only mode
    if args.metadata_only:
        episode_metadata = load_episode_metadata(args.dataset_dir)
        print("\n===== EPISODE METADATA =====")
        print(episode_metadata.to_string())
        return
    
    # Auto-derive paths if requested or if no output dirs specified
    # Force auto-derived paths; ignore CLI overrides
    reconstruction_paths = get_reconstruction_paths(args.dataset_dir)
    args.states_output_dir = reconstruction_paths['reconstructed_data_dir']
    print(f"[auto-paths] Using auto-derived states output: {args.states_output_dir}")
    if not args.disable_rendering:
        args.images_output_dir = reconstruction_paths['reconstructed_data_dir'] + "/images"
        print(f"[auto-paths] Using auto-derived images output: {args.images_output_dir}")
    else:
        args.images_output_dir = None
    
    # Allow concepts-only mode; with fixed auto paths we simply skip images if disabled
    if not args.images_output_dir and not args.states_output_dir and not args.enable_state_io:
        print("[info] Running in concepts-only mode (no images/state outputs).")
    
    # Validate output directories are in fast storage
    for output_dir in [args.images_output_dir, args.states_output_dir]:
        if output_dir and '/work/nvme/' not in str(output_dir):
            print(f"[debug-recon] WARNING: {output_dir} should be in /work/nvme/ for fast storage")
    
    # Build episode filter
    episode_filter = {}
    if args.filter_success:
        episode_filter['success'] = True
    if args.filter_task_ids:
        episode_filter['task_id'] = args.filter_task_ids
    
    if args.episode_idx is not None:
        # Reconstruct single episode by index
        single_writer = StateChunkWriter(dataset_root=(args.states_output_dir or args.dataset_dir), process_id=0) if args.enable_state_io else None
        concepts_recorders: Dict[str, CSVRelationsRecorder] = {}
        result = reconstruct_trajectory_episode(
            dataset_dir=args.dataset_dir,
            episode_idx=args.episode_idx,
            task_suite_name=args.task_suite_name,
            images_output_dir=args.images_output_dir,
            states_output_dir=args.states_output_dir,
            enable_rendering=not args.disable_rendering,
            state_writer=single_writer,
            concepts_recorders=concepts_recorders,
            concepts_root_dir=str(resolve_paths(args.states_output_dir or args.dataset_dir)["concepts"]),
            render_concepts=args.render_concepts,
            concepts_only_changing=(not args.concepts_all),
        )
        print(f"Single episode reconstruction complete: {result}")
        if args.enable_state_io and single_writer is not None:
            print("[debug-recon] Flushing chunk files...")
            single_writer.flush()
        # Save per-task CSV(s)
        for key, rec in concepts_recorders.items():
            csv_path = rec.save_as_task_csv(str(resolve_paths(args.states_output_dir or args.dataset_dir)["concepts"]))
            print(f"[debug-recon] Saved task relations CSV: {csv_path}")
        if args.enable_state_io and not args.no_combine:
            print(f"[debug-recon] Combining state chunks into final sim_states/...")
            summary = combine_state_chunks(args.states_output_dir or args.dataset_dir)
            print(f"[debug-recon] Combine summary: {summary}")
    else:
        # Reconstruct dataset
        reconstruct_dataset(
            dataset_dir=args.dataset_dir,
            images_output_dir=args.images_output_dir,
            states_output_dir=args.states_output_dir,
            task_suite_name=args.task_suite_name,
            max_episodes=args.max_episodes,
            episode_filter=episode_filter if episode_filter else None,
            enable_rendering=not args.disable_rendering,
            combine_after=(not args.no_combine),
            render_concepts=args.render_concepts,
            concepts_only_changing=(not args.concepts_all),
            enable_state_io=args.enable_state_io,
        )


if __name__ == "__main__":
    main()
