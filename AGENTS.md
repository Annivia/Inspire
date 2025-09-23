## Overview

This was adapted from the InSpire repository - an implementation of Vision-Language-Action models with intrinsic spatial reasoning for robotics. 

## Development Goal

I want to probe the VLA activations to understand how action tokenization methods influence the representation flow inside transformer layers (from VLMs to VLAs). For that, I need:

- Data collection scripts that saves relevant meta data during model evaluation.
- State Reconstruction scripts that replay collected action data in the simulator, and record the corresponding simulator states.
- Scripts that perform linear probing on collected data.
- Supporting different VLA models other than miniVLA-vq (which is the default for current data collection scripts)


## Data Schema (Optimized Trajectories)

Available modalities (stored in HDF5 under `/optimized_trajectory_data/`):

- Hidden states: Transformer activations for every layer at each action generation step.
- Actions: Robot control tokens (action chunking aware; see caveat below).
- Vision features: Patch features from vision backbone (DINOv2 / CLIP / SigLIP).
- VLM embeddings: Processed visual embeddings used by the VLA.
- Episode index: Minimal info for reconstruction (task / episode / seed / indices / success / description).
- Images: Reconstructable from episode metadata via LIBERO.
- Simulator states: Reconstructable by replaying actions in LIBERO.

Storage layout:

```
/optimized_trajectory_data/
├── actions.h5                    # [N_samples, action_horizon]
├── hidden_states/
│   ├── generation_step_0.h5     # datasets: layer_00..layer_24 → [N_samples, hidden_dim]
│   ├── generation_step_1.h5
│   └── ... generation_step_6.h5
├── vision_features.h5           # [N_samples, num_patches, vision_dim]
├── vlm_embeddings.h5            # [N_samples, vlm_embed_dim]
├── episode_index.h5             # episode metadata + [start_idx, end_idx]
└── dataset_summary.json         # stats and metadata
```

Detailed specs:

- Actions (Generated Tokens)
  - What: Autoregressively generated action tokens per embodiment dimension; action‑chunked per step.
  - Shape per sample: `[action_horizon]` (e.g., 7 for miniVLA‑vq; 1 for most models).
  - Storage: `actions.h5` as `[N_samples, action_horizon]`.

- Hidden States (During Generation)
  - What: Transformer layer activations captured at each generation step (one file per step).
  - Files: `hidden_states/generation_step_k.h5` with datasets `layer_00..layer_24` → `[N_samples, hidden_dim]`.
  - Requirement: Consistent `hidden_dim` across layers and steps.

- Vision Features
  - Shape: `[N_samples, num_patches, vision_dim]`.
  - Storage: `vision_features.h5`.

- VLM Visual Embeddings
  - Shape: `[N_samples, vlm_embed_dim]`.
  - Storage: `vlm_embeddings.h5`.

- Episode Metadata (Reconstruction Clues)
  - Contents: `img_task_id`, `img_episode_id`, `img_env_seed`, `num_timesteps`, `task_description`, `success`, `start_idx`, `end_idx`.
  - Storage: `episode_index.h5`.

Action chunking caveat (important):

- Some models emit action chunks per autoregressive step. For action horizon `n`, each step contains a vector of length `n` for the current embodiment token.
- miniVLA‑vq uses `n=7`; most non‑chunked models use `n=1`.
- Implications:
  - `actions.h5` is `[N_samples, action_horizon]` (not a scalar per step).
  - `hidden_states/generation_step_k.h5` correspond to generating the k‑th embodiment token, whose value is a length‑`n` chunk.
  - Probing targets must account for action horizon when mapping steps → targets.

## Visual Concepts and State Reconstruction

- Visual concepts extractor: `vla_scripts/visual_concepts_extractor.py`
  - `extract_simulator_state(env)`: Structured MuJoCo state (robot joints / eef / object poses / contacts).
  - `identify_key_objects(...)`: Pick T (target), R (region), P (dominant plane), G (gripper).
  - `compute_spatial_relationships(...)`: on / above / near / aligned / touching + distances and heights.
  - Output: `concept_vector` (flat), `concept_names`, plus raw `sim_state` and `key_objects`.

- Trajectory replay & reconstruction: `vla_scripts/reconstruct_trajectory_data.py`
  - `load_episode_metadata(...)`: Fast, metadata‑only loading from `episode_index.h5`.
  - `ParallelReconstructionCollector`: Batch state/image accumulation with HDF5 compression.
  - Replays actions and extracts states/images for concept probing.

- LIBERO upstream geometry and predicates (for reference):
  - State access: `LIBERO/.../envs/env_wrapper.py#get_sim_state`, `sim.data.*`.
  - Object geometry wrappers: `LIBERO/.../envs/object_states/base_object_states.py` (`check_contact`, `check_contain`, `check_ontop`).
  - Region geometry: `LIBERO/.../envs/objects/site_object.py` (`in_box`, `under`) and `.../objects/target_zones.py`.
  - Predicates: `LIBERO/.../envs/predicates/*` and `eval_predicate_fn`.
  - Details guide: `vla_scripts/README_visual_concepts.md` (overview + pointers).

## Linear Probing Experiments

All experiments support three baselines: Normal (original), Randomized pairs (shuffled trajectories), Noise baseline.

1) Hidden → Actions (COMPLETED)
2) Vision → Actions (COMPLETED)
3) Hidden → Visual concepts (IN PROGRESS)
4) Vision → Visual concepts (IN PROGRESS)

Primary scripts:

- `probing/probe.py`: Experiment dispatcher and parameters.
- `probing/linear_probe.py`: Shared linear probe implementation.
- `probing/experiments/experiment_1_hidden_to_actions.py`
- `probing/experiments/experiment_2_vision_to_actions.py`
- `probing/visualize_results.py`, `probing/visualize_experiment_2.py`
- `probing/probe1.sh`, `probing/probe2.sh` (probe3/4 to be added)

## Commands

Linear probing
```bash
cd probing
bash probe1.sh  # Hidden → Actions (completed)
bash probe2.sh  # Vision → Actions (completed)

# Upcoming
bash probe3.sh  # Hidden → Visual concepts (requires reconstruction)
bash probe4.sh  # Vision → Visual concepts (requires reconstruction)
```

State reconstruction
```bash
cd vla_scripts
python test_reconstruction.py
python reconstruct_trajectory_data.py /path/to/optimized_trajectory_data --metadata-only
# Concepts-only replay (no images, no sim_states):
python reconstruct_trajectory_data.py /path/to/optimized_trajectory_data \
  --disable-rendering
# Save concepts and simulator states (enable state IO):
python reconstruct_trajectory_data.py /path/to/optimized_trajectory_data \
  --states-output-dir /path/to/reconstructed_states --enable-state-io \
  --filter-success --max-episodes 5
python reconstruct_trajectory_data.py /path/to/optimized_trajectory_data \
  --states-output-dir /path/to/reconstructed_states --enable-state-io --episode-idx 0
```

HPC / SLURM

- `run.sbatch`: Full evaluation pipeline with data transfer.
- `test.sbatch`: Quick testing script.

## Next Steps

1) Validate reconstruction (`test_reconstruction.py`).
2) Extract and persist concept vectors for selected episodes.
3) Implement `probe3.sh` and `probe4.sh` and wire targets.
4) Tune target definitions for concepts (positions, contacts, relationships) and align with action chunking.
