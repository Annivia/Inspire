# Visual Concepts Extraction Guide

Purpose: Map the exact files and functions involved in simulator state access, geometry / predicate logic (LIBERO), and InSpire’s visual‑concepts extraction and trajectory reconstruction so you know where to read, modify, and debug when implementing probes 3–4.

## Primary Entry Points (InSpire)

- vla_scripts/visual_concepts_extractor.py
  - Key: `VisualConceptsExtractor`, `extract_simulator_state`, `identify_key_objects`, `compute_spatial_relationships`, `extract_visual_concepts`, `extract_visual_concepts_from_state`.
  - What: Reads MuJoCo state, identifies T/R/P/G (Target object, Region, dominant Plane, Gripper), computes on / above / near / aligned / touching, distances and heights, and returns a flat `concept_vector` + `concept_names`.

- vla_scripts/reconstruct_trajectory_data.py
  - Key: `extract_simulator_state` (delegates to the shared extractor), `load_episode_metadata`, `ParallelReconstructionCollector`.
  - What: Replays episodes from the optimized dataset (using stored actions), reconstructing per‑timestep simulator state and optional images, ready for concept extraction.

## LIBERO Upstream (State & Geometry)

- LIBERO/libero/libero/envs/env_wrapper.py
  - Key: `ControlEnv.get_sim_state()`, `sim`, `set_state`, `regenerate_obs_from_state`.
  - What: Provides MuJoCo flat state getter and state control utilities used in replay.

- LIBERO/libero/libero/envs/bddl_base_domain.py
  - Key: `_create_obj_sensors()` defines `obj_pos` (`sim.data.body_xpos`), `obj_quat` (`sim.data.body_xquat`), and relative transforms to the gripper (`obj_to_eef_pos`, `obj_to_eef_quat`) via `world_pose_in_gripper`.
  - What: Canonical object pose observables and relative transforms used by tasks.

- LIBERO/libero/libero/envs/object_states/base_object_states.py
  - Key: `ObjectState.get_geom_state`, `check_contact`, `check_contain`, `check_ontop`; plus `SiteObjectState` site‑based variants.
  - What: Geometry wrappers that predicates use (contacts via environment, containment via site geometry, “on top” via z‑order + contact + xy proximity).

- LIBERO/libero/libero/envs/objects/site_object.py
  - Key: `SiteObject.in_box`, `SiteObject.under`.
  - What: Axis‑aligned containment and “under/on” checks in site (region) coordinates.

- LIBERO/libero/libero/envs/objects/target_zones.py
  - Key: `TargetZone.in_box`, `TargetZone.on_top`.
  - What: Region logic for flat support zones (plates/coasters/trays).

- LIBERO/libero/libero/envs/predicates/base_predicates.py
  - Key: `InContactPredicateFn`, `In`, `On`, `Up`, etc.
  - What: High‑level predicates built on the object‑state geometry.

- LIBERO/libero/libero/envs/predicates/__init__.py
  - Key: `VALIDATE_PREDICATE_FN_DICT`, `eval_predicate_fn`.
  - What: Predicate registry and evaluator (note: `incontact` is commented out by default; can be re‑enabled via `update_predicate_fn_dict`).

- LIBERO/libero/libero/envs/venv.py
  - Key: Forwards `get_sim_state()` through vectorized envs.
  - What: Matching `env_wrapper.get_sim_state()` in vectorized settings.

- LIBERO/libero/libero/envs/bddl_utils.py
  - Key: `robosuite_parse_problem`, `get_problem_info`.
  - What: Parses BDDL to produce `initial_state`, `goal_state`, objects, regions, and language strings.

## Experiment Helpers (Env + Images)

- experiments/robot/libero/libero_utils.py
  - Key: `get_libero_env`, `get_libero_image`, `get_libero_dummy_action`.
  - What: Initialize a LIBERO env for a task, extract and resize images.

## How Things Connect

- Simulator state → positions / orientations / contacts:
  - LIBERO exposes MuJoCo state (`get_sim_state`, `sim.data.*`) and geometry utilities (`ObjectState`, `SiteObject`, predicates). 

- InSpire concepts:
  - `vla_scripts/visual_concepts_extractor.py` reads LIBERO state directly, finds key objects (T/R/P/G), and computes relationships. It implements robust contact / near / above / on / aligned logic using MuJoCo contacts and per‑body extents.

- Trajectory replay:
  - `vla_scripts/reconstruct_trajectory_data.py` replays actions from `/optimized_trajectory_data/` and calls the shared extractor to produce per‑step concept vectors for probes 3–4.

## PDDL / BDDL State Note

- LIBERO does not ship a one‑call “export full symbolic state” API. You can enumerate and evaluate predicates yourself with:
  - `env.object_states_dict` (objects), and
  - `from libero.libero.envs.predicates import eval_predicate_fn` to test `on`, `in`, etc. over object pairs.
  - Enable additional predicates by updating the registry:
    ```python
    from libero.libero.envs.predicates import update_predicate_fn_dict
    update_predicate_fn_dict("incontact", "InContactPredicateFn")
    ```

## Common Workflows

- Online concept extraction during evaluation
  1) Build env with `experiments/robot/libero/libero_utils.get_libero_env(...)`.
  2) Call `from vla_scripts.visual_concepts_extractor import extract_visual_concepts` with `(env, task_description)`.

- Extract concepts from pre‑extracted state
  1) Use `extract_visual_concepts_from_state(sim_state, task_description)` if you already have state dicts.

- Reconstruct from dataset and extract
  1) Concepts-only replay (no images, no `sim_states/`):
     ```bash
     python vla_scripts/reconstruct_trajectory_data.py /path/to/optimized_trajectory_data --disable-rendering
     ```
     This replays episodes and writes per-task concept CSVs under `<dataset_or_states_root>/concepts/`.

  2) Replay and also save simulator states (`sim_states/`):
     ```bash
     python vla_scripts/reconstruct_trajectory_data.py /path/to/optimized_trajectory_data \
       --states-output-dir /path/to/reconstructed_states --enable-state-io
     ```
     This enables `state_io` and merges chunks into `sim_states/` for offline analysis.

## Debugging Pointers

- Object identity: `sim_state['object_names']` (bytes) and `sim_state['object_positions']` must align; extractor warns and truncates to min length on mismatch.
- Contacts: Check `sim.data.ncon` and contact arrays (`contact_*`) when validating touching / on.
- Size‑normalized distances: `_get_body_extent` and thresholds in `compute_spatial_relationships` control “near” sensitivity.
- Key object selection: `identify_key_objects` uses keyword lists + heuristics; adjust if task naming differs.

## Quick Reference (Paths → Responsibilities)

- InSpire
  - `vla_scripts/visual_concepts_extractor.py`: InSpire concepts: state extraction, key objects, relationships, concept vector.
  - `vla_scripts/reconstruct_trajectory_data.py`: Trajectory replay, metadata loading, state/image saving, extractor integration.

- LIBERO
  - `LIBERO/.../envs/env_wrapper.py`: `get_sim_state`, state control.
  - `LIBERO/.../envs/bddl_base_domain.py`: Object pose sensors, relative transforms, object lookup.
  - `LIBERO/.../envs/object_states/base_object_states.py`: `get_geom_state`, contact / containment / on‑top checks.
  - `LIBERO/.../envs/objects/site_object.py`: Region containment (`in_box`) and “under/on” (`under`).
  - `LIBERO/.../envs/objects/target_zones.py`: Region variants for target zones.
  - `LIBERO/.../envs/predicates/*`: Predicate registry and implementations (symbolic truth evaluation).
  - `LIBERO/.../envs/venv.py`: Vectorized `get_sim_state` wrappers.
  - `LIBERO/.../envs/bddl_utils.py`: BDDL parsing to `initial_state`, `goal_state`, etc.
