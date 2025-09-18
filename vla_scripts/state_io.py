#!/usr/bin/env python3
"""
Shared IO utilities for simulator state and concept relations.

This module defines a single source of truth for:
- Paths resolution
- Per-process chunk writing (HDF5 + JSON manifests)
- Final combination / merge into the stable layout under sim_states/

It is used by both trajectory data collection and reconstruction scripts to
guarantee an identical on-disk structure.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np


SCHEMA_VERSION = "1.0"


def _sanitize(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]", "", s)
    return s or "task"


def resolve_paths(dataset_root: str) -> Dict[str, Path]:
    root = Path(dataset_root)
    paths = {
        "root": root,
        "sim_states": root / "sim_states",
        "concepts": root / "concepts",
        "temp_recon": root / "temp_state_reconstruction",
        "temp_collect": root / "temp_trajectory_processing",
    }
    return paths


def task_dir(dataset_root: str, task_name: str) -> Path:
    return resolve_paths(dataset_root)["sim_states"] / "tasks" / _sanitize(task_name)


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class _TaskObjectsBuffer:
    names: Optional[List[str]] = None
    body_ids: Optional[np.ndarray] = None
    extents: Optional[np.ndarray] = None
    positions_list: List[np.ndarray] = field(default_factory=list)
    orientations_list: List[np.ndarray] = field(default_factory=list)
    # contacts per-sample
    contact_body1: List[np.ndarray] = field(default_factory=list)
    contact_body2: List[np.ndarray] = field(default_factory=list)
    contact_pos: List[np.ndarray] = field(default_factory=list)
    contact_dist: List[np.ndarray] = field(default_factory=list)


class StateChunkWriter:
    """Accumulates per-sample states and writes chunk files for later combination.

    Usage:
        w = StateChunkWriter(dataset_root, process_id, temp_dir=None)
        ... append_* ...
        w.flush()
    """

    def __init__(self, dataset_root: str, process_id: int, temp_dir: Optional[str] = None):
        self.dataset_root = Path(dataset_root)
        self.process_id = int(process_id)

        paths = resolve_paths(dataset_root)
        if temp_dir:
            self.temp_dir = Path(temp_dir)
        else:
            # default under reconstruction-style path
            self.temp_dir = paths["temp_recon"] / f"reconstruction_process_{self.process_id}"
        _ensure_dir(self.temp_dir)

        # Accumulators for core states
        self.core_robot_qpos: List[np.ndarray] = []
        self.core_robot_qvel: List[np.ndarray] = []
        self.core_ee_pos: List[np.ndarray] = []
        self.core_ee_quat: List[np.ndarray] = []
        self.core_time: List[float] = []

        # Tasks buffers
        self.tasks: Dict[str, _TaskObjectsBuffer] = {}

        # Episodes metadata
        self.episodes: List[Dict] = []

    # ------------- Append APIs -------------
    def append_core(self, robot_joint_pos: np.ndarray, robot_joint_vel: np.ndarray,
                    ee_pos: np.ndarray, ee_quat: np.ndarray, time: float):
        self.core_robot_qpos.append(np.asarray(robot_joint_pos, dtype=np.float32))
        self.core_robot_qvel.append(np.asarray(robot_joint_vel, dtype=np.float32))
        self.core_ee_pos.append(np.asarray(ee_pos, dtype=np.float32))
        self.core_ee_quat.append(np.asarray(ee_quat, dtype=np.float32))
        self.core_time.append(float(time))

    def _get_task_buf(self, task_name: str) -> _TaskObjectsBuffer:
        key = str(task_name)
        if key not in self.tasks:
            self.tasks[key] = _TaskObjectsBuffer()
        return self.tasks[key]

    def set_task_objects_metadata(self, task_name: str,
                                  object_names: List[str],
                                  object_body_ids: np.ndarray,
                                  object_extents: np.ndarray):
        buf = self._get_task_buf(task_name)
        if buf.names is None:
            buf.names = list(object_names)
            buf.body_ids = np.asarray(object_body_ids, dtype=np.int32)
            buf.extents = np.asarray(object_extents, dtype=np.float32)
        else:
            # validate consistency
            assert buf.names == list(object_names), "Inconsistent object_names across chunks for task"
            assert np.array_equal(buf.body_ids, np.asarray(object_body_ids, dtype=np.int32)), "Inconsistent object_body_ids"
            assert np.allclose(buf.extents, np.asarray(object_extents, dtype=np.float32)), "Inconsistent object_extents"

    def append_task_objects(self, task_name: str,
                            object_positions: np.ndarray,
                            object_orientations: np.ndarray):
        buf = self._get_task_buf(task_name)
        self.tasks[task_name].positions_list.append(np.asarray(object_positions, dtype=np.float32))
        self.tasks[task_name].orientations_list.append(np.asarray(object_orientations, dtype=np.float32))

    def append_contacts(self, task_name: str,
                        body1_ids: np.ndarray,
                        body2_ids: np.ndarray,
                        contact_pos: np.ndarray,
                        contact_dist: np.ndarray):
        buf = self._get_task_buf(task_name)
        buf.contact_body1.append(np.asarray(body1_ids, dtype=np.int32))
        buf.contact_body2.append(np.asarray(body2_ids, dtype=np.int32))
        buf.contact_pos.append(np.asarray(contact_pos, dtype=np.float32))
        buf.contact_dist.append(np.asarray(contact_dist, dtype=np.float32))

    def append_episode_meta(self, meta: Dict):
        """Append per-episode metadata.

        Expected fields (some optional):
        - episode_idx, task_name, task_id, language_instruction, success,
          num_timesteps, start_idx, end_idx, orig_start_idx, orig_end_idx
        """
        self.episodes.append(dict(meta))

    # ------------- Flush chunk to disk -------------
    def flush(self) -> Dict:
        summary = {"process_id": self.process_id}
        comp = dict(compression="gzip", compression_opts=6, shuffle=True)

        # Core states chunk
        if self.core_robot_qpos:
            core_path = self.temp_dir / "core_states_chunk.h5"
            with h5py.File(core_path, "w") as f:
                f.create_dataset("robot_joint_pos", data=np.stack(self.core_robot_qpos, axis=0), **comp)
                f.create_dataset("robot_joint_vel", data=np.stack(self.core_robot_qvel, axis=0), **comp)
                f.create_dataset("ee_pos", data=np.stack(self.core_ee_pos, axis=0), **comp)
                f.create_dataset("ee_quat", data=np.stack(self.core_ee_quat, axis=0), **comp)
                f.create_dataset("time", data=np.asarray(self.core_time, dtype=np.float32), **comp)
            summary["core_states_chunk"] = str(core_path)

        # Task-specific chunks
        for task_name, buf in self.tasks.items():
            tdir = self.temp_dir / "tasks" / _sanitize(task_name)
            _ensure_dir(tdir)

            # Object states
            if buf.positions_list:
                obj_path = tdir / "object_states_chunk.h5"
                positions = np.stack(buf.positions_list, axis=0)  # [N_chunk_samples, N_obj, 3]
                orientations = np.stack(buf.orientations_list, axis=0)
                with h5py.File(obj_path, "w") as f:
                    f.create_dataset("object_positions", data=positions, **comp)
                    f.create_dataset("object_orientations", data=orientations, **comp)
                    if buf.names is not None:
                        names_arr = np.array([n.encode("utf-8") for n in buf.names], dtype="S64")
                        f.create_dataset("object_names", data=names_arr)
                    if buf.body_ids is not None:
                        f.create_dataset("object_body_ids", data=buf.body_ids)
                    if buf.extents is not None:
                        f.create_dataset("object_extents", data=buf.extents)
                summary.setdefault("task_object_chunks", {})[_sanitize(task_name)] = str(obj_path)

            # Contacts CSR
            # Build CSR arrays from per-sample lists within this chunk
            if buf.contact_body1:
                indptr = [0]
                b1_all = []
                b2_all = []
                pos_all = []
                dist_all = []
                for i in range(len(buf.contact_body1)):
                    n = int(len(buf.contact_body1[i]))
                    indptr.append(indptr[-1] + n)
                    if n:
                        b1_all.append(buf.contact_body1[i])
                        b2_all.append(buf.contact_body2[i])
                        pos_all.append(buf.contact_pos[i])
                        dist_all.append(buf.contact_dist[i])
                b1 = np.concatenate(b1_all, axis=0) if b1_all else np.zeros((0,), dtype=np.int32)
                b2 = np.concatenate(b2_all, axis=0) if b2_all else np.zeros((0,), dtype=np.int32)
                pos = np.concatenate(pos_all, axis=0) if pos_all else np.zeros((0, 3), dtype=np.float32)
                dst = np.concatenate(dist_all, axis=0) if dist_all else np.zeros((0,), dtype=np.float32)
                csr_path = tdir / "contacts_chunk_csr.h5"
                with h5py.File(csr_path, "w") as f:
                    f.create_dataset("indptr", data=np.asarray(indptr, dtype=np.int64))
                    f.create_dataset("body1_ids", data=b1)
                    f.create_dataset("body2_ids", data=b2)
                    f.create_dataset("contact_pos", data=pos, **comp)
                    f.create_dataset("contact_dist", data=dst, **comp)
                summary.setdefault("task_contact_chunks", {})[_sanitize(task_name)] = str(csr_path)

        # Episodes list
        if self.episodes:
            episodes_path = self.temp_dir / "episodes_chunk.json"
            with open(episodes_path, "w") as f:
                json.dump({"episodes": self.episodes, "schema_version": SCHEMA_VERSION}, f, indent=2)
            summary["episodes_chunk"] = str(episodes_path)

        # Manifest
        manifest = {
            "process_id": self.process_id,
            "temp_dir": str(self.temp_dir),
            "schema_version": SCHEMA_VERSION,
            "summary": summary,
        }
        with open(self.temp_dir / "chunk_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        return summary


def _find_process_dirs(dataset_root: str) -> List[Path]:
    paths = resolve_paths(dataset_root)
    dirs: List[Path] = []
    for base in [paths["temp_recon"], paths["temp_collect"]]:
        if base.exists():
            for d in base.glob("**/*"):
                if d.is_dir() and (d.name.startswith("reconstruction_process_") or d.name.startswith("gpu_process_")):
                    dirs.append(d)
    # sort by trailing integer id if present
    def _pid(p: Path) -> int:
        m = re.search(r"(\d+)$", p.name)
        return int(m.group(1)) if m else 0
    dirs.sort(key=_pid)
    return dirs


def combine_state_chunks(dataset_root: str) -> Dict:
    """Combine all chunk files under temp dirs into the stable sim_states/ layout."""
    root = Path(dataset_root)
    paths = resolve_paths(dataset_root)
    out_sim = paths["sim_states"]
    _ensure_dir(out_sim)
    _ensure_dir(out_sim / "tasks")

    proc_dirs = _find_process_dirs(dataset_root)
    if not proc_dirs:
        return {"merged": False, "reason": "no process dirs found"}

    # Aggregators
    core_qpos = []
    core_qvel = []
    core_ee_pos = []
    core_ee_quat = []
    core_time = []

    # Episodes
    episodes_all: List[Dict] = []

    # Task aggregators
    task_objects: Dict[str, Dict] = {}
    task_contacts: Dict[str, Dict] = {}
    # For building task-local episode ranges
    task_episode_order: Dict[str, List[Tuple[int, int]]] = {}  # list of (episode_idx, num_timesteps)

    # Read chunks in process order
    for pd in proc_dirs:
        # Core
        core_path = pd / "core_states_chunk.h5"
        if core_path.exists():
            with h5py.File(core_path, "r") as f:
                core_qpos.append(f["robot_joint_pos"][...])
                core_qvel.append(f["robot_joint_vel"][...])
                core_ee_pos.append(f["ee_pos"][...])
                core_ee_quat.append(f["ee_quat"][...])
                core_time.append(f["time"][...])

        # Episodes
        ep_path = pd / "episodes_chunk.json"
        if ep_path.exists():
            with open(ep_path, "r") as f:
                data = json.load(f)
                episodes = data.get("episodes", [])
                episodes_all.extend(episodes)
                for ep in episodes:
                    tn = _sanitize(ep.get("task_name", ""))
                    nl = int(ep.get("num_timesteps", 0))
                    task_episode_order.setdefault(tn, []).append((int(ep.get("episode_idx", 0)), nl))

        # Task-specific
        troot = pd / "tasks"
        if troot.exists():
            for tdir in troot.iterdir():
                if not tdir.is_dir():
                    continue
                tkey = tdir.name
                # Object states
                obj_chunk = tdir / "object_states_chunk.h5"
                if obj_chunk.exists():
                    with h5py.File(obj_chunk, "r") as f:
                        positions = f["object_positions"][...]
                        orientations = f["object_orientations"][...]
                        names = f.get("object_names")
                        body_ids = f.get("object_body_ids")
                        extents = f.get("object_extents")
                        tgt = task_objects.setdefault(tkey, {
                            "positions": [], "orientations": [],
                            "names": None, "body_ids": None, "extents": None,
                        })
                        tgt["positions"].append(positions)
                        tgt["orientations"].append(orientations)
                        if names is not None:
                            nval = [x.decode("utf-8") for x in names[...]]
                            if tgt["names"] is None:
                                tgt["names"] = nval
                            else:
                                assert tgt["names"] == nval, "Inconsistent object_names across chunks"
                        if body_ids is not None:
                            bval = body_ids[...]
                            if tgt["body_ids"] is None:
                                tgt["body_ids"] = bval
                            else:
                                assert np.array_equal(tgt["body_ids"], bval), "Inconsistent body_ids across chunks"
                        if extents is not None:
                            eval_ = extents[...]
                            if tgt["extents"] is None:
                                tgt["extents"] = eval_
                            else:
                                assert np.allclose(tgt["extents"], eval_), "Inconsistent extents across chunks"

                # Contacts CSR
                csr_chunk = tdir / "contacts_chunk_csr.h5"
                if csr_chunk.exists():
                    with h5py.File(csr_chunk, "r") as f:
                        indptr = f["indptr"][...]
                        b1 = f["body1_ids"][...]
                        b2 = f["body2_ids"][...]
                        pos = f["contact_pos"][...]
                        dist = f["contact_dist"][...]
                        tgt = task_contacts.setdefault(tkey, {
                            "indptr": [], "b1": [], "b2": [], "pos": [], "dist": []
                        })
                        tgt["indptr"].append(indptr)
                        tgt["b1"].append(b1)
                        tgt["b2"].append(b2)
                        tgt["pos"].append(pos)
                        tgt["dist"].append(dist)

    # Write core_states.h5
    if core_qpos:
        with h5py.File(out_sim / "core_states.h5", "w") as f:
            comp = dict(compression="gzip", compression_opts=6, shuffle=True)
            f.create_dataset("robot_joint_pos", data=np.concatenate(core_qpos, axis=0), **comp)
            f.create_dataset("robot_joint_vel", data=np.concatenate(core_qvel, axis=0), **comp)
            f.create_dataset("ee_pos", data=np.concatenate(core_ee_pos, axis=0), **comp)
            f.create_dataset("ee_quat", data=np.concatenate(core_ee_quat, axis=0), **comp)
            f.create_dataset("time", data=np.concatenate(core_time, axis=0), **comp)

    # Build episodes_index.h5 with global and task-local ranges
    # Global start/end are cumulative over episodes_all in read order
    g_starts = []
    g_ends = []
    cur = 0
    for ep in episodes_all:
        n = int(ep.get("num_timesteps", 0))
        g_starts.append(cur)
        cur += n
        g_ends.append(cur - 1 if n > 0 else cur)

    # Task-local ranges: cumulative per task according to task_episode_order
    task_local_map: Dict[str, Dict[int, Tuple[int, int]]] = {}
    for tkey, eps in task_episode_order.items():
        tl_cur = 0
        tmap = {}
        for ep_idx, n in eps:
            tl_start = tl_cur
            tl_cur += n
            tl_end = tl_cur - 1 if n > 0 else tl_cur
            tmap[int(ep_idx)] = (tl_start, tl_end)
        task_local_map[tkey] = tmap

    # Prepare columns
    if episodes_all:
        epi_path = out_sim / "episodes_index.h5"
        with h5py.File(epi_path, "w") as f:
            def _arr(vals, dtype=None):
                return np.asarray(vals if vals is not None else [], dtype=dtype) if vals else np.zeros((0,), dtype=dtype or np.int64)

            f.create_dataset("episode_idx", data=_arr([int(ep.get("episode_idx", 0)) for ep in episodes_all], np.int64))
            if any("episode_id" in ep for ep in episodes_all):
                f.create_dataset("episode_id", data=_arr([int(ep.get("episode_id", -1)) for ep in episodes_all], np.int64))
            names = [str(ep.get("task_name", "")) for ep in episodes_all]
            f.create_dataset("task_name", data=np.array([n.encode("utf-8") for n in names], dtype="S128"))
            # Language instruction if available
            if any("language_instruction" in ep for ep in episodes_all):
                langs = [str(ep.get("language_instruction", "")) for ep in episodes_all]
                f.create_dataset("language_instruction", data=np.array([s.encode("utf-8") for s in langs], dtype="S256"))
            if any("task_id" in ep for ep in episodes_all):
                f.create_dataset("task_id", data=_arr([int(ep.get("task_id", -1)) for ep in episodes_all], np.int64))
            f.create_dataset("num_timesteps", data=_arr([int(ep.get("num_timesteps", 0)) for ep in episodes_all], np.int64))
            f.create_dataset("success", data=_arr([int(bool(ep.get("success", False))) for ep in episodes_all], np.int8))
            f.create_dataset("start_idx", data=_arr(g_starts, np.int64))
            f.create_dataset("end_idx", data=_arr(g_ends, np.int64))
            # Optional carry-through of original alignment
            if any("orig_start_idx" in ep for ep in episodes_all):
                f.create_dataset("orig_start_idx", data=_arr([int(ep.get("orig_start_idx", -1)) for ep in episodes_all], np.int64))
            if any("orig_end_idx" in ep for ep in episodes_all):
                f.create_dataset("orig_end_idx", data=_arr([int(ep.get("orig_end_idx", -1)) for ep in episodes_all], np.int64))
            # Task-local using sanitized task key
            tl_starts = []
            tl_ends = []
            for ep in episodes_all:
                tkey = _sanitize(ep.get("task_name", ""))
                ep_idx = int(ep.get("episode_idx", 0))
                s, e = task_local_map.get(tkey, {}).get(ep_idx, (-1, -1))
                tl_starts.append(s)
                tl_ends.append(e)
            f.create_dataset("task_local_start_idx", data=_arr(tl_starts, np.int64))
            f.create_dataset("task_local_end_idx", data=_arr(tl_ends, np.int64))

    # Write per-task object_states and contacts CSR
    for tkey, data in task_objects.items():
        tdir = out_sim / "tasks" / tkey
        _ensure_dir(tdir)
        with h5py.File(tdir / "object_states.h5", "w") as f:
            comp = dict(compression="gzip", compression_opts=6, shuffle=True)
            f.create_dataset("object_positions", data=np.concatenate(data["positions"], axis=0), **comp)
            f.create_dataset("object_orientations", data=np.concatenate(data["orientations"], axis=0), **comp)
            if data["names"] is not None:
                f.create_dataset("object_names", data=np.array([n.encode("utf-8") for n in data["names"]], dtype="S64"))
            if data["body_ids"] is not None:
                f.create_dataset("object_body_ids", data=data["body_ids"])
            if data["extents"] is not None:
                f.create_dataset("object_extents", data=data["extents"])

    for tkey, data in task_contacts.items():
        tdir = out_sim / "tasks" / tkey
        _ensure_dir(tdir)
        # Concatenate CSR chunks by offsetting indptr
        indptr_all = []
        offset = 0
        for ind in data["indptr"]:
            indptr_all.append(ind + offset)
            offset += int(ind[-1])
        indptr = np.concatenate(indptr_all, axis=0) if indptr_all else np.zeros((0,), dtype=np.int64)
        b1 = np.concatenate(data["b1"], axis=0) if data["b1"] else np.zeros((0,), dtype=np.int32)
        b2 = np.concatenate(data["b2"], axis=0) if data["b2"] else np.zeros((0,), dtype=np.int32)
        pos = np.concatenate(data["pos"], axis=0) if data["pos"] else np.zeros((0, 3), dtype=np.float32)
        dist = np.concatenate(data["dist"], axis=0) if data["dist"] else np.zeros((0,), dtype=np.float32)
        with h5py.File(tdir / "contacts_csr.h5", "w") as f:
            comp = dict(compression="gzip", compression_opts=6, shuffle=True)
            f.create_dataset("indptr", data=indptr)
            f.create_dataset("body1_ids", data=b1)
            f.create_dataset("body2_ids", data=b2)
            f.create_dataset("contact_pos", data=pos, **comp)
            f.create_dataset("contact_dist", data=dist, **comp)

    # Summary
    summary = {
        "merged": True,
        "schema_version": SCHEMA_VERSION,
        "process_dirs": [str(p) for p in proc_dirs],
        "num_core_samples": int(sum(arr.shape[0] for arr in core_qpos)) if core_qpos else 0,
        "num_episodes": len(episodes_all),
        "tasks": sorted(set(task_objects.keys()) | set(task_contacts.keys())),
    }
    with open(out_sim / "dataset_state_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary
