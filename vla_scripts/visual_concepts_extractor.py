#!/usr/bin/env python3
"""
Visual Concepts Extractor

Shared infrastructure for extracting visual concepts from LIBERO simulator states.
Used by both trajectory data collection and reconstruction scripts.

Key Objects:
- T = target object (Container role: cup, mug, bowl)
- R = target region (FlatSupport role: coaster, plate, tray, taped spot). If no explicit target exists, set R = P.
- P = dominant support surface (workspace plane; often the table)  
- G = gripper / end effector

Key Relationships:
- on(x, y) → Container supported by FlatSupport
- above(x, y) → vertical displacement without contact
- near(x, y) → within a size-normalized distance threshold
- aligned_to(x, y, tol) → horizontal COM alignment within tolerance
- touching(x, y) → contact detected
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict


class VisualConceptsExtractor:
    """
    Extracts visual/spatial concepts from LIBERO simulator states for probes 3 and 4.
    """
    
    def __init__(self, distance_threshold: float = 0.1, alignment_tolerance: float = 0.05):
        """
        Args:
            distance_threshold: Distance threshold for near() relationship (in meters)
            alignment_tolerance: Tolerance for aligned_to() relationship (in meters)
        """
        self.distance_threshold = distance_threshold
        self.alignment_tolerance = alignment_tolerance
        
        # LIBERO object categorization
        self.container_keywords = ['cup', 'mug', 'bowl', 'pot', 'can', 'container']
        self.flat_support_keywords = ['coaster', 'plate', 'tray', 'spot', 'surface', 'table', 'desk']
        self.workspace_keywords = ['table', 'desk', 'surface', 'workspace', 'plane']
        
    def extract_simulator_state(self, env) -> Dict[str, Any]:
        """
        Extract comprehensive simulator state as structured numpy arrays.
        Replacement for the function in reconstruct_trajectory_data.py
        
        Args:
            env: LIBERO environment
            
        Returns:
            Dict containing structured simulator state data
        """
        sim_state = {}
        
        try:
            # Get MuJoCo simulation data
            sim = env.sim
            
            # Robot joint positions and velocities (fixed size tensors)
            sim_state['robot_joint_pos'] = sim.data.qpos[:7].copy().astype(np.float32)
            sim_state['robot_joint_vel'] = sim.data.qvel[:7].copy().astype(np.float32)
            
            # End-effector position and orientation with fallbacks
            ee_pos, ee_quat, gripper_body_id = self._get_gripper_info(sim)
            sim_state['ee_pos'] = ee_pos.astype(np.float32)  # [3]
            sim_state['ee_quat'] = ee_quat.astype(np.float32)  # [4]
            sim_state['gripper_body_id'] = np.int32(gripper_body_id) if gripper_body_id is not None else np.int32(-1)
            
            # Extract object information with proper filtering
            object_positions = []
            object_orientations = []
            object_names = []
            object_body_ids = []
            object_extents = []  # For size-normalized distance calculations
            
            for body_id in range(sim.model.nbody):
                body_name = sim.model.body_id2name(body_id)
                # Skip robot bodies, world body, and None names
                if (body_name and 
                    not body_name.startswith('robot0') and 
                    body_name.lower() not in ['world', 'worldbody']):
                    
                    pos = sim.data.body_xpos[body_id].copy()
                    quat = sim.data.body_xquat[body_id].copy()
                    
                    # Calculate object extent (bounding box size)
                    extent = self._get_body_extent(sim, body_id)
                    
                    object_positions.append(pos)
                    object_orientations.append(quat)
                    object_names.append(body_name)
                    object_body_ids.append(body_id)
                    object_extents.append(extent)
            
            # Convert to numpy arrays for efficient storage
            if object_positions:
                sim_state['object_positions'] = np.stack(object_positions, axis=0).astype(np.float32)  # [N_objects, 3]
                sim_state['object_orientations'] = np.stack(object_orientations, axis=0).astype(np.float32)  # [N_objects, 4]
                sim_state['object_body_ids'] = np.array(object_body_ids, dtype=np.int32)  # [N_objects]
                sim_state['object_extents'] = np.array(object_extents, dtype=np.float32)  # [N_objects]
                # Store names separately for indexing
                sim_state['object_names'] = np.array([name.encode('utf-8') for name in object_names], dtype='S64')
            else:
                sim_state['object_positions'] = np.zeros((0, 3), dtype=np.float32)
                sim_state['object_orientations'] = np.zeros((0, 4), dtype=np.float32)
                sim_state['object_body_ids'] = np.array([], dtype=np.int32)
                sim_state['object_extents'] = np.array([], dtype=np.float32)
                sim_state['object_names'] = np.array([], dtype='S64')
            
            # Enhanced contact information with body mapping
            if sim.data.ncon > 0:
                contact_geom1 = np.array([sim.data.contact[i].geom1 for i in range(sim.data.ncon)], dtype=np.int32)
                contact_geom2 = np.array([sim.data.contact[i].geom2 for i in range(sim.data.ncon)], dtype=np.int32)
                contact_pos = np.array([sim.data.contact[i].pos.copy() for i in range(sim.data.ncon)], dtype=np.float32)
                contact_dist = np.array([sim.data.contact[i].dist for i in range(sim.data.ncon)], dtype=np.float32)
                
                # Map geoms to bodies for contact analysis
                contact_body1 = np.array([sim.model.geom_bodyid[geom1] for geom1 in contact_geom1], dtype=np.int32)
                contact_body2 = np.array([sim.model.geom_bodyid[geom2] for geom2 in contact_geom2], dtype=np.int32)
                
                sim_state['contact_geom1'] = contact_geom1
                sim_state['contact_geom2'] = contact_geom2  
                sim_state['contact_body1'] = contact_body1  # [N_contacts]
                sim_state['contact_body2'] = contact_body2  # [N_contacts]
                sim_state['contact_pos'] = contact_pos  # [N_contacts, 3]
                sim_state['contact_dist'] = contact_dist  # [N_contacts]
            else:
                sim_state['contact_geom1'] = np.array([], dtype=np.int32)
                sim_state['contact_geom2'] = np.array([], dtype=np.int32)
                sim_state['contact_body1'] = np.array([], dtype=np.int32)
                sim_state['contact_body2'] = np.array([], dtype=np.int32)
                sim_state['contact_pos'] = np.zeros((0, 3), dtype=np.float32)
                sim_state['contact_dist'] = np.array([], dtype=np.float32)
            
            # Time and physics info as scalars
            sim_state['time'] = np.float32(sim.data.time)
            
        except Exception as e:
            print(f"[visual-concepts] WARNING: Could not extract simulator state: {e}")
            # Fallback to empty tensors
            sim_state = {
                'robot_joint_pos': np.zeros(7, dtype=np.float32),
                'robot_joint_vel': np.zeros(7, dtype=np.float32),
                'ee_pos': np.zeros(3, dtype=np.float32),
                'ee_quat': np.array([0, 0, 0, 1], dtype=np.float32),
                'gripper_body_id': np.int32(-1),
                'object_positions': np.zeros((0, 3), dtype=np.float32),
                'object_orientations': np.zeros((0, 4), dtype=np.float32),
                'object_body_ids': np.array([], dtype=np.int32),
                'object_extents': np.array([], dtype=np.float32),
                'object_names': np.array([], dtype='S64'),
                'contact_geom1': np.array([], dtype=np.int32),
                'contact_geom2': np.array([], dtype=np.int32),
                'contact_body1': np.array([], dtype=np.int32),
                'contact_body2': np.array([], dtype=np.int32),
                'contact_pos': np.zeros((0, 3), dtype=np.float32),
                'contact_dist': np.array([], dtype=np.float32),
                'time': np.float32(0.0),
                'error': str(e)
            }
            
        return sim_state
    
    def _get_gripper_info(self, sim) -> Tuple[np.ndarray, np.ndarray, Optional[int]]:
        """
        Get gripper position, orientation, and body ID with fallbacks.
        
        Returns:
            Tuple of (position, quaternion, body_id)
        """
        # Try primary gripper site names
        gripper_sites = ['gripper0_grip_site', 'grip_site', 'eef_site', 'gripper_site']
        ee_pos = None
        
        for site_name in gripper_sites:
            try:
                site_id = sim.model.site_name2id(site_name)
                ee_pos = sim.data.site_xpos[site_id].copy()
                break
            except:
                continue
        
        if ee_pos is None:
            # Fallback to end-effector body position
            print("[visual-concepts] WARNING: Could not find gripper site, using fallback")
            ee_pos = np.array([0.0, 0.0, 0.0])
        
        # Try primary gripper body names for orientation and body ID
        gripper_bodies = ['gripper0_eef', 'eef', 'gripper_eef', 'gripper']
        ee_quat = None
        gripper_body_id = None
        
        for body_name in gripper_bodies:
            try:
                gripper_body_id = sim.model.body_name2id(body_name)
                ee_quat = sim.data.get_body_xquat(body_name).copy()
                break
            except:
                continue
        
        if ee_quat is None:
            print("[visual-concepts] WARNING: Could not find gripper body, using default quaternion")
            ee_quat = np.array([0.0, 0.0, 0.0, 1.0])
        
        return ee_pos, ee_quat, gripper_body_id
    
    def _get_body_extent(self, sim, body_id: int) -> float:
        """
        Calculate object extent (size) for size-normalized distance calculations.
        
        Args:
            sim: MuJoCo simulation
            body_id: Body ID
            
        Returns:
            Characteristic size (maximum dimension of bounding box)
        """
        try:
            # Find all geoms associated with this body
            body_geoms = []
            for geom_id in range(sim.model.ngeom):
                if sim.model.geom_bodyid[geom_id] == body_id:
                    body_geoms.append(geom_id)
            
            if not body_geoms:
                return 0.1  # Default size
            
            # Calculate bounding box of all geoms
            max_extent = 0.0
            for geom_id in body_geoms:
                geom_size = sim.model.geom_size[geom_id]
                # Use maximum dimension as characteristic size
                if len(geom_size) > 0:
                    max_extent = max(max_extent, np.max(geom_size) * 2)  # Diameter
            
            return max(max_extent, 0.01)  # Minimum size threshold
            
        except Exception as e:
            return 0.1  # Fallback size
    
    def identify_key_objects(self, sim_state: Dict[str, Any], task_description: str = "") -> Dict[str, Optional[int]]:
        """
        Identify key objects T, R, P, G from simulator state.
        
        Args:
            sim_state: Simulator state from extract_simulator_state()
            task_description: Task description for context
            
        Returns:
            Dict mapping object roles to object indices (or None if not found)
        """
        object_names = [name.decode('utf-8') if hasattr(name, 'decode') else str(name) 
                       for name in sim_state['object_names']]
        object_positions = sim_state['object_positions']
        ee_pos = sim_state['ee_pos']
        
        key_objects = {
            'T': None,  # Target object (Container)
            'R': None,  # Target region (FlatSupport)  
            'P': None,  # Dominant support surface
            'G': None   # Gripper (always use end-effector position)
        }
        
        if len(object_names) == 0 or len(object_positions) == 0:
            return key_objects
        
        # Validate that object_names and object_positions have the same length
        if len(object_names) != len(object_positions):
            print(f"[visual-concepts] WARNING: Mismatch between object_names ({len(object_names)}) and object_positions ({len(object_positions)})")
            # Use the minimum length to avoid index errors
            max_objects = min(len(object_names), len(object_positions))
            object_names = object_names[:max_objects]
        else:
            max_objects = len(object_names)
        
        # G = Gripper/end-effector (always available)
        key_objects['G'] = -1  # Special index for end-effector
        
        # Find P = dominant support surface (workspace plane, often table)
        workspace_candidates = []
        for i in range(max_objects):
            name = object_names[i]
            name_lower = name.lower()
            if any(keyword in name_lower for keyword in self.workspace_keywords):
                workspace_candidates.append((i, name, object_positions[i]))
        
        if workspace_candidates:
            # Choose the largest/lowest table-like surface
            lowest_z = min(pos[2] for _, _, pos in workspace_candidates)
            P_candidates = [(i, name) for i, name, pos in workspace_candidates if abs(pos[2] - lowest_z) < 0.1]
            key_objects['P'] = P_candidates[0][0]  # Take first if multiple at same height
        else:
            # Fallback: find lowest large object
            if len(object_positions) > 0:
                lowest_idx = np.argmin(object_positions[:max_objects, 2])
                key_objects['P'] = lowest_idx
        
        # Find T = target object (Container role)
        container_candidates = []
        for i in range(max_objects):
            name = object_names[i]
            name_lower = name.lower()
            if any(keyword in name_lower for keyword in self.container_keywords):
                container_candidates.append((i, name, object_positions[i]))
        
        if container_candidates:
            # Prefer container closest to gripper or mentioned in task
            distances_to_gripper = [np.linalg.norm(pos - ee_pos) for _, _, pos in container_candidates]
            closest_idx = np.argmin(distances_to_gripper)
            key_objects['T'] = container_candidates[closest_idx][0]
        
        # Find R = target region (FlatSupport role)
        flatsupport_candidates = []
        for i in range(max_objects):
            name = object_names[i]
            name_lower = name.lower()
            if any(keyword in name_lower for keyword in self.flat_support_keywords):
                # Skip if this is already chosen as P
                if i != key_objects['P']:
                    flatsupport_candidates.append((i, name, object_positions[i]))
        
        if flatsupport_candidates:
            # Prefer flat support closest to target object if available
            if key_objects['T'] is not None:
                target_pos = object_positions[key_objects['T']]
                distances_to_target = [np.linalg.norm(pos - target_pos) for _, _, pos in flatsupport_candidates]
                closest_idx = np.argmin(distances_to_target)
                key_objects['R'] = flatsupport_candidates[closest_idx][0]
            else:
                key_objects['R'] = flatsupport_candidates[0][0]
        else:
            # If no explicit target region, set R = P
            key_objects['R'] = key_objects['P']
        
        return key_objects
    
    def compute_spatial_relationships(self, sim_state: Dict[str, Any], key_objects: Dict[str, Optional[int]]) -> Dict[str, float]:
        """
        Compute spatial relationships between key objects.
        
        Args:
            sim_state: Simulator state
            key_objects: Key object indices from identify_key_objects()
            
        Returns:
            Dict with relationship values (1.0 = true, 0.0 = false, or continuous values)
        """
        relationships = {}
        
        object_positions = sim_state['object_positions']
        ee_pos = sim_state['ee_pos']
        
        # Handle backwards compatibility with old state format
        object_body_ids = sim_state.get('object_body_ids', np.array([], dtype=np.int32))
        object_extents = sim_state.get('object_extents', np.array([], dtype=np.float32))
        gripper_body_id = sim_state.get('gripper_body_id', -1)
        contact_body1 = sim_state.get('contact_body1', np.array([], dtype=np.int32))
        contact_body2 = sim_state.get('contact_body2', np.array([], dtype=np.int32))
        
        # Fallback for missing contact body mapping
        if len(contact_body1) == 0 and 'contact_geom1' in sim_state:
            print("[visual-concepts] WARNING: Using fallback contact detection (old state format)")
            contact_geom1 = sim_state.get('contact_geom1', np.array([], dtype=np.int32))
            contact_geom2 = sim_state.get('contact_geom2', np.array([], dtype=np.int32))
            has_contacts = len(contact_geom1) > 0
        else:
            has_contacts = len(contact_body1) > 0
        
        # Generate default object extents if missing
        if len(object_extents) == 0 and len(object_positions) > 0:
            object_extents = np.full(len(object_positions), 0.1, dtype=np.float32)  # Default size
        
        # Helper functions
        def get_position(obj_key):
            if key_objects[obj_key] is None:
                return None
            if key_objects[obj_key] == -1:  # End-effector
                return ee_pos
            return object_positions[key_objects[obj_key]]
        
        def get_body_id(obj_key):
            if key_objects[obj_key] is None:
                return None
            if key_objects[obj_key] == -1:  # End-effector
                return gripper_body_id
            if len(object_body_ids) > key_objects[obj_key]:
                return object_body_ids[key_objects[obj_key]]
            else:
                # Fallback for old state format - use index as approximate body ID
                return key_objects[obj_key]
        
        def get_extent(obj_key):
            if key_objects[obj_key] is None or key_objects[obj_key] == -1:
                return 0.1  # Default size for gripper
            if len(object_extents) > key_objects[obj_key]:
                return object_extents[key_objects[obj_key]]
            else:
                return 0.1  # Default size for old state format
        
        def check_contact(body_id_1, body_id_2):
            """Check if two bodies are in contact"""
            if not has_contacts:
                return False
            
            if len(contact_body1) > 0:
                # Use proper body-based contact detection if available
                if body_id_1 is None or body_id_2 is None:
                    return False
                contacts = ((contact_body1 == body_id_1) & (contact_body2 == body_id_2)) | \
                          ((contact_body1 == body_id_2) & (contact_body2 == body_id_1))
                return np.any(contacts)
            else:
                # Fallback to heuristic contact detection for old state format
                pos_1 = get_position('G' if body_id_1 == gripper_body_id else None)
                pos_2 = get_position('T' if body_id_2 is not None else None)
                if pos_1 is not None and pos_2 is not None:
                    distance = np.linalg.norm(pos_1 - pos_2)
                    return distance < 0.05  # Heuristic threshold
                return False
        
        def get_size_normalized_threshold(obj_key_1, obj_key_2):
            """Get size-normalized distance threshold for near() relationship"""
            extent_1 = get_extent(obj_key_1)
            extent_2 = get_extent(obj_key_2)
            # Use average of object extents as the threshold, with a minimum
            return max((extent_1 + extent_2) / 2, self.distance_threshold)
        
        # Get positions and body IDs
        T_pos = get_position('T')
        R_pos = get_position('R') 
        P_pos = get_position('P')
        G_pos = get_position('G')
        
        T_body = get_body_id('T')
        R_body = get_body_id('R')
        P_body = get_body_id('P')
        G_body = get_body_id('G')
        
        # on(x, y) → Container supported by FlatSupport AND in contact
        if T_pos is not None and P_pos is not None:
            vertical_dist = abs(T_pos[2] - P_pos[2])
            is_above = T_pos[2] > P_pos[2]
            is_close_vertically = vertical_dist < 0.05
            has_contact = check_contact(T_body, P_body)
            relationships['on_T_P'] = 1.0 if (is_close_vertically and is_above and has_contact) else 0.0
        else:
            relationships['on_T_P'] = 0.0
            
        if T_pos is not None and R_pos is not None:
            vertical_dist = abs(T_pos[2] - R_pos[2])
            is_above = T_pos[2] > R_pos[2]
            is_close_vertically = vertical_dist < 0.05
            has_contact = check_contact(T_body, R_body)
            relationships['on_T_R'] = 1.0 if (is_close_vertically and is_above and has_contact) else 0.0
        else:
            relationships['on_T_R'] = 0.0
        
        # above(x, y) → vertical displacement WITHOUT contact
        if T_pos is not None and P_pos is not None:
            vertical_dist = T_pos[2] - P_pos[2] 
            horizontal_dist = np.linalg.norm(T_pos[:2] - P_pos[:2])
            is_higher = vertical_dist > 0.05
            is_aligned = horizontal_dist < 0.2
            no_contact = not check_contact(T_body, P_body)
            relationships['above_T_P'] = 1.0 if (is_higher and is_aligned and no_contact) else 0.0
        else:
            relationships['above_T_P'] = 0.0
            
        if G_pos is not None and T_pos is not None:
            vertical_dist = G_pos[2] - T_pos[2]
            horizontal_dist = np.linalg.norm(G_pos[:2] - T_pos[:2])
            is_higher = vertical_dist > 0.02
            is_aligned = horizontal_dist < 0.1
            no_contact = not check_contact(G_body, T_body)
            relationships['above_G_T'] = 1.0 if (is_higher and is_aligned and no_contact) else 0.0
        else:
            relationships['above_G_T'] = 0.0
        
        # near(x, y) → within a size-normalized distance threshold
        if G_pos is not None and T_pos is not None:
            distance = np.linalg.norm(G_pos - T_pos)
            threshold = get_size_normalized_threshold('G', 'T')
            relationships['near_G_T'] = 1.0 if distance < threshold else 0.0
        else:
            relationships['near_G_T'] = 0.0
            
        if T_pos is not None and R_pos is not None:
            distance = np.linalg.norm(T_pos - R_pos)
            threshold = get_size_normalized_threshold('T', 'R')
            relationships['near_T_R'] = 1.0 if distance < threshold else 0.0
        else:
            relationships['near_T_R'] = 0.0
        
        # aligned_to(x, y, tol) → horizontal COM alignment within tolerance
        if G_pos is not None and T_pos is not None:
            horizontal_dist = np.linalg.norm(G_pos[:2] - T_pos[:2])
            relationships['aligned_G_T'] = 1.0 if horizontal_dist < self.alignment_tolerance else 0.0
        else:
            relationships['aligned_G_T'] = 0.0
            
        if T_pos is not None and R_pos is not None:
            horizontal_dist = np.linalg.norm(T_pos[:2] - R_pos[:2])
            relationships['aligned_T_R'] = 1.0 if horizontal_dist < self.alignment_tolerance else 0.0
        else:
            relationships['aligned_T_R'] = 0.0
        
        # touching(x, y) → contact detected (proper body-based contact detection)
        relationships['touching_G_T'] = 1.0 if check_contact(G_body, T_body) else 0.0
        relationships['touching_T_P'] = 1.0 if check_contact(T_body, P_body) else 0.0
        relationships['touching_T_R'] = 1.0 if check_contact(T_body, R_body) else 0.0
        
        # Additional useful relationships
        if T_pos is not None and P_pos is not None and R_pos is not None:
            # Distance metrics as continuous features
            relationships['dist_T_P'] = np.linalg.norm(T_pos - P_pos)
            relationships['dist_T_R'] = np.linalg.norm(T_pos - R_pos)
            relationships['height_T'] = T_pos[2]
            relationships['height_P'] = P_pos[2]
            relationships['height_R'] = R_pos[2]
        
        if G_pos is not None:
            relationships['gripper_x'] = G_pos[0]
            relationships['gripper_y'] = G_pos[1] 
            relationships['gripper_z'] = G_pos[2]
        
        # Size-normalized distance features
        if G_pos is not None and T_pos is not None:
            distance = np.linalg.norm(G_pos - T_pos)
            threshold = get_size_normalized_threshold('G', 'T')
            relationships['normalized_dist_G_T'] = distance / threshold
        
        return relationships
    
    def extract_visual_concepts(self, env, task_description: str = "") -> Dict[str, Any]:
        """
        Main interface: Extract all visual concepts from environment state.
        
        Args:
            env: LIBERO environment
            task_description: Task description for context
            
        Returns:
            Dict containing:
                - 'sim_state': Raw simulator state
                - 'key_objects': Identified key object indices  
                - 'relationships': Spatial relationship values
                - 'concept_vector': Flat array of all concept values for ML
        """
        # Extract simulator state
        sim_state = self.extract_simulator_state(env)
        
        # Identify key objects
        key_objects = self.identify_key_objects(sim_state, task_description)
        
        # Compute spatial relationships
        relationships = self.compute_spatial_relationships(sim_state, key_objects)
        
        # Create flat concept vector for machine learning
        concept_names = sorted(relationships.keys())
        concept_vector = np.array([relationships[name] for name in concept_names], dtype=np.float32)
        
        return {
            'sim_state': sim_state,
            'key_objects': key_objects,
            'relationships': relationships,
            'concept_vector': concept_vector,
            'concept_names': concept_names
        }


def extract_visual_concepts_from_state(sim_state: Dict[str, Any], task_description: str = "") -> Dict[str, Any]:
    """
    Convenience function to extract visual concepts from pre-extracted simulator state.
    
    Args:
        sim_state: Pre-extracted simulator state
        task_description: Task description for context
        
    Returns:
        Dict with visual concepts (same as VisualConceptsExtractor.extract_visual_concepts)
    """
    extractor = VisualConceptsExtractor()
    
    # Identify key objects
    key_objects = extractor.identify_key_objects(sim_state, task_description)
    
    # Compute spatial relationships
    relationships = extractor.compute_spatial_relationships(sim_state, key_objects)
    
    # Create flat concept vector
    concept_names = sorted(relationships.keys())
    concept_vector = np.array([relationships[name] for name in concept_names], dtype=np.float32)
    
    return {
        'sim_state': sim_state,
        'key_objects': key_objects,
        'relationships': relationships,
        'concept_vector': concept_vector,
        'concept_names': concept_names
    }


# Global extractor instance for efficiency
_global_extractor = VisualConceptsExtractor()

def extract_simulator_state(env) -> Dict[str, Any]:
    """
    Drop-in replacement for reconstruct_trajectory_data.extract_simulator_state()
    """
    return _global_extractor.extract_simulator_state(env)

def extract_visual_concepts(env, task_description: str = "") -> Dict[str, Any]:
    """
    Convenience function using global extractor instance.
    """
    return _global_extractor.extract_visual_concepts(env, task_description)