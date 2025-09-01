"""
data_collecting_vla.py

Wrapper around OpenVLA that collects hidden states during action prediction.
Focus only on hidden states extraction with extensive debug prints.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union
from PIL.Image import Image as Img
from transformers import LlamaTokenizerFast
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

from prismatic.models.vlas.openvla import OpenVLA
from prismatic.models.vlms.prismatic import PrismaticVLM


class DataCollectingVLA(OpenVLA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collect_data = False
        self.current_episode_data = []
        print(f"[DATA_VLA] DataCollectingVLA initialized")
    
    def enable_data_collection(self):
        """Enable data collection for current episode"""
        self.collect_data = True
        self.current_episode_data = []
        print(f"[DATA_VLA] Data collection ENABLED")
    
    def disable_data_collection(self):
        """Disable data collection"""
        self.collect_data = False
        print(f"[DATA_VLA] Data collection DISABLED")
    
    def get_episode_data(self):
        """Get collected data for current episode"""
        print(f"[DATA_VLA] Returning episode data with {len(self.current_episode_data)} timesteps")
        return self.current_episode_data
    
    def clear_episode_data(self):
        """Clear collected episode data"""
        self.current_episode_data = []
        print(f"[DATA_VLA] Episode data cleared")

    @torch.inference_mode()
    def predict_action_with_data_collection(
        self, observation: Union[Img, List[Img], Dict], instruction: str, unnorm_key: Optional[str] = None, **kwargs: str
    ) -> tuple[np.ndarray, Dict]:
        """
        Modified predict_action that collects hidden states.
        Returns: (actions, collected_data_dict)
        """
        print(f"[DATA_VLA] predict_action_with_data_collection called")
        print(f"[DATA_VLA] collect_data flag: {self.collect_data}")
        print(f"[DATA_VLA] instruction: {instruction}")
        print(f"[DATA_VLA] unnorm_key: {unnorm_key}")
        
        # Handle observation format - could be dict with 'full_image' key or direct image
        if isinstance(observation, dict) and 'full_image' in observation:
            raw_images = observation['full_image']
            print(f"[DATA_VLA] Extracted raw_images from observation dict, type: {type(raw_images)}")
            
            # Process images same as get_prismatic_vla_action
            if not isinstance(raw_images, list):
                raw_images = [raw_images]
            
            processed_images = []
            from PIL import Image
            import numpy as np
            
            for img in raw_images:
                # Convert numpy array to PIL Image
                if isinstance(img, np.ndarray):
                    pil_image = Image.fromarray(img)
                else:
                    pil_image = img
                
                pil_image = pil_image.convert("RGB")
                
                # Apply center crop if specified
                center_crop = kwargs.get('center_crop', False)
                if center_crop:
                    import math
                    temp_image = np.array(pil_image)  # (H, W, C)
                    crop_scale = 0.9
                    sqrt_crop_scale = math.sqrt(crop_scale)
                    
                    # Simple center crop calculation
                    h, w = temp_image.shape[:2]
                    t_h = int(sqrt_crop_scale * h)
                    t_w = int(sqrt_crop_scale * w)
                    
                    # Center crop
                    start_h = (h - t_h) // 2
                    start_w = (w - t_w) // 2
                    temp_image_cropped = temp_image[start_h:start_h+t_h, start_w:start_w+t_w]
                    
                    temp_image = Image.fromarray(temp_image_cropped)
                    temp_image = temp_image.resize(pil_image.size, Image.Resampling.BILINEAR)
                    pil_image = temp_image
                
                processed_images.append(pil_image)
            
            # If single image, unwrap from list
            if len(processed_images) == 1:
                image = processed_images[0]
            else:
                image = processed_images
                
            print(f"[DATA_VLA] Processed images, final type: {type(image)}")
        else:
            image = observation
            print(f"[DATA_VLA] Using observation as image, type: {type(image)}")
        
        # Standard preprocessing (same as original predict_action)
        image_transform, tokenizer = self.vision_backbone.get_image_transform(), self.llm_backbone.tokenizer
        
        # Build VLA Prompt
        prompt_builder = self.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()
        print(f"[DATA_VLA] Prompt text length: {len(prompt_text)}")

        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        print(f"[DATA_VLA] Input IDs shape: {input_ids.shape}")
        
        # Handle tokenizer specifics (copied from original)
        if isinstance(tokenizer, LlamaTokenizerFast):
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
                )
        elif isinstance(tokenizer, Qwen2TokenizerFast):
            pass
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        # Preprocess Image
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
            print(f"[DATA_VLA] Pixel values shape: {pixel_values.shape}")
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
            print(f"[DATA_VLA] Pixel values dict keys: {list(pixel_values.keys())}")
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Initialize collected data
        collected_data = {}
        
        # Modified generation with hidden state collection
        autocast_dtype = self.llm_backbone.half_precision_dtype
        print(f"[DATA_VLA] Using autocast dtype: {autocast_dtype}")
        
        # Filter out parameters that are used for image preprocessing but not for generation
        # center_crop is consumed during image preprocessing above and should not be passed to generate()
        generation_kwargs = {k: v for k, v in kwargs.items() if k not in ['center_crop']}
        print(f"[DATA_VLA] Filtered generation kwargs: {list(generation_kwargs.keys())}")
        
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            if self.collect_data:
                print(f"[DATA_VLA] Generating with hidden states collection...")
                
                # Enable hidden state output for data collection  
                # Use same call as original predict_action: super(PrismaticVLM, self).generate
                generated_output = super(PrismaticVLM, self).generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    max_new_tokens=self.get_action_dim(unnorm_key),
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    **generation_kwargs
                )
                
                print(f"[DATA_VLA] Generated output type: {type(generated_output)}")
                print(f"[DATA_VLA] Generated output keys: {generated_output.keys() if hasattr(generated_output, 'keys') else 'No keys'}")
                
                # Extract hidden states from all layers
                if hasattr(generated_output, 'hidden_states') and generated_output.hidden_states is not None:
                    print(f"[DATA_VLA] Found hidden_states in output")
                    print(f"[DATA_VLA] Hidden states structure: {len(generated_output.hidden_states)} generation steps")
                    
                    hidden_states = {}
                    
                    # DETAILED DEBUG: Analyze hidden states structure
                    print(f"[DATA_VLA] DEBUGGING HIDDEN STATES STRUCTURE:")
                    print(f"[DATA_VLA] Total generation steps: {len(generated_output.hidden_states)}")
                    
                    # Process each generation step
                    for step_idx, step_hidden in enumerate(generated_output.hidden_states):
                        print(f"[DATA_VLA] === GENERATION STEP {step_idx} ===")
                        print(f"[DATA_VLA] Number of layers: {len(step_hidden)}")
                        
                        # Process each layer in this step
                        for layer_idx, layer_hidden in enumerate(step_hidden):
                            print(f"[DATA_VLA] Step {step_idx}, Layer {layer_idx}: shape {layer_hidden.shape}, dtype {layer_hidden.dtype}")
                            
                            if layer_idx not in hidden_states:
                                hidden_states[layer_idx] = []
                            
                            # Store as numpy array (detach from GPU)
                            # Convert BFloat16 to Float32 since NumPy doesn't support BFloat16
                            if layer_hidden.dtype == torch.bfloat16:
                                layer_data = layer_hidden.detach().cpu().float().numpy()
                            else:
                                layer_data = layer_hidden.detach().cpu().numpy()
                            
                            print(f"[DATA_VLA] Converted to numpy: shape {layer_data.shape}, dtype {layer_data.dtype}")
                            hidden_states[layer_idx].append(layer_data)
                    
                    # DETAILED DEBUG: Analyze collected data before stacking
                    print(f"[DATA_VLA] === ANALYZING COLLECTED DATA BEFORE STACKING ===")
                    for layer_idx in sorted(hidden_states.keys()):
                        layer_data_list = hidden_states[layer_idx]
                        print(f"[DATA_VLA] Layer {layer_idx}: {len(layer_data_list)} generation steps collected")
                        for i, data in enumerate(layer_data_list):
                            print(f"[DATA_VLA]   Step {i}: shape {data.shape}")
                        
                        # Check if all shapes are consistent
                        shapes = [data.shape for data in layer_data_list]
                        if len(set(shapes)) == 1:
                            print(f"[DATA_VLA] Layer {layer_idx}: ALL SHAPES CONSISTENT - {shapes[0]}")
                        else:
                            print(f"[DATA_VLA] Layer {layer_idx}: INCONSISTENT SHAPES! {shapes}")
                    
                    # Keep hidden states as lists - don't try to stack incompatible shapes
                    for layer_idx in hidden_states:
                        layer_data_list = hidden_states[layer_idx]
                        print(f"[DATA_VLA] Layer {layer_idx}: {len(layer_data_list)} generation steps")
                        for i, data in enumerate(layer_data_list):
                            print(f"[DATA_VLA] Layer {layer_idx}, step {i}: shape {data.shape}")
                        # Keep as list - shapes are incompatible for stacking due to autoregressive generation
                        print(f"[DATA_VLA] Layer {layer_idx}: keeping as list due to variable sequence lengths")
                    
                    collected_data['hidden_states'] = hidden_states
                    print(f"[DATA_VLA] Collected hidden states for layers: {list(hidden_states.keys())}")
                    
                else:
                    print(f"[DATA_VLA] WARNING: No hidden_states found in generated output!")
                    collected_data['hidden_states'] = {}
                
                # Extract the generated token ids
                if hasattr(generated_output, 'sequences'):
                    action_token_ids = generated_output.sequences[0, -self.get_action_dim(unnorm_key):]
                    print(f"[DATA_VLA] Action token IDs shape: {action_token_ids.shape}")
                else:
                    print(f"[DATA_VLA] ERROR: No sequences found in generated output!")
                    return np.array([]), collected_data
                    
            else:
                print(f"[DATA_VLA] Generating without data collection (standard mode)")
                # Standard generation without data collection - same call as original
                generated_ids = super(PrismaticVLM, self).generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    max_new_tokens=self.get_action_dim(unnorm_key),
                    **generation_kwargs
                )
                action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key):]
                print(f"[DATA_VLA] Action token IDs shape: {action_token_ids.shape}")

        # Decode action (same as original)
        print(f"[DATA_VLA] Decoding action tokens...")
        normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(action_token_ids.cpu().numpy())
        actions = self._unnormalize_action(normalized_actions, unnorm_key)
        print(f"[DATA_VLA] Final actions shape: {np.array(actions).shape}")
        print(f"[DATA_VLA] Actions: {actions}")
        
        # Store step data if collecting
        if self.collect_data:
            collected_data['actions'] = actions.copy() if isinstance(actions, np.ndarray) else np.array(actions)
            self.current_episode_data.append(collected_data)
            print(f"[DATA_VLA] Added step data to episode (now {len(self.current_episode_data)} steps)")
        
        return actions, collected_data if self.collect_data else {}


def wrap_model_for_data_collection(model):
    """
    Convert a regular OpenVLA model into a DataCollectingVLA.
    This is a helper function for the evaluator.
    """
    print(f"[DATA_VLA] Wrapping model {type(model)} for data collection")
    
    if isinstance(model, DataCollectingVLA):
        print(f"[DATA_VLA] Model already wrapped")
        return model
    
    if not isinstance(model, OpenVLA):
        print(f"[DATA_VLA] WARNING: Model is not OpenVLA type: {type(model)}")
        return model
    
    # Create new DataCollectingVLA with same parameters
    wrapped_model = DataCollectingVLA(
        model.model_id,
        model.vision_backbone,
        model.llm_backbone,
        enable_mixed_precision_training=model.enable_mixed_precision_training,
        arch_specifier=getattr(model, 'arch_specifier', 'gelu-mlp'),
        norm_stats=model.norm_stats,
        action_tokenizer=model.action_tokenizer
    )
    
    # Copy any additional attributes that might be missing
    for attr_name in ['arch_specifier', 'all_module_keys', 'trainable_module_keys']:
        if hasattr(model, attr_name):
            setattr(wrapped_model, attr_name, getattr(model, attr_name))
    
    # Copy state dict
    wrapped_model.load_state_dict(model.state_dict())
    wrapped_model.to(model.device)
    
    print(f"[DATA_VLA] Model successfully wrapped")
    return wrapped_model