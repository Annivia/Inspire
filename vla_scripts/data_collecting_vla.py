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
    
    def enable_data_collection(self):
        """Enable data collection for current episode"""
        self.collect_data = True
        self.current_episode_data = []
    
    def disable_data_collection(self):
        """Disable data collection"""
        self.collect_data = False
    
    def get_episode_data(self):
        """Get collected data for current episode"""
        return self.current_episode_data
    
    def clear_episode_data(self):
        """Clear collected episode data"""
        self.current_episode_data = []

    @torch.inference_mode()
    def predict_action_with_data_collection(
        self, observation: Union[Img, List[Img], Dict], instruction: str, unnorm_key: Optional[str] = None, **kwargs: str
    ) -> tuple[np.ndarray, Dict]:
        """
        Modified predict_action that collects hidden states.
        Returns: (actions, collected_data_dict)
        """
        
        # Handle observation format - could be dict with 'full_image' key or direct image
        if isinstance(observation, dict) and 'full_image' in observation:
            raw_images = observation['full_image']
            
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
                
        else:
            image = observation
        
        # Standard preprocessing (same as original predict_action)
        image_transform, tokenizer = self.vision_backbone.get_image_transform(), self.llm_backbone.tokenizer
        
        # Build VLA Prompt
        prompt_builder = self.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()

        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        
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
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Initialize collected data
        collected_data = {}
        
        # Modified generation with hidden state and vision collection
        autocast_dtype = self.llm_backbone.half_precision_dtype
        
        # Filter out parameters that are used for image preprocessing but not for generation
        # center_crop is consumed during image preprocessing above and should not be passed to generate()
        generation_kwargs = {k: v for k, v in kwargs.items() if k not in ['center_crop']}
        
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            if self.collect_data:
                
                # Capture vision encoder features before generation
                vision_features = self._extract_vision_features(pixel_values)
                if vision_features is not None:
                    collected_data['vision_features'] = vision_features
                
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
                
                # Extract hidden states from all layers
                if hasattr(generated_output, 'hidden_states') and generated_output.hidden_states is not None:
                    
                    
                    # CORRECTED: Organize by generation_step -> layer (not layer -> generation_step)
                    hidden_states = {}
                    
                    # Process each generation step (0 through 6 for 7 action tokens)
                    for step_idx, step_hidden in enumerate(generated_output.hidden_states):
                        if step_idx not in hidden_states:
                            hidden_states[step_idx] = {}
                        
                        
                        # Process each layer in this generation step
                        for layer_idx, layer_hidden in enumerate(step_hidden):
                            
                            
                            # For Step 0, extract only the final token to match Steps 1-6 dimensions
                            if step_idx == 0 and len(layer_hidden.shape) == 3:
                                # Extract final token: [1, seq_len, hidden] -> [1, hidden]  
                                current_token_hidden = layer_hidden[:, -1, :]
                            elif step_idx == 0:
                                current_token_hidden = layer_hidden
                            else:
                                # Steps 1-6: should be single tokens with shape [1, 1, 896], need to squeeze to [1, 896]
                                if len(layer_hidden.shape) == 3:
                                    if layer_hidden.shape[1] == 1:
                                        # Single token: [1, 1, hidden] -> [1, hidden]
                                        current_token_hidden = layer_hidden.squeeze(1)  # Remove middle dimension
                                    else:
                                        # Multiple tokens, extract final
                                        current_token_hidden = layer_hidden[:, -1, :]
                                else:
                                    # Already 2D
                                    current_token_hidden = layer_hidden
                                    # print(f"[DEBUG_HIDDEN] Step {step_idx}, Layer {layer_idx}: using original 2D shape {current_token_hidden.shape}")
                            
                            # Store as numpy array (detach from GPU)
                            # Convert BFloat16 to Float32 since NumPy doesn't support BFloat16
                            if current_token_hidden.dtype == torch.bfloat16:
                                layer_data = current_token_hidden.detach().cpu().float().numpy()
                            else:
                                layer_data = current_token_hidden.detach().cpu().numpy()

                            # CORRECTED: hidden_states[generation_step][layer] structure
                            hidden_states[step_idx][layer_idx] = layer_data
                    
                    collected_data['hidden_states'] = hidden_states
                    
                else:
                    print(f"WARNING: No hidden_states found in generated output!")
                    collected_data['hidden_states'] = {}
                
                # Extract the generated token ids
                if hasattr(generated_output, 'sequences'):
                    action_token_ids = generated_output.sequences[0, -self.get_action_dim(unnorm_key):]
                else:
                    print(f"ERROR: No sequences found in generated output!")
                    return np.array([]), collected_data
                    
            else:
                # Standard generation without data collection - same call as original
                generated_ids = super(PrismaticVLM, self).generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    max_new_tokens=self.get_action_dim(unnorm_key),
                    **generation_kwargs
                )
                action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key):]

        # Decode action (same as original)
        normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(action_token_ids.cpu().numpy())
        actions = self._unnormalize_action(normalized_actions, unnorm_key)
        
        # Store step data if collecting
        if self.collect_data:
            collected_data['actions'] = actions.copy() if isinstance(actions, np.ndarray) else np.array(actions)
            self.current_episode_data.append(collected_data)
        
        return actions, collected_data if self.collect_data else {}
    
    def _extract_vision_features(self, pixel_values):
        """
        Extract vision encoder patch features from pixel values.
        This captures the raw output from the vision backbone (DINOv2/CLIP/SigLIP etc.)
        before projection to LLM embedding space.
        """
        try:
            
            
            # Run Visual Feature Extraction (same logic as PrismaticVLM.forward line 370-372)
            with torch.set_grad_enabled(self.vision_backbone_requires_grad):
                if isinstance(pixel_values, dict):
                    patch_features = self.vision_backbone(pixel_values)
                    print(f"[debug-visual] Dict input - vision backbone output shape: {patch_features.shape}")
                else:
                    patch_features = self.vision_backbone(pixel_values)
                    print(f"[debug-visual] Tensor input - vision backbone output shape: {patch_features.shape}")
            
            # Convert to numpy for storage (same approach as hidden states)
            if patch_features.dtype == torch.bfloat16:
                vision_features_np = patch_features.detach().cpu().float().numpy()
            else:
                vision_features_np = patch_features.detach().cpu().numpy()

            return vision_features_np
            
        except Exception as e:
            print(f"ERROR extracting vision features: {e}")
            import traceback
            traceback.print_exc()
            return None


def wrap_model_for_data_collection(model):
    """
    Convert a regular OpenVLA model into a DataCollectingVLA.
    This is a helper function for the evaluator.
    """
    
    if isinstance(model, DataCollectingVLA):
        return model
    
    if not isinstance(model, OpenVLA):
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
    
    return wrapped_model