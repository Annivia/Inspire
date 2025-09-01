import os
# os.environ["MUJOCO_GL"] = "osmesa"
os.environ["MUJOCO_GL"] = "egl" 
os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"


import argparse
import math
import multiprocessing
import numpy as np
import traceback
from PIL import Image
from pathlib import Path
import torch, random, numpy as np
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

import sys
sys.path.append('.')
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
)
from utils.logger import Logger, reset_logging
from utils.visualize import write_video


def get_image_resize_size(cfg):
    if cfg.model_family == "prismatic":
        resize_size = 224
    elif cfg.model_family == "openvla":
        resize_size = 224
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return resize_size



def apply_center_crop(im, t_h, t_w):
    assert im.shape[-3] >= t_h and im.shape[-2] >= t_w
    assert im.shape[-1] in [1, 3, 6]
    crop_h = int((im.shape[-3] - t_h) / 2)
    crop_w = int((im.shape[-2] - t_w) / 2)
    return im[..., crop_h : crop_h + t_h, crop_w : crop_w + t_w, :]


def get_prismatic_vla_action(vla, obs, task_label, unnorm_key, center_crop=False, **kwargs):
    if not isinstance(obs["full_image"], list):
        obs["full_image"] = [obs["full_image"]]

    processed_images = []

    for img in obs["full_image"]:
        image = Image.fromarray(img)
        image = image.convert("RGB")

        if center_crop:
            temp_image = np.array(image)  # (H, W, C)
            crop_scale = 0.9
            sqrt_crop_scale = math.sqrt(crop_scale)
            temp_image_cropped = apply_center_crop(
                temp_image,
                t_h=int(sqrt_crop_scale * temp_image.shape[0]),
                t_w=int(sqrt_crop_scale * temp_image.shape[1]),
            )
            temp_image = Image.fromarray(temp_image_cropped)
            temp_image = temp_image.resize(
                image.size, Image.Resampling.BILINEAR
            )  # IMPORTANT: dlimp uses BILINEAR resize
            image = temp_image

        processed_images.append(image)

    if len(processed_images) == 1:
        processed_images = processed_images[0]

    outputs = vla.predict_action(processed_images, task_label, unnorm_key=unnorm_key, **kwargs)
    if isinstance(outputs, tuple):
        action, text = outputs
    else:
        action, text = outputs, None
    return action, text


def normalize_gripper_action(action, binarize=True):
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action


def invert_gripper_action(action):
    action[..., -1] = action[..., -1] * -1.0
    return action


class GenerateConfig:
    def __init__(self,
                 model_family="prismatic",
                 hf_token=Path(".hf_token"),
                 pretrained_checkpoint="/work/nvme/bfbo/xzhang42/Inspire/runs/prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90+n0+b16+x7/",
                 load_step=None,
                 load_in_8bit=False,
                 load_in_4bit=False,
                 center_crop=True,
                 obs_history=1,
                 use_wrist_image=False,
                 seed=7,
                 task_suite_name="libero_90",
                 num_steps_wait=10,
                 num_trials_per_task=10,
                 num_gpus=8,
                 num_processes=32,
                 save_root="./results",
                 fps=30,
                 with_vqa=False,
                 check_catch=True,
                 check_close=True,
                 vqa_mode='coarse_direction',
                 collect_trajectory_data=False,
                 trajectory_data_save_path="./trajectory_data",
                 max_total_trajectories=None,  # For early stopping
                 ):
        self.model_family = model_family
        self.hf_token = hf_token
        self.pretrained_checkpoint = pretrained_checkpoint
        self.load_step = load_step
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.center_crop = center_crop
        self.obs_history = obs_history
        self.use_wrist_image = use_wrist_image
        self.seed = seed
        self.task_suite_name = task_suite_name
        self.num_steps_wait = num_steps_wait
        self.num_trials_per_task = num_trials_per_task
        self.num_gpus = num_gpus
        self.num_processes = num_processes
        self.save_root = save_root
        self.fps = fps
        self.with_vqa = with_vqa
        self.check_catch = check_catch
        self.check_close = check_close
        self.vqa_mode = vqa_mode
        self.collect_trajectory_data = collect_trajectory_data
        self.trajectory_data_save_path = trajectory_data_save_path
        self.max_total_trajectories = max_total_trajectories

        self.image_sequence_len = 1
        if self.obs_history == 2 or self.use_wrist_image:
            self.image_sequence_len = 2


class ParallelLiberoEvaluator:
    def __init__(self, cfg, opts=None):
        # [Note] Data root is not used for evaluation
        # os.environ["PRISMATIC_DATA_ROOT"] = '/projects/bfbo/xzhang42/Inspire' #### @Polina remember to change it to /work
        # [Note] Tokenizers parallelism is set to true for faster tokenization
        os.environ["TOKENIZERS_PARALLELISM"] = 'true'

        assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
        if "image_aug" in cfg.pretrained_checkpoint:
            assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
        assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

        self.cfg = cfg
        # self.cfg.unnorm_key = self.cfg.task_suite_name
        self.cfg.unnorm_key = 'libero_90'

        if cfg.task_suite_name == "libero_spatial":
            self.cfg.max_steps = 220  # longest training demo has 193 steps
        elif cfg.task_suite_name == "libero_object":
            self.cfg.max_steps = 280  # longest training demo has 254 steps
        elif cfg.task_suite_name == "libero_goal":
            self.cfg.max_steps = 300  # longest training demo has 270 steps
        elif cfg.task_suite_name == "libero_10":
            self.cfg.max_steps = 520  # longest training demo has 505 steps
        elif cfg.task_suite_name == "libero_90":
            self.cfg.max_steps = 400  # longest training demo has 373 steps

        if opts is not None:
            for key, value in opts.items():
                setattr(self.cfg, key, value)

        if self.cfg.load_step is None:
            checkpoint_files = os.listdir(os.path.join(self.cfg.pretrained_checkpoint, 'checkpoints'))
            steps = [int(file.split('-')[1]) for file in checkpoint_files]
            self.cfg.load_step = max(steps)

        # name like step-000000-epoch-00-loss=0.0000.pt
        checkpoint_files = os.listdir(os.path.join(self.cfg.pretrained_checkpoint, 'checkpoints'))
        load_step = '0' + str(self.cfg.load_step) if self.cfg.load_step < 100000 else str(self.cfg.load_step)
        checkpoint_file = [file for file in checkpoint_files if f"step-{load_step}" in file][0]
        self.cfg.pretrained_checkpoint = os.path.join(self.cfg.pretrained_checkpoint, 'checkpoints', checkpoint_file)
        
    def evaluate(self):
        from libero.libero import benchmark

        multiprocessing.set_start_method('spawn', force=True)

        self._set_results()
        self._build_logger()
        self.logger.infos('Config', vars(self.cfg))

        self.resize_size = get_image_resize_size(self.cfg)

        # Data collector will be created in each subprocess to avoid pickle issues
        if self.cfg.collect_trajectory_data:
            print(f"[EVALUATOR] Trajectory data collection enabled (will create collector in each subprocess)")
        else:
            print(f"[EVALUATOR] Trajectory data collection disabled")

        # benchmark_dict = benchmark.get_benchmark_dict()
        # self.task_suite = benchmark_dict[self.cfg.task_suite_name]()
        # num_tasks_in_suite = self.task_suite.n_tasks

        # Polina: move task suite creation to subprocesses
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite_class = benchmark_dict[self.cfg.task_suite_name] 
        num_tasks_in_suite = task_suite_class().n_tasks 

        gpus = self._check_free_gpus()
        if self.cfg.num_gpus < len(gpus):
            gpus = gpus[:self.cfg.num_gpus]
        
        task_ids_and_episodes_all_processes = [[] for _ in range(self.cfg.num_processes)]
        idx = 0
        total_trajectories = 0
        
        for task_id in range(num_tasks_in_suite):
            # task = self.task_suite.get_task(task_id).language
            for episode in range(self.cfg.num_trials_per_task):
                # Early stopping check
                if self.cfg.max_total_trajectories is not None and total_trajectories >= self.cfg.max_total_trajectories:
                    print(f"[EVALUATOR] Early stopping: reached max_total_trajectories ({self.cfg.max_total_trajectories})")
                    break
                    
                task_ids_and_episodes_all_processes[idx % self.cfg.num_processes].append((task_id, episode))
                idx += 1
                total_trajectories += 1
                
            # Break outer loop too if early stopping triggered
            if self.cfg.max_total_trajectories is not None and total_trajectories >= self.cfg.max_total_trajectories:
                break
        
        print(f"[EVALUATOR] Total trajectories to evaluate: {total_trajectories}")

        processes = []
        manager = multiprocessing.Manager()
        summaries = manager.list()
        
        for idx, task_ids_and_episodes in enumerate(task_ids_and_episodes_all_processes):
            gpu = gpus[idx % len(gpus)]
            self.logger.info(f'GPU {gpu}: {task_ids_and_episodes}')
            process = multiprocessing.Process(target=self.evaluate_episodes,
                                              args=(gpu, task_ids_and_episodes, idx == 0, summaries))
            processes.append(process)
            
        for process in processes:
            process.start()
        for process in processes:
            process.join()

        self._build_logger(mode='a')
        task_ids = set([summary["task_id"] for summary in summaries])
        for task_id in task_ids:
            task_summaries = [summary for summary in summaries if summary["task_id"] == task_id]
            success_rate = sum([summary["success"] for summary in task_summaries]) / len(task_summaries)
            task_description = task_summaries[0]['task']
            self.logger.info(f"Task {task_id} {task_description} success rate: {success_rate:.2f}")
        
        success_rate = sum([summary["success"] for summary in summaries]) / len(summaries)
        self.logger.info(f"Overall success rate: {success_rate:.2f}")
        self.logger.info("Evaluation finished.")

    def evaluate_episodes(self, gpu, task_ids_and_episodes, show_detail, summaries):
        os.environ["MUJOCO_GL"] = "egl" 
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
        import sys
        import time

        try:
            print(f"Process starting on GPU {gpu}")
            from libero.libero import benchmark
            benchmark_dict = benchmark.get_benchmark_dict()
            task_suite = benchmark_dict[self.cfg.task_suite_name]()
            print(f"GPU {gpu}: Task suite loaded, n_tasks: {task_suite.n_tasks}", flush=True)
            
            model, processor = self._build_policy(gpu)
            print(f"[DEBUG] GPU {gpu}: Model loaded successfully", flush=True)
            reset_logging()
            self._build_logger(mode='a')

            # Create data collector in subprocess to avoid pickle issues
            data_collector = None
            if self.cfg.collect_trajectory_data:
                print(f"[DEBUG] GPU {gpu}: Creating data collector in subprocess", flush=True)
                from vla_scripts.trajectory_data_collector import TrajectoryDataCollector
                data_collector = TrajectoryDataCollector(
                    self.cfg.trajectory_data_save_path, 
                    self.cfg.task_suite_name,
                    process_id=gpu  # Use GPU ID as process ID
                )

            for i, (task_id, episode) in enumerate(task_ids_and_episodes):
                self.logger.info(f"GPU {gpu}: task {task_id} episode {episode}")
                summary = self.evalute_single(model, task_suite, processor, task_id, episode, show_detail, data_collector)
                summaries.append(summary)
                    
            
        except Exception as e:
            print(f"[ERROR] GPU {gpu}: Process failed with error: {str(e)}", flush=True)
            print(f"[ERROR] GPU {gpu}: Error type: {type(e)}", flush=True)
            print(f"[ERROR] GPU {gpu}: Full traceback:", flush=True)
            traceback.print_exc()
            
            # Write error to file
            try:
                error_file = os.path.join(self.save_dir, f'error_gpu{gpu}_pid{os.getpid()}.log')
                with open(error_file, 'w') as f:
                    f.write(f"Process PID: {os.getpid()}\n")
                    f.write(f"GPU: {gpu}\n")
                    f.write(f"Error: {str(e)}\n")
                    f.write(f"Error type: {type(e)}\n")
                    f.write("Full traceback:\n")
                    traceback.print_exc(file=f)
            except Exception as file_error:
                print(f"[ERROR] GPU {gpu}: Failed to write error file: {file_error}", flush=True)


    def evalute_single(self, model, task_suite, processor, task_id, episode, show_detail, data_collector=None):
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, self.cfg.model_family, resolution=self.resize_size)
        env.seed(episode)
        env.reset()

        # for libero object, we reset the environment
        # so the initial state is not the same as the training data
        if not self.cfg.task_suite_name == 'libero_object':
            initial_states = task_suite.get_task_init_states(task_id)
            obs = env.set_init_state(initial_states[episode])
        
        replay_images, replay_wrist_images = [], []
        texts = []
        timestep = 0
        success = False
        
        # Enable data collection for this episode if collector is available
        episode_data = []
        if data_collector is not None and hasattr(model, 'enable_data_collection'):
            print(f"[EVAL_SINGLE] Enabling data collection for task_{task_id}/episode_{episode}")
            model.enable_data_collection()
        else:
            print(f"[EVAL_SINGLE] Data collection not available for task_{task_id}/episode_{episode}")

        while timestep < self.cfg.max_steps + self.cfg.num_steps_wait:
            if timestep < self.cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(self.cfg.model_family))
                self._add_observation(obs, replay_images, replay_wrist_images)
                timestep += 1
                continue

            observation = self._prepare_inputs(obs, replay_images, replay_wrist_images)
            
            # Use data collecting method if available
            if data_collector is not None and hasattr(model, 'predict_action_with_data_collection'):
                print(f"[EVAL_SINGLE] Using data collecting action prediction, timestep {timestep}")
                action, step_data = model.predict_action_with_data_collection(
                    observation, task_description, self.cfg.unnorm_key, center_crop=self.cfg.center_crop
                )
                texts.append(None)  # No text from data collecting method
                print(f"[EVAL_SINGLE] Step data keys: {list(step_data.keys()) if step_data else 'None'}")
            else:
                print(f"[EVAL_SINGLE] Using standard action prediction, timestep {timestep}")
                action, text = get_prismatic_vla_action(model, observation, task_description, 
                                                        self.cfg.unnorm_key, center_crop=self.cfg.center_crop)
                texts.append(text)
            if isinstance(action, list):
                action = [normalize_gripper_action(a, binarize=True) for a in action]
            else:
                action = normalize_gripper_action(action, binarize=True)
            # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
            # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
            if self.cfg.model_family in ["openvla", "prismatic"]:
                if isinstance(action, list):
                    action = [invert_gripper_action(a) for a in action]
                else:
                    action = invert_gripper_action(action)

            if isinstance(action, list):
                for a in action:
                    obs, reward, done, info = env.step(a.tolist())
                    self._add_observation(obs, replay_images, replay_wrist_images)

                    timestep += 1
                    if show_detail:
                        self.logger.info(f"Step {timestep}: done {done}, {info}")
                    if done:
                        success = True
                        break
                if success:
                    break
            else:
                obs, reward, done, info = env.step(action.tolist())
                self._add_observation(obs, replay_images, replay_wrist_images)

                timestep += 1
                if show_detail:
                    self.logger.info(f"Step {timestep}: done {done}, {info}")
                if done:
                    success = True
                    break
        
        video_save_dir = os.path.join(self.save_dir, f'{task_id}_{task_description}')
        os.makedirs(video_save_dir, exist_ok=True)
        write_video(replay_images, os.path.join(video_save_dir, f'episode{episode}_success={success}.gif'), 
                    texts=None, fps=self.cfg.fps)
        
        # Save episode data if data collection is enabled
        if data_collector is not None and hasattr(model, 'get_episode_data'):
            print(f"[EVAL_SINGLE] Saving episode data for task_{task_id}/episode_{episode}")
            try:
                episode_hidden_states = model.get_episode_data()
                print(f"[EVAL_SINGLE] Retrieved {len(episode_hidden_states)} timesteps of data")
                
                if len(episode_hidden_states) > 0:
                    data_collector.save_episode_hidden_states(
                        task_id=task_id,
                        episode=episode,
                        hidden_states_data=episode_hidden_states,
                        task_description=task_description,
                        success=success
                    )
                    print(f"[EVAL_SINGLE] Successfully saved episode data")
                else:
                    print(f"[EVAL_SINGLE] WARNING: No hidden states data to save")
                
                # Clear episode data from model
                model.clear_episode_data()
                print(f"[EVAL_SINGLE] Cleared episode data from model")
                
            except Exception as e:
                print(f"[EVAL_SINGLE] ERROR saving episode data: {e}")
                import traceback
                traceback.print_exc()
        
        self.logger.info(f'Task {task_id} {task_description} episode {episode}: success {success}')
        return {"task_id": task_id, "task": task_description, "episode": episode, "success": success}
            
    def _set_results(self):
        self.save_dir = os.path.join(self.cfg.save_root, 
                                     f'{self.cfg.task_suite_name}-{self.cfg.model_family}', 
                                     f'step_{self.cfg.load_step}-vqa_{self.cfg.with_vqa}')
        os.makedirs(self.save_dir, exist_ok=True)
    
    def _build_logger(self, mode='w'):
        self.logger = Logger(os.path.join(self.save_dir, '000.log'), mode=mode)

    # def _check_free_gpus(self):
    #     """ Check free GPUs. Incompatible with HPC system"""
    #     used_memorys = os.popen(f"nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader").readlines()
    #     used_memorys = [int(memory.strip()) for memory in used_memorys]
    #     return [i for i, memory in enumerate(used_memorys) if memory < 1000]

    def _check_free_gpus(self):
        """ Get GPUs allocated to this job. Compatible with HPC system"""
        # Try to get GPUs from SLURM environment
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            cuda_devices = os.environ["CUDA_VISIBLE_DEVICES"]
            if cuda_devices:
                return [int(x) for x in cuda_devices.split(',')]
        
        # Try SLURM_STEP_GPUS or SLURM_JOB_GPUS
        for env_var in ["SLURM_STEP_GPUS", "SLURM_JOB_GPUS"]:
            if env_var in os.environ:
                gpu_ids = os.environ[env_var]
                return [int(x) for x in gpu_ids.split(',')]
        
        # Fallback: check nvidia-smi but be more lenient
        print("ERRRRORRRRR: shouldn't make it to this step")
        try:
            used_memorys = os.popen(f"nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader").readlines()
            used_memorys = [int(memory.strip()) for memory in used_memorys]
            # Use more lenient threshold for HPC
            available_gpus = [i for i, memory in enumerate(used_memorys) if memory < 5000]  # 5GB threshold
            
            if not available_gpus:
                print("Warning: No GPUs found with reasonable memory usage. Using all detected GPUs.")
                return list(range(len(used_memorys)))
            
            return available_gpus
        except Exception as e:
            print(f"Error checking GPU memory: {e}")
            print("Falling back to using GPU 0")
            return [0]

    def _set_gpu(self, gpu):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
        # list_physical devices can avoid cuda error, don't know why
        import tensorflow as tf
        tf.config.list_physical_devices("GPU")

        #Polina added
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_device(0)
    

    def _build_policy(self, gpu):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        self._set_gpu(gpu)

        from experiments.robot.openvla_utils import get_processor
        from experiments.robot.robot_utils import get_model, set_seed_everywhere
        
        set_seed_everywhere(self.cfg.seed)
        
        try:
            model = get_model(self.cfg)
            print(f"[DEBUG] GPU {gpu}: get_model successful, model type: {type(model)}", flush=True)
        except Exception as e:
            print(f"[ERROR] GPU {gpu}: get_model failed: {e}", flush=True)
            print(f"[ERROR] GPU {gpu}: get_model traceback:", flush=True)
            traceback.print_exc()
            raise
        
        if self.cfg.with_vqa:
            print(f"[DEBUG] GPU {gpu}: Loading VQA wrapper", flush=True)
            try:
                from vla_scripts.openvla_with_vqa import OpenVLAWithVQA
                model = OpenVLAWithVQA(model, self.cfg.check_catch, self.cfg.check_close, self.cfg.vqa_mode)
                print(f"[DEBUG] GPU {gpu}: VQA wrapper loaded successfully", flush=True)
            except Exception as e:
                print(f"[ERROR] GPU {gpu}: VQA wrapper failed: {e}", flush=True)
                traceback.print_exc()
                raise

        # [OpenVLA] Check that the model contains the action un-normalization key
        if self.cfg.model_family in ["openvla", "prismatic"]:
            print(f"[DEBUG] GPU {gpu}: Checking unnorm_key: {self.cfg.unnorm_key}", flush=True)
            print(f"[DEBUG] GPU {gpu}: Available norm_stats keys: {list(model.norm_stats.keys()) if hasattr(model, 'norm_stats') else 'No norm_stats'}", flush=True)
            
            if hasattr(model, 'norm_stats'):
                if self.cfg.unnorm_key not in model.norm_stats and f"{self.cfg.unnorm_key}_no_noops" in model.norm_stats:
                    print(f"[DEBUG] GPU {gpu}: Using {self.cfg.unnorm_key}_no_noops instead", flush=True)
                    self.cfg.unnorm_key = f"{self.cfg.unnorm_key}_no_noops"
                
                if self.cfg.unnorm_key not in model.norm_stats:
                    print(f"[ERROR] GPU {gpu}: unnorm_key {self.cfg.unnorm_key} not found in norm_stats", flush=True)
                    raise AssertionError(f"Action un-norm key {self.cfg.unnorm_key} not found in VLA `norm_stats`!")
            else:
                print(f"[ERROR] GPU {gpu}: Model has no norm_stats attribute", flush=True)

        processor = None
        if self.cfg.model_family == "openvla":
            print(f"[DEBUG] GPU {gpu}: Getting processor for openvla", flush=True)
            try:
                processor = get_processor(self.cfg)
                print(f"[DEBUG] GPU {gpu}: Processor loaded successfully", flush=True)
            except Exception as e:
                print(f"[ERROR] GPU {gpu}: get_processor failed: {e}", flush=True)
                traceback.print_exc()
                raise
        
        print(f"[DEBUG] GPU {gpu}: _build_policy completed successfully", flush=True)
        
        # Wrap model for data collection if enabled
        if self.cfg.collect_trajectory_data:
            print(f"[DEBUG] GPU {gpu}: Wrapping model for data collection", flush=True)
            try:
                from vla_scripts.data_collecting_vla import wrap_model_for_data_collection
                model = wrap_model_for_data_collection(model)
                print(f"[DEBUG] GPU {gpu}: Model wrapped successfully", flush=True)
            except Exception as e:
                print(f"[ERROR] GPU {gpu}: Failed to wrap model for data collection: {e}", flush=True)
                traceback.print_exc()
                raise
        
        return model, processor
    
    def _add_observation(self, obs, replay_images, replay_wrist_images):
        image = get_libero_image(obs, self.resize_size)
        # Image.fromarray(image).save('test.png')
        replay_images.append(image)

        # use_wrist_image
        if self.cfg.use_wrist_image:
            wrist_img = get_libero_image(obs, self.resize_size, key="robot0_eye_in_hand_image")
            replay_wrist_images.append(wrist_img)

    def _prepare_inputs(self, obs, replay_images, replay_wrist_images):
        # buffering #obs_history images, optionally
        image_history = replay_images[-self.cfg.obs_history :]
        if len(image_history) < self.cfg.obs_history:
            image_history.extend([replay_images[-1]] * (self.cfg.obs_history - len(image_history)))

        # same but for optional wrist images
        if self.cfg.use_wrist_image:
            wrist_image_history = replay_wrist_images[-self.cfg.obs_history :]
            if len(wrist_image_history) < self.cfg.obs_history:
                wrist_image_history.extend(
                    [replay_wrist_images[-1]] * (self.cfg.obs_history - len(wrist_image_history))
                )
            # interleaved images [... image_t, wrist_t ...]
            image_history = [val for tup in zip(image_history, wrist_image_history) for val in tup]

        # Prepare observations dict
        # Note: OpenVLA does not take proprio state as input
        return {
            "full_image": image_history,
            "state": np.concatenate(
                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
            ),
        }


def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1', 'yes'):
        return True
    elif v.lower() in ('false', '0', 'no'):
        return False
    else:
        raise ValueError(f"Cannot convert {v} to boolean.")


def main(args):
    for step in args.steps:
        cfg = GenerateConfig(
            load_step=step, 
            pretrained_checkpoint=args.pretrained_checkpoint,
            num_trials_per_task=args.num_trails_per_task,
            num_gpus=args.num_gpus,
            num_processes=args.num_processes,
            task_suite_name=args.task_suite_name,
            save_root=args.save_root,
            with_vqa=str_to_bool(args.with_vqa),
            check_catch=str_to_bool(args.check_catch),
            check_close=str_to_bool(args.check_close),
            vqa_mode=args.vqa_mode,
            obs_history=args.obs_history,
            use_wrist_image=args.use_wrist_image,
            collect_trajectory_data=getattr(args, 'collect_trajectory_data', False),
            trajectory_data_save_path=getattr(args, 'trajectory_data_save_path', './trajectory_data'),
            max_total_trajectories=getattr(args, 'max_total_trajectories', None),
        )
        evaluator = ParallelLiberoEvaluator(cfg)
        evaluator.evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--num-processes', type=int, default=32)
    parser.add_argument('--task-suite-name', default='libero_90')
    parser.add_argument('--num-trails-per-task', type=int, default=10)
    parser.add_argument('--pretrained-checkpoint', default='')
    parser.add_argument('--save-root', default='./results')
    parser.add_argument('--with-vqa', type=str, default='False')
    parser.add_argument('--check-catch', type=str, default='True')
    parser.add_argument('--check-close', type=str, default='False')
    parser.add_argument('--vqa-mode', default='coarse_direction')
    parser.add_argument('--steps', nargs='+', type=int)
    parser.add_argument('--obs-history', type=int, default=1)
    parser.add_argument('--use-wrist-image', action='store_true')
    
    # Data collection arguments
    parser.add_argument('--collect-trajectory-data', action='store_true', 
                       help='Enable trajectory data collection for linear probe training')
    parser.add_argument('--trajectory-data-save-path', type=str, default='./trajectory_data',
                       help='Path to save trajectory data')
    parser.add_argument('--max-total-trajectories', type=int, default=None,
                       help='Maximum total trajectories to evaluate (for early stopping)')
    
    args = parser.parse_args()
    main(args)
