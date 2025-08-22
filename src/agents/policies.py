import copy
import json
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from functools import partial, reduce
from multiprocessing import resource_tracker, shared_memory
from operator import getitem
from pathlib import Path
from typing import Any, Union, Optional

import numpy as np
from PIL import Image


@dataclass
class SharedMemoryPayload:
    shm_name: str
    shape: tuple[int, ...]
    dtype: str = "uint8"


@dataclass
class Obs:
    cameras: dict[str, Union[np.ndarray, SharedMemoryPayload]] = field(default_factory=dict)
    gripper: Optional[float] = None
    info: dict[str, Any] = field(default_factory=dict)
    camera_data_in_shared_memory: bool = False


@dataclass
class Act:
    action: np.ndarray
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class Agent:
    def __init__(
        self, default_checkpoint_path: str, checkpoint_path: Optional[str] = None, checkpoint_step: Optional[int] = None
    ) -> None:
        self.instruction = None
        self.step = -1
        self.episode = -1
        self.checkpoint_step = checkpoint_step
        self.default_checkpoint_path = default_checkpoint_path
        self.checkpoint_path = checkpoint_path
        self._shm: dict[str, shared_memory.SharedMemory] = {}

    def initialize(self):
        # heavy initialization, e.g. loading models
        pass

    def _from_shared_memory(self, obs: Obs) -> Obs:
        """transparently uses shared memory if configured and modifies obs in place"""
        if obs.camera_data_in_shared_memory:
            camera_dict = {}
            for camera_name, camera_data in obs.cameras.items():
                assert isinstance(camera_data, SharedMemoryPayload)
                if camera_data.shm_name not in self._shm:
                    self._shm[camera_data.shm_name] = shared_memory.SharedMemory(camera_data.shm_name)
                camera_dict[camera_name] = np.ndarray(
                    camera_data.shape, dtype=camera_data.dtype, buffer=self._shm[camera_data.shm_name].buf
                )
            obs.cameras = camera_dict
        return obs

    def act(self, obs: Obs) -> Act:
        assert self.instruction is not None, "forgot reset?"
        self.step += 1
        self._from_shared_memory(obs)

        return Act(action=np.zeros(7, dtype=np.float32), done=False, info={})

    def reset(self, obs: Obs, instruction: Any, **kwargs) -> dict[str, Any]:
        logging.info(f"Resetting agent, new instruction: {instruction} ###############")
        self.step = 0
        self.episode += 1
        self.instruction = instruction
        self._from_shared_memory(obs)
        # info
        return {}

    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        self.close()

    def close(self, *args, **kwargs):
        for shm in self._shm.values():
            shm.close()
            resource_tracker.unregister(shm._name, "shared_memory")
        self._shm = {}


class TestAgent(Agent):

    def __init__(self, **kwargs) -> None:
        super().__init__(default_checkpoint_path="", **kwargs)
        self.i = 0

    def act(self, obs: Obs) -> Act:
        super().act(obs)
        # echo data back for testing
        info = {
            "shapes": {k: v.shape for k, v in obs.cameras.items()},
            "dtype": {k: v.dtype.name for k, v in obs.cameras.items()},
            "data": {k: v for k, v in obs.cameras.items()},
        }
        a = Act(action=np.array([0, 0, 0, 0, 0, 0, self.i % 2], dtype=np.float32), done=False, info=info)
        self.i += 1
        return a

    def reset(self, obs: Obs, instruction: Any, **kwargs) -> dict[str, Any]:
        super().reset(obs, instruction, **kwargs)
        info = {
            "shapes": {k: v.shape for k, v in obs.cameras.items()},
            "dtype": {k: v.dtype.name for k, v in obs.cameras.items()},
            "data": {k: v for k, v in obs.cameras.items()},
            "instruction": instruction,
        }
        return info


class OpenVLAModel(Agent):
    # === Utilities ===
    SYSTEM_PROMPT = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )

    def __init__(
        self,
        attn_implementation: str,
        device: str,
        unnorm_key: str,
        default_checkpoint_path: str = "openvla/openvla-7b",
        **kwargs,
    ) -> None:
        super().__init__(default_checkpoint_path=default_checkpoint_path, **kwargs)
        self.unnorm_key = unnorm_key
        logging.info(f"Using unnorm_key: {self.unnorm_key}")
        self.attn_implementation = attn_implementation
        if self.checkpoint_step is None or self.checkpoint_path is None:
            self.openvla_path = self.default_checkpoint_path
            logging.info(f"Using default checkpoint path: {self.openvla_path}")
            if self.unnorm_key != "viola":
                logging.warning(
                    "unnorm_key should be 'viola' when using default path, ignoring unnorm_key and setting it to 'viola'"
                )
                self.unnorm_key = "viola"
        else:
            self.openvla_path = self.checkpoint_path.format(checkpoint_step=self.checkpoint_step)
            logging.info(
                f"Using custom checkpoint path: {self.openvla_path} with checkpoint step: {self.checkpoint_step}"
            )
        self.device = device
        self.attn_implementation = attn_implementation

    def initialize(self):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.device = torch.device(self.device) if torch.cuda.is_available() else torch.device("cpu")

        # Load VLA Model using HF AutoClasses
        self.processor = AutoProcessor.from_pretrained(self.openvla_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.openvla_path,
            attn_implementation=self.attn_implementation,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        print("==========================")
        print(self.vla.norm_stats.keys())

        # [Hacky] Load Dataset Statistics from Disk (if passing a path to a fine-tuned model)
        if os.path.isdir(self.openvla_path):
            with open(Path(self.openvla_path) / "dataset_statistics.json", "r") as f:
                self.vla.norm_stats = json.load(f)

    def get_openvla_prompt(self, instruction: str, openvla_path: Union[str, Path]) -> str:
        if "v01" in openvla_path:
            return f"{self.SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
        else:
            return f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    def act(self, obs: Obs) -> Act:
        # no batch dimension here
        import torch

        super().act(obs)
        # Parse payload components
        assert obs.cameras["rgb_side"].shape == (256, 256, 3), "wrong shape, use lanczos"
        image = obs.cameras["rgb_side"]
        unnorm_key = self.unnorm_key

        # Run VLA Inference
        prompt = self.get_openvla_prompt(self.instruction, self.openvla_path)
        inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(self.device, dtype=torch.bfloat16)
        # to use temperature use: do_sample=True, temperature=50.0
        action = self.vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
        # unsqueeze to add horizon dimension
        return Act(action=action[None])


class OctoModel(Agent):
    """
    This model is trained with a window size of 2, predicting 7 dimensional actions 4 steps into the future.
    Observations and tasks conform to the following spec:

    Observations: {
        image_primary: ('batch', 'history_window', 256, 256, 3),
        image_wrist: ('batch', 'history_window', 128, 128, 3),
    }
    Tasks: {
        image_primary: ('batch', 256, 256, 3),
        image_wrist: ('batch', 128, 128, 3),
        language_instruction: {
            attention_mask: ('batch', 16),
            input_ids: ('batch', 16),
        },
    }

    At inference, you may pass in any subset of these observation and task keys, with a history window up to 2 timesteps.
    """

    def __init__(
        self,
        horizon: int = 2,
        unnorm_key: Optional[list[str]] = None,
        default_checkpoint_path: str = "hf://rail-berkeley/octo-base-1.5",
        **kwargs,
    ) -> None:
        # default window size is 2 in octo
        # default unnorm is viola as it used the fr3
        super().__init__(default_checkpoint_path=default_checkpoint_path, **kwargs)
        self.horizon = horizon
        if unnorm_key is None:
            self.unnorm_key = []
        else:
            self.unnorm_key = unnorm_key

        # log checkpoint path and step and kwargs
        logging.info(f"checkpoint_path: {self.checkpoint_path}, checkpoint_step: {self.checkpoint_step}")
        logging.info(f"horizon: {self.horizon}")
        logging.info(f"unnorm_key: {self.unnorm_key}")
        logging.info(f"kwargs: {kwargs}")
        if self.checkpoint_path is None:
            self.octo_path = self.default_checkpoint_path
            logging.info(f"Using default checkpoint path: {self.octo_path}")
            if self.checkpoint_step is not None:
                logging.warning(
                    "checkpoint_step should be None when using default path, ignoring checkpoint_step and setting it to None"
                )
                self.checkpoint_step = None
            if self.unnorm_key != ["viola"]:
                logging.warning(
                    "unnorm_key should be ['viola'] when using default path, ignoring unnorm_key and setting it to ['viola']"
                )
                self.unnorm_key = ["viola"]
        else:
            self.octo_path = self.checkpoint_path
            logging.info(f"Using custom checkpoint path: {self.octo_path} with checkpoint step: {self.checkpoint_step}")
        logging.info(f"Using unnorm_key: {self.unnorm_key}")

    def initialize(self):
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = (
            "false"  # disable preallocation of memory in jax (might make it less efficient)
        )
        from octo.model.octo_model import OctoModel as _OctoModel
        from octo.utils.train_callbacks import supply_rng

        self.model = _OctoModel.load_pretrained(self.octo_path, self.checkpoint_step)

        window_size = self.model.example_batch["observation"]["timestep_pad_mask"].shape[1]
        if window_size < self.horizon:
            logging.warning(
                f"Horizon {self.horizon} is greater than the model's window size {window_size} with which the model has been trained with. "
            )

        self.trained_obs = self.model.example_batch["observation"].keys()

        self.horizon = self.horizon
        self.history = deque(maxlen=self.horizon)
        self.num_obs = 0

        logging.info("==========================")
        logging.info(self.model.dataset_statistics.keys())
        self.policy_fn = supply_rng(
            partial(
                self.model.sample_actions,
                unnormalization_statistics=reduce(getitem, self.unnorm_key, self.model.dataset_statistics)["action"],
            ),
        )
        self.task = None

    def act(self, obs: Obs) -> Act:
        # from octo.model.octo_model import _verify_shapes
        import jax
        from octo.utils.gym_wrappers import stack_and_pad

        super().act(obs)
        assert self.task is not None, "forgot reset?"
        # _verify_shapes(obs, <name>, self.model.example_batch["observation"])

        self.num_obs += 1

        # single image
        assert obs.cameras["rgb_side"].shape == (256, 256, 3), "wrong shape, use lanczos"
        obs = {"image_primary": obs.cameras["rgb_side"]}

        self.history.append(obs)
        assert len(self.history) == self.horizon, "forgot reset?"
        full_obs = stack_and_pad(self.history, self.num_obs)

        actions = self.policy_fn(
            jax.tree_map(
                lambda x: x[None],
                full_obs,
            ),
            self.task,
        )
        # remove the batch dimension (batch, horizon, action)
        return Act(action=np.array(actions[0, :, :]))

    def reset(self, obs: Obs, instruction: Any):
        super().reset(obs, instruction)
        assert obs.cameras["rgb_side"].shape == (256, 256, 3), "wrong shape"
        obs = {"image_primary": obs.cameras["rgb_side"]}
        self.task = self.model.create_tasks(texts=[instruction])
        self.num_obs = 1
        self.history.extend([obs] * self.horizon)
        return {}


class OctoActionDistribution(OctoModel):
    """
    this model does not support history window
    dont use self.step and self.episode as this model is not used sequentially
    """

    def __init__(self, **kwargs) -> None:
        assert kwargs["horizon"] == 1, "horizon must be 1 for OctoActionDistribution"
        super().__init__(**kwargs)

    def act(self, obs: Obs) -> Act:
        """
        Args:
            Obs:
                cameras:
                    rgb_side: np.ndarray[tuple[BATCH, H, W, Literal[3]], np.dtype[np.int8]]
                info:
                    num_samples: int
        Return:
            Act:
                action: None
                info:
                    means: np.ndarray[tuple[BATCH, 7], np.dtype[np.float32]]
                    stds: np.ndarray[tuple[BATCH, 7], np.dtype[np.float32]]
        """
        import jax
        import jax.numpy as jnp

        self._from_shared_memory(obs)

        batch_size = obs.cameras["rgb_side"].shape[0]
        assert obs.cameras["rgb_side"].shape == (batch_size, 256, 256, 3), "wrong shape"
        assert self.instruction is not None, "forgot reset?"
        num_samples = obs.info.get("num_samples", 1)

        x = jnp.array(obs.cameras["rgb_side"])  # BATCH, H, W, 3
        # x_expanded = x[:, None, :, :, :]
        x_expanded = jnp.expand_dims(x, 1)
        x_tiled = jnp.tile(x_expanded, (1, num_samples, 1, 1, 1))  # Shape: [BATCH, N, H, W, 3]
        x_duplicated = x_tiled.reshape(-1, x.shape[1], x.shape[2], x.shape[3])  # Shape: [BATCH*N, H, W, 3]
        full_obs = {
            "image_primary": jnp.expand_dims(x_duplicated, 1),
            "timestep_pad_mask": np.ones((batch_size * num_samples, 1)),
        }
        # full_obs = stack_and_pad(x_duplicated, 1)
        tasks = self.model.create_tasks(texts=[self.instruction] * batch_size * num_samples)
        actions = self.policy_fn(
            full_obs,
            tasks,
        )
        # actions: [num_samples x BATCH, 4, 7]
        # remove the horizon dimension and reshape to [BATCH, num_samples, 7]
        actions = actions[:, 0, :].reshape(batch_size, num_samples, 7)
        stds = jnp.std(actions, axis=1)
        means = jnp.mean(actions, axis=1)

        stds = np.asarray(stds)
        means = np.asarray(means)

        return Act(action=None, info={"means": means, "stds": stds, "actions": np.asarray(actions)})

    def reset(self, obs, instruction):
        self.instruction = instruction
        return {}


class OpenVLADistribution(OpenVLAModel):

    def act(self, obs: Obs) -> Act:
        # no batch dimension here
        import torch

        self._from_shared_memory(obs)

        assert self.instruction is not None, "forgot reset?"
        self.step += 1
        batch_size = obs.cameras["rgb_side"].shape[0]

        # Parse payload components
        images = obs.cameras["rgb_side"]
        actions = []
        unnorm_key = self.unnorm_key
        num_samples = obs.info.get("num_samples", 1)

        # time it
        import time

        t1 = time.time()
        # Run VLA Inference
        prompt = self.get_openvla_prompt(self.instruction, self.openvla_path)

        x_expanded = np.expand_dims(images, 1)
        x_tiled = np.tile(x_expanded, (1, num_samples, 1, 1, 1))  # Shape: [BATCH, N, H, W, 3]
        x_duplicated = x_tiled.reshape(
            -1, images.shape[1], images.shape[2], images.shape[3]
        )  # Shape: [BATCH*N, H, W, 3]

        for image in x_duplicated:
            inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(self.device, dtype=torch.bfloat16)
            # to use temperature use: do_sample=True, temperature=50.0
            action = self.vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
            actions.append(action)

        t2 = time.time()
        logging.info(f"needed time for {len(actions)} was {t2-t1}s")

        # unsqueeze to add horizon dimension
        actions = np.stack(actions).astype(np.float32)
        actions = actions.reshape(batch_size, num_samples, 7)
        means = np.mean(actions, axis=1).astype(np.float32)
        stds = np.std(actions, axis=1).astype(np.float32)
        return Act(action=None, info={"means": means, "stds": stds, "actions": actions})

class DiffusionPolicy(Agent):
    """
    Agent implementation for vanilla diffusion policy
    """

    def __init__(
        self,
        checkpoint_path: str = "/home/walter/Weights/latest.ckpt",
        steps_per_inference: int = 3,
        control_frequency_hz: int = 30,
        **kwargs,
    ) -> None:
        super().__init__(default_checkpoint_path="", **kwargs)

        self.checkpoint_path = checkpoint_path
        self.steps_per_inference = steps_per_inference
        self.control_frequency_hz = control_frequency_hz

        self.step = self.steps_per_inference
        self.last_action_time = None

        # log checkpoint path and step and kwargs
        logging.info(f"checkpoint_path: {self.checkpoint_path}")
        logging.info(f"steps_per_inference: {self.steps_per_inference}")
        logging.info(f"kwargs: {kwargs}")
        if self.checkpoint_path is None:
            logging.error(
                "No checkpoint provided"
            )
        else:
            logging.info(
                f"Loading checkpoint {self.checkpoint_path}"
            )

    def initialize(self):
        import sys
        sys.path.insert(0, "/home/walter/Code/private-diffusion-policy-fork")
        from diffusion_policy.utn.agent import UTNDiffusionPolicy

        self.policy = UTNDiffusionPolicy(self.checkpoint_path)
        self.policy.initialize()
        self.obs_buffer = deque(maxlen=self.policy.cfg.n_obs_steps)

    def act(self, obs: Obs) -> Act:
        from datetime import datetime
        from time import sleep
        # from diffusion_policy.utn.arro import ARROSegmentation

        # segmented_img = arro.segment(obs.rgb_side)

        self.obs_buffer.append(obs.rgb_side)


        if self.step >= self.steps_per_inference:
            print("Computing new actions")
            self.actions = self.policy.act(self.obs_buffer)
            self.step = 0

        print("Action: ", self.actions[self.step])

        # Needs to be activated if robot is running in async mode
        
        # Ensure that actions are not issued faster than the specified frequency
        # if self.last_action_time is not None:
        #     time_diff = (datetime.now() - self.last_action_time).total_seconds()
        #     sleep_time = 1/self.control_frequency_hz - time_diff
        #     if sleep_time > 0:
        #         sleep(sleep_time)                

        self.step += 1
        self.last_action_time = datetime.now()
        return Act(action=self.actions[self.step])        

    def reset(self, obs: Obs, instruction: Any):
        super().reset(obs, instruction="")
        image = obs.rgb_side
        self.obs_buffer.clear()
        self.policy.reset()
        self.last_action_time = None
        self.step = self.steps_per_inference
        self.obs_buffer.extend([image] * self.policy.cfg.n_obs_steps)
        return {}

class ReplayPolicy(Agent):
    def __init__(self,
                 trajectory_path="/home/walter/Data/LowSize_clean_data_zarr"
                 ):
        super().__init__(default_checkpoint_path="")

        self.trajectory_path=trajectory_path

    def initialize(self):
        pass
    
    def act(self, obs):
        if self.step < self.episode_length:
            action = self.replay_episode["action"][self.step]
            self.step += 1
            return Act(action=action)
        else:
            return Act(done=True)
        
    def reset(self, obs=None, instruction=None, **kwargs):
        import sys
        sys.path.insert(0, "/home/walter/Code/private-diffusion-policy-fork")
        from diffusion_policy.common.replay_buffer import ReplayBuffer
        super().reset(obs, instruction, **kwargs)

        buffer = ReplayBuffer.create_from_path(self.trajectory_path)
        self.episode_idx = int(instruction)
        self.replay_episode = buffer.get_episode(self.episode_idx)
        self.episode_length = self.replay_episode["action"].shape[0]

        self.step = 0
        return {}   

AGENTS = dict(
    test=TestAgent,
    octo=OctoModel,
    openvla=OpenVLAModel,
    octodist=OctoActionDistribution,
    openvladist=OpenVLADistribution,
    diffusionpolicy=DiffusionPolicy,
    replaypolicy=ReplayPolicy,
)