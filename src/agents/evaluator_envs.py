import copy
import datetime
import json
import logging
import os
import shlex
import subprocess
from abc import ABC
from dataclasses import asdict, dataclass
from time import sleep
from typing import Any

import gymnasium as gym
import numpy as np
from simple_slurm import Slurm
from tqdm import tqdm

from agents.client import RemoteAgent
from agents.policies import Act, Agent, Obs
from agents.wrappers import HumanCameraWrapper

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


class EvaluatorEnv(ABC):
    ENVS: dict[str, "EvaluatorEnv"] = {}

    def __init__(self, env_id: str, seed: int, **env_kwargs) -> None:
        self.do_import()
        self.env = gym.make(env_id, **env_kwargs)
        self.env.np_random = np.random.RandomState(seed=seed)
        self.env_id = env_id
        self.seed = seed

    def step(self, action: Act) -> tuple[Obs, float, bool, bool, dict]:
        raise NotImplementedError

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Obs, dict[str, Any]]:
        raise NotImplementedError

    @property
    def language_instruction(self) -> str:
        raise NotImplementedError

    @staticmethod
    def register(env_id: str, env: "EvaluatorEnv") -> None:
        EvaluatorEnv.ENVS[env_id] = env

    @staticmethod
    def make(env_id: str, seed: int, **env_kwargs) -> "EvaluatorEnv":
        return EvaluatorEnv.ENVS[env_id](env_id, seed, **env_kwargs)

    @staticmethod
    def do_import():
        raise NotImplementedError


class ManiSkill(EvaluatorEnv):
    INSTRUCTIONS = {
        "LiftPegUpright-v1": "lift the peg upright",
        "PegInsertionSide-v1": "insert the peg from the side",
        "PickCube-v1": "pick up the cube",
        "PlugCharger-v1": "plug the charger in",
        "PullCube-v1": "pull the cube towards the robot base",
        "PullCubeTool-v1": "pull the cube by using the red tool",
        "PushCube-v1": "push the cube away from the robot base",
        "PushT-v1": "align the T shape",
        "RollBall-v1": "push the ball",
        "StackCube-v1": "stack the red cube on the green cube",
        "PokeCube-v1": "push the cube by using the blue tool",
    }

    def __init__(self, env_id, seed, **env_kwargs):
        # TODO: one could save only every nth episode by adding an episode counter which steps the record env only
        # when the counter is divisible by n otherwise steps the normal env
        logging.info(f"Creating ManiSkill env {env_id}")
        if "video_dir" in env_kwargs:
            output_dir = env_kwargs["video_dir"]
            del env_kwargs["video_dir"]
        else:
            output_dir = None
        super().__init__(env_id, seed, **env_kwargs)
        logging.info(f"Created ManiSkill env {env_id}")
        if "human_render_camera_configs" in env_kwargs:
            self.env = HumanCameraWrapper(self.env)

        if output_dir is not None:
            logging.info(f"Recording to {output_dir}")
            from mani_skill.utils import wrappers

            self.env = wrappers.RecordEpisode(
                self.env,
                output_dir,
                save_on_reset=True,
                save_trajectory=True,
                trajectory_name=f"eval-{env_id}",
                save_video=True,
                video_fps=30,
                record_reward=True,
            )
        logging.info(f"Done Created ManiSkill env {env_id}")

    def translate_obs(self, obs: dict[str, Any]) -> Obs:
        # does not include history
        return Obs(
            rgb_side=obs["sensor_data"]["base_camera"]["rgb"].squeeze(0).numpy(),
            # gripper=float(not obs["extra"]["is_grasped"]),
        )

    def step(self, action: Act) -> tuple[Obs, float, bool, bool, dict]:
        # includes horizon
        # careful with gripper action: the model needs to be trained on [-1, 1] interval

        a = copy.copy(action.action[0])
        # a[-1] = -1.0 if a[-1] < 0.9 else 1.0
        if self.env_id == "PushT-v1":
            a = a[:-1]
        else:
            a[-1] = a[-1] * 2 - 1.0
        obs, reward, success, truncated, info = self.env.step(a)
        return self.translate_obs(obs), reward, success, truncated, info

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Obs, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self.translate_obs(obs), info

    @property
    def language_instruction(self) -> str:
        return self.INSTRUCTIONS[self.env_id]

    @staticmethod
    def do_import():
        import mani_skill.envs


EvaluatorEnv.register("LiftPegUpright-v1", ManiSkill)
EvaluatorEnv.register("PegInsertionSide-v1", ManiSkill)
EvaluatorEnv.register("PickCube-v1", ManiSkill)
EvaluatorEnv.register("PlugCharger-v1", ManiSkill)
EvaluatorEnv.register("PullCube-v1", ManiSkill)
EvaluatorEnv.register("PullCubeTool-v1", ManiSkill)
EvaluatorEnv.register("PushCube-v1", ManiSkill)
EvaluatorEnv.register("PushT-v1", ManiSkill)
EvaluatorEnv.register("RollBall-v1", ManiSkill)
EvaluatorEnv.register("StackCube-v1", ManiSkill)
EvaluatorEnv.register("PokeCube-v1", ManiSkill)


@dataclass
class EvalConfig:
    env_id: str
    env_kwargs: dict[str, Any]
    max_steps_per_episode: int = 100


@dataclass
class ClientConfig:
    host: str
    port: int
    model: str


def single_eval(env: EvaluatorEnv, agent: Agent, max_steps: int) -> tuple[list[float], list[float], list[float]]:
    logging.info(f"Starting evaluation of {env.env.unwrapped.spec.id}")
    obs, _ = env.reset(options={})
    logging.info(f"Reset env {env.env.unwrapped.spec.id}")
    agent.reset(obs, env.language_instruction)
    logging.info(f"Reset agent {env.env.unwrapped.spec.id}")
    done = False
    truncated = False
    step = 0.0
    rewards = []
    while not done and not truncated and max_steps > step:
        action = agent.act(obs)
        obs, reward, done, truncated, _ = env.step(action)
        reward = float(reward)
        done, truncated = bool(done), bool(truncated)
        step += 1
        rewards.append(reward)

    env.reset(options={})
    logging.info(
        f"Finished evaluation of {env.env.unwrapped.spec.id} with {step} steps and reward {reward}, success {done}"
    )
    # success, last reward and number of steps
    return done, rewards, step


per_process_cache = {}


def create_env_agent(client_config: ClientConfig, cfg: EvalConfig, seed: int) -> tuple[EvaluatorEnv, RemoteAgent]:
    logging.info(f"retrieving env {cfg.env_id} and agent")
    if cfg.env_id not in per_process_cache:
        logging.info(f"env {cfg.env_id} not available, creating new env and agent")
        env = EvaluatorEnv.make(cfg.env_id, seed=seed, **cfg.env_kwargs)
        logging.info("done creating env")
        agent = RemoteAgent(client_config.host, client_config.port, client_config.model)
        logging.info("done creating agent")
        per_process_cache[cfg.env_id] = (env, agent)
    return per_process_cache[cfg.env_id]


def per_process(args: tuple[int, list[EvalConfig], int, ClientConfig]) -> tuple[float, float, float]:
    logging.info(f"Starting process {args}")
    i, cfgs, episodes, client_cfg = args
    cfg = cfgs[i // episodes]
    env, agent = create_env_agent(client_cfg, cfg, seed=i)
    # busy wait for server to finish initialization
    while not agent.is_initialized():
        logging.info("Waiting for agent to initialize...")
        sleep(5)
    return single_eval(env, agent, cfg.max_steps_per_episode)


def multi_eval(
    client_cfg: ClientConfig, cfgs: list[EvalConfig], episodes: int = 100, n_processes: int = 1
) -> tuple[np.ndarray, list[list[list[float]]]]:
    # return is [envs, episodes, 3(success, reward, steps)], [envs, episodes, rewards for all steps in the episode]
    logging.info(f"Starting evaluation with {len(cfgs)} environments and {episodes} episodes each")

    # with process
    # with Pool(n_processes) as p:
    #     args = [(i, cfgs, episodes, client_cfg) for i in range(len(cfgs) * episodes)]
    #     single_results = p.map(per_process, args)

    # without process
    args = [(i, cfgs, episodes, client_cfg) for i in range(len(cfgs) * episodes)]
    single_results = [per_process(arg) for arg in tqdm(args)]

    single_results_last_reward = np.array([(i[0], i[1][-1], i[2]) for i in single_results])

    # this works because row-major order
    # per_env_results = single_results.reshape(len(cfgs), episodes, 3)
    per_env_results_last_reward = single_results_last_reward.reshape(len(cfgs), episodes, 3)
    per_env_results_rewards = [
        [i[1] for i in single_results[i : i + episodes]] for i in range(0, len(single_results), episodes)
    ]
    return per_env_results_last_reward, per_env_results_rewards


def start_server(
    agent_name: str, kwargs: dict[str, Any], port=8080, host="localhost", python_path: str = "python"
) -> subprocess.Popen:
    """Start the agent server in a subprocess.

    Args:
        agent_name (str): Name of the agent to start.
        kwargs (dict[str, Any]): Additional keyword arguments for the agent.
        port (int): Port to start the server on. Defaults to 8080.
        host (str): Host to bind the server to. Defaults to "localhost".
        python_path (str): Path to the Python interpreter to use. If you use conda you can look up the path with `conda info --envs`.
            It can also be a format string that will be formatted with the agent_name, e.g. "conda run -n {agent_name} python".
            Defaults to "python".
    Returns:
        subprocess.Popen: The process running the server.
    """

    logging.info(
        f"Server starting with command: {python_path.format(agent_name=agent_name)} -m agents start-server {agent_name} --port={port} --host={host} --kwargs={json.dumps(kwargs)}"
    )
    p = subprocess.Popen(
        [
            python_path.format(agent_name=agent_name),
            "-m",
            "agents",
            "start-server",
            f"{agent_name}",
            f"--port={port}",
            f"--host={host}",
            f"--kwargs={json.dumps(kwargs)}",
        ]
    )
    logging.info("successfully started")
    return p


def evaluation(
    agent_name: str,
    kwargs: dict[str, Any],
    eval_cfgs: list[EvalConfig],
    port: int = 8080,
    host: str = "localhost",
    episodes: int = 100,
    n_processes: int = 1,
    python_path: str = "python",
):
    logging.info(f"Starting evaluation with {agent_name} and {kwargs}")
    with start_server(agent_name, kwargs, port, host, python_path) as p:
        client_cfg = ClientConfig(host, port, agent_name)
        res = multi_eval(client_cfg, eval_cfgs, episodes, n_processes)
        logging.info("Evaluation finished")
        # send ctrl c signal
        p.send_signal(subprocess.signal.SIGINT)

    logging.info(f"Results (success, reward, steps) for all envs: {res[0].mean(axis=1)}")
    logging.info(
        f"Mean reward for all envs: {[np.mean([np.mean(ep_rewards) for ep_rewards in env_rewards]) for env_rewards in res[1]]}"
    )
    # print indices of successful episodes
    for idx, env in enumerate(res[0]):
        logging.info(f"Env {eval_cfgs[idx].env_id} successful episodes: {np.where(env[:, 0])[0]}")
    return res


def run_eval_during_training(
    agent_name: str,
    kwargs: dict[str, Any],
    eval_cfgs: list[EvalConfig],
    wandb_id: str,
    wandb_entity: str,
    wandb_group: str,
    wandb_project: str,
    wandb_note: str,
    wandb_name: str,
    slurm: Slurm,
    output_path: str,
    wandb_first: bool = False,
    port=8080,
    host="localhost",
    episodes: int = 100,
    n_processes: int | None = None,
):
    cmd = [
        "python",
        "-m",
        "agents" "run-eval-during-training",
        agent_name,
        f"--kwargs={json.dumps(kwargs)}",
        f"--port={port}",
        f"--host={host}",
        f"--episodes={episodes}",
        f"--n-processes={n_processes}",
        f"--eval-cfgs={json.dumps([asdict(cfg) for cfg in eval_cfgs])}",
        f"--wandb-id={wandb_id}",
        f"--wandb-group={wandb_group.replace(':', '_')}",
        f"--wandb-project={wandb_project}",
        f"--wandb-entity={wandb_entity}",
        f"--wandb-note={wandb_note}",
        f"--wandb-name={wandb_name}",
        f"--output-path={output_path}",
    ]
    if wandb_first:
        cmd.append("--wandb-first")
    slurm.sbatch(shlex.join(cmd))


def run_eval_post_training(
    agent_name: str,
    kwargs: dict[str, Any],
    eval_cfgs: list[EvalConfig],
    wandb_entity: str,
    wandb_project: str,
    wandb_note: str,
    wandb_name: str,
    checkpoint_steps: list[int],
    slurm: Slurm,
    output_path: str,
    wandb_group: str | None = None,
    port=8080,
    host="localhost",
    episodes: int = 100,
    n_processes: int | None = None,
    video: bool = False,
    n_gpus: int = 1,
):
    if video:
        run_recordings = os.path.join(output_path, "run_recordings")
        os.mkdir(run_recordings)
        for cfg in eval_cfgs:
            cfg.env_kwargs["video_dir"] = run_recordings
    slurm.sbatch(
        shlex.join(
            [
                "python",
                "-m",
                "agents",
                "run-eval-post-training",
                agent_name,
                f"--kwargs={json.dumps(kwargs)}",
                f"--port={port}",
                f"--host={host}",
                f"--episodes={episodes}",
                f"--n-processes={n_processes}",
                f"--eval-cfgs={json.dumps([asdict(cfg) for cfg in eval_cfgs])}",
                f"--wandb-group={wandb_group.replace(':', '_') if wandb_group else ''}",
                f"--wandb-project={wandb_project}",
                f"--wandb-entity={wandb_entity}",
                f"--wandb-note={wandb_note}",
                f"--wandb-name={wandb_name}",
                f"--n-gpus={n_gpus}",
                f"--steps={json.dumps(checkpoint_steps)}",
                f"--run-path={output_path}",
            ]
        ),
    )


def write_results(
    results: np.ndarray,
    rewards: list[list[list[float]]],
    eval_cfgs: list[EvalConfig],
    model_cfg: dict[str, Any],
    out: str = "",
) -> str:
    # first read json, if not exists write empty list
    path = os.path.join(out, f"results_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump([], f)
    with open(path, "r") as f:
        prev_results = json.load(f)
    assert isinstance(prev_results, list)

    flatten_rewards = [[item for sublist in env_rewards for item in sublist] for env_rewards in rewards]
    mean_rewards = [np.mean(env_rewards) for env_rewards in flatten_rewards]

    for idx, cfg in enumerate(eval_cfgs):
        success_mean, reward_mean, steps_mean = results[idx].mean(axis=0, keepdims=False)
        success_max, reward_max, steps_max = results[idx].max(axis=0, keepdims=False)
        success_min, reward_min, steps_min = results[idx].min(axis=0, keepdims=False)
        sucess_std, reward_std, steps_std = results[idx].std(axis=0, keepdims=False)
        success_median, reward_median, steps_median = np.median(results[idx], axis=0, keepdims=False)
        prev_results.append(
            {
                "success": {
                    "mean": success_mean,
                    "max": success_max,
                    "min": success_min,
                    "std": sucess_std,
                    "median": success_median,
                    "values": results[idx, :, 0].tolist(),
                },
                "reward_last_step": {
                    "mean": reward_mean,
                    "max": reward_max,
                    "min": reward_min,
                    "std": reward_std,
                    "median": reward_median,
                    "values": results[idx, :, 1].tolist(),
                },
                "rewards": {
                    "mean": mean_rewards[idx],
                    "values": rewards[idx],
                },
                "steps": {
                    "mean": steps_mean,
                    "max": steps_max,
                    "min": steps_min,
                    "std": steps_std,
                    "median": steps_median,
                    "values": results[idx, :, 2].tolist(),
                },
                "episodes": len(results),
                "timestamp": datetime.datetime.now().isoformat(),
                "env_cfg": asdict(cfg),
                "model_cfg": model_cfg,
            }
        )

    with open(path, "w") as f:
        json.dump(prev_results, f, indent=2)
    return path
