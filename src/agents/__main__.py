import copy
import json
import logging
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import rpyc
import typer
import wandb

# to use non-inline backend, necessary for the case
# when started from jupyter notebook
os.environ["MPLBACKEND"] = "Agg"

from agents.evaluator_envs import EvalConfig, evaluation, write_results
from agents.policies import AGENTS
from agents.server import AgentService

main_app = typer.Typer(help="CLI tool for the agents library.")


def wandb_log_git_diff(path: str):
    path = Path(path)
    git_path = path / "git"
    git_path.mkdir(parents=True, exist_ok=True)
    log_git_diff(git_path)
    wandb.log_artifact(git_path, type="directory", name="git")


def log_git_diff(path: str):
    # git id
    git_id = os.path.join(path, "git_id.txt")
    os.system(f'git log --format="%H" -n 1 > {git_id}')

    # submodule git ids
    git_submodules = os.path.join(path, "git_submodules.txt")
    os.system(f"git submodule status > {git_submodules}")

    # get git diff
    git_diff = os.path.join(path, "git_diff.txt")
    os.system(f"git diff --submodule=diff > {git_diff}")


@main_app.command()
def start_server(
    agent_name: Annotated[str, typer.Argument(help="Agent name to run.")],
    kwargs: Annotated[str, typer.Option(help="args to start the agent.")] = "{}",
    port: Annotated[int, typer.Option(help="Port to run the server on.")] = 8080,
    host: Annotated[str, typer.Option(help="Host to run the server on.")] = "localhost",
):
    """Runs eval server."""
    agent = AGENTS[agent_name](**json.loads(kwargs))
    service = AgentService(agent, agent_name)
    with service:
        t = rpyc.ThreadedServer(
            service, port=port, hostname=host, protocol_config={"allow_pickle": True, "allow_public_attrs": True}
        )
        t.start()


def _per_process(
    args: tuple[int, dict, list[EvalConfig], str, int, str, int, Optional[int], int],
) -> tuple[np.ndarray, list[list[list[float]]], list[float], int]:
    step, kwargs, eval_cfgs, agent_name, port, host, episodes, n_processes, nth_gpu = args
    logging.info(f"Starting evaluation for step {step}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(nth_gpu)
    job_kwargs = copy.deepcopy(kwargs)
    job_kwargs["checkpoint_step"] = step
    per_env_results_last_reward, per_env_results_rewards = evaluation(
        agent_name, job_kwargs, eval_cfgs, port, host, episodes, n_processes
    )
    logging.info(f"Finished evaluation for step {step}")
    flatten_rewards = [[item for sublist in env_rewards for item in sublist] for env_rewards in per_env_results_rewards]
    mean_rewards = [np.mean(env_rewards) if env_rewards else 0.0 for env_rewards in flatten_rewards]
    logging.info("Returning results for step %s", step)
    return per_env_results_last_reward, per_env_results_rewards, mean_rewards, step


@main_app.command()
def run_eval_post_training(
    agent_name: Annotated[str, typer.Argument(help="Agent name to run.")],
    wandb_project: Annotated[str, typer.Option(help="weights and biases logging project.")],
    wandb_entity: Annotated[str, typer.Option(help="weights and biases logging entity.")],
    wandb_note: Annotated[str, typer.Option(help="weights and biases logging note.")],
    wandb_name: Annotated[str, typer.Option(help="weights and biases logging name.")],
    output_path: Annotated[str, typer.Option(help="Path to store the run results.")],
    wandb_group: Annotated[Optional[str], typer.Option(help="weights and biases logging name.")] = None,
    steps: Annotated[Optional[str], typer.Option(help="steps to evaluate.")] = None,
    kwargs: Annotated[str, typer.Option(help="args to start the agent.")] = "{}",
    port: Annotated[int, typer.Option(help="Port to run the server on.")] = 8080,
    host: Annotated[str, typer.Option(help="Host to run the server on.")] = "localhost",
    episodes: Annotated[int, typer.Option(help="Number of episodes to run.")] = 100,
    n_processes: Annotated[Optional[int], typer.Option(help="Number of processes to run.")] = None,
    n_gpus: Annotated[int, typer.Option(help="Number of gpus to run.")] = 1,
    eval_cfgs: Annotated[
        str, typer.Option(help="Evaluation configurations.")
    ] = '[{"env": "rcs/SimplePickUpSim-v0", "kwargs": {}}]',
):
    """
    post training eval which goes over all checkpoints
    - each checkpoint with many envs
    """
    if steps is None:
        steps = [None]
    else:
        steps = json.loads(steps)

    if wandb_group == "":
        wandb_group = None

    wandb.init(
        entity=wandb_entity,
        resume="allow",
        project=wandb_project,
        config=dict(agent_name=agent_name, agent_kwargs=json.loads(kwargs), eval_cfgs=json.loads(eval_cfgs)),
        notes=wandb_note,
        job_type="eval",
        name=wandb_name,
        group=wandb_group,
    )
    wandb_log_git_diff(output_path)
    wandb.run.log_code(".")

    wandb.define_metric(
        "total/success",
        step_metric="train_step",
        overwrite=False,
        step_sync=False,
        hidden=False,
        summary="max",
    )
    wandb.define_metric(
        "total/last_step_reward",
        step_metric="train_step",
        overwrite=False,
        step_sync=False,
        hidden=False,
        summary="max",
    )
    wandb.define_metric(
        "total/total_steps",
        step_metric="train_step",
        overwrite=False,
        step_sync=False,
        hidden=False,
        summary="min",
    )
    wandb.define_metric(
        "total/mean_reward",
        step_metric="train_step",
        overwrite=False,
        step_sync=False,
        hidden=False,
        summary="max",
    )
    eval_cfgs = [EvalConfig(**cfg) for cfg in json.loads(eval_cfgs)]
    kwargs = json.loads(kwargs)
    for idx, env in enumerate(eval_cfgs):
        wandb.define_metric(
            f"{env.env_id}/success",
            step_metric="train_step",
            overwrite=False,
            step_sync=False,
            hidden=False,
            summary="max",
        )
        wandb.define_metric(
            f"{env.env_id}/last_step_reward",
            step_metric="train_step",
            overwrite=False,
            step_sync=False,
            hidden=False,
            summary="max",
        )
        wandb.define_metric(
            f"{env.env_id}/total_steps",
            step_metric="train_step",
            overwrite=False,
            step_sync=False,
            hidden=False,
            summary="min",
        )
        wandb.define_metric(
            f"{env.env_id}/mean_reward",
            step_metric="train_step",
            overwrite=False,
            step_sync=False,
            hidden=False,
            summary="max",
        )

    # distribute gpus equally
    gpus_ids = [i % n_gpus for i in range(len(steps))]

    # spawn n processes and run in parallel
    with Pool(n_processes) as p:
        args = [
            (step, kwargs, eval_cfgs, agent_name, port + idx, host, episodes, 1, gpus_ids[idx])
            for idx, step in enumerate(steps)
        ]
        results = p.map(_per_process, args)
    logging.info("Finished evaluation")

    for result in results:
        per_env_results_last_reward, per_env_results_rewards, mean_rewards, step = result
        step = step if step is not None else 0
        wandb_log_dict = {
            "total/success": per_env_results_last_reward.mean(axis=(0, 1))[0],
            "total/last_step_reward": per_env_results_last_reward.mean(axis=(0, 1))[1],
            "total/total_steps": per_env_results_last_reward.mean(axis=(0, 1))[2],
            "total/mean_reward": np.mean(mean_rewards),
            "train_step": step,
        }
        # log for each env
        for idx, env in enumerate(eval_cfgs):
            wandb_log_dict.update(
                {
                    f"{env.env_id}/success": per_env_results_last_reward[idx].mean(axis=0)[0],
                    f"{env.env_id}/last_step_reward": per_env_results_last_reward[idx].mean(axis=0)[1],
                    f"{env.env_id}/total_steps": per_env_results_last_reward[idx].mean(axis=0)[2],
                    f"{env.env_id}/mean_reward": mean_rewards[idx],
                }
            )
        wandb.log(wandb_log_dict, step=step, commit=True)

        path = write_results(
            per_env_results_last_reward,
            per_env_results_rewards,
            eval_cfgs,
            model_cfg={"agent_name": agent_name, "kwargs": kwargs},
            out=output_path,
        )
        wandb.log_artifact(path, type="file", name="results", aliases=[f"step_{step}"])


@main_app.command()
def run_eval_during_training(
    agent_name: Annotated[str, typer.Argument(help="Agent name to run.")],
    wandb_id: Annotated[str, typer.Option(help="weights and biases logging id.")],
    wandb_group: Annotated[str, typer.Option(help="weights and biases logging group.")],
    wandb_project: Annotated[str, typer.Option(help="weights and biases logging project.")],
    wandb_entity: Annotated[str, typer.Option(help="weights and biases logging entity.")],
    wandb_note: Annotated[str, typer.Option(help="weights and biases logging note.")],
    wandb_name: Annotated[str, typer.Option(help="weights and biases logging name.")],
    output_path: Annotated[str, typer.Option(help="Path to store the run results.")],
    wandb_first: Annotated[bool, typer.Option(help="whether its the first eval.")] = False,
    kwargs: Annotated[str, typer.Option(help="args to start the agent.")] = "{}",
    port: Annotated[int, typer.Option(help="Port to run the server on.")] = 8080,
    host: Annotated[str, typer.Option(help="Host to run the server on.")] = "localhost",
    episodes: Annotated[int, typer.Option(help="Number of episodes to run.")] = 100,
    n_processes: Annotated[Optional[int], typer.Option(help="Number of processes to run.")] = None,
    eval_cfgs: Annotated[
        str, typer.Option(help="Evaluation configurations.")
    ] = '[{"env": "rcs/SimplePickUpSim-v0", "kwargs": {}}]',
    python_path: Annotated[str, typer.Option(help="Full path to the policy environment's python")] = "python",
):
    """
    during training eval, all need to use the same id
    - just for one model, but many envs
    - can be new run but at least in the same project and same group as the training
    """

    if wandb_first:
        wandb.init(
            id=wandb_id,
            entity=wandb_entity,
            resume="allow",
            group=wandb_group,
            project=wandb_project,
            config=dict(agent_name=agent_name, agent_kwargs=json.loads(kwargs), eval_cfgs=json.loads(eval_cfgs)),
            notes=wandb_note,
            job_type="eval",
            name=wandb_name,
        )

        wandb_log_git_diff(output_path)
        wandb.run.log_code(".")
    else:
        wandb.init(id=wandb_id, entity=wandb_entity, resume="must", project=wandb_project)

    eval_cfgs = [EvalConfig(**cfg) for cfg in json.loads(eval_cfgs)]
    kwargs = json.loads(kwargs)

    step = kwargs.get("checkpoint_step", 0)
    step = step if step is not None else 0
    # Genius TODO
    per_env_results_last_reward, per_env_results_rewards = evaluation(
        agent_name, kwargs, eval_cfgs, port, host, episodes, n_processes, python=python_path
    )

    # return is [envs, episodes, 3(success, reward, steps)], [envs, episodes, rewards for all steps in the episode]

    flatten_rewards = [[item for sublist in env_rewards for item in sublist] for env_rewards in per_env_results_rewards]
    mean_rewards = [np.mean(env_rewards) if env_rewards else 0.0 for env_rewards in flatten_rewards]

    # these new define metric to also not work with several jobs
    # wandb says that logging can only be done in run groups
    if wandb_first:
        wandb.define_metric(
            "total/success",
            step_metric="train_step",
            overwrite=False,
            step_sync=False,
            hidden=False,
            summary="max",
        )
        wandb.define_metric(
            "total/last_step_reward",
            step_metric="train_step",
            overwrite=False,
            step_sync=False,
            hidden=False,
            summary="max",
        )
        wandb.define_metric(
            "total/total_steps",
            step_metric="train_step",
            overwrite=False,
            step_sync=False,
            hidden=False,
            summary="min",
        )
        wandb.define_metric(
            "total/mean_reward",
            step_metric="train_step",
            overwrite=False,
            step_sync=False,
            hidden=False,
            summary="max",
        )
        for idx, env in enumerate(eval_cfgs):
            wandb.define_metric(
                f"{env.env_id}/success",
                step_metric="train_step",
                overwrite=False,
                step_sync=False,
                hidden=False,
                summary="max",
            )
            wandb.define_metric(
                f"{env.env_id}/last_step_reward",
                step_metric="train_step",
                overwrite=False,
                step_sync=False,
                hidden=False,
                summary="max",
            )
            wandb.define_metric(
                f"{env.env_id}/total_steps",
                step_metric="train_step",
                overwrite=False,
                step_sync=False,
                hidden=False,
                summary="min",
            )
            wandb.define_metric(
                f"{env.env_id}/mean_reward",
                step_metric="train_step",
                overwrite=False,
                step_sync=False,
                hidden=False,
                summary="max",
            )

    wandb_log_dict = {
        "total/success": per_env_results_last_reward.mean(axis=(0, 1))[0],
        "total/last_step_reward": per_env_results_last_reward.mean(axis=(0, 1))[1],
        "total/total_steps": per_env_results_last_reward.mean(axis=(0, 1))[2],
        "total/mean_reward": np.mean(mean_rewards),
        "train_step": step,
    }

    # log for each env
    for idx, env in enumerate(eval_cfgs):
        wandb_log_dict.update(
            {
                f"{env.env_id}/success": per_env_results_last_reward[idx].mean(axis=0)[0],
                f"{env.env_id}/last_step_reward": per_env_results_last_reward[idx].mean(axis=0)[1],
                f"{env.env_id}/total_steps": per_env_results_last_reward[idx].mean(axis=0)[2],
                f"{env.env_id}/mean_reward": mean_rewards[idx],
            }
        )
    wandb.log(wandb_log_dict, step=step, commit=True)
    path = write_results(
        per_env_results_last_reward,
        per_env_results_rewards,
        eval_cfgs,
        model_cfg={"agent_name": agent_name, "kwargs": kwargs},
        out=output_path,
    )
    wandb.log_artifact(path, type="file", name="results", aliases=[f"step_{step}"])


if __name__ == "__main__":
    main_app()
