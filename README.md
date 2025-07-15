# Agents
Agents is a python library that allows to septate next action prediction from policy networks from action execution in simulated or real environments.
It defines an interface for policies and for environments.
The policies run independent in their own virtual environment, potentially on a different computer, and can be queried for an action (in principle similar to the chatgpt api).

Why is this useful?
- Separation of dependencies by using two different python environments: Some times dependencies contradict e.g. pytorch and jax
- Some robot hardware requires a real time linux kernel which does not easily allow you to use an Nvidia GPU.
- Separate deployment and model code

This library is a byproduct of the [Refined Policy Distillation (RPD)](https://refined-policy-distillation.github.io/) paper which distilled VLAs into expert policies using Reinforcement Learning.
The work also includes a section on related engineering challenges regarding jax and pytorch.

## Installation

### Local Installation
```shell
git clone https://github.com/juelg/agents.git
cd agents
pip install -ve .
```

### Repo Installation
```shell
pip install git+https://github.com/juelg/agents.git
```

### Environment and Policy Installation
On top of agents you can then install a simulation environment where the agent acts.
We currently support [maniskill](https://github.com/haosulab/ManiSkill) with more to come.
In order to avoid dependency conflicts, use a second conda/pip environment to install your policy.
We currently support [octo](https://github.com/octo-models/octo) and [openvla](https://github.com/openvla/openvla).

### Octo
To use Octo as an agent/policy you need to create a new conda environment:
```shell
conda create -n octo python=3.10
conda activate octo
conda install nvidia/label/cuda-11.8.0::cuda --no-channel-priority
conda install conda-forge::cudnn=8.9
# for gpu support:
pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Verify that the jax installation was successful and that jax finds your gpu.
Open a python terminal in the same conda env and type
```python
from jax.lib import xla_bridge
# this should output gpu if the installation was successful
print(xla_bridge.get_backend().platform)
```

Install Octo and dependencies:
```shell
pip install -r agents/utils/fixed_octo_requirements.txt
pip install git+https://github.com/octo-models/octo.git
```

Install the agents library on top:
```shell
pip install git+https://github.com/octo-models/octo.git
```

For more details, see the [Octo github page](https://github.com/octo-models/octo).

### OpenVLA
To use OpenVLA, create a new conda environment:
```shell
conda create -n openvla python=3.10 -y
conda activate openvla
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

Install [flash attention](https://github.com/Dao-AILab/flash-attention):
```shell
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
# if you run into issues try `pip cache remove flash_attn` first
```

Install OpenVLA
```shell
pip install git+https://github.com/openvla/openvla.git
```

For more details, see the [OpenVLA github page](https://github.com/openvla/openvla).

## Usage
To start an agents server use the `start-server` command where `kwargs` is a dictionary of the constructor arguments of the policy you want to start e.g.
```shell
python -m agents start-server octo --host localhost --port 8080 --kwargs '{"checkpoint_path": "hf://Juelg/octo-base-1.5-finetuned-maniskill", "checkpoint_step": 60000, "horizon": 1, "unnorm_key": []}'
```

There is also the `run-eval-during-training` command to evaluate a model during training, so a single checkpoint.
The `run-eval-post-training` command evaluates a range of checkpoints in parallel.
In both cases environment and arguments as well as policy and arguments and wandb config for logging can be passed as CLI arguments.


## Contribution

### New Policy
In order to extend the library with a new policy network, extend the `Agent` class in [policies.py](src/agents/policies.py).
It is important to only invoke policy specific imports in the class functions, as each policy can have its own dependencies.


### New Environment
In order to extend the library with a new agent environment, extend the `EvaluatorEnv` class in [evaluator_envs.py](src/agents/evaluator_envs.py).


### Developer Tools
Install the following dev dependencies:
```shell
pip install -ve '.[dev]'
```

The following dev tools are provided:
```shell
# format the code
make format

# lint the code
make lint

# run tests
make test
```

## Citation
If you find the agent useful for your work, please consider citing the original work behind it:
```
@inproceedings{juelg2025refinedpolicydistillationvla,
    title={{Refined Policy Distillation}: {F}rom {VLA} Generalists to {RL} Experts}, 
    author={Tobias Jülg and Wolfram Burgard and Florian Walter},
    year={2025},
    booktitle={Proc.~of the IEEE/RSJ Int.~Conf.~on Intelligent Robots and Systems (IROS)},
    note={Accepted for publication.}
}
```