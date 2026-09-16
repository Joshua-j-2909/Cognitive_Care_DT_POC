# Cognitive_Care_Digital_Twin_Proof_of_Concept

## Submission Notes

This folder was renamed to Cognitive_Care_Digital_Twin_Proof_of_Concept for submission. Earlier development versions used different folder names. The internal package name remains pymarlzooplus because the project was developed by extending the PyMARLZoo+ framework.

The submitted project keeps parts of the original PyMARLZoo+, EPyMARL and RWARE frameworks. During development, removing files that appeared unnecessary caused dependency errors and prevented the project from running correctly. The wider framework structure has therefore been kept to keep the proof of concept runnable and reproducible.

This README explains how the project works, how to set it up and run it, how to reproduce the experiments reported in the dissertation and which parts of the original framework were modified.


## Project Purpose

This project implements a proof of concept for uncertainty aware interpretation within a future Cognitive Care Digital Twin decision support framework.

The implementation uses a modified PyMARLZoo+ and RWARE multi agent reinforcement learning setup. RWARE is used as an abstract multi agent environment because it includes multiple interacting agents, partial observability, sequential actions, sparse rewards and coordination requirements.

The main addition to the existing codebase is a Bayesian mentalizing module. The module interprets observable agent behaviour and estimates a probability distribution across three simplified hidden behaviour states:
1. delivery_focused
2. collision_avoiding
3. random_or_untrained
These states are abstract behavioural categories created for the proof of concept. They are used to test whether the system can represent different possible explanations for observed behaviour and show uncertainty between them.


## Relationship to the Original Code

The base of the project is PyMARLZoo+, which provides the multi agent reinforcement learning framework used for training, environment integration, logging, controllers and learners. The project also retains components and design patterns inherited from PyMARL and EPyMARL, with support for environments such as RWARE.

This dissertation does not replace the original frameworks. Selected parts of PyMARLZoo+ were extended to add Bayesian mentalizing, uncertainty aware interpretation, optional mental state inputs and reward shaping for the RWARE experiments.


## How the Code Works

The project works as follows:
* pymarlzooplus/main.py loads the selected algorithm and environment configuration.
* The IQL configuration is used to train the independent learning agents.
* The gymma environment configuration loads the RWARE environment through the environment wrapper used by PyMARLZoo+.
* pymarlzooplus/runners/episode_runner.py runs each episode, collects observations, selects actions and receives rewards from the environment.
* When mentalizing is enabled, the runner records behavioural evidence during the episode and passes it to the mentalizing module.
* pymarlzooplus/mentalizing/paper_style_rware_mentalizer.py uses this evidence to perform approximate posterior inference over the three hidden behaviour states.
* The validation scripts use the resulting probability distribution to calculate uncertainty measures such as maximum probability, probability margin and normalised entropy.
* When use_mental_state_input=True, a six value mental state vector is added to the episode batch and passed to the controller as an additional agent input.
* When reward shaping is enabled, a small additional reward is applied when an agent newly picks up a requested shelf.
* Training logs and experiment outputs are saved for later evaluation.


## Main Files Changed or Added

### Added File

pymarlzooplus/mentalizing/paper_style_rware_mentalizer.py

This file contains the Bayesian mentalizing module developed for the project. It stores observed transition evidence, converts trajectories into behavioural summaries, performs approximate posterior inference over the three hidden behaviour states and maintains a six value mental state vector that can optionally be provided to the learning agents.

### Added Validation Scripts

pymarlzooplus/uncertainty_tests/mental_state_confidence_test.py

pymarlzooplus/uncertainty_tests/ambiguous_state_uncertainty_test.py

pymarlzooplus/uncertainty_tests/balanced_ambiguous_uncertainty_test.py

pymarlzooplus/uncertainty_tests/balanced_ambiguous_low_reward_test.py

These scripts run the controlled validation experiments used to test the mentalizing module. They generate predefined behavioural evidence, pass it into the mentalizer and use the resulting probability distributions to calculate confidence and uncertainty measures.

### Modified Configuration File

pymarlzooplus/config/default.yaml

The default configuration was extended with the following options:

paper_mentalizing_enabled: False
paper_mentalizing_output_dir: "mentalizing_results_paper_style"
paper_mentalizing_n_evidence: 50
paper_mentalizing_n_samples: 100
mental_state_scale: 1.0
use_mental_state_input: False
mental_state_dim: 6
reward_shaping_enabled: False
correct_pickup_reward: 0.1
reward_shaping_debug: False

### Modified Runner

pymarlzooplus/runners/episode_runner.py

The episode runner was modified to:
* Initialise the mentalizing module when enabled
* Collect behavioural evidence during each episode
* Pass transition evidence into the mentalizing module
* Retrieve the current mental state vector
* Add the mental state vector to the transition data when enabled
* Apply a shaped reward when an agent newly picks up a requested shelf

### Modified Run Setup

pymarlzooplus/run.py

The training data scheme was modified so that a mental_state field can be added to the episode batch when use_mental_state_input=True.

### Modified Controller

pymarlzooplus/controllers/basic_controller.py

The controller was modified so that the mental state vector can be appended to the agent input when use_mental_state_input=True.

### Bayesian Mentalizing Module

The mentalizing module was written specifically for this project. Its design was informed by probabilistic mentalizing approaches, including the work of Zhinin-Vera et al. (2025). However, the inference procedure was adapted for use within the RWARE and PyMARLZoo+ setup.

The module observes agent behaviour and converts the recorded trajectory into a small set of summary statistics. These include:
* Episode return
* Frequency of positive rewards
* Idle behaviour
* Repeated actions
* Action entropy
* Whether the episode terminated

Candidate hidden parameter values are then sampled and used to generate simulated behavioural summaries. The observed and simulated summaries are compared using Euclidean distance. The closest matching candidates are used to form an approximate posterior distribution.

The inference produces:
* Posterior probabilities for delivery_focused, collision_avoiding and random_or_untrained
* A most likely hidden behaviour state
* Posterior summaries of the hidden theta parameters
* A best matching distance between observed and simulated behaviour

The validation scripts use the inferred probability distribution to calculate additional confidence and uncertainty measures, including maximum probability, probability margin and normalised entropy.

During an episode, the module also maintains a six value live mental state vector derived from the behaviour observed so far. When use_mental_state_input=True, this vector can be provided to the learning agents as an additional input.

### Reward Shaping Addition

A small reward shaping mechanism was added for Experiment 5 to provide additional learning feedback for correct shelf pickups. When reward shaping is enabled, the runner checks whether an agent has newly picked up a shelf that is currently requested by the environment. If this occurs, an additional shaped reward is added to the environment reward.

The reward value used in the experiments was:

correct_pickup_reward = 0.1

The reward shaping behaviour is controlled by the following configuration options:

reward_shaping_enabled
correct_pickup_reward
reward_shaping_debug

Setting reward_shaping_enabled=True activates the additional pickup reward. When it is disabled, the original environment reward is used without the additional pickup reward.



## Setup Instructions

Run the project from the root folder:

cd Cognitive_Care_Digital_Twin_Proof_of_Concept

The setup below was tested on Windows using Conda and Python 3.8.18.

### Create the Conda Environment

Create and activate a clean Python 3.8.18 environment:

conda create -y -n ccdt_clean python=3.8.18
conda activate ccdt_clean

Check that the correct Python installation is active:

python --version
where.exe python

The Python version should be:

Python 3.8.18

### Update the Python Installation Tools

python -m pip install --upgrade pip setuptools wheel

### Register the Local PyMARLZoo+ Project

Install the local project in editable mode without automatically installing all dependencies:

python -m pip install -e . --no-deps

The --no-deps option is used because the original PyMARLZoo+ requirements include dependencies for several optional environments that are not required for this dissertation.

The proof of concept uses the RWARE environment, so only the packages required by the submitted implementation are installed.

### Install the Required Dependencies

python -m pip install PyYAML sacred six tensorboard-logger pandas matplotlib "numpy<2.0.0" torch scikit-learn "pyglet==1.5.27" tqdm rich wandb seaborn networkx "rware>1.0.3" gymnasium "pettingzoo==1.24.3" opencv-python-headless scipy pygame dill

PettingZoo is installed without the [all] extra because the additional PettingZoo environments are not required for the RWARE experiments used in this project.

### Check the Installation

Check that the main dependencies can be imported:

python -c "import yaml, sacred, gymnasium, rware, torch, numpy, sklearn, seaborn, rich; print('Core imports OK')"

A successful installation should print:

Core imports OK

Then check that the local PyMARLZoo+ package can be imported:

python -c "import pymarlzooplus; print('PyMARLZoo+ import OK'); print(pymarlzooplus.__file__)"

### Check the RWARE Environment

Before running PyMARLZoo+, RWARE can be tested independently:

python -c "import gymnasium as gym, rware; env=gym.make('rware-tiny-2ag-hard-v2'); obs, info=env.reset(seed=42); print('RWARE reset OK'); print('Agents:', len(obs)); env.close()"

The expected output is:

RWARE reset OK
Agents: 2

### Run a Short PyMARLZoo+ Test

A short IQL training run can be used to confirm that the complete PyMARLZoo+ and RWARE pipeline is working:

python pymarlzooplus/main.py --config=iql --env-config=gymma --capture=no with t_max=1000 env_args.time_limit=50 env_args.key="rware:rware-tiny-2ag-hard-v2"

The --capture=no option disables Sacred console capture. This is recommended on Windows because Sacred console capture can cause the training output to be hidden even when the experiment itself completes successfully.

A successful run should include output similar to:

Running command 'my_main'
Beginning training for 1000 timesteps
Finished Training
Creating plots ...
Exiting Main
Completed after ...

Each experiment is stored in a numbered run folder containing files such as:

config.json
cout.txt
info.json
metrics.json
run.json
plots/

### Check the Bayesian Mentalizing Module

The first controlled Bayesian mentalizing experiment can be used as a final setup check:

python pymarlzooplus/uncertainty_tests/mental_state_confidence_test.py

This runs 50 controlled delivery focused trials and saves the individual trial results and aggregate summary.

A successful run creates:

uncertainty_results_delivery_focused/delivery_focused_confidence_results.csv
uncertainty_results_delivery_focused/delivery_focused_confidence_summary.json


## Base RWARE Command

Once the setup has been checked, the standard RWARE IQL configuration can be run using:

python pymarlzooplus/main.py --config=iql --env-config=gymma --capture=no with env_args.time_limit=500 env_args.key="rware:rware-tiny-2ag-hard-v2"

This uses:

algorithm = iql
environment config = gymma
environment key = rware:rware-tiny-2ag-hard-v2
episode time limit = 500

Training is headless by default, so a graphical warehouse window is not expected during a normal training run.


## Reproducing Experiments 1 to 4

Experiments 1 to 4 are controlled validation tests for the mentalizing module. They do not train the RWARE agents. Instead, each script generates predefined behavioural evidence and passes it directly into the mentalizer so that the resulting probability distributions and uncertainty measures can be examined.

Run the experiments from the project root with the ccdt_clean Conda environment activated:

python pymarlzooplus/uncertainty_tests/mental_state_confidence_test.py

python pymarlzooplus/uncertainty_tests/ambiguous_state_uncertainty_test.py

python pymarlzooplus/uncertainty_tests/balanced_ambiguous_uncertainty_test.py

python pymarlzooplus/uncertainty_tests/balanced_ambiguous_low_reward_test.py

Each script runs 50 trials and saves its trial level and aggregate outputs to a separate results folder.

## Reproducing Experiment 5

Experiment 5 compares four RWARE IQL conditions to examine the effects of pickup reward shaping and the additional mental state input. Each condition uses the same environment, training length and random seed so that the configuration changes can be compared directly.

Only one seed was used for each condition, so the Experiment 5 results are treated as descriptive comparisons rather than evidence of statistically significant performance differences.

The shared settings are:

algorithm = iql
environment configuration = gymma
env_args.time_limit = 500
env_args.key = rware:rware-tiny-2ag-hard-v2
env_args.seed = 2024
seed = 2024
t_max = 50000
correct_pickup_reward = 0.1
mental_state_dim = 6
mental_state_scale = 1.0
paper_mentalizing_n_evidence = 50
paper_mentalizing_n_samples = 100

The commands should be run from the project root folder with the Conda environment activated.

The --capture=no option is used to disable the Sacred console capture. This allows the training output to display correctly on Windows and also allows the console output to be saved using Tee-Object.

### A. Baseline, no shaping and no mentalizing

This condition uses IQL without pickup reward shaping or mental state input.

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outdir = "raw_results\experiment_5_A_baseline_no_shaping_no_mentalizing_$timestamp"
New-Item -ItemType Directory -Force -Path $outdir | Out-Null


python pymarlzooplus/main.py --config=iql --env-config=gymma --capture=no with `
env_args.time_limit=500 `
env_args.key="rware:rware-tiny-2ag-hard-v2" `
env_args.seed=2024 `
seed=2024 `
t_max=50000 `
reward_shaping_enabled=False `
correct_pickup_reward=0.1 `
reward_shaping_debug=True `
paper_mentalizing_enabled=False `
use_mental_state_input=False `
mental_state_dim=6 `
mental_state_scale=1.0 `
2>&1 | Tee-Object -FilePath "$outdir\console_output.txt"

### B. Pickup shaping only

This condition enables the additional reward for correctly picking up a requested shelf but does not provide mental state information to the agents.

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outdir = "raw_results\experiment_5_B_pickup_shaping_only_$timestamp"
New-Item -ItemType Directory -Force -Path $outdir | Out-Null


python pymarlzooplus/main.py --config=iql --env-config=gymma --capture=no with `
env_args.time_limit=500 `
env_args.key="rware:rware-tiny-2ag-hard-v2" `
env_args.seed=2024 `
seed=2024 `
t_max=50000 `
reward_shaping_enabled=True `
correct_pickup_reward=0.1 `
reward_shaping_debug=True `
paper_mentalizing_enabled=False `
use_mental_state_input=False `
mental_state_dim=6 `
mental_state_scale=1.0 `
2>&1 | Tee-Object -FilePath "$outdir\console_output.txt"

### C. Pickup shaping plus mentalizing

This condition combines pickup reward shaping with the six value mental state input produced by the mentalizing module.

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outdir = "raw_results\experiment_5_C_pickup_shaping_plus_mentalizing_$timestamp"
New-Item -ItemType Directory -Force -Path $outdir | Out-Null


python pymarlzooplus/main.py --config=iql --env-config=gymma --capture=no with `
env_args.time_limit=500 `
env_args.key="rware:rware-tiny-2ag-hard-v2" `
env_args.seed=2024 `
seed=2024 `
t_max=50000 `
reward_shaping_enabled=True `
correct_pickup_reward=0.1 `
reward_shaping_debug=True `
paper_mentalizing_enabled=True `
paper_mentalizing_n_evidence=50 `
paper_mentalizing_n_samples=100 `
paper_mentalizing_output_dir="$outdir\mentalizing_outputs" `
use_mental_state_input=True `
mental_state_dim=6 `
mental_state_scale=1.0 `
2>&1 | Tee-Object -FilePath "$outdir\console_output.txt"

### D. Mentalizing only

This condition provides the mental state input to the agents without adding the pickup shaped reward.

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outdir = "raw_results\experiment_5_D_mentalizing_only_$timestamp"
New-Item -ItemType Directory -Force -Path $outdir | Out-Null


python pymarlzooplus/main.py --config=iql --env-config=gymma --capture=no with `
env_args.time_limit=500 `
env_args.key="rware:rware-tiny-2ag-hard-v2" `
env_args.seed=2024 `
seed=2024 `
t_max=50000 `
reward_shaping_enabled=False `
correct_pickup_reward=0.1 `
reward_shaping_debug=True `
paper_mentalizing_enabled=True `
paper_mentalizing_n_evidence=50 `
paper_mentalizing_n_samples=100 `
paper_mentalizing_output_dir="$outdir\mentalizing_outputs" `
use_mental_state_input=True `
mental_state_dim=6 `
mental_state_scale=1.0 `
2>&1 | Tee-Object -FilePath "$outdir\console_output.txt"

Each condition creates a timestamped folder inside raw_results. This prevents console logs from different runs from overwriting one another.

The console output for each condition is saved as:

console_output.txt

For Conditions C and D, the outputs produced by the mentalizing module are also stored inside the corresponding experiment folder under:

mentalizing_outputs

PyMARLZoo+ also records Sacred run information separately under:

pymarlzooplus/results/sacred/iql/rware_rware-tiny-2ag-hard-v2/

The numbered Sacred folders contain the configuration, metrics and run information generated during training.

## Notes on Interpreting the Code

This repository should be understood as a technical proof of concept. It tests whether Bayesian inference can be used to represent uncertainty when interpreting behaviour in a simplified multi agent simulation.

The three hidden behaviour states used in the RWARE experiments are:
* delivery_focused
* collision_avoiding
* random_or_untrained
These are simplified behavioural categories created for testing the inference mechanism. In a future Cognitive Care Digital Twin, a similar approach could be used to represent richer care related explanations.


## References and Attribution

### Original Framework and Environment

G. Papadopoulos, A.  Kontogiannis, F. Papadopoulou, C. Poulianou, I. Koumentis, and G. Vouros, "An Extended Benchmarking of Multi-Agent Reinforcement Learning Algorithms in Complex Fully Cooperative Tasks," in Proc. 24th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2025), International Foundation for Autonomous Agents and Multiagent Systems (IFAAMAS), 2025.

### Bayesian Mentalizing Inspiration

L. Zhinin-Vera, E. Pretel, Victor Lopez-Jaquero, E. Navarro, and Pascual Gonzalez, "Mindful Human Digital Twins: Integrating Theory of Mind With Multi-Agent Reinforcement Learning," Applied Soft Computing, vol. 174, Art. no. 112939, Apr. 2025, doi: 10.1016/j.asoc.2025.112939. 
