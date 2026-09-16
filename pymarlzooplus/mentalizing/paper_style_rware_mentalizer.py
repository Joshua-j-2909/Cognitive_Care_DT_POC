import json
import math
import os
from collections import Counter

import numpy as np


class PaperStyleRwareMentalizer:
    """
    Mentalizing module for RWARE and PyMARLzoo+, inspired by
    'Human Mentalizing as Rational Probabilistic Inference'.

    The module observes IQL behaviour, creates behavioural summaries,
    compares them with simulated behaviour and estimates the hidden
    theta parameters that best explain the observed trajectory.

    The inference process:
    1. Defines hidden theta parameters.
    2. Generates simulated behaviour from theta.
    3. Summarises observed and simulated behaviour.
    4. Compares the summaries using a distance function.
    5. Builds an approximate posterior from the closest matches.
    """

    def __init__(
        self,
        output_dir="mentalizing_results_paper_style",
        n_evidence=200,
        n_samples=1000,
        episode_time_limit=50,
        seed=None
    ):
        # Folder where JSON output files will be saved.
        self.output_dir = os.path.abspath(output_dir)

        # Number of possible theta settings tested per episode.
        self.n_evidence = n_evidence

        # Number of posterior samples drawn from the best theta settings.
        self.n_samples = n_samples

        # store the RWARE episode length for later use.
        self.episode_time_limit = episode_time_limit

        # Random number generator for reproducible sampling.
        self.random_generator = np.random.default_rng(seed)

        # Stores transitions from the current episode only.
        self.trajectory = []

        # Stores posterior outputs across episodes.
        self.posterior_history = []

        # Stores readable summaries across episodes.
        self.episode_summaries = []

        # Live mental state used by the agents during the episode.
        # Starts neutral before any behaviour has been observed.
        self.live_mental_state_vector = [
            0.0,
            0.0,
            0.0,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0
        ]

        # Hidden parameter bounds.
        self.bounds = {
            "theta0": (-2.0, 0.0),     # Movement or collision cost
            "theta1": (0.0, 2.0),      # Delivery goal strength.
            "theta2": (0.0, 2.0)       # Randomness or weak policy control
        }

        os.makedirs(self.output_dir, exist_ok=True)

    def observe_transition(
        self,
        timestep,
        observations,
        actions,
        reward,
        terminated,
        env_info=None
    ):
        """
        Store one transition from the real PyMARLzoo+ RWARE episode.
        """

        transition = {
            "timestep": self._to_python(timestep),
            "observations": self._to_python(observations),
            "actions": self._to_python(actions),
            "reward": self._to_python(reward),
            "terminated": self._to_python(terminated),
            "env_info": self._to_python(env_info or {})
        }

        self.trajectory.append(transition)

        self.update_live_mental_state()

    def finish_episode(self, episode_return=None, episode_length=None):
        """
        Run inference at the end of one episode.
        """

        if len(self.trajectory) == 0:
            return None

        # Build summaries from the real IQL trajectory.
        observed_summary = self.build_observed_summary(self.trajectory)

        # Infer which theta values best explain the summary.
        posterior_result = self.fit_posterior(observed_summary)

        episode_summary = {
            "episode_return": self._to_python(episode_return),
            "episode_length": self._to_python(episode_length),
            "observed_summary": observed_summary,
            "posterior_theta_mean": posterior_result["theta_mean"],
            "posterior_theta_std": posterior_result["theta_std"],
            "agent_type_distribution": posterior_result["agent_type_distribution"],
            "most_likely_type": posterior_result["most_likely_type"],
            "best_distance": posterior_result["best_distance"]
        }

        self.episode_summaries.append(episode_summary)
        self.posterior_history.append(posterior_result)

        self.save_outputs()

        # Clear the current trajectory before the next episode.
        self.trajectory = []

        return episode_summary

    def fit_posterior(self, observed_summary):
        """
        Approximate posterior inference.

        The original mentalizing approach uses ELFI and BOLFI. This implementation
        uses lightweight sampling so that inference can operate within PyMARLzoo+
        without substantially increasing training time.
        """

        evidence = []

        # Sample possible theta values and generate simulated summaries.
        for _ in range(self.n_evidence):
            theta = self.sample_prior()
            simulated_summary = self.stimulus_one_generator(theta)
            distance = self.euclidean_distance(observed_summary, simulated_summary)

            evidence.append({
                "theta": theta,
                "simulated_summary": simulated_summary,
                "distance": distance
            })

        # Sort theta settings from best to worst match.
        evidence = sorted(evidence, key=lambda item: item["distance"])

        # Retain the best quarter of sampled theta settings.
        best_evidence_count = max(20, int(self.n_evidence * 0.25))
        best_evidence = evidence[:best_evidence_count]

        distances = np.array([item["distance"] for item in best_evidence])

        # Convert distances into weights.
        # Smaller distance = higher weight.
        shifted_distances = distances - np.min(distances)
        scale = np.std(shifted_distances) + 1e-8

        weights = np.exp(-shifted_distances / scale)
        weight_total = np.sum(weights)

        if weight_total == 0 or np.isnan(weight_total):
            weights = np.ones(len(best_evidence)) / len(best_evidence)
        else:
            weights = weights / weight_total

        # Draw posterior samples from best theta explanations.
        selected_indices = self.random_generator.choice(
            len(best_evidence),
            size=self.n_samples,
            replace=True,
            p=weights
        )

        theta_samples = [
            best_evidence[index]["theta"]
            for index in selected_indices
        ]

        theta_array = np.array([
            [
                sample["theta0"],
                sample["theta1"],
                sample["theta2"]
            ]
            for sample in theta_samples
        ])

        theta_mean = {
            "theta0": float(np.mean(theta_array[:, 0])),
            "theta1": float(np.mean(theta_array[:, 1])),
            "theta2": float(np.mean(theta_array[:, 2]))
        }

        theta_std = {
            "theta0": float(np.std(theta_array[:, 0])),
            "theta1": float(np.std(theta_array[:, 1])),
            "theta2": float(np.std(theta_array[:, 2]))
        }

        # Convert theta samples into readable behavioural type probabilities.
        agent_type_distribution = self.classify_theta_samples(theta_samples)

        most_likely_type = max(
            agent_type_distribution,
            key=agent_type_distribution.get
        )

        return {
            "theta_samples": theta_samples,
            "theta_mean": theta_mean,
            "theta_std": theta_std,
            "agent_type_distribution": agent_type_distribution,
            "most_likely_type": most_likely_type,
            "best_distance": float(best_evidence[0]["distance"]),
            "best_simulated_summary": best_evidence[0]["simulated_summary"]
        }

    def sample_prior(self):
        """
        Sample hidden theta values from simple uniform priors.
        """

        theta0_low, theta0_high = self.bounds["theta0"]
        theta1_low, theta1_high = self.bounds["theta1"]
        theta2_low, theta2_high = self.bounds["theta2"]

        return {
            "theta0": float(self.random_generator.uniform(theta0_low, theta0_high)),
            "theta1": float(self.random_generator.uniform(theta1_low, theta1_high)),
            "theta2": float(self.random_generator.uniform(theta2_low, theta2_high))
        }

    def stimulus_one_generator(self, theta):
        """
        RWARE adaptation of the stimulus generator.

        The original approach trains an agent under hidden theta values and
        then observes its trajectory. Here, this process is approximated by
        generating a behavioural summary directly from theta.
        """

        theta0 = theta["theta0"]
        theta1 = theta["theta1"]
        theta2 = theta["theta2"]

        # Convert theta values into behavioural tendencies.
        collision_caution = self.sigmoid(-theta0)
        delivery_drive = self.sigmoid(2.0 * theta1 - 1.0)
        randomness = self.sigmoid(2.0 * theta2 - 2.0)

        # Estimate useful warehouse behaviour from delivery drive and randomness.
        efficiency = delivery_drive * (1.0 - randomness)

        simulated_summary = {
            "return_norm": self.clip01(efficiency * (1.0 - 0.25 * collision_caution)),
            "positive_reward_norm": self.clip01(delivery_drive * (1.0 - randomness)),
            "idle_ratio": self.clip01(0.05 + 0.35 * collision_caution + 0.15 * randomness),
            "repeat_action_ratio": self.clip01(0.10 + 0.70 * randomness),
            "action_entropy_norm": self.clip01(0.20 + 0.70 * randomness),
            "termination": self.clip01(efficiency)
        }

        return simulated_summary

    def build_observed_summary(self, trajectory):
        """
        Convert a real RWARE trajectory into summary statistics.

        These summaries play the same role as the paper's white area counts,
        grey area counts and goal reaching statistics.
        """

        rewards = [self._reward_to_float(item["reward"]) for item in trajectory]
        actions = [self._flatten_action(item["actions"]) for item in trajectory]

        total_reward = sum(rewards)
        positive_reward_count = sum(1 for reward in rewards if reward > 0)

        # Count action 0 as idle or no operation.
        idle_count = 0
        total_action_count = 0

        for joint_action in actions:
            for action in joint_action:
                total_action_count += 1

                if action == 0:
                    idle_count += 1

        if total_action_count == 0:
            idle_ratio = 0.0
        else:
            idle_ratio = idle_count / total_action_count

        # Count how often the joint action repeats from one step to the next.
        repeated_count = 0

        for index in range(1, len(actions)):
            if actions[index] == actions[index - 1]:
                repeated_count += 1

        if len(actions) <= 1:
            repeat_action_ratio = 0.0
        else:
            repeat_action_ratio = repeated_count / (len(actions) - 1)

        # Entropy measures how varied the chosen actions are.
        action_entropy_norm = self.normalised_action_entropy(actions)

        # Record whether the episode reached a terminal state.
        terminated = 1.0 if any(item["terminated"] for item in trajectory) else 0.0

        return {
            "return_norm": self.clip01(total_reward / 10.0),
            "positive_reward_norm": self.clip01(positive_reward_count / 5.0),
            "idle_ratio": self.clip01(idle_ratio),
            "repeat_action_ratio": self.clip01(repeat_action_ratio),
            "action_entropy_norm": self.clip01(action_entropy_norm),
            "termination": terminated
        }

    def euclidean_distance(self, observed_summary, simulated_summary):
        """
        Compare observed and simulated summaries using Euclidean distance.
        """

        keys = [
            "return_norm",
            "positive_reward_norm",
            "idle_ratio",
            "repeat_action_ratio",
            "action_entropy_norm",
            "termination"
        ]

        total = 0.0

        for key in keys:
            observed_value = observed_summary[key]
            simulated_value = simulated_summary[key]
            total += (observed_value - simulated_value) ** 2

        return math.sqrt(total)

    def classify_theta_samples(self, theta_samples):
        """
        Convert posterior theta samples into behavioural type probabilities.
        """

        counts = Counter()

        for sample in theta_samples:
            collision_score = -sample["theta0"]
            delivery_score = sample["theta1"]
            random_score = sample["theta2"]

            scores = {
                "collision_avoiding": collision_score,
                "delivery_focused": delivery_score,
                "random_or_untrained": random_score
            }

            inferred_type = max(scores, key=scores.get)
            counts[inferred_type] += 1

        total = sum(counts.values())

        if total == 0:
            return {
                "delivery_focused": 1.0 / 3.0,
                "collision_avoiding": 1.0 / 3.0,
                "random_or_untrained": 1.0 / 3.0
            }

        return {
            "delivery_focused": counts["delivery_focused"] / total,
            "collision_avoiding": counts["collision_avoiding"] / total,
            "random_or_untrained": counts["random_or_untrained"] / total
        }

    def update_live_mental_state(self):
        """
        Update the live mental state vector during an episode.

        Observed behaviour is converted into a six value mental state vector
        that can be provided to each agent at the next timestep.
        """

        if len(self.trajectory) == 0:
            return

        observed_summary = self.build_observed_summary(self.trajectory)

        return_norm = self.clip01(
            observed_summary.get("return_norm", 0.0)
        )

        positive_reward_norm = self.clip01(
            observed_summary.get("positive_reward_norm", 0.0)
        )

        idle_ratio = self.clip01(
            observed_summary.get("idle_ratio", 0.0)
        )

        repeat_action_ratio = self.clip01(
            observed_summary.get("repeat_action_ratio", 0.0)
        )

        action_entropy_norm = self.clip01(
            observed_summary.get("action_entropy_norm", 0.0)
        )

        # Estimate behavioural type scores from the observed summary.
        delivery_score = (
            0.50 * positive_reward_norm
            + 0.25 * return_norm
            + 0.15 * action_entropy_norm
            + 0.10 * (1.0 - idle_ratio)
        )

        collision_avoiding_score = (
            0.40 * idle_ratio
            + 0.30 * (1.0 - positive_reward_norm)
            + 0.20 * (1.0 - repeat_action_ratio)
            + 0.10 * (1.0 - return_norm)
        )

        random_or_untrained_score = (
            0.40 * action_entropy_norm
            + 0.30 * repeat_action_ratio
            + 0.20 * (1.0 - positive_reward_norm)
            + 0.10 * (1.0 - return_norm)
        )

        total_score = (
            delivery_score
            + collision_avoiding_score
            + random_or_untrained_score
        )

        if total_score <= 0.0:
            delivery_prob = 1.0 / 3.0
            collision_prob = 1.0 / 3.0
            random_prob = 1.0 / 3.0
        else:
            delivery_prob = delivery_score / total_score
            collision_prob = collision_avoiding_score / total_score
            random_prob = random_or_untrained_score / total_score

        # Convert behavioural probabilities back into theta values.
        theta0 = -2.0 * collision_prob
        theta1 = 2.0 * delivery_prob
        theta2 = 2.0 * random_prob

        self.live_mental_state_vector = [
            float(theta0),
            float(theta1),
            float(theta2),
            float(delivery_prob),
            float(collision_prob),
            float(random_prob)
        ]


    def get_mental_state_vector(self):
        """
        Return the current live mental state vector.

        This vector is used by the agents as part of their input when
        use_mental_state_input is enabled.
        """

        return list(self.live_mental_state_vector)

    def save_outputs(self):
        """
        Save the current inference outputs as JSON files.
        """

        trajectory_path = os.path.join(
            self.output_dir,
            "paper_style_trajectory.json"
        )    

        posterior_path = os.path.join(
           self.output_dir,
            "paper_style_posterior.json"
        )

        summary_path = os.path.join(
            self.output_dir,
            "paper_style_summary.json"
        )

        with open(trajectory_path, "w", encoding="utf-8") as output_file:
            json.dump(
                self._to_python(self.trajectory),
                output_file,
                indent=4
            )

        with open(posterior_path, "w", encoding="utf-8") as output_file:
            json.dump(
                self._to_python(self.posterior_history),
                output_file,
                indent=4
            )

        with open(summary_path, "w", encoding="utf-8") as output_file:
            json.dump(
                self._to_python(self.episode_summaries[-1]),
                output_file,
                indent=4
            )

    def normalised_action_entropy(self, actions):
        """
        Measure how varied the agents' actions are.
        """

        flat_actions = []

        for joint_action in actions:
            for action in joint_action:
                flat_actions.append(action)

        if len(flat_actions) == 0:
            return 0.0

        counts = Counter(flat_actions)
        total = sum(counts.values())

        entropy = 0.0

        for count in counts.values():
            probability = count / total
            entropy -= probability * math.log(probability + 1e-8)

        max_entropy = math.log(max(len(counts), 2))

        return entropy / max_entropy

    def sigmoid(self, value):
        """
        Convert any number into the range 0 to 1.
        """

        return 1.0 / (1.0 + math.exp(-value))

    def clip01(self, value):
        """
        Keep a value inside the range 0 to 1.
        """

        return max(0.0, min(1.0, float(value)))

    def _reward_to_float(self, reward):
        """
        Convert reward values into one float.
        """

        if isinstance(reward, (int, float)):
            return float(reward)

        if isinstance(reward, np.ndarray):
            return float(np.sum(reward))

        if isinstance(reward, list):
            return float(sum(self._reward_to_float(item) for item in reward))

        if isinstance(reward, tuple):
            return float(sum(self._reward_to_float(item) for item in reward))

        return 0.0

    def _flatten_action(self, actions):
        """
        Convert action formats into a flat list of integers.
        """

        if isinstance(actions, np.ndarray):
            actions = actions.tolist()

        if isinstance(actions, np.integer):
            return [int(actions)]

        if isinstance(actions, int):
            return [actions]

        if isinstance(actions, float):
            return [int(actions)]

        if isinstance(actions, tuple):
            actions = list(actions)

        if isinstance(actions, list):
            flat_actions = []

            for item in actions:
                if isinstance(item, list):
                    flat_actions.extend(self._flatten_action(item))
                elif isinstance(item, tuple):
                    flat_actions.extend(self._flatten_action(list(item)))
                else:
                    flat_actions.append(int(item))

            return flat_actions

        return [0]

    def _to_python(self, value):
        """
        Convert numpy values into JSON friendly Python values.
        """

        # Convert PyTorch tensors to standard Python values.
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            return value.detach().cpu().tolist()

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, np.floating):
            return float(value)

        if isinstance(value, np.bool_):
            return bool(value)

        if isinstance(value, dict):
            return {
                key: self._to_python(item)
                for key, item in value.items()
            }

        if isinstance(value, tuple):
            return [
                self._to_python(item)
                for item in value
            ]

        if isinstance(value, list):
            return [
                self._to_python(item)
                for item in value
            ]

        return value