import csv
import json
import math
import os
import random

import numpy as np

from pymarlzooplus.mentalizing.paper_style_rware_mentalizer import (
    PaperStyleRwareMentalizer
)

def calculate_entropy(distribution):
    """ 
    Calculates entropy for the inferred agent type distribution.

    Higher entropy means the mentalizer is less certain about the hidden type.
    """
    
    entropy = 0.0

    for probability in distribution.values():
        if probability > 0.0:
            entropy -= probability * math.log(probability)

    # Normalise entropy so values are easier to compare across tests.
    normalised_entropy = entropy / math.log(len(distribution))
    return entropy, normalised_entropy

def probability_margin(distribution):
    """ 
    Calculates the difference between the highest and second highest probability.

    A smaller margin means the mentalizer is less confident in its top prediction.
    """
    
    probabilities = sorted(
        distribution.values(),
        reverse=True
    )

    return probabilities[0] - probabilities[1]

def run_balanced_low_reward_trial(trial_id, output_root):
    """ 
    Runs one trial using a balanced mix of delivery, collision avoiding

    and random/untrained behaviour with only weak positive reward evidence.
    """
    
    # Set seeds so each trial can be reproduced.
    random.seed(3000 + trial_id)
    np.random.seed(3000 + trial_id)

    # Create output folder for this trial.
    output_dir = os.path.join(
        output_root,
        "balanced_low_reward_trial_" + str(trial_id)
    )

    # Initialise the mentalizer used to infer hidden agent type.
    mentalizer = PaperStyleRwareMentalizer(
        output_dir=output_dir,
        n_evidence=50,
        n_samples=500,
        episode_time_limit=50
    )

    # Create balanced episode containing delivery-like,
    # collision-like and random/untrained behaviour.
    behaviour_modes = (
        ["delivery"] * 16
        + ["collision"] * 17
        + ["random"] * 17
    )

    # Shuffle behaviour order so evidence is mixed across the episode.
    random.shuffle(behaviour_modes)

    # Action pattern used to represent delivery-focused behaviour.
    delivery_pattern = [
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 1],
        [1, 3],
        [2, 4]
    ]

    # Action pattern used to represent collision-avoiding or cautious behaviour.
    collision_pattern = [
        [0, 0],
        [0, 1],
        [1, 0],
        [0, 0]
    ]

    # Available RWARE-style discrete actions.
    possible_actions = [0, 1, 2, 3, 4]

    delivery_index = 0
    collision_index = 0
    delivery_reward_index = 0
    previous_actions = None

    total_return = 0.0
    positive_reward_steps = 0

    # Track the number of timesteps assigned to each behaviour mode.
    mode_counts = {
        "delivery": 0,
        "collision": 0,
        "random": 0
    }

    # Run the 50 timestep synthetic episode.
    for timestep, behaviour_mode in enumerate(behaviour_modes):
        mode_counts[behaviour_mode] += 1
        reward = 0.0

        if behaviour_mode == "delivery":
            # Use the next delivery-style action pattern.
            actions = delivery_pattern[
                delivery_index % len(delivery_pattern)
            ]

            delivery_index += 1

            # Include three weak positive rewards across the episode.
            # this retains more delivery evidence without making
            # the episode strongly delivery focused.
            if delivery_reward_index in [0, 6, 12]:
                reward = 0.1
                positive_reward_steps += 1

            delivery_reward_index += 1

        elif behaviour_mode == "collision":
            # Use the next collision-avoiding action pattern.
            actions = collision_pattern[
                collision_index % len(collision_pattern)
            ]

            collision_index += 1

        else:
            # Random behaviour sometimes repeats previous action,
            # making evidence less clearly structured.
            if previous_actions is not None and random.random() < 0.35:
                actions = list(previous_actions)
            else:
                actions = [
                    random.choice(possible_actions),
                    random.choice(possible_actions)
                ]

        previous_actions = list(actions)
        total_return += reward

        # Dummy observations used because test focuses on action
        # and reward patterns rather than real environment observations.
        observations = [
            [0.0 for _ in range(20)],
            [0.0 for _ in range(20)]
        ]

        # Send the synthetic transition to the mentalizer.
        mentalizer.observe_transition(
            timestep=timestep,
            observations=observations,
            actions=actions,
            reward=reward,
            terminated=False,
            env_info={}
        )

    # Finish the episode and get the inference summary.
    summary = mentalizer.finish_episode(
        episode_return=total_return,
        episode_length=50
    )

    distribution = summary["agent_type_distribution"]
    observed_summary = summary["observed_summary"]

    entropy, normalised_entropy = calculate_entropy(
        distribution
    )

    # Store trial-level metrics for later CSV and summary analysis.
    result = {
        "trial_id": trial_id,
        "episode_return": total_return,
        "positive_reward_steps": positive_reward_steps,
        "delivery_mode_steps": mode_counts["delivery"],
        "collision_mode_steps": mode_counts["collision"],
        "random_mode_steps": mode_counts["random"],
        "delivery_focused_probability": distribution[
            "delivery_focused"
        ],
        "collision_avoiding_probability": distribution[
            "collision_avoiding"
        ],
        "random_or_untrained_probability": distribution[
            "random_or_untrained"
        ],
        "most_likely_type": summary["most_likely_type"],
        "max_probability": max(distribution.values()),
        "probability_margin": probability_margin(
            distribution
        ),
        "entropy": entropy,
        "normalised_entropy": normalised_entropy,
        "observed_idle_ratio": observed_summary[
            "idle_ratio"
        ],
        "observed_repeat_action_ratio": observed_summary[
            "repeat_action_ratio"
        ],
        "observed_action_entropy": observed_summary[
            "action_entropy_norm"
        ],
        "theta0_mean": summary[
            "posterior_theta_mean"
        ]["theta0"],
        "theta1_mean": summary[
            "posterior_theta_mean"
        ]["theta1"],
        "theta2_mean": summary[
            "posterior_theta_mean"
        ]["theta2"],
        "best_distance": summary["best_distance"]
    }

    return result


def main():
    # Create the output folder for all test results.
    output_root = "uncertainty_results_balanced_low_reward"
    os.makedirs(output_root, exist_ok=True)

    results = []

    # Run 50 independent trials.
    for trial_id in range(1, 51):
        result = run_balanced_low_reward_trial(
            trial_id=trial_id,
            output_root=output_root
        )

        results.append(result)

    # Save full trial results to CSV.
    csv_path = os.path.join(
        output_root,
        "balanced_low_reward_results.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(results[0].keys())
        )

        writer.writeheader()
        writer.writerows(results)

    # Count how often each agent type is selected as most likely.
    type_counts = {}

    for result in results:
        result_type = result["most_likely_type"]
        type_counts[result_type] = type_counts.get(result_type, 0) + 1

    # Create aggregate summary across all 50 trials.
    aggregate_summary = {
        "number_of_trials": len(results),
        "type_counts": type_counts,
        "mean_delivery_probability": float(
            np.mean([
                result["delivery_focused_probability"]
                for result in results
            ])
        ),
        "mean_collision_probability": float(
            np.mean([
                result["collision_avoiding_probability"]
                for result in results
            ])
        ),
        "mean_random_probability": float(
            np.mean([
                result["random_or_untrained_probability"]
                for result in results
            ])
        ),
        "mean_max_probability": float(
            np.mean([
                result["max_probability"]
                for result in results
            ])
        ),
        "mean_probability_margin": float(
            np.mean([
                result["probability_margin"]
                for result in results
            ])
        ),
        "mean_normalised_entropy": float(
            np.mean([
                result["normalised_entropy"]
                for result in results
            ])
        ),
        "mean_idle_ratio": float(
            np.mean([
                result["observed_idle_ratio"]
                for result in results
            ])
        ),
        "mean_repeat_action_ratio": float(
            np.mean([
                result["observed_repeat_action_ratio"]
                for result in results
            ])
        ),
        "mean_action_entropy": float(
            np.mean([
                result["observed_action_entropy"]
                for result in results
            ])
        ),

        # Record behaviour design so test setup is clear in output.
        "behaviour_design": {
            "delivery_steps_per_trial": 16,
            "collision_steps_per_trial": 17,
            "random_steps_per_trial": 17,
            "positive_delivery_steps_per_trial": 3,
            "reward_per_positive_step": 0.1
        },
        "interpretation": (
            "Mixed behaviour with weak positive reward evidence is expected "
            "to produce greater uncertainty in the inferred agent type distribution."
        )
    }

    # Save aggregate summary as JSON.
    json_path = os.path.join(
        output_root,
        "balanced_low_reward_summary.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as output_file:
        json.dump(
            aggregate_summary,
            output_file,
            indent=4
        )

    # Print result locations and summary.
    print("Saved trial results to:", csv_path)
    print("Saved aggregate summary to:", json_path)
    print(json.dumps(aggregate_summary, indent=4))


if __name__ == "__main__":
    main()