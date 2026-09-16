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
    
    Lower entropy means the mentalizer is more confident.
    """

    values = list(distribution.values())
    entropy = 0.0

    for value in values:
        if value > 0.0:
            entropy -= value * math.log(value)

    # Normalise entropy so it can be compared across different tests.
    normalised_entropy = entropy / math.log(len(values))
    return entropy, normalised_entropy

def probability_margin(distribution):
    """ 
    Calculates the difference between the highest and second highest probability.
    
    A larger margin means the mentalizer is more confident in its top prediction.
    """

    sorted_values = sorted(distribution.values(), reverse=True)
    return sorted_values[0] - sorted_values[1]

def run_delivery_focused_trial(trial_id, output_root):
    """ 
    Runs one delivery focused trial.

    This test gives the mentalizer clear delivery style behaviour and rewards.
    """

    # Set random seeds so each trial can be reproduced.
    random.seed(trial_id)
    np.random.seed(trial_id)

    # Create a separate output folder for trial.
    output_dir = os.path.join(
        output_root,
        "delivery_focused_trial_" + str(trial_id)
    )

    # Initialise the mentalizer used to infer hidden agent type.
    mentalizer = PaperStyleRwareMentalizer(
        output_dir=output_dir,
        n_evidence=50,
        n_samples=500,
        episode_time_limit=50
    )

    total_return = 0.0
    positive_reward_steps = 0
    idle_action_count = 0
    repeated_action_count = 0
    previous_actions = None

    # Repeated non-idle action pattern used to represent purposeful behaviour.
    action_pattern = [
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 1],
        [1, 3]
    ]

    # Run a fixed length synthetic episode of 50 timesteps.
    for timestep in range(50):
        # Cycle through the delivery focused action pattern.
        actions = action_pattern[timestep % len(action_pattern)]

        # Track repeated actions for behavioural summary metrics.
        if previous_actions is not None and actions == previous_actions:
            repeated_action_count += 1

        previous_actions = list(actions)

        # Count idle actions, which should stay low in delivery focused test.
        idle_action_count += actions.count(0)

        # Give a positive reward every 5 timesteps to create clear delivery evidence.
        if timestep % 5 == 0:
            reward = 1.0
            positive_reward_steps += 1
        else:
            reward = 0.0

        total_return += reward

        # Dummy observations used because test focuses on actions and rewards.
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

    # Finish the episode and collect the inference summary.
    summary = mentalizer.finish_episode(
        episode_return=total_return,
        episode_length=50
    )

    distribution = summary["agent_type_distribution"]
    entropy, normalised_entropy = calculate_entropy(distribution)
    margin = probability_margin(distribution)

    # Store trial level metrics for later analysis.
    result = {
        "trial_id": trial_id,
        "episode_return": total_return,
        "positive_reward_steps": positive_reward_steps,
        "idle_action_count": idle_action_count,
        "repeated_action_count": repeated_action_count,
        "delivery_focused_probability": distribution["delivery_focused"],
        "collision_avoiding_probability": distribution["collision_avoiding"],
        "random_or_untrained_probability": distribution["random_or_untrained"],
        "most_likely_type": summary["most_likely_type"],
        "probability_margin": margin,
        "entropy": entropy,
        "normalised_entropy": normalised_entropy,
        "theta0_mean": summary["posterior_theta_mean"]["theta0"],
        "theta1_mean": summary["posterior_theta_mean"]["theta1"],
        "theta2_mean": summary["posterior_theta_mean"]["theta2"],
        "best_distance": summary["best_distance"]
    }

    return result


def main():
    # create the output folder for the confidence test.
    output_root = "uncertainty_results_delivery_focused"
    os.makedirs(output_root, exist_ok=True)

    results = []

    # Run 50 independent delivery focused trials.
    for trial_id in range(1, 51):
        result = run_delivery_focused_trial(
            trial_id=trial_id,
            output_root=output_root
        )

        results.append(result)

    # Save trial level results to CSV.
    csv_path = os.path.join(
        output_root,
        "delivery_focused_confidence_results.csv"
    )

    fieldnames = list(results[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    # Calculate the average delivery focused probability across all trials.
    mean_delivery_probability = np.mean(
        [
            result["delivery_focused_probability"]
            for result in results
        ]
    )

    # Calculate the average confidence margin across all trials.
    mean_margin = np.mean(
        [
            result["probability_margin"]
            for result in results
        ]
    )

    # Calculate the average uncertainty across all trials.
    mean_normalised_entropy = np.mean(
        [
            result["normalised_entropy"]
            for result in results
        ]
    )

    # Count how often mentalizer classified trial as delivery focused.
    delivery_classification_count = sum(
        1 for result in results
        if result["most_likely_type"] == "delivery_focused"
    )

    # Store main aggregate results for confidence test.
    aggregate_summary = {
        "number_of_trials": len(results),
        "delivery_classification_count": delivery_classification_count,
        "mean_delivery_probability": float(mean_delivery_probability),
        "mean_probability_margin": float(mean_margin),
        "mean_normalised_entropy": float(mean_normalised_entropy),
        "interpretation": (
            "Clear delivery focused behaviour and reward evidence are expected "
            "to produce a high delivery focused probability, a larger probability "
            "margin and lower entropy across trials."
        )
    }

    # Save aggregate summary as JSON.
    json_path = os.path.join(
        output_root,
        "delivery_focused_confidence_summary.json"
    )

    with open(json_path, "w", encoding="utf-8") as output_file:
        json.dump(
            aggregate_summary,
            output_file,
            indent=4
        )

    # Print saved file paths and summary for quick checking.
    print("Saved trial results to:", csv_path)
    print("Saved aggregate summary to:", json_path)
    print(json.dumps(aggregate_summary, indent=4))


if __name__ == "__main__":
    main()