import csv
import json
import math
import os
import random

import numpy as np

from pymarlzooplus.mentalizing.paper_style_rware_mentalizer import (
    PaperStyleRwareMentalizer )

def calculate_entropy(distribution):
    """ 
    Calculates the entropy of the probability distribution.

    Higher entropy means the mentalizer is more uncertain.
    """
    
    values = list(distribution.values())
    entropy = 0.0

    for value in values:
        if value > 0.0:
            entropy -= value * math.log(value)

    # Normalise entropy so it's easier to compare across tests.
    normalised_entropy = entropy / math.log(len(values))
    return entropy, normalised_entropy

def probability_margin(distribution):
    """ 
    Calculates the gap between the most likely and second most likely type.
    
    A smaller margin suggests more uncertainty.
    """
    
    sorted_values = sorted(distribution.values(), reverse=True)
    return sorted_values[0] - sorted_values[1]

def max_probability(distribution):
    """ 
    Returns the highest probability assigned to any agent type. 
    """

    return max(distribution.values())

def run_ambiguous_trial(trial_id, output_root):
    """ 
    Runs one ambiguous behaviour trial.
    
    The actions are deliberately mixed so mentalizer receives unclear evidence. 
    """
    
    # Set seeds so each trial is reproducible.
    random.seed(1000 + trial_id)
    np.random.seed(1000 + trial_id)

    # Create separate output folder for this trial.
    output_dir = os.path.join(
        output_root,
        "ambiguous_trial_" + str(trial_id)
    )

    # Initialise the mentalizer used to infer hidden agent type.
    mentalizer = PaperStyleRwareMentalizer(
        output_dir=output_dir,
        n_evidence=50,
        n_samples=500,
        episode_time_limit=50
    )

    total_return = 0.0
    idle_action_count = 0
    repeated_action_count = 0
    previous_actions = None

    # RWARE actions represented as integer values.
    possible_actions = [0, 1, 2, 3, 4]

    # Run a fixed length episode of 50 timesteps.
    for timestep in range(50):
        behaviour_choice = random.random()

        # Sometimes both agents perform the idle action.
        if behaviour_choice < 0.35:
            actions = [0, 0]

        # Sometimes agents repeat their previous actions.
        elif behaviour_choice < 0.65 and previous_actions is not None:
            actions = list(previous_actions)

        # Otherwise, select actions randomly.
        else:
            actions = [
                random.choice(possible_actions),
                random.choice(possible_actions)
            ]

        # Count repeated actions as an indicator of possible ambiguous behaviour.
        if previous_actions is not None and actions == previous_actions:
            repeated_action_count += 1

        previous_actions = list(actions)

        # Count how often idle actions occur.
        idle_action_count += actions.count(0)

        # Reward kept at zero to avoid giving clear delivery evidence.
        reward = 0.0
        total_return += reward

        # Dummy observations are used because test focuses on action patterns.
        observations = [
            [0.0 for _ in range(20)],
            [0.0 for _ in range(20)]
        ]

        # Pass transition to the mentalizer.
        mentalizer.observe_transition(
            timestep=timestep,
            observations=observations,
            actions=actions,
            reward=reward,
            terminated=False,
            env_info={}
        )

    # Finish the episode and collect inference summary.
    summary = mentalizer.finish_episode(
        episode_return=total_return,
        episode_length=50
    )

    distribution = summary["agent_type_distribution"]
    entropy, normalised_entropy = calculate_entropy(distribution)
    margin = probability_margin(distribution)

    # Store key outputs from the trial.
    result = {
        "trial_id": trial_id,
        "episode_return": total_return,
        "idle_action_count": idle_action_count,
        "repeated_action_count": repeated_action_count,
        "delivery_focused_probability": distribution["delivery_focused"],
        "collision_avoiding_probability": distribution["collision_avoiding"],
        "random_or_untrained_probability": distribution["random_or_untrained"],
        "most_likely_type": summary["most_likely_type"],
        "max_probability": max_probability(distribution),
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
    # Create the output folder for all test results
    output_root = "uncertainty_results_ambiguous"
    os.makedirs(output_root, exist_ok=True)

    results = []

    # Run 50 independent ambiguous trials.
    for trial_id in range(1, 51):
        result = run_ambiguous_trial(
            trial_id=trial_id,
            output_root=output_root
        )

        results.append(result)

    # Save individual trial results to CSV.
    csv_path = os.path.join(
        output_root,
        "ambiguous_uncertainty_results.csv"
    )

    fieldnames = list(results[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    # Count how often each agent type is selected as most likely.
    type_counts = {}

    for result in results:
        result_type = result["most_likely_type"]

        if result_type not in type_counts:
            type_counts[result_type] = 0

        type_counts[result_type] += 1

    # Calculate mean probabilities across all trials.
    mean_delivery_probability = np.mean(
        [
            result["delivery_focused_probability"]
            for result in results
        ]
    )

    mean_collision_probability = np.mean(
        [
            result["collision_avoiding_probability"]
            for result in results
        ]
    )

    mean_random_probability = np.mean(
        [
            result["random_or_untrained_probability"]
            for result in results
        ]
    )

    mean_max_probability = np.mean(
        [
            result["max_probability"]
            for result in results
        ]
    )

    mean_margin = np.mean(
        [
            result["probability_margin"]
            for result in results
        ]
    )

    mean_normalised_entropy = np.mean(
        [
            result["normalised_entropy"]
            for result in results
        ]
    )

    # Create the aggregate summary for the ambiguous uncertainty test.
    aggregate_summary = {
        "number_of_trials": len(results),
        "type_counts": type_counts,
        "mean_delivery_probability": float(mean_delivery_probability),
        "mean_collision_probability": float(mean_collision_probability),
        "mean_random_probability": float(mean_random_probability),
        "mean_max_probability": float(mean_max_probability),
        "mean_probability_margin": float(mean_margin),
        "mean_normalised_entropy": float(mean_normalised_entropy),
        "interpretation": (
            "Ambiguous behaviour is expected to produce a lower maximum probability, "
            "a smaller probability margin and higher entropy than the clear "
            "delivery focused confidence test."
        )
    }

    # Compare ambiguous with the clear delivery focused test if available.
    comparison_path = os.path.join(
        "uncertainty_results_delivery_focused",
        "delivery_focused_confidence_summary.json"
    )

    if os.path.exists(comparison_path):
        with open(comparison_path, "r", encoding="utf-8") as input_file:
            delivery_summary = json.load(input_file)

        aggregate_summary["comparison_with_delivery_test"] = {
            "delivery_test_mean_delivery_probability": delivery_summary.get(
                "mean_delivery_probability"
            ),
            "delivery_test_mean_probability_margin": delivery_summary.get(
                "mean_probability_margin"
            ),
            "delivery_test_mean_normalised_entropy": delivery_summary.get(
                "mean_normalised_entropy"
            ),
            "ambiguous_test_mean_max_probability": float(mean_max_probability),
            "ambiguous_test_mean_probability_margin": float(mean_margin),
            "ambiguous_test_mean_normalised_entropy": float(
                mean_normalised_entropy
            )
        }

    # Save the aggregate summary as JSON file.
    json_path = os.path.join(
        output_root,
        "ambiguous_uncertainty_summary.json"
    )

    with open(json_path, "w", encoding="utf-8") as output_file:
        json.dump(
            aggregate_summary,
            output_file,
            indent=4
        )

    # Print result locations and summary
    print("Saved trial results to:", csv_path)
    print("Saved aggregate summary to:", json_path)
    print(json.dumps(aggregate_summary, indent=4))


if __name__ == "__main__":
    main()