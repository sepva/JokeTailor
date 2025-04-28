import statistics
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datasets import load_dataset
from scipy.stats import entropy


def compute_entropy(rankings_dict):
    """
    Computes the entropy for each joke's rankings.
    """
    entropies = {}
    for joke_id, rankings in rankings_dict.items():
        counts = np.bincount(rankings, minlength=5)[
            1:
        ]  # Count occurrences of each rank (1-4)
        prob_dist = counts / counts.sum()  # Convert to probability distribution
        entropies[joke_id] = entropy(prob_dist, base=2)  # Compute entropy (base 2)
    return entropies


def compute_krippendorff_alpha(rankings_dict):
    """
    Computes Krippendorff's Alpha for ordinal rankings, handling missing values.
    """
    joke_ids = list(rankings_dict.keys())
    max_len = max(len(rankings) for rankings in rankings_dict.values())
    ranking_matrix = np.full((len(joke_ids), max_len), np.nan)

    for i, rankings in enumerate(rankings_dict.values()):
        ranking_matrix[i, : len(rankings)] = rankings

    valid_entries = ~np.isnan(ranking_matrix)

    # Compute observed disagreement
    D_o = 0
    count = 0
    for i in range(len(joke_ids)):
        valid_ranks = ranking_matrix[i, valid_entries[i]]
        for j in range(len(valid_ranks)):
            for k in range(j + 1, len(valid_ranks)):
                D_o += (valid_ranks[j] - valid_ranks[k]) ** 2
                count += 1
    D_o /= count if count > 0 else 1

    # Compute expected disagreement
    D_e = 0
    global_mean = np.nanmean(ranking_matrix)
    for i in range(len(joke_ids)):
        valid_ranks = ranking_matrix[i, valid_entries[i]]
        for j in range(len(valid_ranks)):
            D_e += (valid_ranks[j] - global_mean) ** 2
    D_e /= np.sum(valid_entries)

    alpha = 1 - (D_o / D_e) if D_e != 0 else 1
    return alpha


def compute_quadratic_weighted_entropy(rankings_dict):
    """
    Computes the quadratic weighted entropy for each joke's rankings.
    """
    weighted_entropies = {}
    for joke_id, rankings in rankings_dict.items():
        counts = np.bincount(rankings, minlength=5)[
            1:
        ]  # Count occurrences of each rank (1-4)
        prob_dist = counts / counts.sum()  # Convert to probability distribution
        mean_rank = np.sum(np.arange(1, 5) * prob_dist)  # Weighted mean of rankings
        weights = (np.arange(1, 5) - mean_rank) ** 2  # Quadratic weights
        weighted_entropy = -np.sum(
            weights * prob_dist * np.log2(prob_dist, where=(prob_dist > 0))
        )
        weighted_entropies[joke_id] = weighted_entropy
    return weighted_entropies


def plot_histogram(dict, title, xlabel, ylabel, bins=20):
    """
    Plots a histogram.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(dict.values(), bins=bins, color="skyblue", edgecolor="black")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.box(False)
    # plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def compute_disagreement(rankings_dict):
    """
    Computes the disagreement for each joke based on mode agreement.
    """
    disagreements = {}
    for joke_id, rankings in rankings_dict.items():
        ranking_counts = Counter(rankings)
        max_count = max(ranking_counts.values())
        modes = [rank for rank, count in ranking_counts.items() if count == max_count]
        agreement = sum(max_count / len(rankings) for mode in modes) / len(modes)
        disagreements[joke_id] = 1 - agreement  # Disagreement is 1 - agreement
    return disagreements


def get_rankings(dataset_id):
    ds = load_dataset(dataset_id, split="train")

    rankings_for_joke = {}

    for entry in ds:
        for ranking in range(7):
            for i, jokeId in enumerate(entry[f"ranking{ranking}"]):
                if not jokeId in rankings_for_joke:
                    rankings_for_joke[jokeId] = [i + 1]

                rankings_for_joke[jokeId].append(i + 1)
    return rankings_for_joke


rankings_for_joke = get_rankings("SeppeV/survey_results_final")
disagreements = compute_disagreement(rankings_for_joke)
print("Average disagreement:", statistics.mean(disagreements.values()))
plot_histogram(
    disagreements,
    "Disagreement Distribution",
    "Disagreement",
    "Frequency",
    bins=13,
)
print("Krippendorff alpha: ", compute_krippendorff_alpha(rankings_for_joke))
# entropies = compute_entropy(rankings_for_joke)
# plot_histogram(entropies, "Entropy Distribution", "Entropy", "Frequency", bins=20)
# plot_histogram(
#     compute_quadratic_weighted_entropy(rankings_for_joke),
#     "Quadratic Weighted Entropy Distribution",
#     "Quadratic Weighted Entropy",
#     "Frequency",
#     bins=20,
# )
