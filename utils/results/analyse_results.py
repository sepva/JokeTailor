import json
import os
import secrets
from itertools import combinations

import choix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from scipy.stats import wilcoxon

load_dotenv()

RANKING_NAMES = [
    "FullGen",
    "FullMinusBon",
    "OnlyRAG",
    "OnlyFT",
    "OnlySFT",
    "RH",
    "BaseModel",
    "RandomUID",
]


def get_rankings():
    wins_matrix = [
        [0 for _ in range(len(RANKING_NAMES))] for _ in range(len(RANKING_NAMES))
    ]

    totals_matrix = [
        [0 for _ in range(len(RANKING_NAMES))] for _ in range(len(RANKING_NAMES))
    ]

    collection = get_collection()
    for survey_result in collection.find():
        results = json.loads(survey_result["resultJson"])
        for key, ranking in results.items():
            if "ranking" in key:
                for i in range(len(ranking) - 1):
                    higher_ranking_name = ranking[i].split(f"_")[0]
                    higher_index = RANKING_NAMES.index(higher_ranking_name)
                    for j in range(i + 1, len(ranking)):
                        lower_ranking_name = ranking[j].split(f"_")[0]
                        lower_index = RANKING_NAMES.index(lower_ranking_name)
                        wins_matrix[higher_index][lower_index] += 1

                        totals_matrix[min(lower_index, higher_index)][
                            max(lower_index, higher_index)
                        ] += 1

    result_matrix = [
        [0 for _ in range(len(RANKING_NAMES))] for _ in range(len(RANKING_NAMES))
    ]

    for i in range(len(RANKING_NAMES)):
        for j in range(len(RANKING_NAMES)):
            if wins_matrix[i][j] > 0:
                result_matrix[i][j] = (
                    wins_matrix[i][j] / totals_matrix[min(i, j)][max(i, j)]
                )

    df = pd.DataFrame(
        result_matrix, index=pd.Index(RANKING_NAMES), columns=pd.Index(RANKING_NAMES)
    )
    return df


def get_rank_differences():
    """Extracts pairwise ranking differences from the survey responses."""

    # Dictionary to store ranking differences for each technique pair
    ranking_differences = {pair: [] for pair in combinations(RANKING_NAMES, 2)}

    collection = get_collection()

    for survey_result in collection.find():
        results = json.loads(survey_result["resultJson"])

        for key, ranking in results.items():
            if "ranking" in key:
                # Store ranks of techniques in the current survey page
                rank_positions = {
                    ranking[i].split("_")[0]: i for i in range(len(ranking))
                }

                # Compute differences for each pair of techniques
                for tech1, tech2 in ranking_differences.keys():
                    if tech1 in rank_positions and tech2 in rank_positions:
                        diff = (
                            rank_positions[tech2] - rank_positions[tech1]
                        )  # Higher-ranked is positive
                        ranking_differences[(tech1, tech2)].append(diff)

    return ranking_differences


def compute_wilcoxon_tests():
    """Performs Wilcoxon Signed-Rank Test for each technique pair."""

    results = []
    ranking_differences = get_rank_differences()

    for (tech1, tech2), differences in ranking_differences.items():
        if len(differences) > 0:
            # Perform Wilcoxon test if there are at least two rankings
            stat, p_value = wilcoxon(
                differences, alternative="greater"
            )  # Tests if tech1 > tech2

            results.append(
                {
                    "Technique A": tech1,
                    "Technique B": tech2,
                    "Wilcoxon Statistic": stat,
                    "p-value": p_value,
                }
            )

    return pd.DataFrame(results)


def plot_win_matrix_heatmap(wins_matrix):
    """Plot a heatmap of the wins matrix."""
    # Convert the wins matrix to a numpy array
    wins_matrix = np.array(wins_matrix)

    # Set up the figure size and the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        wins_matrix,
        annot=True,
        cmap="Blues",
        xticklabels=RANKING_NAMES,
        yticklabels=RANKING_NAMES,
    )

    # Add titles and labels
    plt.title("Wins Matrix Heatmap")
    plt.xlabel("Technique (Winner)")
    plt.ylabel("Technique (Loser)")
    plt.xticks(rotation=45)

    # Display the plot
    plt.show()


def estimate_strengths():
    """Estimates technique strengths using the Bradley-Terry model and returns a Pandas DataFrame."""
    pairwise_data = []
    collection = get_collection()

    # Generate pairwise comparisons from survey data
    for survey_result in collection.find():
        results = json.loads(survey_result["resultJson"])
        for key, ranking in results.items():
            if "ranking" in key:
                ranked_techniques = [item.split("_")[0] for item in ranking]
                for higher_ranked, lower_ranked in combinations(ranked_techniques, 2):
                    higher_index = RANKING_NAMES.index(higher_ranked)
                    lower_index = RANKING_NAMES.index(lower_ranked)
                    pairwise_data.append((higher_index, lower_index))

    # Estimate strengths using Bradley-Terry model
    strengths = choix.lsr_pairwise(len(RANKING_NAMES), pairwise_data)

    # Create a DataFrame with ranked techniques
    ranked_df = pd.DataFrame(
        sorted(zip(RANKING_NAMES, strengths), key=lambda x: x[1], reverse=True),
        columns=["Technique", "Strength"],
    )

    return ranked_df


def estimate_strengths_with_confidence_interval(
    num_iterations=1000, percentage_of_data=0.8
):
    """Bootstrap resampling to estimate variability of strength values."""
    collection = get_collection()
    pairwise_data = []

    # Generate pairwise comparisons from survey data
    for survey_result in collection.find():
        results = json.loads(survey_result["resultJson"])
        for key, ranking in results.items():
            if "ranking" in key:
                ranked_techniques = [item.split("_")[0] for item in ranking]
                for i, higher_ranked in enumerate(ranked_techniques):
                    for lower_ranked in ranked_techniques[i + 1 :]:
                        higher_index = RANKING_NAMES.index(higher_ranked)
                        lower_index = RANKING_NAMES.index(lower_ranked)
                        pairwise_data.append((higher_index, lower_index))

    # Bootstrap procedure
    bootstrap_strengths = []
    pairwise_array = np.array(pairwise_data)
    all_strengths = {
        technique: [] for technique in RANKING_NAMES
    }  # To store strengths for violin plot

    for _ in range(num_iterations):
        resampled_data = np.random.choice(
            len(pairwise_data),
            size=int(len(pairwise_data) * percentage_of_data),
            replace=True,
        )
        resampled_pairwise_data = pairwise_array[resampled_data]

        # Compute strength values on the resampled data
        bootstrap_strength = choix.lsr_pairwise(
            len(RANKING_NAMES), resampled_pairwise_data, alpha=1e-6
        )
        bootstrap_strengths.append(bootstrap_strength)

        # Store the strengths for each technique
        for i, technique in enumerate(RANKING_NAMES):
            all_strengths[technique].append(bootstrap_strength[i])

    # Convert to a DataFrame for the confidence intervals and plotting
    bootstrap_strengths = np.array(bootstrap_strengths)
    bootstrap_df = pd.DataFrame(bootstrap_strengths, columns=RANKING_NAMES)

    # Calculate confidence intervals (e.g., 95% CI)
    ci_lower = bootstrap_df.quantile(0.025)
    ci_upper = bootstrap_df.quantile(0.975)

    # Mean strengths and CI
    mean_strengths = bootstrap_df.mean()

    # Add all_strengths to result_df
    result_df = pd.DataFrame(
        {
            "Technique": RANKING_NAMES,
            "Mean Strength": mean_strengths,
            "Lower 95% CI": ci_lower,
            "Upper 95% CI": ci_upper,
            "All Strengths": [all_strengths[technique] for technique in RANKING_NAMES],
        }
    ).sort_values("Mean Strength", ascending=False)

    return result_df


def plot_strengths(result_df):
    """Plot the mean strengths as points with confidence intervals as error bars."""

    result_df = result_df.sort_values("Mean Strength")

    # Plot
    plt.figure(figsize=(10, 6))

    # Error bar plot with mean strength as point and confidence intervals as the error bars
    plt.errorbar(
        result_df["Mean Strength"],
        result_df["Technique"],
        xerr=[
            result_df["Mean Strength"] - result_df["Lower 95% CI"],
            result_df["Upper 95% CI"] - result_df["Mean Strength"],
        ],
        fmt="o",
        color="skyblue",
        capsize=5,
        markersize=8,
        linestyle="None",
        ecolor="grey",
        elinewidth=2,
    )

    plt.xlabel("Mean Strength")
    plt.title("Technique Strengths with Confidence Intervals")
    plt.show()


def plot_violin_strengths(result_df):
    """Plot a violin plot for the distribution of strengths for each technique."""
    # Convert the 'All Strengths' column into a DataFrame for plotting
    strengths_df = pd.DataFrame(
        {
            technique: result_df["All Strengths"].iloc[i]
            for i, technique in enumerate(result_df["Technique"])
        }
    )
    print(strengths_df)

    # Plot using seaborn's violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=strengths_df, inner="quart", palette="muted")
    plt.xlabel("Technique")
    plt.ylabel("Strength")
    plt.title("Distribution of Technique Strengths Across Bootstrapped Samples")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()


def get_collection():
    try:
        mongosecret = os.getenv("MONGODB")
        uri = f"mongodb+srv://seppevanswegenoven:{mongosecret}@cluster0.p77cc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

        client = MongoClient(uri, server_api=ServerApi("1"))

        return client["JokeSurvey"]["secondsurveyresults"]
    except Exception as e:
        raise Exception("The following error occurred: ", e)


wins_matrix = get_rankings()
print(wins_matrix)
print(compute_wilcoxon_tests())
plot_win_matrix_heatmap(wins_matrix)
print(estimate_strengths())
strengths_with_confidence_interval = estimate_strengths_with_confidence_interval()
print(strengths_with_confidence_interval)
# plot_strengths(strengths_with_confidence_interval)
plot_violin_strengths(strengths_with_confidence_interval)
