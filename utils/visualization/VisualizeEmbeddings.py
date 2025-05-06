import io
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from sklearn.manifold import TSNE


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


class VisualizeEmbeddings:

    def __init__(
        self, joke_ds_id, item_embeddings_ds_id, user_embeddings_ds_id, cpu=False
    ):
        self.joke_ds = load_dataset(joke_ds_id, split="train")
        self.embeddings = self.load_item_embeddings(
            "jokeTailor_embeddings.pkl", repo_name=item_embeddings_ds_id, cpu=cpu
        )
        self.user_embeddings = load_dataset(user_embeddings_ds_id, split="train")[
            "user_embedding"
        ]

    def load_item_embeddings(self, fname, hf=True, repo_name="", cpu=False):
        if not os.path.exists(fname):
            snapshot_download(
                repo_id=repo_name,
                allow_patterns="*.pkl",
                repo_type="dataset",
                local_dir="./",
            )

        if cpu:
            with open(fname, "rb") as f:
                cache_data = CPU_Unpickler(f).load()
                return cache_data["embeddings"]
        else:
            with open(fname, "rb") as f:
                cache_data = pickle.load(f)
                return cache_data["embeddings"]

    def visualize_joke_and_user_embeddings(self, file_name=None):
        color_to_topic = {}
        topic_to_color = {}
        colors = []
        last_topic_color = 0
        for topic in self.joke_ds["broad_topic"]:
            if topic in topic_to_color:
                colors.append(topic_to_color[topic])
            else:
                last_topic_color += 1
                color_to_topic[last_topic_color] = topic
                topic_to_color[topic] = last_topic_color
                colors.append(last_topic_color)

        colors.extend([100] * len(self.user_embeddings))
        all_embeddings = self.embeddings.tolist() + self.user_embeddings

        projected_embeddings = TSNE(perplexity=5).fit_transform(
            np.array(all_embeddings)
        )
        fig, ax = plt.subplots()
        ax.scatter(
            projected_embeddings[:, 0],
            projected_embeddings[:, 1],
            marker="o",
            c=colors,
            s=1,
            alpha=0.8,
        )
        if file_name is None:
            plt.show()
        else:
            plt.savefig(file_name)

    def visualize_joke_embeddings(self, file_name=None):
        color_to_topic = {}
        topic_to_color = {}
        colors = []
        last_topic_color = 0
        for topic in self.joke_ds["broad_topic"]:
            if topic in topic_to_color:
                colors.append(topic_to_color[topic])
            else:
                last_topic_color += 1
                color_to_topic[last_topic_color] = topic
                topic_to_color[topic] = last_topic_color
                colors.append(last_topic_color)

        projected_embeddings = TSNE(perplexity=5).fit_transform(self.embeddings.cpu())
        fig, ax = plt.subplots()
        ax.scatter(
            projected_embeddings[:, 0],
            projected_embeddings[:, 1],
            marker="o",
            c=colors,
            s=1,
            alpha=0.8,
        )
        if file_name is None:
            plt.show()
        else:
            plt.savefig(file_name)

    def visualize_user_embeddings(self, file_name=None):
        projected_embeddings = TSNE(perplexity=5).fit_transform(
            np.array(self.user_embeddings)
        )
        fig, ax = plt.subplots()
        ax.scatter(
            projected_embeddings[:, 0],
            projected_embeddings[:, 1],
            marker="o",
            s=1,
            alpha=0.8,
        )
        if file_name is None:
            plt.show()
        else:
            plt.savefig(file_name)


VE = VisualizeEmbeddings(
    "SeppeV/JokeTailor_big_set_annotated",
    "SeppeV/jokeTailor_embeddings",
    "SeppeV/jokeTailor_user_embeddings",
    cpu=True,
)

VE.visualize_joke_embeddings("joke_embeddings")
