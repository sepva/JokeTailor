from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import pickle
import os
import shutil
import numpy as np
from huggingface_hub import HfApi, create_repo, snapshot_download
import torch
from transformers.utils.logging import disable_progress_bar
from random import sample
import pandas as pd
from dotenv import load_dotenv
import logging
logger = logging.getLogger(__name__)

load_dotenv()

disable_progress_bar()


class RecRag:

    def __init__(
        self,
        itemDB_id,
        userRatingDB_id,
        bi_encoder_id,
        cross_encoder_id,
        item_column_name,
        item_embedding_file=None,
        item_repo_name="",
        user_repo_name="",
        best_k=10,
    ):

        self.bi_encoder = SentenceTransformer(bi_encoder_id)
        self.cross_encoder = CrossEncoder(cross_encoder_id)
        self.item_column_name = item_column_name

        if item_embedding_file:
            self.load_item_embeddings(item_embedding_file, repo_name=item_repo_name)
        else:
            self.itemDB = load_dataset(itemDB_id, split="train")
            self.calculate_item_embeddings()

        if user_repo_name:
            self.load_user_embeddings(repo_name=user_repo_name)
        else:
            self.userRatingDB = load_dataset(userRatingDB_id, split="train")
            self.calculate_user_embeddings(best_k)

    def calculate_item_embeddings(self):
        self.items = self.itemDB[self.item_column_name]
        self.item_embeddings = self.bi_encoder.encode(
            self.items, convert_to_tensor=True, show_progress_bar=True
        )

    def mean_pooling(self, jokes, weights):
        embeddings = self.bi_encoder.encode(
            jokes, convert_to_tensor=True, show_progress_bar=False
        ).cpu()
        mean_pooled_embedding = np.average(embeddings, weights=weights, axis=0)
        return mean_pooled_embedding

    def calculate_user_embeddings(self, best_k):
        userIds = list(set(self.userRatingDB["userId"]))
        pd_userRatingDB = self.userRatingDB.to_pandas()
        user_groups = pd_userRatingDB.groupby(["userId"])

        self.user_embeddings = []
        self.userIds = []
        for i, userId in enumerate(userIds):
            print(i * 100 / len(userIds), end="\r")
            user_scores = user_groups.get_group(userId).sort_values(
                "rating", ascending=False
            )
            jokes, weights = (
                user_scores[self.item_column_name].tolist(),
                user_scores["rating"].tolist(),
            )
            max_index = best_k if best_k < len(weights) else len(weights)
            jokes, weights = jokes[:max_index], weights[:max_index]
            if sum(weights) > 0:
                self.user_embeddings.append(self.mean_pooling(jokes, weights))
                self.userIds.append(userId)

        user_dict = {"userId": self.userIds, "user_embedding": self.user_embeddings}
        self.user_ds = Dataset.from_dict(user_dict)

    def save_user_embeddings(self, repo_name=""):
        self.user_ds.push_to_hub(repo_name)

    def save_item_embeddings(self, fname, hf=True, make_repo=True, repo_name=""):
        with open(fname, "wb") as f:
            pickle.dump({"items": self.items, "embeddings": self.item_embeddings}, f)
        if hf:
            self.save_vector_store_to_hf(fname, make_repo, repo_name)

    def save_vector_store_to_hf(self, fname, make_repo, repo_name):
        api = HfApi()
        if make_repo:
            create_repo(repo_name, repo_type="dataset")
        api.upload_file(
            path_or_fileobj=fname,
            repo_id=repo_name,
            path_in_repo=fname,
            repo_type="dataset",
        )

    def load_item_embeddings(self, fname, hf=True, repo_name=""):
        if not os.path.exists(fname):
            snapshot_download(
                repo_id=repo_name,
                allow_patterns="*.pkl",
                repo_type="dataset",
                local_dir="./",
            )

        with open(fname, "rb") as f:
            cache_data = pickle.load(f)
            self.items = cache_data["items"]
            self.item_embeddings = cache_data["embeddings"]

    def load_user_embeddings(self, repo_name=""):
        self.user_ds = load_dataset(repo_name, split="train")
        self.user_df = self.user_ds.to_pandas()
        self.userIds = self.user_ds["userId"]
        self.user_embeddings = self.user_ds["user_embedding"]

    def semantic_search(
        self,
        userId,
        query_template,
        top_k,
        output_k=5,
        re_ranker=False,
        item_corpus_ids=None,
        question_embedding=None,
        cross_encoder_query=None
    ):
        ##### Semantic Search #####
        if question_embedding == None:
            logger.debug(f"query sem search: {query_template.format(userId=userId)}")
            question_embedding = self.bi_encoder.encode(
                query_template.format(userId=userId), convert_to_tensor=True
            )

        question_embedding = question_embedding.cuda()
        if item_corpus_ids:
            filtered_items = np.array(self.items)[item_corpus_ids]
            filtered_embeddings = self.item_embeddings[item_corpus_ids].cuda()
        else:
            filtered_items = np.array(self.items)
            filtered_embeddings = self.item_embeddings.cuda()

        hits = util.semantic_search(
            question_embedding, filtered_embeddings, top_k=top_k
        )
        hits = hits[0]  # Get the hits for the first query
        
        ##### Re-Ranking #####
        if cross_encoder_query is None:
            cross_encoder_query = query_template
        
        logger.debug(f"query cross encoder: {cross_encoder_query.format(userId=userId)}")
        cross_inp = [[cross_encoder_query.format(userId=userId), filtered_items[hit["corpus_id"]]] for hit in hits]
        cross_scores = self.cross_encoder.predict(cross_inp)

        if re_ranker:
            # Sort results by the cross-encoder scores
            for idx in range(len(cross_scores)):
                hits[idx]["cross-score"] = cross_scores[idx]
            hits = sorted(hits, key=lambda x: x["cross-score"], reverse=True)
        else:
            hits = sorted(hits, key=lambda x: x["score"], reverse=True)

        return [filtered_items[hit["corpus_id"]] for hit in hits[0:output_k]]

    def user_rec_search(self, userId, top_k, re_ranker=False):
        user_embedding = self.user_df[self.user_df["userId"] == userId][
            "user_embedding"
        ]
        if len(user_embedding) > 0:
            user_embedding = torch.FloatTensor(user_embedding.item())
            hits = util.semantic_search(
                user_embedding, self.item_embeddings, top_k=top_k
            )
            hits = hits[0]

            hits = sorted(hits, key=lambda x: x["score"], reverse=True)
            return [self.items[hit["corpus_id"]] for hit in hits], [
                hit["corpus_id"] for hit in hits
            ]
        else:
            raise Exception("The given user doesn't have a saved user embedding!")

    def user_rec_then_semantic_search(
        self, userId, query, user_rec_size, top_k, output_k, re_ranker=False, cross_encoder_query=None,
    ):
        _, user_recs = self.user_rec_search(userId, user_rec_size)
        jokes = self.semantic_search(userId, query, top_k, output_k, re_ranker, user_recs, cross_encoder_query=cross_encoder_query)
        return jokes

    def user_rec_search_without_topic(
        self,
        userId,
        user_rec_size,
        semantic_search_prompt,
        top_k,
        output_k,
        re_ranker=False,
        min_community_size=1,
        treshold=0.75,
    ):
        _, user_recs = self.user_rec_search(userId, user_rec_size)
        topic_embedding = self.choose_topic_embedding(
            user_recs, min_community_size, treshold
        )
        jokes = self.semantic_search(
            userId,
            semantic_search_prompt,
            top_k,
            output_k,
            re_ranker,
            user_recs,
            question_embedding=topic_embedding,
        )

        return jokes

    def choose_topic_embedding(self, user_recs, min_community_size=1, treshold=0.75):
        joke_embeddings = self.item_embeddings[user_recs]
        communities = util.community_detection(
            joke_embeddings, min_community_size=min_community_size, threshold=treshold
        )

        probs = [len(community) for community in communities]
        probs = np.exp(probs) / sum(np.exp(probs))

        index = np.random.choice(len(communities), p=probs)
        chosen_community = communities[index]

        community_embeddings = [
            joke_embeddings[jokeIndex].cpu() for jokeIndex in chosen_community
        ]
        mean_embedding = np.average(community_embeddings, axis=0)

        topic_embedding = torch.FloatTensor(mean_embedding)

        return topic_embedding
