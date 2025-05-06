from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import torch
from datasets import Dataset
from dotenv import load_dotenv
import logging
from sentence_transformers import SentenceTransformer
logger = logging.getLogger(__name__)

load_dotenv()


device = "cuda:0" if torch.cuda.is_available() else "cpu"

class BoN:

    def __init__(self, score_models, model_weights, extra_gen, batch_size, sim_model, sim_limit=0.3):
        self.tokenizers, self.models = self.initialize_test_models(score_models)
        self.weights = model_weights
        self.extra_gen = extra_gen
        self.batch_size = batch_size
        self.similarity_model = SentenceTransformer(sim_model)
        self.sim_limit = sim_limit

    def filter_best_responses(self, responses, userIds):
        df = self.score_generation_results(responses, userIds).to_pandas()
        grouped_df = df.sort_values("score", ascending=False).groupby("userId")
        logger.info(f"DF for BoN: {df.head()}")
        best_jokes = []
        for userId, ranked_jokes in grouped_df.groups.items():
            logger.info(f"UserId: {userId}")
            amount_of_jokes = userIds.count(userId)
            logger.info(f"Amount of jokes: {amount_of_jokes}")
            logger.info(f"ranked jokes: {ranked_jokes}")
            best_jokes.extend(self.get_best_diverse_jokes(df.iloc[ranked_jokes]["jokeText"].to_list(), amount_of_jokes))
            logger.info(f"best jokes: {best_jokes}")
        return best_jokes

    def get_best_diverse_jokes(self, jokes, amount_of_jokes):
        best_jokes = [jokes[0]]
        best_joke_embeddings = [self.similarity_model.encode(jokes[0])]
        for joke in jokes[1:]:
            joke_embedding = self.similarity_model.encode(joke)
            if not self.too_similar(joke_embedding, best_joke_embeddings):
                best_jokes.append(joke)
                best_joke_embeddings.append(joke_embedding)

        while len(best_jokes) < amount_of_jokes:
            logger.info("not enough jokes, have to add one that is similar")
            for joke in jokes[1:]:
                if joke not in best_jokes:
                    best_jokes.append(joke)
        
        return best_jokes[:amount_of_jokes]
    
    def too_similar(self, new_joke, joke_list):
        sims_ranking = self.similarity_model.similarity([new_joke], joke_list)
        logger.info(f"sims ranking: {sims_ranking}")
        return any(sims_ranking[0] > self.sim_limit)

    def initialize_test_models(self, model_ids):
        print("Get tokenizers and models")
        tokenizers = []
        models = []
        for model_id in model_ids:
            tokenizers.append(AutoTokenizer.from_pretrained(model_id))
            models.append(
                AutoModelForSequenceClassification.from_pretrained(
                    model_id, low_cpu_mem_usage=True
                ).to(device)
            )
        return tokenizers, models

    def score_generation_results(self, responses, userIds):
        def transform_dataset(rows):
            userIds = rows["userId"]

            input_texts = [
                f"User {userId}: {jokeText}"
                for userId, jokeText in zip(userIds, rows["jokeText"])
            ]
            preds = []
            for tokenizer, model in zip(self.tokenizers, self.models):
                encoding = tokenizer(
                    input_texts,
                    truncation=True,
                    padding="max_length",
                    max_length=256,
                    return_tensors="pt",
                ).to(device)
                preds.append(torch.flatten(model(**encoding).logits).tolist())

            rows["score"] = torch.Tensor(self.weights) @ torch.Tensor(preds)
            return rows
        
        ds = Dataset.from_dict(
            {"userId": [userId for userId in userIds for _ in range(self.extra_gen)],
             "jokeText": responses}
        )

        ds = ds.map(transform_dataset, batched=True, batch_size=self.batch_size)
        return ds