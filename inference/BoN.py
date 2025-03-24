from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import torch
from datasets import Dataset
from dotenv import load_dotenv

load_dotenv()


device = "cuda:0" if torch.cuda.is_available() else "cpu"

class BoN:

    def __init__(self, score_models, model_weights, extra_gen, batch_size):
        self.tokenizers, self.models = self.initialize_test_models(score_models)
        self.weights = model_weights
        self.extra_gen = extra_gen
        self.batch_size = batch_size

    def filter_best_responses(self, responses, userIds):
        gen_per_userId = userIds.count(userIds[0])
        df = self.score_generation_results(responses, userIds).to_pandas().sort_values("score", ascending=False).groupby("userId").head(gen_per_userId).sort_values("userId")
        return df["jokeText"].to_list()

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