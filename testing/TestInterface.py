from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from datasets import load_dataset
import torch
import statistics
import random

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class TestInterface:

    def __init__(
        self,
        model_ids,
        weigths,
        generated_jokes_db_name,
        result_ds_name,
        push_db=True,
        batch_size=5,
        test_user_dependency=False,
    ):
        self.model_ids = model_ids
        self.weights = weigths
        self.generated_jokes_db_name = generated_jokes_db_name
        self.results_ds_name = result_ds_name
        self.push_db = push_db
        self.batch_size = batch_size
        self.test_user_dependency = test_user_dependency
        self.tokenizers, self.models = self.initialize_test_models()

    def initialize_test_models(self):
        print("Get tokenizers and models")
        tokenizers = []
        models = []
        for model_id in self.model_ids:
            tokenizers.append(AutoTokenizer.from_pretrained(model_id))
            models.append(
                AutoModelForSequenceClassification.from_pretrained(
                    model_id, low_cpu_mem_usage=True
                ).to(device)
            )
        return tokenizers, models

    def score_generation_results(self):
        print("Get dataset")
        ds = load_dataset(self.generated_jokes_db_name, split="train")

        def transform_dataset(rows):
            if self.test_user_dependency:
                nr_of_users = max(rows["userId"])
                userIds = random.sample(range(1, nr_of_users + 1), self.batch_size)
            else:
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

        ds = ds.map(transform_dataset, batched=True, batch_size=self.batch_size)
        return ds

    def test_joke_generation_output(self):
        ds = self.score_generation_results(
            self.tokenizers, self.models, self.weights, self.generated_jokes_db_name
        )
        if self.push_db:
            ds.push_to_hub(self.result_ds_name, split="train")
        else:
            scores = ds["score"]
            print("Mean:", statistics.mean(scores))
            print("Median:", statistics.median(scores))
            return ds
