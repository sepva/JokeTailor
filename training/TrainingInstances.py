from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification
from peft import LoraConfig
from datasets import load_dataset
from TrainingInterface import TrainingInterface
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample
import random
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers import models, datasets, losses
from torch.utils.data import DataLoader


class DPO(TrainingInterface):

    def __init__(
        self,
        model_id: str,
        dataset_id: str,
        percentage_data: int,
        trainer_config: dict,
        prompt: str,
        lora_config=None,
        bnb_config=None,
        lora=None,
    ):
        self.prompt = prompt
        self.initialize_all(
            model_id,
            dataset_id,
            percentage_data,
            trainer_config,
            lora_config,
            bnb_config,
            lora,
        )

    def prompt_function(self, userId):
        return [{"role": "user", "content": self.prompt.format(userId=userId)}]

    def process(self, row):
        prompt_in_chat = self.prompt_function(row["userId"])
        row["chosen"] = self.tokenizer.apply_chat_template(
            [*prompt_in_chat, row["chosen"][0]], tokenize=False
        )
        row["rejected"] = self.tokenizer.apply_chat_template(
            [*prompt_in_chat, row["rejected"][0]], tokenize=False
        )
        row["prompt"] = self.tokenizer.apply_chat_template(
            prompt_in_chat, tokenize=False
        )
        return row

    def get_data(self, dataset_id, percentage_data):
        print("Getting data...")
        ds = load_dataset(dataset_id, split=f"train[:{percentage_data}%]")

        ds = ds.map(self.process)
        ds = ds.select_columns(["prompt", "chosen", "rejected"])

        split_db = ds.train_test_split(test_size=0.1)
        train_dataset = split_db["train"]
        eval_dataset = split_db["test"]
        return train_dataset, eval_dataset

    def get_trainer(self, trainer_config, lora_config):
        print("Getting trainer...")

        trainer_config = DPOConfig(**trainer_config)
        lora_config = LoraConfig(**lora_config) if lora_config != None else None

        trainer = DPOTrainer(
            self.model,
            ref_model=None,
            args=trainer_config,
            peft_config=lora_config,
            train_dataset=self.data[0],
            eval_dataset=self.data[1],
            tokenizer=self.tokenizer,
        )

        return trainer


class CurryDPO(DPO):

    def __init__(
        self,
        model_id,
        dataset_id,
        percentage_data,
        trainer_config,
        prompt,
        lora_config=None,
        bnb_config=None,
        lora=None,
        iteration=0,
        max_iterations=4,
    ):
        self.dataset_id = dataset_id
        self.percentage_data = percentage_data
        self.trainer_config = trainer_config
        self.output_dir_base = trainer_config["output_dir"]
        self.run_name_base = trainer_config["run_name"]
        self.lora_config = lora_config
        self.iteration = iteration
        self.max_iterations = max_iterations
        self.initialize_all(
            model_id,
            dataset_id,
            percentage_data,
            trainer_config,
            prompt,
            lora_config,
            bnb_config,
            lora,
        )

    def get_data(self, dataset_id, percentage_data):
        print("Getting data...")
        ds = load_dataset(dataset_id, split=f"train[:{percentage_data}%]")

        sorted_by_score_difference_ds = ds.sort(column_names=["margin"], reverse=True)
        sorted_by_score_difference_ds = sorted_by_score_difference_ds.shard(
            num_shards=self.max_iterations, index=self.iteration
        )

        ds = ds.map(self.process)
        ds = ds.select_columns(["prompt", "chosen", "rejected"])

        split_db = ds.train_test_split(test_size=0.1)
        train_dataset = split_db["train"]
        eval_dataset = split_db["test"]
        return train_dataset, eval_dataset

    def start_training(self):
        print("Start Training!")
        while self.iteration < self.max_iterations:
            self.data = self.get_data(self.dataset_id, self.percentage_data)
            self.trainer_config["output_dir"] = (
                f"{self.output_dir_base}_{self.iteration}"
            )
            self.trainer_config["run_name"] = f"{self.run_name_base}_{self.iteration}"
            self.trainer = self.get_trainer(self.trainer_config, self.lora_config)
            self.trainer.train()
            self.iteration += 1


class SFT(TrainingInterface):

    def __init__(
        self,
        model_id,
        dataset_id,
        percentage_data,
        trainer_config,
        prompt,
        # reasoning_column_name,
        lora_config=None,
        bnb_config=None,
        lora=None,
    ):
        self.prompt_function = lambda userId: [
            {"role": "user", "content": prompt.format(userId=userId)}
        ]
        # self.reasoning_column_name = reasoning_column_name
        self.initialize_all(
            model_id,
            dataset_id,
            percentage_data,
            trainer_config,
            lora_config,
            bnb_config,
            lora,
        )

    def joke_with_reasoning_steps(self, row):
        joke = row["jokeText"]
        # reasoning = row[self.reasoning_column_name]
        return f"<joke>{joke}</joke>"

    def process(self, row):
        prompt_in_chat = self.prompt_function(random.randint(0, 200))
        row["messages"] = [
            *prompt_in_chat,
            {"role": "assistant", "content": self.joke_with_reasoning_steps(row)},
        ]
        return row

    def get_data(self, dataset_id, percentage_data):
        print("Getting data...")
        ds = load_dataset(dataset_id, split=f"train[:{percentage_data}%]")

        ds = ds.map(self.process)
        ds = ds.select_columns(["messages"])

        split_db = ds.train_test_split(test_size=0.1)
        train_dataset = split_db["train"]
        eval_dataset = split_db["test"]
        return train_dataset, eval_dataset

    def get_trainer(self, trainer_config, lora_config):
        print("Getting trainer...")

        trainer_config = SFTConfig(**trainer_config)
        lora_config = LoraConfig(**lora_config) if lora_config != None else None

        trainer = SFTTrainer(
            self.model,
            args=trainer_config,
            peft_config=lora_config,
            train_dataset=self.data[0],
            eval_dataset=self.data[1],
            tokenizer=self.tokenizer,
        )

        return trainer


class TestModelTrainer(TrainingInterface):

    def __init__(
        self,
        model_id,
        dataset_id,
        percentage_data,
        trainer_config,
        max_error_for_accuracy,
        prompt,
        lora_config=None,
        bnb_config=None,
        lora=None,
    ):
        self.max_error_for_accuracy = max_error_for_accuracy
        self.prompt = prompt
        self.initialize_all(
            model_id,
            dataset_id,
            percentage_data,
            trainer_config,
            lora_config,
            bnb_config,
            lora,
        )

    def compute_metrics_for_regression(self, eval_pred):
        logits, labels = eval_pred
        labels = labels.reshape(-1, 1)

        mse = mean_squared_error(labels, logits)
        mae = mean_absolute_error(labels, logits)
        r2 = r2_score(labels, logits)
        single_squared_errors = ((logits - labels).flatten() ** 2).tolist()

        # Compute accuracy
        # Based on the fact that the predicted score ~= true score only if |error| < max_error_for_accuracy => error^2 < max_error_for_accuracy^2
        accuracy = sum(
            [1 for e in single_squared_errors if e < self.max_error_for_accuracy**2]
        ) / len(single_squared_errors)

        return {"mse": mse, "mae": mae, "r2": r2, "accuracy": accuracy}

    def get_model(self, model_id, bnb_config, lora):
        print("Getting model...")

        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=1,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="cuda",
        )

        return model

    def process(self, rows):
        labels = rows["rating"]
        input_texts = [
            self.prompt.format(userId=userId, jokeText=jokeText)
            for userId, jokeText in zip(rows["userId"], rows["jokeText"])
        ]
        rows = self.tokenizer(
            input_texts,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        )
        rows["label"] = [float(label) for label in labels]
        return rows

    def get_data(self, dataset_id, percentage_data):
        print("Getting data...")
        ds = load_dataset(dataset_id, split=f"train[:{percentage_data}%]")

        ds = ds.map(self.process, batched=True, batch_size=100)
        ds = ds.select_columns(
            ["input_ids", "attention_mask", "label"]
        )

        split_db = ds.train_test_split(test_size=0.1)
        train_dataset = split_db["train"]
        eval_dataset = split_db["test"]
        return train_dataset, eval_dataset

    def get_trainer(self, trainer_config, lora_config):
        print("Getting trainer...")

        trainer_config = TrainingArguments(**trainer_config)
        self.tokenizer.push_to_hub(trainer_config.hub_model_id)

        trainer = Trainer(
            self.model,
            args=trainer_config,
            train_dataset=self.data[0],
            eval_dataset=self.data[1],
            compute_metrics=self.compute_metrics_for_regression,
        )

        return trainer


class CrossEncoderTrainer(TrainingInterface):

    def __init__(
        self,
        model_id,
        dataset_id,
        percentage_data,
        trainer_config,
        prompt,
        hub_model_id,
        lora_config=None,
        bnb_config=None,
        lora=None,
    ):  
        self.prompt = prompt
        self.hub_model_id = hub_model_id
        self.initialize_all(
            model_id,
            dataset_id,
            percentage_data,
            trainer_config,
            lora_config,
            bnb_config,
            lora,
        )

    def get_model(self, model_id, bnb_config, lora):
        print("Getting model...")
        model = CrossEncoder(model_id, num_labels=1)
        return model

    def process(self, rows):
        labels = rows["rating"]
        rows["text"] = [
            [self.prompt.format(userId=userId), f"{jokeText}"]
            for userId, jokeText in zip(rows["userId"], rows["jokeText"])
        ]
        rows["label"] = [float(label) for label in labels]
        return rows

    def get_data(self, dataset_id, percentage_data):
        print("Getting data...")
        ds = load_dataset(dataset_id, split=f"train[:{percentage_data}%]")

        ds = ds.map(self.process, batched=True, batch_size=100)
        ds = ds.select_columns(["text", "label"])

        input_examples = [InputExample(texts=r["text"], label=r["label"]) for r in ds]
        return DataLoader(input_examples, batch_size=8)

    def get_trainer(self, trainer_config, lora_config):
        print("Getting trainer...")
        self.trainer_config = trainer_config
        return None

    def start_training(self):
        print("Start Training!")
        self.model.fit(self.data, **self.trainer_config)
        self.model.push_to_hub(self.hub_model_id)

class TSDAE_BiEncoderTrainer(TrainingInterface):

    def __init__(self, model_id, dataset_id, percentage_data, trainer_config, hub_model_id, lora_config=None, bnb_config=None, lora=None):
        self.hub_model_id = hub_model_id
        self.initialize_all(
            model_id,
            dataset_id,
            percentage_data,
            trainer_config,
            lora_config,
            bnb_config,
            lora,
        )

    def get_model(self, model_id, bnb_config, lora):
        print("Getting model...")
        word_embedding_model = models.Transformer(model_id)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), "cls")
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        self.model_id = model_id
        return model
    
    def get_data(self, dataset_id, percentage_data):
        print("Getting data...")
        ds = load_dataset(dataset_id, split=f"train[:{percentage_data}%]")
        jokes = ds["jokeText"]
        train_dataset = datasets.DenoisingAutoEncoderDataset(jokes)
        return DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    def get_trainer(self, trainer_config, lora_config):
        print("Getting trainer...")
        self.trainer_config = trainer_config
        return None
    
    def start_training(self):
        print("Start Training!")
        train_loss = losses.DenoisingAutoEncoderLoss(
            self.model, decoder_name_or_path=self.model_id, tie_encoder_decoder=True
        )

        self.model.fit(
            train_objectives=[(self.data, train_loss)],
            **self.trainer_config
        )
        self.model.push_to_hub(self.hub_model_id)

class BiEncoderOnUserdataTrainer(TrainingInterface):

    def __init__(self, model_id, dataset_id, percentage_data, trainer_config, prompt, lora_config=None, bnb_config=None, lora=None):
        self.prompt = prompt
        self.initialize_all(
            model_id,
            dataset_id,
            percentage_data,
            trainer_config,
            lora_config,
            bnb_config,
            lora,
        )

    def get_model(self, model_id, bnb_config, lora):
        print("Getting model...")
        model = SentenceTransformer(model_id)
        return model
    
    def process(self, row):
        row["anchor"] = self.prompt.format(userId=row["userId"])
        row["positive"] = row["chosen"][0]["content"]
        row["negative"] = row["rejected"][0]["content"]
        return row
    
    def get_data(self, dataset_id, percentage_data):
        print("Getting data...")
        ds = load_dataset(dataset_id, split=f"train[:{percentage_data}%]")
        ds = ds.map(self.process)
        ds = ds.select_columns(["anchor", "positive", "negative"])
        split_db = ds.train_test_split(test_size=0.1)
        train_dataset = split_db["train"]
        eval_dataset = split_db["test"]
        return train_dataset, eval_dataset
    
    def get_trainer(self, trainer_config, lora_config):
        trainer_config = SentenceTransformerTrainingArguments(**trainer_config)

        trainer = SentenceTransformerTrainer(
            self.model,
            args=trainer_config,
            train_dataset=self.data[0],
            eval_dataset=self.data[1],
            loss=losses.MultipleNegativesRankingLoss(self.model)
        )

        return trainer