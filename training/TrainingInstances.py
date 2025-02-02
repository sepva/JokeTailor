from trl import DPOConfig, DPOTrainer
from peft import LoraConfig
from datasets import load_dataset
from TrainingInterface import TrainingInterface


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
        super().__init__(
            model_id,
            dataset_id,
            percentage_data,
            trainer_config,
            lora_config,
            bnb_config,
            lora,
        )
        self.prompt_function = lambda userId: [
            {"role": "user", "content": prompt.format(userId=userId)}
        ]

    def get_data(self, dataset_id, percentage_data):
        print("Getting data...")
        ds = load_dataset(dataset_id, split=f"train[:{percentage_data}%]")

        def process(row):
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

        ds = ds.map(process)
        ds = ds.select_columns(["prompt", "chosen", "rejected"])

        split_db = ds.train_test_split(test_size=0.1)
        train_dataset = split_db["train"]
        eval_dataset = split_db["test"]
        return train_dataset, eval_dataset

    def get_trainer(self, trainer_config, lora_config):
        print("Getting trainer...")

        trainer_config = DPOConfig(**trainer_config)
        lora_config = LoraConfig(**lora_config)

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
