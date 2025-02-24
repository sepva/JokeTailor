from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
import torch
import TrainingInstances
import os

dirname = os.path.dirname(__file__)


class TrainingInterface:

    def __init__(
        self,
        model_id: str,
        dataset_id: str,
        percentage_data: int,
        trainer_config: dict,
        lora_config=None,
        bnb_config=None,
        lora=None,
    ):
        self.model = self.get_model(model_id, bnb_config, lora)
        self.tokenizer = self.get_tokenizer(model_id)
        self.data = self.get_data(dataset_id, percentage_data)
        self.trainer = self.get_trainer(trainer_config, lora_config)

    @staticmethod
    def get_trainer_from_config(config_file):
        with open(config_file, "r") as f:
            config = json.load(f)
            training_class = getattr(TrainingInstances, config["training_class"])
            return training_class(**config["training_args"])

    def get_model(self, model_id, bnb_config, lora):
        print("Getting model...")
        compute_dtype = getattr(torch, "float16")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            quantization_config=(
                BitsAndBytesConfig(**bnb_config, bnb_4bit_compute_dtype=compute_dtype)
                if bnb_config != None
                else None
            ),
        )

        if lora != None:
            model.load_adapter(lora)

        return model

    def get_tokenizer(self, model_id):
        print("Getting tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.chat_template is None:
            tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n\n'}}{% endfor %}{{ eos_token }}"
        return tokenizer

    def get_data(self, dataset_id, percentage_data):
        pass

    def get_trainer(self, trainer_config):
        pass

    def start_training(self):
        print("Start Training!")
        self.trainer.train()


# TODO: make config file an argument
if __name__ == "__main__":
    trainer = TrainingInterface.get_trainer_from_config(
        os.path.join(dirname, "config.json")
    )
    trainer.start_training()
