from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from datasets import load_dataset, Dataset
from RecRAG import RecRag
from BoN import BoN
from dotenv import load_dotenv

load_dotenv()


class InferenceInterface:

    def __init__(
        self,
        model_id: str,
        system_prompt,
        user_prompt_template,
        generate_config,
        bnb_config=None,
        lora=None,
        rag_config=None,
        rag_query_config=None,
        bon_config=None,
    ):
        self.model = self.get_model(model_id, bnb_config, lora)
        self.tokenizer = self.get_tokenizer(model_id)
        self.pipeline = self.get_pipeline(
            system_prompt,
            user_prompt_template,
            generate_config,
            rag_config,
            rag_query_config,
            bon_config,
        )

    def get_model(self, model_id, bnb_config, lora):
        print("Getting model...")
        compute_dtype = getattr(torch, "float16")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
            attn_implementation="flash_attention_2",
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

    def get_pipeline(
        self,
        system_prompt,
        user_prompt_template,
        generate_config,
        rag_config,
        rag_query_config,
        bon_config,
    ):
        rag_used = False
        if not rag_config is None:
            rag_used = True
            rag = RecRag(**rag_config)
            rag_function = getattr(rag, rag_query_config.pop("method"))

        generation_per_userId = 1
        bon_used = False
        if not bon_config is None:
            bon_used = True
            generation_per_userId = bon_config["extra_gen"]
            bon_interface = BoN(**bon_config)

        def pipeline(userIds):
            suggestions = {}
            if rag_used:
                suggestions = {
                    userId: rag_function(userId, **rag_query_config)
                    for userId in userIds
                }

            input_texts = [
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": user_prompt_template.format(
                                userId=str(userId),
                                suggestions="\n".join(suggestions.get(userId, [])),
                            ),
                        },
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for userId in userIds
                for _ in range(generation_per_userId)
            ]

            model_inputs = self.tokenizer(
                input_texts, return_tensors="pt", padding=True, truncation=True
            ).to(self.model.device)

            generated_ids = self.model.generate(
                **model_inputs, max_new_tokens=512, **generate_config
            )
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            responses = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            if bon_used:
                responses = bon_interface.filter_best_responses(responses, userIds)

            return responses

        return pipeline

    def prompt_pipeline(self, userId):
        print(self.pipeline([userId])[0])

    def generate_jokes_for_users(
        self, userIds, jokes_per_user, batch_size, output_ds_id, jokeId_template
    ):
        ds = Dataset.from_dict(
            {"userId": [userId for userId in userIds for _ in range(jokes_per_user)]}
        )

        def gen_map(rows, idx):
            rows["jokeText"] = self.pipeline(rows["userId"])
            rows["jokeId"] = [jokeId_template.format(userId=userId, jokeNr=id) for id, userId in zip(idx, rows["userId"])]
            return rows

        ds = ds.map(gen_map, batched=True, batch_size=batch_size, with_indices=True)
        ds.push_to_hub(output_ds_id)
