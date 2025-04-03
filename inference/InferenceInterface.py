import logging
import random
import re

import torch
from BoN import BoN
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from RecRAG import RecRag
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

load_dotenv()


class InferenceInterface:

    def __init__(
        self,
        model_id: str,
        system_prompt,
        user_prompt_template,
        generate_config,
        topics=None,
        joke_tag=None,
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
            joke_tag,
            generate_config,
            rag_config,
            rag_query_config,
            bon_config,
            topics,
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
        joke_tag,
        generate_config,
        rag_config,
        rag_query_config,
        bon_config,
        topics,
    ):
        rag_used = False
        if not rag_config is None:
            rag_used = True
            rag = RecRag(**rag_config)
            rag_function = getattr(rag, rag_query_config.pop("method"))
            rag_query_template = rag_query_config["query"]

        generation_per_userId = 1
        bon_used = False
        if not bon_config is None:
            bon_used = True
            generation_per_userId = bon_config["extra_gen"]
            bon_interface = BoN(**bon_config)

        def pipeline(userIds):
            logger.info(f"Starting pipeline for userIds {userIds}")

            if not topics is None:
                chosen_topics = random.choices(topics, k=len(userIds))
                logger.info(f"Chosen topics: {chosen_topics}")

            suggestions = []
            if rag_used:
                logger.info(f"Calculating suggestions for userIds")
                if not topics is None:
                    for i, userId in enumerate(userIds):
                        rag_query_config["query"] = rag_query_template.format(
                            userId=userId, topic=chosen_topics[i]
                        )
                        logger.info(
                            f"Getting suggestions for: {rag_query_config['query']}"
                        )
                        suggestions.append(rag_function(userId, **rag_query_config))
                        logger.info(
                            f"RAG returned following suggestions: {suggestions[i]} "
                        )
                else:
                    for userId in userIds:
                        suggestions.append(rag_function(userId, **rag_query_config))

            input_texts = [
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": user_prompt_template.format(
                                userId=str(userId),
                                suggestions=(
                                    "\n- ".join(suggestions[i])
                                    if len(suggestions) > 0
                                    else ""
                                ),
                                topic=chosen_topics[i],
                            ),
                        },
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for i, userId in enumerate(userIds)
                for _ in range(generation_per_userId)
            ]

            model_inputs = self.tokenizer(
                input_texts, return_tensors="pt", padding=True, truncation=True
            ).to(self.model.device)

            logger.info("Generating outputs")
            generated_ids = self.model.generate(**model_inputs, **generate_config)
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            responses = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            if not joke_tag is None:
                jokes = []
                failed = 0
                logger.info("Extracting jokes")
                for response in responses:
                    joke = re.search(
                        rf"<{joke_tag}>(.*?)</{joke_tag}>", response, flags=re.DOTALL
                    )
                    if joke != None:
                        try:
                            jokes.append(joke.group(1))
                        except:
                            jokes.append("")
                            failed += 1
                    else:
                        jokes.append("")
                        failed += 1
                if failed > 0:
                    logger.warning(f"Failed to extract {failed} jokes")
            else:
                jokes = responses

            if bon_used:
                logger.info("BoN filtering")
                jokes = bon_interface.filter_best_responses(jokes, userIds)

            return jokes

        return pipeline

    def prompt_pipeline(self, userId):
        logging.basicConfig(filename="inference.log", level=logging.DEBUG)
        print(self.pipeline([userId])[0])

    def generate_jokes_for_users(
        self, userIds, jokes_per_user, batch_size, output_ds_id, jokeId_template
    ):
        logging.basicConfig(filename="inference.log", level=logging.WARNING)
        ds = Dataset.from_dict(
            {"userId": [userId for userId in userIds for _ in range(jokes_per_user)]}
        )

        def gen_map(rows, idx):
            rows["jokeText"] = self.pipeline(rows["userId"])
            rows["jokeId"] = [
                jokeId_template.format(userId=userId, jokeNr=id)
                for id, userId in zip(idx, rows["userId"])
            ]
            return rows

        ds = ds.map(gen_map, batched=True, batch_size=batch_size, with_indices=True)
        ds.push_to_hub(output_ds_id)
