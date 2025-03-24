from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
)
import torch
import wandb
import re
from dotenv import load_dotenv

load_dotenv()

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    device_map="auto",
    attn_implementation="flash_attention_2",
    # max_memory={0: "10GiB", 1: "60GiB"},
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.chat_template is None:
    tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"


def add_joke_rating(rows):
    jokes = rows["jokeText"]
    texts = [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": """You are a humor expert. You will get jokes and are asked to classify them on a few levels.
                    For every level, use the predefined classes:
                    - Broad Topic: Work, Relationships, Family, School, Technology, Science, Nature, Food, Health, Money, Politics, Entertainment, Sports, News
                    - Joke Type: Pun, Observational humor, Satire, Sarcasm, Dark humor, Self-deprecating, One-liner, Anti-joke, Absurd, Riddle
                    - Complexity: Simple (one-liner), Setup & Punchline, Story, Multi-part jokes
                    - Tone: Lighthearted, Dark, Sarcastic, Absurd, Clever
                    For every joke, return your final judgement by using XML tags for the different levels:
                    <broad_topic> "One of the predefined classes" </broad_topic>
                    <joke_type> "One of the predefined classes" </joke_type>
                    <complexity> "One of the predefined classes" </complexity>
                    <tone> "One of the predefined classes" </tone>
                    """,
                },
                {
                    "role": "user",
                    "content": f"Classify the following joke, remember to put your final judgements in XML tags: {joke}",
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for joke in jokes
    ]

    model_inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    ).to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
    )
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    classification_keys = ["broad_topic", "joke_type", "complexity", "tone"]

    for key in classification_keys:
        results = []
        for response in responses:
            classification = re.search(
                rf"<{key}>(.*?)</{key}>", response, flags=re.DOTALL
            )
            if classification != None:
                try:
                    results.append(classification.group(1))
                except:
                    results.append(None)
            else:
                results.append(None)

        rows[key] = results

    return rows


ds = load_dataset("SeppeV/JokeTailor_big_set_filtered", split="train")
ds = ds.map(add_joke_rating, batched=True, batch_size=100)
ds.push_to_hub("SeppeV/JokeTailor_big_set_annotated")
