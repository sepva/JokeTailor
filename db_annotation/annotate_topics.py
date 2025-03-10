from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re
import wandb

model_name = "google/flan-t5-xxl"

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
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
                    For every joke, return your expert idea in a following structured way:
                    {
                        "explanation": "<your explanation>",
                        "Broad Topic": "<One of the Broad topics>",
                        "Joke Type": "<One of the Joke Types>",
                        "Complexity": "<One of the Complexity>",
                        "Tone": "<One of the Tone>",
                    }
                    """,
                },
                {
                    "role": "user",
                    "content": f"Classify the following joke: {joke}",
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
    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    scores = []
    for response in responses:
        score = re.search(r"<score>(.*?)</score>", response, flags=re.DOTALL)
        if score != None:
            scores.append(int(score.group(1)))
        else:
            scores.append(1)

    rows["score"] = scores

    return rows


ds = load_dataset("SeppeV/JokeTailor_big_set", split="train[:100]")
ds = ds.map(add_joke_rating, batched=True, batch_size=100)
print(ds["score"][:5])
