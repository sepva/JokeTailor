from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re
import wandb

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

model = AutoModelForCausalLM.from_pretrained(
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
                    "content": """You are a comedy critic. When given a joke, you give a score out of 5.
            Score 0: This is not a joke
            Score 1: This looks like a joke, but I don't think it is funny
            Score 2: Most people would find this joke amusing
            Score 3: Most people would laugh with this joke
            Score 4: Everyone would laugh really hard with this joke
            Score 5: People will pee in their pants if they read this joke
            First, think about why this joke would be funny. Then give the rating.
            Your final score decision should be in xml format, like so: <score> 3 </score>""",
                },
                {
                    "role": "user",
                    "content": f"Score the following joke and put it between xml tags <score></score>: {joke}",
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
            try:
                scores.append(int(score.group(1)))
            except:
                scores.append(-1)
        else:
            scores.append(-1)

    rows["score"] = scores

    return rows


ds = load_dataset("SeppeV/JokeTailor_big_set_filtered", split="train")
ds = ds.map(add_joke_rating, batched=True, batch_size=100)
ds.push_to_hub("SeppeV/JokeTailor_big_set_filtered")

