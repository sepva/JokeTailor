import random

from datasets import Dataset, load_dataset
from dotenv import load_dotenv

load_dotenv()

jokes_ds_id = "SeppeV/JokeTailor_big_set_annotated"
output_ds_id = "SeppeV/Human_jokes"
userIds = range(1)
amount_of_jokes_per_user = 50

jokes_ds = load_dataset(jokes_ds_id, split="train")

ds_dict = {"userId": [], "jokeText": [], "jokeId": []}

for userId in userIds:
    joke_indices = random.sample(range(len(jokes_ds)), amount_of_jokes_per_user)
    jokes = jokes_ds.select(joke_indices)["jokeText"]
    jokeIds = jokes_ds.select(joke_indices)["jokeId"]
    ds_dict["userId"].extend([userId] * amount_of_jokes_per_user)
    ds_dict["jokeText"].extend(jokes)
    ds_dict["jokeId"].extend([f"RH_{userId}_{jokeId}" for jokeId in jokeIds])

Dataset.from_dict(ds_dict).push_to_hub(output_ds_id)
