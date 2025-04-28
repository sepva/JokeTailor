from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()


def change_jokeId(row, idx):
    row["jokeId"] = f"RandomUID_{idx}"
    return row


ds = load_dataset("SeppeV/FullGen_for_survey")
ds = ds.map(change_jokeId, with_indices=True).select_columns(["jokeText", "jokeId"])

ds.push_to_hub("SeppeV/RandomUID_for_survey")
