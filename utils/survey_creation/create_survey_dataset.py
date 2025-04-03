import random

from datasets import concatenate_datasets, load_dataset, Dataset


def add_dataset_info(r, dataset):
    r["dataset"] = dataset
    return r


def remove_edits(r):
    r["jokeText"] = r["jokeText"].split("Edit")[0].split("edit")[0].split("EDIT")[0]
    return r


def get_reddit_jokes(num_jokes):
    ds = load_dataset("SeppeV/reddit_jokes_annotated", split="train[:5000]")
    ds = ds.filter(lambda x: x["joke_or_not"])
    ds = ds.select(range(0, num_jokes)).select_columns(["jokeText"])
    ds = ds.map(lambda r: add_dataset_info(r, "reddit"))
    ds = ds.map(remove_edits)
    return ds


def get_scoutlife_jokes(num_jokes):
    ds = load_dataset("SeppeV/scoutlife_db_annotated", split="train")
    ds = ds.filter(lambda x: x["joke_or_not"])
    ds = ds.select(random.sample(range(0, len(ds)), num_jokes)).select_columns(["joke"])
    ds = ds.map(lambda r: add_dataset_info(r, "scoutlife"))
    ds = ds.rename_column("joke", "jokeText")
    return ds


def get_short_jokes(num_jokes):
    ds = load_dataset("SeppeV/short_jokes_annotated", split="train")
    ds = ds.filter(lambda x: x["joke_or_not"])
    ds = ds.select(random.sample(range(0, len(ds)), num_jokes)).select_columns(["Joke"])
    ds = ds.map(lambda r: add_dataset_info(r, "short_jokes"))
    ds = ds.rename_column("Joke", "jokeText")
    return ds


def assemble_dataset():
    ds = concatenate_datasets(
        [get_reddit_jokes(3000), get_scoutlife_jokes(4000), get_short_jokes(6000)]
    )
    ds.push_to_hub("JokeTailor_big_set")


def filter_jokes_null_category():
    ds = load_dataset("SeppeV/JokeTailor_big_set_annotated")
    ds = ds.filter(
        lambda r: r["broad_topic"] != None
        and r["joke_type"] != None
        and r["tone"] != None
    )
    ds = ds.remove_columns(["topics_hierarch_id", "topics_hierarch_name", "complexity"])
    ds.push_to_hub("SeppeV/JokeTailor_big_set_annotated")


def add_jokeIds():
    ds = load_dataset("SeppeV/JokeTailor_big_set_annotated")

    def joke_id_mapper(r, idx):
        r["jokeId"] = idx
        return r

    ds = ds.map(joke_id_mapper, with_indices=True)
    ds.push_to_hub("SeppeV/JokeTailor_big_set_annotated")


def remove_trailing_and_leading_whitespace():
    ds = load_dataset("SeppeV/JokeTailor_big_set_annotated")

    def trimmer(r):
        r["jokeText"] = r["jokeText"].strip()
        return r

    ds = ds.map(trimmer)

    ds.push_to_hub("SeppeV/JokeTailor_big_set_annotated")


def remove_non_standard_topics():
    ds = load_dataset("SeppeV/JokeTailor_big_set_annotated")

    def topic_filter(r):
        return r["broad_topic"] in [
            "Work",
            "Relationships",
            "Family",
            "School",
            "Technology",
            "Science",
            "Nature",
            "Food",
            "Health",
            "Money",
            "Politics",
            "Entertainment",
            "Sports",
            "News",
        ]

    ds = ds.filter(topic_filter)

    ds.push_to_hub("SeppeV/JokeTailor_big_set_annotated")


def remove_too_long_jokes():
    ds = load_dataset("SeppeV/JokeTailor_big_set_annotated")

    def too_long_filter(r):
        return len(r["jokeText"]) < 800

    ds = ds.filter(too_long_filter)

    ds.push_to_hub("SeppeV/JokeTailor_big_set_annotated")

def deduplicate_dataset():
    df = load_dataset("SeppeV/JokeTailor_big_set_annotated", split="train").to_pandas()
    df = df.drop_duplicates("jokeText")
    ds = Dataset.from_pandas(df, preserve_index=False)
    ds.push_to_hub("SeppeV/JokeTailor_big_set_annotated")



deduplicate_dataset()
