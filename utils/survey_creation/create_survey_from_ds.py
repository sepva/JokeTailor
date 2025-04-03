import json
import os
import random

from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sentence_transformers import SentenceTransformer

load_dotenv()


class SurveyCreator:

    def __init__(
        self, ds_id, survey_method, nr_of_surveys, nr_of_jokes_per_survey, joke_column
    ):
        self.ds = load_dataset(ds_id, split="train")
        self.collection = self.get_collection()
        self.survey_method = survey_method
        self.nr_of_surveys = nr_of_surveys
        self.nr_of_jokes_per_survey = nr_of_jokes_per_survey
        self.joke_column = joke_column

    def make_surveys_and_push(self):
        surveys = self.survey_method(
            self.ds, self.nr_of_surveys, self.nr_of_jokes_per_survey, self.joke_column
        )
        if self.insert_surveys(surveys):
            print("Succeeded!")

    def get_collection(self):
        try:
            mongosecret = os.getenv("MONGODB")
            uri = f"mongodb+srv://seppevanswegenoven:{mongosecret}@cluster0.p77cc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

            client = MongoClient(uri, server_api=ServerApi("1"))

            return client["JokeSurvey"]["surveys"]
        except Exception as e:
            raise Exception("The following error occurred: ", e)

    def insert_surveys(self, surveys):
        return self.collection.insert_many(surveys)


def create_ranking_question(jokes, jokeIds, index):
    return {
        "name": f"page{index+2}",
        "elements": [
            {
                "type": "ranking",
                "name": f"ranking{index}",
                "title": "Rank (drag-and-drop) the following jokes from most to least funny.",
                "isRequired": True,
                "choices": [
                    {"value": jokeId, "text": joke}
                    for jokeId, joke in zip(jokeIds, jokes)
                ],
            }
        ],
    }


def create_last_rank_choice(jokes, jokeIds, index):
    output = []
    for jokeId, joke in zip(jokeIds, jokes):
        output.append(
            {
                "value": jokeId,
                "text": joke,
                "visibleIf": "{ranking" + str(index) + "[0]} = '" + str(jokeId) + "'",
            }
        )
    return output


def too_similar(new_joke, jokes_in_ranking, jokes_in_survey, model):
    if new_joke is None:
        return True

    if not jokes_in_ranking:
        return False

    ranking_sim_limit = 0.15
    survey_sim_limit = 0.3

    sims_ranking = model.similarity([new_joke], jokes_in_ranking)
    r = any(sims_ranking[0] > ranking_sim_limit)

    if jokes_in_survey:
        sims_survey = model.similarity([new_joke], jokes_in_survey)
        return r and any(sims_survey[0] > survey_sim_limit)

    return r


def topic_dependend_survey(
    ds: Dataset,
    nr_of_surveys,
    nr_of_jokes_per_survey,
    joke_column,
):
    topic_column = "broad_topic"
    num_jokes_per_rank = 4
    nr_ranks = nr_of_jokes_per_survey // num_jokes_per_rank
    topics = list(set(ds[topic_column]))
    all_embeddings = []
    model = SentenceTransformer("all-roberta-large-v1")

    surveys = []
    for _ in range(nr_of_surveys):
        print("Creating survey...")
        pages = [
            {
                "name": "page1",
                "elements": [
                    {
                        "type": "text",
                        "name": "e-mail",
                        "width": "100%",
                        "minWidth": "256px",
                        "title": "Email address:",
                        "inputType": "email",
                        "isRequired": True,
                    }
                ],
            },
        ]
        last_ranking_choices = []
        for i in range(nr_ranks):
            jokes_to_rank = []
            jokeIds = []
            joke_embeddings = []

            for _ in range(num_jokes_per_rank):
                random_embedding = None
                while too_similar(
                    random_embedding, joke_embeddings, all_embeddings, model
                ):
                    topic = random.choice(topics)
                    jokes_topic = ds.filter(lambda j: j[topic_column] == topic)
                    random_joke_row = jokes_topic.select(
                        [random.randint(0, len(jokes_topic) - 1)]
                    )
                    random_joke = random_joke_row[joke_column][0]
                    random_jokeId = random_joke_row["jokeId"][0]
                    random_embedding = random_joke_row["embedding"][0]

                jokes_to_rank.append(random_joke)
                jokeIds.append(random_jokeId)
                joke_embeddings.append(random_embedding)

            all_embeddings.extend(joke_embeddings)
            pages.append(create_ranking_question(jokes_to_rank, jokeIds, i))
            last_ranking_choices.extend(
                create_last_rank_choice(jokes_to_rank, jokeIds, i)
            )

        pages.append(
            {
                "name": "page9",
                "elements": [
                    {
                        "type": "ranking",
                        "name": f"ranking{i+1}",
                        "title": "Rank (drag-and-drop) the following jokes from most to least funny.",
                        "isRequired": True,
                        "choices": last_ranking_choices,
                    }
                ],
            }
        )

        surveys.append(
            {
                "json": json.dumps(
                    {
                        "title": "JokeTailor",
                        "logoPosition": "right",
                        "pages": pages,
                    }
                )
            }
        )

    return surveys


if __name__ == "__main__":
    sc = SurveyCreator(
        "SeppeV/JokeTailor_big_set_annotated",
        topic_dependend_survey,
        10,
        28,
        "jokeText",
    )
    sc.make_surveys_and_push()
