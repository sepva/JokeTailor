import json
import os
import secrets

from datasets import load_dataset
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

load_dotenv()


class GenerationSurveyCreator:

    def __init__(
        self,
        generation_ds_ids,
        email_to_userId_ds_id,
        distribution_of_jokes_per_ds_per_ranking,
        random_gen_ds_ids,
        num_random_jokes_per_ranking,
        num_rankings,
    ):
        self.email_to_userId = load_dataset(
            email_to_userId_ds_id, split="train"
        ).to_pandas()
        self.generation_dfs = [
            load_dataset(generation_ds_id, split="train").to_pandas()
            for generation_ds_id in generation_ds_ids
        ]
        self.distribution_of_jokes_per_ds_per_ranking = (
            distribution_of_jokes_per_ds_per_ranking
        )

        self.random_gen_dfs = [
            load_dataset(random_gen_ds_id, split="train").to_pandas()
            for random_gen_ds_id in random_gen_ds_ids
        ]

        self.num_random_jokes_per_ranking = num_random_jokes_per_ranking

        self.num_rankings = num_rankings

    def create_ranking_question(self, jokes, jokeIds, index):
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

    def create_surveys(self):
        userIds = self.generation_dfs[0]["userId"].drop_duplicates().to_list()
        surveys = []

        for userId in userIds:
            survey, passcode = self.create_survey(userId)
            email = self.email_to_userId.loc[self.email_to_userId["id"] == userId][
                "e-mail"
            ].values[0]
            surveys.append(
                {
                    "userId": userId,
                    "e-mail": email,
                    "survey": json.dumps(survey),
                    "passcode": passcode,
                }
            )

        if self.insert_surveys(surveys):
            print("Succes!")

    def create_survey(self, userId):
        indices = [0] * len(self.distribution_of_jokes_per_ds_per_ranking)
        passcode = secrets.token_urlsafe(8)
        pages = [
            {
                "name": "page1",
                "elements": [
                    {
                        "type": "text",
                        "name": "passcode",
                        "width": "100%",
                        "minWidth": "256px",
                        "title": "Personal passcode:",
                        "isRequired": True,
                        "validators": [
                            {
                                "type": "expression",
                                "text": "Wrong passcode...",
                                "expression": "{passcode} = '" + passcode + "'",
                            }
                        ],
                    }
                ],
            },
        ]
        for index in range(self.num_rankings):
            jokes_to_rank = []
            jokeIds_to_rank = []
            for i, generation_df in enumerate(self.generation_dfs):
                start = indices[i]
                amount_of_jokes = self.distribution_of_jokes_per_ds_per_ranking[i][
                    index
                ]
                next_rows = generation_df.loc[generation_df["userId"] == userId].iloc[
                    start : start + amount_of_jokes
                ]
                jokes_to_rank.extend(next_rows["jokeText"].to_list())
                jokeIds_to_rank.extend(next_rows["jokeId"].to_list())
                indices[i] += amount_of_jokes

            for i, random_gen_df in enumerate(self.random_gen_dfs):
                amount_of_jokes = self.num_random_jokes_per_ranking[i][index]
                next_rows = random_gen_df.sample(amount_of_jokes)
                jokes_to_rank.extend(next_rows["jokeText"].to_list())
                jokeIds_to_rank.extend(next_rows["jokeId"].to_list())

            pages.append(
                self.create_ranking_question(jokes_to_rank, jokeIds_to_rank, index)
            )

        return {
            "title": "JokeTailor",
            "logoPosition": "right",
            "pages": pages,
        }, passcode

    def insert_surveys(self, surveys):
        collection = self.get_collection()
        return collection.insert_many(surveys)

    def get_collection(self):
        try:
            mongosecret = os.getenv("MONGODB")
            uri = f"mongodb+srv://seppevanswegenoven:{mongosecret}@cluster0.p77cc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

            client = MongoClient(uri, server_api=ServerApi("1"))

            return client["JokeSurvey"]["secondsurveys"]
        except Exception as e:
            raise Exception("The following error occurred: ", e)


gsc = GenerationSurveyCreator(
    [
        "SeppeV/FullGen_for_survey",
        "SeppeV/OnlyRAG_for_survey",
        "SeppeV/OnlyFT_for_survey",
        "SeppeV/FullMinusBon_for_survey",
    ],
    "SeppeV/email_to_id_test",
    [
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 1, 0, 1],
    ],
    [
        "SeppeV/SFT_inference_survey",
        "SeppeV/base_model_inference_survey",
        "SeppeV/Human_jokes",
        "SeppeV/RandomUID_for_survey",
    ],
    [
        [0, 0, 0, 1, 0, 0, 1, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 1, 0, 1, 1, 0, 0],
    ],
    10,
)
gsc.create_surveys()
