from datasets import load_dataset
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
import json
from dotenv import load_dotenv

load_dotenv()



class GenerationSurveyCreator():

    def __init__(self, generation_ds_ids, distribution_of_jokes_per_ds_per_ranking, num_rankings):
        self.generation_dfs = [load_dataset(generation_ds_id, split="train").to_pandas() for generation_ds_id in generation_ds_ids]
        self.distribution_of_jokes_per_ds_per_ranking = distribution_of_jokes_per_ds_per_ranking
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
            survey = self.create_survey(userId)
            surveys.append(
            {
                "userId": userId,
                "survey": json.dumps(survey)
            }
        )
        
        if self.insert_surveys(surveys):
            print("Succes!")


    def create_survey(self, userId): 
        indices = [0]*len(self.distribution_of_jokes_per_ds_per_ranking)
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
                }
            ],
        },
    ]
        for index in range(self.num_rankings):
            jokes_to_rank = []
            jokeIds_to_rank = []
            for i, generation_df in enumerate(self.generation_dfs):
                start = indices[i]
                amount_of_jokes = self.distribution_of_jokes_per_ds_per_ranking[i][index]
                next_rows = generation_df.loc[generation_df["userId"] == userId].iloc[start:start+amount_of_jokes]
                jokes_to_rank.extend(next_rows["jokeText"].to_list())
                jokeIds_to_rank.extend(next_rows["jokeId"].to_list())
                indices[i] += amount_of_jokes
            
            pages.append(self.create_ranking_question(jokes_to_rank, jokeIds_to_rank, index))

        return {
                "title": "JokeTailor",
                "logoPosition": "right",
                "pages": pages,
            }

    def insert_surveys(self, surveys):
        collection = self.get_collection()
        return collection.insert_many(surveys)
    
    def get_collection(self):
        try:
            mongosecret = os.getenv("MONGODB")
            uri = f"mongodb+srv://seppevanswegenoven:{mongosecret}@cluster0.p77cc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

            client = MongoClient(uri, server_api=ServerApi("1"))

            return client["JokeSurvey"]["second_surveys"]
        except Exception as e:
            raise Exception("The following error occurred: ", e)
        
gsc = GenerationSurveyCreator(["SeppeV/RecRag_inference", "SeppeV/Human_jokes_test"], [[1,1,1,1,1,0,0], [1]*7], 7)
gsc.create_surveys()