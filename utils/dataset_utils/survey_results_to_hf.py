import json
import os

from datasets import Dataset
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

load_dotenv()


class SurveyResultDSCreator:

    def __init__(self):
        self.email_to_id_ds_id = "SeppeV/email_to_id_0304"
        self.result_ds_id = "SeppeV/survey_results_0304"
        self.results = self.get_results()

    def add_results(self):
        email_to_ids = {"e-mail": [], "id": []}
        results = {"userId": []}
        for idx, entrie in enumerate(self.results):
            email_to_ids["e-mail"].append(entrie["e-mail"])
            email_to_ids["id"].append(idx)

            results["userId"].append(idx)
            for key, item in entrie.items():
                if key != "e-mail":
                    if key in results:
                        results[key].append(item)
                    else:
                        results[key] = [item]

        Dataset.from_dict(email_to_ids).push_to_hub(self.email_to_id_ds_id)
        Dataset.from_dict(results).push_to_hub(self.result_ds_id)

    def get_results(self):
        try:
            mongosecret = os.getenv("MONGODB")
            uri = f"mongodb+srv://seppevanswegenoven:{mongosecret}@cluster0.p77cc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

            client = MongoClient(uri, server_api=ServerApi("1"))

            return [
                json.loads(doc["resultJson"])
                for doc in client["JokeSurvey"]["surveyresults"].find({})
            ]
        except Exception as e:
            raise Exception("The following error occurred: ", e)


sr = SurveyResultDSCreator()
sr.add_results()
