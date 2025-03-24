from datasets import load_dataset, Dataset
from dotenv import load_dotenv

load_dotenv()

class RatedDatasetCreator():

    def __init__(self):
        self.result_ds = load_dataset("SeppeV/survey_results_test", split="train")
        self.joke_ds = load_dataset("SeppeV/JokeTailor_big_set_annotated", split="train").to_pandas()
        self.num_jokes_per_survey = 28

    def create_rated_ds(self, ds_id):
        ds_dict = {"userId": [], "jokeText": [], "rating": []}
        userIds = self.result_ds["userId"]
        for index, userId in enumerate(userIds):
            last_ranking = self.result_ds[index]["ranking7"] 
            ds_dict["userId"].extend([userId]*self.num_jokes_per_survey)
            ds_dict["jokeText"].extend([self.joke_ds.loc[self.joke_ds["jokeId"] == jokeId, "jokeText"].item() for jokeId in last_ranking])
            ds_dict["rating"].extend(range(10, 3,-1))
            for i in range(7):
                rankings = self.result_ds[index][f"ranking{i}"][1:]
                ds_dict["jokeText"].extend([self.joke_ds.loc[self.joke_ds["jokeId"] == jokeId, "jokeText"].item() for jokeId in rankings])
                ds_dict["rating"].extend([3, 2, 1])
            
        Dataset.from_dict(ds_dict).push_to_hub(ds_id)

rd = RatedDatasetCreator()
rd.create_rated_ds("rated_ds_test")