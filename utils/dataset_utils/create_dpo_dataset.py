from datasets import load_dataset, Dataset
from dotenv import load_dotenv

load_dotenv()

class DPODatasetCreator():

    def __init__(self):
        self.result_ds = load_dataset("SeppeV/survey_results_test", split="train")
        self.joke_ds = load_dataset("SeppeV/JokeTailor_big_set_annotated", split="train").to_pandas()

    def create_dpo_ds(self, dataset_id):
        ds_dict = {"userId": [], "chosen": [], "rejected": []}
        userIds = self.result_ds["userId"]

        for index, userId in enumerate(userIds):
            for i in range(7):
                rankings = self.result_ds[index][f"ranking{i}"]
                ranked_jokes = [self.joke_ds.loc[self.joke_ds["jokeId"] == jokeId, "jokeText"].item() for jokeId in rankings]
                for joke in ranked_jokes[1:]:
                    ds_dict["userId"].append(userId)
                    ds_dict["chosen"].append([{"role": "assistant", "content": ranked_jokes[0]}])
                    ds_dict["rejected"].append([{"role": "assistant", "content": joke}])

        Dataset.from_dict(ds_dict).push_to_hub(dataset_id)


dpo = DPODatasetCreator()
dpo.create_dpo_ds("dpo_ds_test")