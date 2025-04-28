import os

from datasets import load_dataset
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

load_dotenv()


def add_password(r):
    r["passcode"] = SEC_SURVEYS.find_one({"userId": int(r["id"])})["passcode"]
    return r


def get_not_filled_in_mail_addresses():
    id_to_email = load_dataset("SeppeV/email_to_id_final", split="train")
    filled_in_userIds = [row["userId"] for row in get_second_survey_results().find()]

    return id_to_email.filter(lambda r: r["id"] not in filled_in_userIds).map(
        add_password
    )


def get_filled_in_mail_addresses():
    id_to_email = load_dataset("SeppeV/email_to_id_final", split="train")
    filled_in_userIds = [row["userId"] for row in get_second_survey_results().find()]

    return id_to_email.filter(lambda r: r["id"] in filled_in_userIds)


def get_second_survey_results():
    try:
        mongosecret = os.getenv("MONGODB")
        uri = f"mongodb+srv://seppevanswegenoven:{mongosecret}@cluster0.p77cc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

        client = MongoClient(uri, server_api=ServerApi("1"))

        return client["JokeSurvey"]["secondsurveyresults"]
    except Exception as e:
        raise Exception("The following error occurred: ", e)


def get_second_surveys():
    try:
        mongosecret = os.getenv("MONGODB")
        uri = f"mongodb+srv://seppevanswegenoven:{mongosecret}@cluster0.p77cc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

        client = MongoClient(uri, server_api=ServerApi("1"))

        return client["JokeSurvey"]["secondsurveys"]
    except Exception as e:
        raise Exception("The following error occurred: ", e)


SEC_SURVEYS = get_second_surveys()
# not_filled_in = get_not_filled_in_mail_addresses().to_pandas()
# not_filled_in["link"] = (
#     "https://jokesurvey.me/second_survey/"
#     + not_filled_in["id"].astype(str)
#     + "/"
#     + not_filled_in["passcode"]
# )
# not_filled_in[["e-mail", "link"]].to_excel(
#     "not_filled_in.xlsx", index=False, header=["e-mail", "link"]
# )
print(get_not_filled_in_mail_addresses().to_pandas())
