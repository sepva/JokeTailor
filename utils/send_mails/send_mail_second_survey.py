import os

import requests
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

load_dotenv()


def send_second_survey_mail(email, passcode, userId):
    return requests.post(
        "https://api.eu.mailgun.net/v3/jokesurvey.me/messages",
        auth=("api", os.getenv("MAIL", "API_KEY")),
        data={
            "from": "JokeTailor survey <postmaster@jokesurvey.me>",
            "to": f"<{email}>",
            "subject": "Second survey for JokeTailor",
            "text": f"""Hi! 
            
The time has come for the second part of the JokeTailor survey.
The survey will work exactly the same as the first one.
To get access to the survey, you will need to fill in your personal passcode.

Your passcode is: {passcode}
You can access the survey by clicking on the following link: https://jokesurvey.me/second_survey/{userId}

If you have any questions or problems, please contact me at sepvanswe@gmail.com.
Thank you very much for your time and effort!""",
        },
    )


def send_all_second_survey_mails():
    collection = get_collection()
    for survey in collection.find():
        try:
            email = survey["e-mail"]
            passcode = survey["passcode"]
            userId = survey["userId"]
            send_second_survey_mail(email, passcode, userId)
            print("Mail sent to: ", email)
        except Exception as e:
            print(
                "The following error occurred: ",
                e,
                "while sending the mail to: ",
                email,
            )


def get_collection():
    try:
        mongosecret = os.getenv("MONGODB")
        uri = f"mongodb+srv://seppevanswegenoven:{mongosecret}@cluster0.p77cc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

        client = MongoClient(uri, server_api=ServerApi("1"))

        return client["JokeSurvey"]["secondsurveys"]
    except Exception as e:
        raise Exception("The following error occurred: ", e)


send_all_second_survey_mails()
