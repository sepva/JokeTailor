import os
import smtplib
import ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from dotenv import load_dotenv
from get_not_filled_in_mails import get_not_filled_in_mail_addresses

load_dotenv()
OWN_EMAIL = os.getenv("EMAIL_ADDRESS")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD")


def send_email(userId, passcode, to_email):
    # Create the email message
    msg = MIMEMultipart()
    msg["From"] = OWN_EMAIL
    msg["To"] = to_email
    msg["Subject"] = (
        "Reminder second survey for JokeTailor (Thesis Seppe Vanswegenoven)"
    )

    body = f"""Hi! 
            
Here a little reminder to fill in my second survey!

You can access the survey by clicking on the following link: https://jokesurvey.me/second_survey/{userId}/{passcode}

If you have any questions or problems, please contact me at sepvanswe@gmail.com.
Thank you very much for your time and effort!"""

    # Attach the email body
    msg.attach(MIMEText(body, "plain"))

    # Set up the secure SSL context
    context = ssl.create_default_context()

    try:
        # Connect to the mail server using TLS encryption
        with smtplib.SMTP("relay.proximus.be", 587) as server:
            server.starttls(context=context)
            server.login(OWN_EMAIL, MAIL_PASSWORD)
            text = msg.as_string()
            resp = server.sendmail(OWN_EMAIL, to_email, text)

        print(resp)
        print(f"Email sent successfully to {to_email}!")
    except Exception as e:
        print(f"Error: {e}")


def send_not_answered():
    for row in get_not_filled_in_mail_addresses():
        email = row["e-mail"]
        passcode = row["passcode"]
        userId = row["id"]
        send_email(userId, passcode, email)


send_not_answered()
# send_email(0, "46519", "sepvanswe@gmail.com")
