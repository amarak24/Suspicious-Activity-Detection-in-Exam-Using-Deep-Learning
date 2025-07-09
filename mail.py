import smtplib
from email.message import EmailMessage
import sys

# Sender & Receiver details
SENDER_EMAIL = "amarkalukhe20@gmail.com"
APP_PASSWORD = "dgco ugqn uxmc urts"
RECEIVER_EMAIL = "amarkalukhe24@gmail.com"

# Get timestamp and frame number from command-line arguments
timestamp = sys.argv[1] if len(sys.argv) > 1 else "Unknown"
frame_number = sys.argv[2] if len(sys.argv) > 2 else "Unknown"

print("Mail Start")

msg = EmailMessage()
msg['Subject'] = "Suspicious Activity Detection In Exam"
msg['From'] = SENDER_EMAIL
msg['To'] = RECEIVER_EMAIL

# Add timestamp and frame number to the email body
msg.set_content(f'''⚠️ Suspicious Activity Detected!

🕒 Timestamp: {timestamp}
🎞️ Frame Number: {frame_number}

Please check the recorded footage for more details.
''')

try:
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(SENDER_EMAIL, APP_PASSWORD)
        smtp.send_message(msg)
        print("Mail sent successfully!")
except Exception as e:
    print(f"Error sending email: {e}")






# import smtplib
# from email.message import EmailMessage
# import sqlite3

# SENDER_EMAIL = "amarkalukhe20@gmail.com"
# APP_PASSWORD = "dgco ugqn uxmc urts"
# RECEIVER_EMAIL = "amarkalukhe24@gmail.com"

# print("Mail Start")

# msg = EmailMessage()
# msg['Subject'] = "Suspicious Activity Detection In Exam"
# msg['From'] = SENDER_EMAIL
# msg['To'] =RECEIVER_EMAIL
# msg.set_content('Exam Suspicious Activity Detected')

# try:
#     with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
#         smtp.login(SENDER_EMAIL, APP_PASSWORD)
#         smtp.send_message(msg)
#         print("Mail sent successfully!")
# except Exception as e:
#     print(f"Error sending email: {e}")