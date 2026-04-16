import os
import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_URL = input("Enter your public URL (e.g., https://xxxx.ngrok-free.app/webhook): ")

if not TOKEN:
    print("Error: TELEGRAM_BOT_TOKEN not found in .env")
    exit(1)

if not WEBHOOK_URL.endswith("/webhook"):
    WEBHOOK_URL += "/webhook"

print(f"Setting webhook to: {WEBHOOK_URL}")
url = f"https://api.telegram.org/bot{TOKEN}/setWebhook?url={WEBHOOK_URL}"

response = requests.get(url)
print(response.json())
