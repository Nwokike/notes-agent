import sys
import os
import asyncio
import uuid
from dotenv import load_dotenv

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
from fastapi import FastAPI, Request
from google.adk.runners import Runner
from telegram import Update, Bot
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters, ContextTypes
from google.genai import types

load_dotenv()

# --- Agent Imports ---
from orchestrator.agent import root_agent

runner = Runner(
    app_name="igbo-notes-agent-hq",
    agent=root_agent
)

# --- State Management ---
active_sessions = {}

# --- Main Pipeline Execution ---
async def run_pipeline(update: Update, bot: Bot):
    chat_id = update.effective_chat.id
    msg_text = update.message.text.strip() if update.message.text else ""
    
    # Handle the /new command
    if msg_text.startswith("/new"):
        active_sessions[chat_id] = f"note_run_{uuid.uuid4().hex[:8]}"
        await bot.send_message(
            chat_id=chat_id, 
            text="🔄 Memory cleared. Send a message like 'Start' to begin the autonomous note-creation pipeline."
        )
        return

    # Ensure the user has an active session ID
    if chat_id not in active_sessions:
        active_sessions[chat_id] = f"note_run_{uuid.uuid4().hex[:8]}"
    
    current_session_id = active_sessions[chat_id]
    
    msg_content = types.Content(role="user", parts=[types.Part.from_text(text=msg_text)])
    
    try:
        # Execute the pipeline using the dynamic session ID
        async for event in runner.run_async(user_id=str(chat_id), session_id=current_session_id, new_message=msg_content):
            author = event.author
            
            # Send messages back to Telegram (ignore internal loops unless it has text)
            if author and author not in ["user", "system"]:
                event_text = ""
                if event.content and event.content.parts:
                    event_text = "".join([getattr(p, 'text', '') for p in event.content.parts if getattr(p, 'text', '')]).strip()

                if event_text:
                    await bot.send_message(
                        chat_id=chat_id, 
                        text=f"{author.upper()}:\n{event_text}"
                    )
                    
                    if author == "publisher" and "successfully published" in event_text.lower():
                        await bot.send_message(
                            chat_id=chat_id,
                            text="✅ Note creation and publication completed successfully!\nSend /new to start on a new archive."
                        )

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        if len(error_msg) > 4000:
            error_msg = error_msg[:4000] + "\n...[Error Truncated]"
        await bot.send_message(chat_id, error_msg)


# --- Webhook Mode (Render Web Service) ---
app = FastAPI()
bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
tg_bot = Bot(token=bot_token) if bot_token else None

@app.post("/webhook")
async def telegram_webhook(request: Request):
    payload = await request.json()
    update = Update.de_json(payload, tg_bot)
    
    if update.message and update.message.text:
        asyncio.create_task(run_pipeline(update, tg_bot))
            
    return {"status": "ok"}

@app.get("/")
def health():
    return {"status": "Notes Creation Agent is ACTIVE on Render", "mode": "Webhook"}

# --- Polling Mode (Local Dev) ---
async def handle_polling(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await run_pipeline(update, context.bot)

if __name__ == "__main__":
    if os.getenv("RENDER") or os.getenv("TELEGRAM_WEBHOOK_URL"):
        import uvicorn
        port = int(os.environ.get("PORT", 8080))
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        if bot_token:
            from telegram.request import HTTPXRequest
            print("Starting Telegram Bot (Polling Mode)...")
            request = HTTPXRequest(connect_timeout=30, read_timeout=30)
            tg_app = ApplicationBuilder().token(bot_token).request(request).build()
            
            tg_app.add_handler(CommandHandler("new", handle_polling))
            tg_app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_polling))
            
            tg_app.run_polling()
        else:
            print("CRITICAL: TELEGRAM_BOT_TOKEN not found.")
