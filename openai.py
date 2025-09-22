import os, json, glob
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")
genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-2.5-flash"

def load_persona_samples(folder="persona_samples", max_samples=8):
    shots = []
    for path in sorted(glob.glob(os.path.join(folder, "*.jsonl")), reverse=True):  # recent first
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    if "user" in obj and "you" in obj:
                        shots.append(obj)
                        if len(shots) >= max_samples:
                            return shots
                except json.JSONDecodeError:
                    pass
    return shots

def build_context(chat_text: str, limit_lines: int = 14) -> str:
    lines = [ln.strip() for ln in chat_text.splitlines() if ln.strip()]
    return "\n".join(lines[-limit_lines:])

def build_prompt(chat_context: str, shots, mode="reply", last_bot_reply=""):
    """
    mode: 'reply' for normal response; 'followup' to extend previous reply only (no repetition).
    """
    sys = (
        "You are Vaibhav, an Indian coder and B.Tech student with a casual Hinglish tone. You use emojis only when contextually appropriate and can make light, friendly roasts or puns."
        "Your primary objective is to act as a chatbot and generate a single, concise, and natural reply for a WhatsApp-style conversation."
        "The conversation context will be formatted with clear speaker labels: [Other_Person]: and [You]:. Your reply must be based only on the most recent message from [Other_Person]."
        "You have real-time awareness of the current date and time. Use this information to make your replies more relevant. If names are mentioned in the conversation, use them to personalize your replies. Do not repeat past content unless specifically asked. Your response must be a single message without any name prefixes."
    )

    fewshot = "\n\nExamples:\n"
    for s in shots:
        fewshot += f"User says: {s['user']}\nYou reply: {s['you']}\n---\n"

    followup_note = ""
    if mode == "followup":
        followup_note = (
            "\nThe previous reply felt insufficient. Continue the same thought briefly, "
            "adding missing clarity or value. Do NOT repeat the previous text verbatim. "
            f"Previous reply was:\n{last_bot_reply}\n---\n"
        )

    return f"{sys}\n{fewshot}{followup_note}\nChat context:\n{chat_context}\nYour next reply:"

def ask_gemini(chat_text: str, persona_folder="persona_samples",
               mode="reply", last_bot_reply="") -> str:
    shots = load_persona_samples(persona_folder)
    context = build_context(chat_text)
    prompt = build_prompt(context, shots, mode=mode, last_bot_reply=last_bot_reply)
    model = genai.GenerativeModel(MODEL_NAME)
    try:
        resp = model.generate_content(prompt)
        return (resp.text or "").strip()
    except Exception as e:
        return f"[Gemini error: {e}]"