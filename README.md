# 🎓 College Copilot - private, offline AI for one student

Reads scattered college info (WhatsApp exports, notice PDFs, screenshots), matches it to the student's
subjects/attendance/interests and answers: **"What matters for ME right now and what should I do?"**

## Run
1. Install [Ollama](https://ollama.com), then: `ollama pull qwen2.5:7b` (optional vision: `ollama pull qwen2.5vl:7b`)
2. `pip install -r requirements.txt`
3. `streamlit run app.py`
4. Upload `sample_chat.txt` from the sidebar to test, then your friend's real chat.

## Features
- WhatsApp chat parser (Android + iOS formats), Hinglish-aware extraction ("kal", "aaj")
- Notice PDFs + screenshots (local vision model)
- Priority ranking by urgency x your subjects x your interests; other-branch notices are dropped
- Attendance planner: classes you must attend, safe bunks left, bunk simulator (pure code, no LLM math)
- Internship tab with links, Ask-anything chat (Hinglish), ready-to-send morning WhatsApp message
- Calendar (.ics) export, profile saved locally

## Why open source
Private chats and attendance never leave the laptop, works with no internet, model is swappable, costs nothing.
