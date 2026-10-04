"""Core logic: parsing, local-LLM extraction (Ollama), ranking, attendance math, ICS export.
Everything runs on the student's own laptop - no cloud calls."""
import re, json, math, time, base64, hashlib, datetime as dt
import requests

OLLAMA = "http://localhost:11434"
TYPES = {"deadline", "exam", "notice", "internship", "event", "fee"}

# ---------------- Attendance (plain math, never the LLM) ----------------
def attendance_status(attended, total, req=75):
    pct = 100 * attended / total if total else 0.0
    if pct < req:
        need = math.ceil((req / 100 * total - attended) / (1 - req / 100) - 1e-9)
        return {"pct": pct, "state": "danger", "need": need, "skip": 0}
    skip = max(0, math.floor(attended * 100 / req - total + 1e-9))
    return {"pct": pct, "state": "safe" if skip > 1 else "edge", "need": 0, "skip": skip}

def after_skipping(attended, total, n):
    return 100 * attended / (total + n) if total + n else 0.0

# ---------------- WhatsApp export parsing ----------------
LINE = re.compile(
    r"^\[?(\d{1,2}[/.]\d{1,2}[/.]\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?\s?(?:[APap][Mm])?)\]?\s*-?\s*([^:]{1,60}?):\s(.*)$")
KEYWORDS = re.compile(
    r"deadline|last date|submit|submission|exam|test|quiz|viva|mid.?sem|end.?sem|notice|circular|internship|"
    r"intern\b|hiring|placement|apply|register|registration|form|fee|assignment|lab|attendance|due|tomorrow|today|"
    r"kal|aaj|reminder|urgent|important|hackathon|workshop|seminar|webinar|result|admit card|http|www\.", re.I)

def parse_date(s):
    s = s.strip().replace(".", "/")
    for f in ("%d/%m/%y", "%d/%m/%Y", "%m/%d/%y", "%m/%d/%Y"):
        try:
            return dt.datetime.strptime(s, f).date()
        except ValueError:
            pass
    return None

def parse_chat(text):
    msgs = []
    for line in text.splitlines():
        m = LINE.match(line.strip())
        if m:
            msgs.append({"date": parse_date(m.group(1)), "sender": m.group(3), "text": m.group(4)})
        elif msgs and line.strip():
            msgs[-1]["text"] += " " + line.strip()
    return [m for m in msgs if "omitted" not in m["text"].lower()]

def relevant_messages(msgs, last_days=45, only_keywords=True):
    cutoff = dt.date.today() - dt.timedelta(days=last_days)
    out = [m for m in msgs if (m["date"] is None or m["date"] >= cutoff)]
    return [m for m in out if KEYWORDS.search(m["text"])] if only_keywords else out

# ---------------- Files -> text ----------------
def pdf_to_text(file):
    try:
        import pdfplumber
        with pdfplumber.open(file) as pdf:
            return "\n".join((p.extract_text() or "") for p in pdf.pages)
    except Exception:
        from pypdf import PdfReader
        file.seek(0)
        return "\n".join((p.extract_text() or "") for p in PdfReader(file).pages)

# ---------------- LLM layer: Groq cloud (if api_key) or local Ollama ----------------
GROQ = "https://api.groq.com/openai/v1"
_SKIP = ("whisper", "guard", "safeguard", "tts", "orpheus", "playai", "embed")

def list_groq_models(api_key):
    """Live list of chat models on this key's account (Groq retires models often)."""
    try:
        r = requests.get(f"{GROQ}/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
        ids = sorted(m["id"] for m in r.json()["data"] if m.get("active", True))
        return [i for i in ids if not any(x in i.lower() for x in _SKIP)]
    except Exception:
        return []

def pick_default(models):
    for pref in ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b"):
        if pref in models:
            return pref
    return models[0] if models else "openai/gpt-oss-120b"

def list_models():  # local Ollama models
    try:
        return [m["name"] for m in requests.get(f"{OLLAMA}/api/tags", timeout=3).json()["models"]]
    except Exception:
        return []

def _groq_chat(api_key, model, messages, as_json, retries=4):
    body = {"model": model, "messages": messages, "temperature": 0}
    if as_json:
        body["response_format"] = {"type": "json_object"}
    for attempt in range(retries):
        r = requests.post(f"{GROQ}/chat/completions", json=body, timeout=120,
                          headers={"Authorization": f"Bearer {api_key}"})
        if r.status_code == 429:                      # free-tier rate limit: wait and retry
            time.sleep(min(3 * 2 ** attempt, 20)); continue
        if r.status_code == 400 and "response_format" in body:
            body.pop("response_format"); continue     # model without JSON mode: ask in prompt
        if r.status_code >= 400:
            raise RuntimeError(f"Groq error {r.status_code} for model '{model}': {r.text[:300]}")
        return r.json()["choices"][0]["message"]["content"] or ""
    raise RuntimeError("Groq rate limit hit. Wait a minute and press Analyze again.")

def _generate(model, prompt, as_json=True, api_key=None):
    if api_key:
        out = _groq_chat(api_key, model, [{"role": "user", "content": prompt}], as_json)
        return re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
    body = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}}
    if as_json:
        body["format"] = "json"
    return requests.post(f"{OLLAMA}/api/generate", json=body, timeout=600).json()["response"]

def _json_from(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}

def image_to_text(data, vision_model="", api_key=None, **kwargs):
    """Read a notice screenshot with a vision model. Returns '' if no vision model works."""
    if not vision_model:
        return ""
    prompt = "Transcribe all text in this college notice/timetable image exactly. Keep dates and links."
    try:
        if api_key:
            mime = "image/jpeg" if data[:2] == b"\xff\xd8" else "image/png"
            uri = f"data:{mime};base64,{base64.b64encode(data).decode()}"
            msg = [{"role": "user", "content": [{"type": "text", "text": prompt},
                                               {"type": "image_url", "image_url": {"url": uri}}]}]
            return _groq_chat(api_key, vision_model, msg, as_json=False)
        r = requests.post(f"{OLLAMA}/api/generate", json={
            "model": vision_model, "stream": False, "prompt": prompt,
            "images": [base64.b64encode(data).decode()]}, timeout=300)
        return r.json().get("response", "")
    except Exception:
        return ""

EXTRACT = """You help a college student. Today is {today} ({weekday}).
Student: {branch}, semester {sem}. Subjects: {subjects}. Interests: {interests}.
The text may be Hindi/Hinglish. Resolve words like 'kal', 'tomorrow', 'Monday' relative to the MESSAGE DATE.

From the text below extract every item that needs the student's attention.
Return JSON: {{"items":[{{"type":"deadline|exam|notice|internship|event|fee",
"title":"short title","date":"YYYY-MM-DD or null","time":"HH:MM or null","subject":"subject or null",
"action":"one clear sentence: what the student should DO","link":"url or null",
"relevant":true/false (false only if clearly for another branch/year),"source":"who said it"}}]}}
Do not invent dates or links. If nothing needs action return {{"items":[]}}.

TEXT:
{chunk}
"""

def _norm_date(s):
    try:
        return dt.date.fromisoformat(s).isoformat()
    except Exception:
        return None

def extract_items(blocks, profile, model, progress=None, api_key=None, **kwargs):
    """blocks: list of text strings (chat chunks, PDF text, OCR text)."""
    items, seen, errors = [], set(), []
    blocks = [b for b in blocks if b and b.strip()]
    for i, chunk in enumerate(blocks):
        if progress:
            progress((i + 1) / len(blocks))
        prompt = EXTRACT.format(
            today=dt.date.today().isoformat(), weekday=dt.date.today().strftime("%A"),
            branch=profile.get("branch", "unknown"), sem=profile.get("sem", "?"),
            subjects=", ".join(profile["subjects"]), interests=", ".join(profile.get("interests", [])), chunk=chunk)
        try:
            raw = _json_from(_generate(model, prompt, api_key=api_key))
        except Exception as e:
            errors.append(str(e)); continue
        for it in raw.get("items", []) if isinstance(raw, dict) else []:
            if not isinstance(it, dict) or not it.get("title") or it.get("relevant") is False:
                continue
            it["type"] = it.get("type") if it.get("type") in TYPES else "notice"
            it["date"] = _norm_date(it.get("date"))
            links = re.findall(r"https?://\S+", chunk)
            if not it.get("link") and links and it["type"] in ("internship", "event", "deadline"):
                it["link"] = links[0].rstrip(").,")
            key = hashlib.md5((it["title"].lower() + str(it["date"])).encode()).hexdigest()
            if key not in seen:
                seen.add(key)
                items.append(it)
    if errors and not items:
        raise RuntimeError(errors[0])
    return items

def chunk_messages(msgs, size=40):
    lines = [f'[{m["date"]}] {m["sender"]}: {m["text"]}' for m in msgs]
    return ["\n".join(lines[i:i + size]) for i in range(0, len(lines), size)]

# ---------------- Ranking ----------------
def days_left(it):
    try:
        return (dt.date.fromisoformat(it["date"]) - dt.date.today()).days
    except Exception:
        return None

def score(it, profile):
    d = days_left(it)
    if d is None: s = 10
    elif d < 0: return -1
    elif d == 0: s = 100
    elif d == 1: s = 90
    elif d <= 3: s = 70
    elif d <= 7: s = 50
    elif d <= 14: s = 30
    else: s = 15
    subs = [x.lower() for x in profile["subjects"]]
    text = (it.get("title", "") + " " + str(it.get("subject") or "")).lower()
    if any(x in text for x in subs): s += 25
    if any(x.lower() in text for x in profile.get("interests", []) if x): s += 20
    if it["type"] == "exam": s += 15
    if it["type"] == "fee": s += 10
    return s

def rank(items, profile):
    for it in items:
        it["score"], it["days"] = score(it, profile), days_left(it)
    return sorted([i for i in items if i["score"] >= 0], key=lambda x: -x["score"])

def attendance_actions(att_rows, req):
    out = []
    for r in att_rows:
        st = attendance_status(r["attended"], r["total"], req)
        if st["state"] == "danger":
            out.append((0, f"⚠️ {r['subject']} attendance {st['pct']:.0f}% (need {req}%). Attend the next {st['need']} classes without a miss."))
        elif st["state"] == "edge":
            out.append((1, f"🟡 {r['subject']} at {st['pct']:.0f}% - only {st['skip']} safe bunk left. Be careful."))
    return [t for _, t in sorted(out)]

# ---------------- LLM extras ----------------
def morning_brief(items, alerts, profile, model, api_key=None, **kwargs):
    top = [{k: i.get(k) for k in ("type", "title", "date", "action")} for i in items[:6]]
    p = (f"Write a short friendly Hinglish morning message (max 6 lines, with emojis) for {profile['name']}, "
         f"telling what to do today. Use ONLY this data.\nAttendance alerts: {alerts}\nItems: {json.dumps(top)}")
    return _generate(model, p, as_json=False, api_key=api_key)

def ask(question, items, alerts, profile, model, api_key=None, **kwargs):
    ctx = json.dumps([{k: i.get(k) for k in ("type", "title", "date", "subject", "action", "link")} for i in items])
    p = (f"You are a private college assistant for {profile['name']}. Today is {dt.date.today()}. "
         f"Answer in the language of the question (Hinglish ok), briefly, using ONLY this data. "
         f"If the data does not contain the answer say so.\nAttendance alerts: {alerts}\nItems: {ctx}\n\nQuestion: {question}")
    return _generate(model, p, as_json=False, api_key=api_key)

# ---------------- Calendar export ----------------
def to_ics(items):
    out = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//CollegeCopilot//EN"]
    for i in items:
        if not i.get("date"):
            continue
        d = i["date"].replace("-", "")
        uid = hashlib.md5((i["title"] + i["date"]).encode()).hexdigest()
        title = i["title"].replace(",", " ").replace("\n", " ")
        out += ["BEGIN:VEVENT", f"UID:{uid}@copilot", f"DTSTAMP:{dt.datetime.utcnow():%Y%m%dT%H%M%SZ}",
                f"DTSTART;VALUE=DATE:{d}", f"SUMMARY:[{i['type'].upper()}] {title}",
                f"DESCRIPTION:{(i.get('action') or '').replace(chr(10), ' ')}", "END:VEVENT"]
    out.append("END:VCALENDAR")
    return "\r\n".join(out)
