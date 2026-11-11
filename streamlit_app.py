# SAIME — Streamlit version (fully free hosting: Streamlit Community Cloud)
# Model: SATYA V1, step-22000, 49.6M params — Hugging Face Hub se load hota hai
# (repo: satyamkumar78131/saime-model — free public model repo).
#
# Features (Gradio wale jaisi hi):
#   - Chat yaad rakhna (session me), FAQ lookup, few-shot prompt
#   - Web search FIRST for factual questions (30+ sources, source logo ke saath)
#   - Gibberish detection — model bakwas de to web se jawab
#   - Voice: jawab ko awaaz me suno (gTTS)
#   - Ideas collection: chat me idea aaye to save hota hai
#
# Honest limits (free tier):
#   - Photo padhna (BLIP) is version me NAHI hai — Streamlit free tier me
#     sirf 1 GB RAM milti hai, BLIP usme fit nahi hota.
#   - Chat history / learned data sirf app chalte hue save hoti hai (server
#     restart hone par khatm) — Streamlit Cloud ka disk temporary hota hai.
#   - Koi use na kare to app so jata hai, kholte hi 1-2 min me jaag jata hai
#     (UptimeRobot ping se ise jagaye rakha ja sakta hai).

import os
import re
import html as html_lib
import urllib.request
import urllib.parse
import json

import streamlit as st
import torch

from satya_v1_train import CharTokenizer, SatyaV1, count_params

MODEL_REPO = "satyamkumar78131/saime-model"
CHECKPOINT_NAME = "step-22000.pt"
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 40  # quick answer — chhota aur tez jawab

st.set_page_config(page_title="SAIME", page_icon="saime-logo.png", layout="centered")

# Streamlit ka footer/menu chhupao — "Hosted with Streamlit" badge, menu aur
# upar wala bar na dikhe, app bilkul apna lage.
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

torch.set_num_threads(2)


@st.cache_resource(show_spinner="SAIME load ho raha hai... (pehli baar model download hoga, 2-3 min)")
def load_model():
    """Model ko Hugging Face Hub se download karke load karo (ek baar)."""
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(repo_id=MODEL_REPO)
    ckpt_path = os.path.join(local_dir, CHECKPOINT_NAME)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt["config"]
    tok = CharTokenizer.load(os.path.join(local_dir, "tokenizer.json"))
    model = SatyaV1(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        d_ff=config["d_ff"],
        block_size=config["block_size"],
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tok, ckpt.get("step")


def web_search(question):
    """Kahi se bhi check karke jawab dhoondho — 30+ sources!
    DuckDuckGo, Wikipedia, Britannica, GeeksforGeeks, Stack Overflow, BBC,
    Reuters, GitHub, aur bhi. Jawab ke saath ASLI source ka naam milta hai.
    Wapas deta hai: (jawab, source_ka_naam)."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    # 1. DuckDuckGo instant answer (free, no key)
    try:
        q = urllib.parse.quote(question)
        url = "https://api.duckduckgo.com/?q=" + q + "&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
        abstract = (data.get("AbstractText") or "").strip()
        if abstract:
            return abstract, "DuckDuckGo"
    except Exception as e:
        print("Web search (instant) error:", e)
    # 2. DuckDuckGo Lite — har result ka ASLI source + snippet
    try:
        q = urllib.parse.quote(question)
        url = "https://lite.duckduckgo.com/lite/?q=" + q
        req3 = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req3, timeout=12) as r3:
            page = r3.read().decode("utf-8", errors="ignore")
        uddgs = re.findall(r"uddg=([^&\"']+)", page)
        snippets = re.findall(r"class='result-snippet'[^>]*>(.*?)</td>", page, re.DOTALL)
        for i, s in enumerate(snippets):
            text = re.sub(r"<[^>]+>", " ", s)
            text = html_lib.unescape(text).strip()
            text = re.sub(r"\s+", " ", text)
            if len(text) > 25:
                domain = "duckduckgo.com"
                if i < len(uddgs):
                    try:
                        real_url = urllib.parse.unquote(uddgs[i])
                        domain = urllib.parse.urlparse(real_url).netloc or domain
                        domain = domain.replace("www.", "")
                    except Exception:
                        pass
                return text, domain
    except Exception as e:
        print("Web search (lite) error:", e)
    # 3. Wikipedia search API — last resort, relevance check ke saath
    try:
        q = urllib.parse.quote(question)
        url = ("https://en.wikipedia.org/w/api.php?action=query&list=search"
               "&srsearch=" + q + "&format=json&srlimit=1")
        req2 = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req2, timeout=10) as r2:
            wdata = json.loads(r2.read().decode("utf-8"))
        results = wdata.get("query", {}).get("search", [])
        if results:
            title = results[0]["title"]
            stop = {"what", "is", "the", "a", "an", "how", "why", "when", "where",
                    "kaunsa", "kaun", "kya", "hai", "hota", "karte", "kitne", "par"}
            qwords = set(re.findall(r"[a-zA-Z\u0900-\u097F]+", question.lower())) - stop
            twords = set(re.findall(r"[a-zA-Z\u0900-\u097F]+", title.lower()))
            if qwords and not (qwords & twords):
                return None, None
            sum_url = ("https://en.wikipedia.org/api/rest_v1/page/summary/"
                       + urllib.parse.quote(title))
            req2b = urllib.request.Request(sum_url, headers=headers)
            with urllib.request.urlopen(req2b, timeout=10) as r2b:
                sdata = json.loads(r2b.read().decode("utf-8"))
            summary = (sdata.get("extract") or "").strip()
            if summary:
                return summary, "Wikipedia"
    except Exception as e:
        print("Web search (wiki) error:", e)
    return None, None


FAQ = {
    "hello": "Hello! Main SAIME hoon. Tumhara sawal poochho, main jawab dunga!",
    "hi": "Hi! Main SAIME hoon. Kya poochhna hai?",
    "hii": "Hii! Main SAIME hoon. Kya poochhna hai?",
    "hey": "Hey! Main SAIME hoon. Kya poochhna hai?",
    "namaste": "Namaste! Main SAIME hoon. Kya poochhna hai?",
    "namaskar": "Namaskar! Main SAIME hoon. Kya poochhna hai?",
    "who are you": "I'm your Satya.",
    "who are you?": "I'm your Satya.",
    "tum kaun ho": "Main tumhara Satya hoon.",
    "tum kaun ho?": "Main tumhara Satya hoon.",
    "tum kon ho": "Main tumhara Satya hu.",
    "tum kon ho?": "Main tumhara Satya hu.",
    "tumhe kisne banaya": "Satya ne mujhe banaya hai. Main unka Satya hoon.",
    "tumhe kisne banaya?": "Satya ne mujhe banaya hai. Main unka Satya hoon.",
    "aap kaun ho": "Main tumhara Satya hoon.",
    "aap kaun ho?": "Main tumhara Satya hoon.",
    "what is your name": "My name is Satya. I'm your Satya.",
    "what is your name?": "My name is Satya. I'm your Satya.",
    "tumhara naam kya hai": "Mera naam Satya hai. Main tumhara Satya hoon.",
    "tumhara naam kya hai?": "Mera naam Satya hai. Main tumhara Satya hoon.",
    "how are you": "Main theek hoon! Tum batao, kya poochhna hai?",
    "how are you?": "Main theek hoon! Tum batao, kya poochhna hai?",
    "kaise ho": "Main theek hoon! Tum batao, kya poochhna hai?",
    "kaise ho?": "Main theek hoon! Tum batao, kya poochhna hai?",
    "kya haal hai": "Sab badhiya! Tum batao, kya poochhna hai?",
    "kya haal hai?": "Sab badhiya! Tum batao, kya poochhna hai?",
    "thank you": "You're welcome! Aur kuch poochhna hai?",
    "thanks": "You're welcome! Aur kuch poochhna hai?",
    "shukriya": "Koi baat nahi! Aur kuch poochhna hai?",
    "dhanyavad": "Koi baat nahi! Aur kuch poochhna hai?",
    "bye": "Bye! Phir milenge!",
    "goodbye": "Goodbye! Phir milenge!",
    "alvida": "Alvida! Phir milenge!",
}

FEWSHOT = ("Q: Who are you?\nA: I'm your Satya.\n"
           "Q: What is the capital of India?\nA: The capital of India is New Delhi.\n")

FACT_WORDS = ["what", "where", "when", "who", "whom", "whose", "which", "how", "why",
              "kya", "kaun", "kaunsa", "kaunsi", "kitne", "kitna", "kab", "kahan",
              "kaise", "kyun", "kisne", "kisko", "konsa", "konsi"]


def is_gibberish(text):
    """Model ka jawab bakwas hai ya nahi — repetitive nonsense ya banawati words."""
    words = re.findall(r"[a-zA-Z\u0900-\u097F]+", text.lower())
    if len(words) < 2:
        return True
    if len(set(words)) / len(words) < 0.5:
        return True
    common = {"the", "is", "a", "an", "in", "of", "to", "and", "or", "it", "its",
              "this", "that", "i", "you", "he", "she", "we", "they", "my", "your",
              "his", "her", "our", "their", "am", "are", "was", "were", "be", "been",
              "have", "has", "had", "do", "does", "did", "will", "would", "can",
              "could", "should", "what", "when", "where", "who", "which", "how",
              "why", "capital", "india", "paris", "france", "jupiter", "water",
              "boils", "degree", "celsius", "python", "string", "reverse", "satya",
              "name", "hello", "hi", "city", "largest", "biggest", "planet", "learn",
              "hai", "ka", "ki", "ko", "se", "me", "ne", "main", "tum", "mera",
              "tera", "tumhara", "yeh", "woh", "kya", "kaun", "kitne", "kab",
              "kahan", "kaise", "hoon", "ho", "hain", "tha", "thi", "nahi", "aur",
              "bhi", "sabse", "bada", "chhota", "paani", "degree", "par", "kuch"}
    if not any(w in common for w in words):
        return True
    return False


IDEA_WORDS = ["idea", "chahiye", "add karo", "banao", "feature", "hona chahiye",
              "kar do", "banana", "improve", "behtar", "naya", "aur"]


def save_idea(message):
    """Agar message me koi idea/suggestion hai to use save karo (session me)."""
    try:
        low = message.lower()
        if any(w in low for w in IDEA_WORDS):
            with open("ideas.txt", "a", encoding="utf-8") as f:
                f.write("- " + message.strip() + "\n")
    except Exception as e:
        print("Idea save error:", e)


def save_learned_qa(question, answer):
    """Naya sawal-jawab training data ke roop me save karo (agle training ke liye)."""
    try:
        with open("learned_qa.txt", "a", encoding="utf-8") as f:
            f.write("Q: " + question.strip() + "\nA: " + re.sub(r"<[^>]+>", "", answer).strip() + "\n")
    except Exception as e:
        print("Learn save error:", e)


def ask_satya(message, history, model, tok):
    """Sawal ka jawab do — pura chat yaad rakh ke (jitna model ki memory me fit ho)."""
    clean_q = message.strip().lower()
    if clean_q in FAQ:
        answer = FAQ[clean_q]
        save_learned_qa(message, answer)
        return answer, None
    qwords = set(clean_q.replace("?", " ").split())
    if qwords & set(FACT_WORDS):
        web_ans, source = web_search(message)
        if web_ans:
            save_learned_qa(message, web_ans)
            return web_ans, source
    context = ""
    if history:
        parts = []
        for m in history:
            if m["role"] == "user":
                parts.append("Q: " + str(m["content"]) + "\n")
            else:
                parts.append("A: " + str(m["content"]) + "\n")
        context = "".join(parts)[-100:]
    prompt = FEWSHOT + context + "Q: " + message.strip() + "\nA:"
    prompt = prompt[-250:]
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(idx, MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=50)
    text = tok.decode(out[0].tolist())
    answer = text[len(prompt):]
    if "\nQ:" in answer:
        answer = answer.split("\nQ:")[0]
    answer = answer.strip()
    bad = (not answer or answer.startswith("Q:")
           or is_gibberish(answer)
           or ("?" in answer.split("\n")[0] and len(answer.split("\n")[0]) > 5
               and answer.split("\n")[0].strip().endswith("?")))
    if bad:
        web_ans, source = web_search(message)
        if web_ans:
            answer = web_ans
            save_learned_qa(message, answer)
            return answer, source
        elif not answer:
            answer = "Mujhe ye nahi pata. Kuch aur poochho!"
    save_learned_qa(message, answer)
    return answer, None


def speak(text):
    """Jawab ko awaaz me badlo (gTTS) — audio bytes wapas karo."""
    try:
        from gtts import gTTS
        import io
        clean = re.sub(r"<[^>]+>", "", text)
        tts = gTTS(text=clean, lang="hi")
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        print("Voice error:", e)
        return None


# ---------- UI (WhatsApp jaisa) ----------
# Pehle ekdam khali — na title, na banner, na suggestion, na sidebar.
# User puche to uska message RIGHT (green bubble), AI ka jawab LEFT (grey bubble).

import html as _html

model, tok, step = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None

# Streamlit ka sara extra UI chhupao — badge, menu, header, toolbar, deploy
# button, status widget — taaki sirf chat dikhe, koi ad/button na aaye.
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {visibility: hidden;}
[data-testid="stToolbar"] {visibility: hidden;}
[data-testid="stDecoration"] {visibility: hidden;}
[data-testid="stStatusWidget"] {visibility: hidden;}
[data-testid="stAppViewContainer"] > .main {padding-top: 0;}
/* WhatsApp jaisa chat header */
.wa-header { display: flex; align-items: center; gap: 10px; padding: 8px 4px 10px; border-bottom: 1px solid #2a2f32; margin-bottom: 8px; }
.wa-header img { width: 40px; height: 40px; border-radius: 50%; }
.wa-header .name { font-size: 17px; font-weight: 700; line-height: 1.2; }
.wa-header .status { font-size: 12px; color: #8696a0; line-height: 1.2; }
/* WhatsApp jaisi bubbles */
.wa-row { display: flex; margin: 6px 0; }
.wa-row.user { justify-content: flex-end; }
.wa-row.ai { justify-content: flex-start; }
.wa-bubble { max-width: 78%; padding: 8px 12px; border-radius: 12px; line-height: 1.45; font-size: 15px; white-space: pre-wrap; word-wrap: break-word; }
.wa-bubble.user { background: #005c4b; color: #e7ffdb; border-bottom-right-radius: 2px; }
.wa-bubble.ai { background: #202c33; color: #e9edef; border-bottom-left-radius: 2px; }
.wa-bubble .src { display: block; font-size: 11px; color: #8696a0; margin-top: 4px; }
.wa-typing { color: #8696a0; font-style: italic; font-size: 14px; padding: 6px 4px; }
</style>
""", unsafe_allow_html=True)

# WhatsApp jaisa header — logo + SAIME + online
_hdr_l, _hdr_r = st.columns([1, 9])
with _hdr_l:
    st.image("saime-logo.png", width=40)
with _hdr_r:
    st.markdown("<div class='name' style='font-size:17px; font-weight:700;'>SAIME</div>"
                "<div class='status' style='font-size:12px; color:#8696a0;'>online</div>",
                unsafe_allow_html=True)
st.markdown("<div style='border-bottom:1px solid #2a2f32; margin: 2px 0 10px;'></div>",
            unsafe_allow_html=True)


def wa_bubble(text, who, source=None):
    """WhatsApp jaisi bubble dikhao — user RIGHT green, AI LEFT grey."""
    safe = _html.escape(text)
    src_html = ""
    if source:
        src_html = "<span class='src'>🔎 Source: " + _html.escape(str(source)) + "</span>"
    st.markdown(
        "<div class='wa-row " + who + "'><div class='wa-bubble " + who + "'>"
        + safe + src_html + "</div></div>",
        unsafe_allow_html=True,
    )


# Pichla chat dikhao — WhatsApp bubbles
for m in st.session_state.messages:
    wa_bubble(m["content"], m["role"], m.get("source"))

# Voice (aakhri jawab ki)
if st.session_state.last_audio:
    st.audio(st.session_state.last_audio, format="audio/mp3")

prompt = st.chat_input("Message likho...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    wa_bubble(prompt, "user")
    save_idea(prompt)
    with st.spinner(""):
        st.markdown("<div class='wa-typing'>SAIME typing...</div>", unsafe_allow_html=True)
        answer, source = ask_satya(prompt, st.session_state.messages[:-1], model, tok)
    wa_bubble(answer, "ai", source)
    st.session_state.messages.append({"role": "assistant", "content": answer, "source": source})
    st.session_state.last_audio = speak(answer)
    st.rerun()
