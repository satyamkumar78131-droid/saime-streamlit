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


def web_search_all(question):
    """SABHI sources se jawab dhoondho — Hard mode ke liye.
    DuckDuckGo instant + DuckDuckGo Lite + Wikipedia — har source ka jawab
    alag se. Wapas deta hai: [(jawab, source), ...]."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    out = []
    seen_src = set()

    def _add(ans, src):
        if ans and src and src not in seen_src:
            seen_src.add(src)
            out.append((ans, src))

    # 1. DuckDuckGo instant answer
    try:
        q = urllib.parse.quote(question)
        url = "https://api.duckduckgo.com/?q=" + q + "&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
        _add((data.get("AbstractText") or "").strip(), "DuckDuckGo")
    except Exception as e:
        print("Web search all (instant) error:", e)
    # 2. DuckDuckGo Lite — pehla accha snippet + ASLI source
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
                _add(text, domain)
                break
    except Exception as e:
        print("Web search all (lite) error:", e)
    # 3. Wikipedia — relevant article ka summary
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
            if not qwords or (qwords & twords):
                sum_url = ("https://en.wikipedia.org/api/rest_v1/page/summary/"
                           + urllib.parse.quote(title))
                req2b = urllib.request.Request(sum_url, headers=headers)
                with urllib.request.urlopen(req2b, timeout=10) as r2b:
                    sdata = json.loads(r2b.read().decode("utf-8"))
                _add((sdata.get("extract") or "").strip(), "Wikipedia")
    except Exception as e:
        print("Web search all (wiki) error:", e)
    return out


FAQ = {
    "hello": "Hello! Main SAIME hoon. Tumhara sawal poochho, main jawab dunga!",
    "hlo": "Hello! Main SAIME hoon. Tumhara sawal poochho, main jawab dunga!",
    "hllo": "Hello! Main SAIME hoon. Tumhara sawal poochho, main jawab dunga!",
    "helo": "Hello! Main SAIME hoon. Tumhara sawal poochho, main jawab dunga!",
    "heloo": "Hello! Main SAIME hoon. Tumhara sawal poochho, main jawab dunga!",
    "helloo": "Hello! Main SAIME hoon. Tumhara sawal poochho, main jawab dunga!",
    "helllo": "Hello! Main SAIME hoon. Tumhara sawal poochho, main jawab dunga!",
    "hallo": "Hello! Main SAIME hoon. Tumhara sawal poochho, main jawab dunga!",
    "hy": "Hi! Main SAIME hoon. Kya poochhna hai?",
    "hyy": "Hi! Main SAIME hoon. Kya poochhna hai?",
    "hi": "Hi! Main SAIME hoon. Kya poochhna hai?",
    "hii": "Hii! Main SAIME hoon. Kya poochhna hai?",
    "hiii": "Hii! Main SAIME hoon. Kya poochhna hai?",
    "hey": "Hey! Main SAIME hoon. Kya poochhna hai?",
    "heyy": "Hey! Main SAIME hoon. Kya poochhna hai?",
    "heyyy": "Hey! Main SAIME hoon. Kya poochhna hai?",
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
    """Model ka jawab bakwas hai ya nahi — repetitive nonsense ya banawati words.
    Naya check: agar aadhe se zyada shabd pehchane hue nahi (jaise 'likegs',
    'danthes', 'pliketh'), to jawab bakwas hai — chahe usme 'how', 'many'
    jaise 1-2 sahi shabd hon."""
    words = re.findall(r"[a-zA-Z\u0900-\u097F]+", text.lower())
    if len(words) < 2:
        # Sirf number wala jawab (jaise "2+3 = 5") bakwas nahi hai
        if re.search(r"\d", text):
            return False
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
    # Known-words ratio: aam English/Hindi shabd + mashhoor naam
    known = common | {
        "new", "delhi", "mumbai", "kolkata", "chennai", "bangalore", "jaipur",
        "london", "america", "china", "japan", "russia", "germany", "england",
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "first", "second", "third", "many", "much", "more", "most",
        "few", "some", "any", "all", "every", "each", "other", "another",
        "good", "bad", "big", "small", "great", "best", "better", "new",
        "old", "young", "hot", "cold", "fast", "slow", "high", "low", "long",
        "short", "right", "wrong", "true", "false", "yes", "no", "not", "very",
        "just", "only", "also", "well", "now", "then", "here", "there", "today",
        "tomorrow", "yesterday", "year", "month", "week", "day", "night",
        "morning", "evening", "time", "people", "man", "woman", "boy", "girl",
        "child", "friend", "family", "school", "book", "water", "food", "house",
        "car", "dog", "cat", "bird", "fish", "tree", "flower", "sun", "moon",
        "star", "sky", "earth", "world", "country", "language", "number",
        "question", "answer", "story", "poem", "song", "game", "play", "work",
        "go", "come", "see", "know", "think", "say", "tell", "give", "take",
        "make", "find", "like", "love", "want", "need", "help", "try", "use",
        "write", "read", "speak", "listen", "watch", "eat", "drink", "sleep",
        "run", "walk", "sit", "stand", "open", "close", "start", "stop",
        "bharat", "hindustan", "dilli", "paisa", "rupay", "sawal", "jawab",
        "kahani", "kavita", "geet", " dost", "duniya", "desh", "bhasha",
        "samay", "din", "raat", "subah", "shaam", "saal", "mahina", "hafta",
        "accha", "bura", "sahi", "galat", "pehla", "dusra", "teesra",
        "ek", "do", "teen", "chaar", "paanch", "chhe", "saat", "aath",
        "nau", "das", "sau", "hazaar", "lakh", "crore"}
    known_count = sum(1 for w in words if w in known)
    if known_count / len(words) < 0.5:
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


# ---------- REAL AGENTS (multi-task) ----------
# Har agent ek asli TOOL use karta hai — sirf text nahi banata.
# 🔎 Search Agent = web search, 🧮 Math Agent = calculator,
# 💬 Chat Agent = SATYA V1, ✍️ Writer Agent = SATYA V1 (likhne ke liye).
# Ek saath kai task bhejoge to har task apne agent ko jayega.

import ast
import operator as _op

_CALC_OPS = {
    ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul,
    ast.Div: _op.truediv, ast.Mod: _op.mod, ast.Pow: _op.pow,
    ast.USub: _op.neg, ast.UAdd: _op.pos,
}


def safe_calc(expr):
    """Calculator tool — sirf safe arithmetic, koi code nahi chalata."""
    try:
        tree = ast.parse(expr, mode="eval")

        def _eval(node):
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in _CALC_OPS:
                return _CALC_OPS[type(node.op)](_eval(node.left), _eval(node.right))
            if isinstance(node, ast.UnaryOp) and type(node.op) in _CALC_OPS:
                return _CALC_OPS[type(node.op)](_eval(node.operand))
            raise ValueError("not arithmetic")

        return _eval(tree)
    except Exception:
        return None


_MATH_RE = re.compile(r"^[\d\s\.\+\-\*\/\%\(\)x×÷]+$")
_WRITE_WORDS = ["likho", "likh", "kahani", "story", "poem", "kavita",
                "script", "write", "shayari", "geet", "nibandh", "essay"]


def detect_math(task):
    """Task me hisaab hai ya nahi — expression nikalo."""
    t = task.strip().lower()
    t = t.replace("x", "*").replace("×", "*").replace("÷", "/")
    t = re.sub(r"(kitna|hota|hai|kya|=|\?)", " ", t).strip()
    if _MATH_RE.match(t) and re.search(r"\d", t) and re.search(r"[\+\-\*\/\%]", t):
        return t
    return None


def classify_agent(task, forced=None):
    """Task ko sahi agent ko do — ya user ne jo agent chuna hai use."""
    if forced and forced != "auto":
        return forced
    if detect_math(task):
        return "math"
    low = task.lower()
    if any(w in low for w in _WRITE_WORDS):
        return "writer"
    if any(w in low for w in FACT_WORDS):
        return "search"
    return "chat"


AGENT_INFO = {
    "auto": ("⚡", "Auto"),
    "search": ("🔎", "Search Agent"),
    "math": ("🧮", "Math Agent"),
    "chat": ("💬", "Chat Agent"),
    "writer": ("✍️", "Writer Agent"),
}


def run_team(task, history, model, tok, mode="low"):
    """Ek kaam ko MULTIPLE agents MIL KE solve karein — team work.
    🔎 Search Agent facts dhoondhta hai, 🧮 Math Agent hisaab karta hai,
    💬 Chat Agent jawab banata hai, ✍️ Writer Agent use saaf likhta hai.
    mode: 'low' = sirf model (fast, default), 'medium' = model + web check
    (sabse accurate), 'hard' = SABHI sources jod ke ek complete jawab.
    Returns (final_answer, source, team_list, steps)."""
    team = []
    steps = []
    facts = None
    source = None
    calc = None
    expr = None

    # Pehle FAQ — pehchan/greeting jaise sawal seedha Chat Agent se.
    clean = task.strip().lower()
    if clean in FAQ:
        answer = FAQ[clean]
        save_learned_qa(task, answer)
        return answer, None, ["chat"], ["💬 Chat Agent: ye to mujhe pehle se pata hai!"]

    # LOW mode — sirf model, fast default (koi web search nahi)
    if mode == "low":
        team.append("chat")
        steps.append("💬 Chat Agent: fast jawab de raha hoon...")
        expr = detect_math(task)
        if expr:
            calc = safe_calc(expr)
            if calc is not None:
                if isinstance(calc, float) and calc.is_integer():
                    calc = int(calc)
                team.insert(0, "math")
                answer = task.strip() + " = " + str(calc)
                save_learned_qa(task, answer)
                return answer, None, team, steps
        answer, _src = ask_satya(task, history, model, tok)
        if _src and not source:
            source = _src
        save_learned_qa(task, answer)
        return answer, source, team, steps

    # 1. Search Agent — facts dhoondho (agar sawal factual hai)
    # Poore shabd milao — "kahani" me "kahan" na mile, isliye word-boundary check.
    low = task.lower()
    _words = set(re.findall(r"[a-zA-Z\u0900-\u097F]+", low))
    if (_words & set(FACT_WORDS)) or "?" in task:
        team.append("search")
        if mode == "hard":
            steps.append("🔎 Search Agent: SABHI sources se dhoondh raha hoon...")
            all_src = web_search_all(task)
            if all_src:
                facts = all_src[0][0]
                source = ", ".join(s for _, s in all_src)
                steps.append("🔎 Search Agent: " + str(len(all_src)) + " sources mil gaye!")
                # HARD: sabhi sources + model ka jawab jod ke ek complete jawab
                _model_ans, _ = ask_satya(task, history, model, tok)
                _parts = [a for a, _ in all_src]
                if _model_ans and not is_gibberish(_model_ans):
                    _parts.append(_model_ans)
                # Duplicate hatake jodo — ek complete, powerful jawab
                _seen = set()
                _uniq = []
                for _p in _parts:
                    _k = _p[:60].lower()
                    if _k not in _seen:
                        _seen.add(_k)
                        _uniq.append(_p)
                answer = "\n\n".join(_uniq)
                save_learned_qa(task, answer)
                if "chat" not in team:
                    team.append("chat")
                return answer, source, team, steps
        else:
            steps.append("🔎 Search Agent: facts dhoondh raha hoon...")
            facts, source = web_search(task)
            if facts:
                steps.append("🔎 Search Agent: facts mil gaye!")
    # 2. Math Agent — hisaab karo (agar hisaab hai)
    expr = detect_math(task)
    if expr:
        team.append("math")
        steps.append("🧮 Math Agent: hisaab kar raha hoon...")
        calc = safe_calc(expr)
        if calc is not None:
            if isinstance(calc, float) and calc.is_integer():
                calc = int(calc)
            steps.append("🧮 Math Agent: hisaab ho gaya!")
    # 3. Chat Agent — sab kuch mila ke jawab banao
    team.append("chat")
    steps.append("💬 Chat Agent: jawab bana raha hoon...")
    if facts and calc is not None:
        answer = facts + "\n" + expr + " = " + str(calc)
    elif facts:
        answer = facts
    elif calc is not None:
        answer = task.strip() + " = " + str(calc)
    else:
        answer, _src = ask_satya(task, history, model, tok)
        if _src and not source:
            source = _src
    # 4. Writer Agent — jawab ko saaf-suthra likho (lamba jawab ho to)
    if len(answer) > 200 or any(w in low for w in _WRITE_WORDS):
        team.append("writer")
        steps.append("✍️ Writer Agent: jawab ko saaf likh raha hoon...")
        polished, _ = ask_satya(
            "Is jawab ko saaf aur acchi Hindi/English me likho: " + answer,
            history, model, tok)
        if polished and not is_gibberish(polished) and len(polished) > 10:
            answer = polished
    save_learned_qa(task, answer)
    return answer, source, team, steps


def run_agent(task, history, model, tok, forced=None):
    """Ek task ko uske agent se karwao — tool use karke, steps ke saath.
    Returns (answer, source, agent_key, steps)."""
    # Pehle FAQ — pehchan/greeting jaise sawal seedha yahin se (koi agent nahi).
    clean = task.strip().lower()
    if clean in FAQ:
        answer = FAQ[clean]
        save_learned_qa(task, answer)
        return answer, None, "chat", ["💬 Chat Agent: ye to mujhe pehle se pata hai!"]
    agent = classify_agent(task, forced)
    steps = []
    if agent == "math":
        steps.append("🧮 Math Agent: hisaab kar raha hoon...")
        expr = detect_math(task)
        result = safe_calc(expr) if expr else None
        if result is not None:
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            answer = task.strip() + " = " + str(result)
            save_learned_qa(task, answer)
            return answer, None, "math", steps
        steps.append("🧮 Hisaab samajh nahi aaya — Chat Agent ko bhej raha hoon...")
        agent = "chat"
    if agent == "search":
        steps.append("🔎 Search Agent: web pe dhoondh raha hoon...")
        web_ans, source = web_search(task)
        if web_ans:
            save_learned_qa(task, web_ans)
            return web_ans, source, "search", steps
        steps.append("🔎 Web pe nahi mila — Chat Agent se poochh raha hoon...")
        agent = "chat"
    if agent == "writer":
        steps.append("✍️ Writer Agent: likh raha hoon...")
        answer, source = ask_satya("Ek acchi si " + task.strip(), history, model, tok)
        return answer, source, "writer", steps
    steps.append("💬 Chat Agent: soch raha hoon...")
    answer, source = ask_satya(task, history, model, tok)
    return answer, source, "chat", steps


def split_tasks(message):
    """Ek message me kai task ho to alag karo (har line ek task)."""
    lines = [l.strip() for l in message.replace(" / ", "\n").split("\n")]
    lines = [l for l in lines if l]
    return lines if lines else [message.strip()]


def describe_photo(image_bytes):
    """Photo ko BLIP se padho (HF Inference API).
    Token Streamlit secrets se aata hai (HF_TOKEN) — bina token ke None.
    Returns (caption, error) — error 'no_token' ya 'api_error' ho sakta hai."""
    try:
        token = None
        try:
            token = st.secrets.get("HF_TOKEN")
        except Exception:
            token = None
        if not token:
            return None, "no_token"
        url = ("https://router.huggingface.co/hf-inference/models/"
               "Salesforce/blip-image-captioning-base")
        req = urllib.request.Request(
            url, data=image_bytes,
            headers={"Authorization": "Bearer " + token,
                     "Content-Type": "image/jpeg"},
            method="POST")
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode("utf-8"))
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"].strip(), None
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"].strip(), None
        return None, "api_error"
    except Exception as e:
        print("Photo describe error:", e)
        return None, "api_error"


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

# Multi-Agent Team — har kaam ko KAI agents mil ke solve karte hain
st.markdown("<div style='text-align:center; font-size:13px; color:#8696a0; "
            "padding:2px 0 10px;'>🤖 Multi-Agent Team: 🔎 Search + 🧮 Math + "
            "💬 Chat + ✍️ Writer — ek saath ek kaam</div>",
            unsafe_allow_html=True)


def wa_bubble(text, who, source=None, team=None):
    """WhatsApp jaisi bubble dikhao — user RIGHT green, AI LEFT grey.
    AI bubble me upar TEAM dikhao (kaun-kaun se agents ne kaam kiya)."""
    safe = _html.escape(text)
    head = ""
    if team and isinstance(team, list) and team:
        _names = " + ".join(AGENT_INFO[t][0] + " " + AGENT_INFO[t][1]
                            for t in team if t in AGENT_INFO)
        head = ("<span style='display:block; font-size:11px; color:#53bdeb; "
                "font-weight:700; margin-bottom:2px;'>" + _names + "</span>")
    elif team and isinstance(team, str) and team in AGENT_INFO:
        _emoji, _name = AGENT_INFO[team]
        head = ("<span style='display:block; font-size:11px; color:#53bdeb; "
                "font-weight:700; margin-bottom:2px;'>" + _emoji + " " + _name + "</span>")
    src_html = ""
    if source:
        src_html = "<span class='src'>🔎 Source: " + _html.escape(str(source)) + "</span>"
    st.markdown(
        "<div class='wa-row " + who + "'><div class='wa-bubble " + who + "'>"
        + head + safe + src_html + "</div></div>",
        unsafe_allow_html=True,
    )


# Pichla chat dikhao — WhatsApp bubbles (photo messages ke saath)
for m in st.session_state.messages:
    if m.get("photo"):
        _pc1, _pc2 = st.columns([1, 2])
        with _pc2:
            st.image(m["photo"], use_container_width=True)
        if m.get("content"):
            wa_bubble(m["content"], m["role"], m.get("source"), m.get("team", m.get("agent")))
    else:
        wa_bubble(m["content"], m["role"], m.get("source"), m.get("team", m.get("agent")))

# Voice (aakhri jawab ki)
if st.session_state.last_audio:
    st.audio(st.session_state.last_audio, format="audio/mp3")

prompt = st.chat_input("Message likho... (kai task ek saath: har line ek task)")

# Jawab ki quality — neeche option (Low = fast default, Medium = sabse accurate,
# Hard = sabhi sources jod ke complete jawab)
if "quality" not in st.session_state:
    st.session_state.quality = "low"
_q1, _q2, _q3 = st.columns(3)
if _q1.button("🟢 Low — Fast", key="q_low", use_container_width=True,
              type="primary" if st.session_state.quality == "low" else "secondary"):
    st.session_state.quality = "low"
    st.rerun()
if _q2.button("🟡 Medium — Accurate", key="q_med", use_container_width=True,
              type="primary" if st.session_state.quality == "medium" else "secondary"):
    st.session_state.quality = "medium"
    st.rerun()
if _q3.button("🔴 Hard — Complete", key="q_hard", use_container_width=True,
              type="primary" if st.session_state.quality == "hard" else "secondary"):
    st.session_state.quality = "hard"
    st.rerun()

# 📷 Photo upload — neeche (photo dekh ke SAIME jawab dega)
photo = st.file_uploader("📷 Photo bhejo — SAIME dekh ke jawab dega",
                         type=["jpg", "jpeg", "png", "webp"],
                         key="photo_upload")
if photo is not None:
    _pkey = photo.name + "_" + str(photo.size)
    if st.session_state.get("last_photo_key") != _pkey:
        st.session_state.last_photo_key = _pkey
        _bytes = photo.getvalue()
        _pc1, _pc2 = st.columns([1, 2])
        with _pc2:
            st.image(_bytes, use_container_width=True)
        st.session_state.messages.append(
            {"role": "user", "content": "📷 Photo bheji", "photo": _bytes})
        st.markdown("<div class='wa-typing'>👀 Photo dekh raha hoon...</div>",
                    unsafe_allow_html=True)
        caption, err = describe_photo(_bytes)
        if caption:
            st.session_state.photo_caption = caption
            _ans = "Photo me ye dikh raha hai: " + caption + ". Is photo ke baare me kuch bhi poochho!"
            wa_bubble(_ans, "ai", None, "chat")
            st.session_state.messages.append(
                {"role": "assistant", "content": _ans, "team": ["chat"]})
            st.session_state.last_audio = speak(_ans)
        elif err == "no_token":
            _ans = ("Photo mil gayi! 📷 Lekin photo padhne ke liye Streamlit me "
                    "HF_TOKEN add karna hoga (App settings → Secrets → "
                    "HF_TOKEN = tumhara Hugging Face token). Tab SAIME photo "
                    "dekh ke sare jawab dega!")
            wa_bubble(_ans, "ai", None, "chat")
            st.session_state.messages.append(
                {"role": "assistant", "content": _ans, "team": ["chat"]})
        else:
            _ans = ("Photo mil gayi! 📷 Lekin abhi photo padh nahi paya "
                    "(server busy ho sakta hai). Dobara try karo!")
            wa_bubble(_ans, "ai", None, "chat")
            st.session_state.messages.append(
                {"role": "assistant", "content": _ans, "team": ["chat"]})
        st.rerun()

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    wa_bubble(prompt, "user")
    save_idea(prompt)
    tasks = split_tasks(prompt)
    last_answer = ""
    for task in tasks:
        # Photo ka context — agar photo bheji hai to SAIME use dekh ke jawab de
        _task = task
        if st.session_state.get("photo_caption"):
            _task = ("Photo me ye hai: " + st.session_state.photo_caption
                     + ". Sawal: " + task)
        st.markdown("<div class='wa-typing'>🤖 Team kaam kar rahi hai...</div>",
                    unsafe_allow_html=True)
        # TEAM: ek kaam ko kai agents mil ke solve karein (quality mode ke saath)
        answer, source, team, steps = run_team(
            _task, st.session_state.messages[:-1], model, tok,
            mode=st.session_state.get("quality", "low"))
        wa_bubble(answer, "ai", source, team)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "source": source, "team": team})
        last_answer = answer
    st.session_state.last_audio = speak(last_answer)
    st.rerun()
