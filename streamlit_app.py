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
    Hindi sawal ho to English me translate karke search karo — web ko
    English query pe behtar result milta hai.
    Wapas deta hai: (jawab, source_ka_naam)."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    # Hindi (Roman ya Devanagari) sawal ho to English me badlo, phir search karo
    _q = question
    try:
        if detect_language(question) == "hi":
            _tr, _ = translate_text(question, "en")
            if _tr:
                _q = _tr
    except Exception:
        pass
    question = _q

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
    alag se. Hindi sawal ho to English me translate karke search karo.
    Wapas deta hai: [(jawab, source), ...]."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    out = []
    seen_src = set()

    # Hindi sawal ho to English me badlo, phir search karo
    try:
        if detect_language(question) == "hi":
            _tr, _ = translate_text(question, "en")
            if _tr:
                question = _tr
    except Exception:
        pass

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
    "hello": "Hello! Main Satya hoon. Tumhara sawal poochho, main jawab dunga!",
    "hlo": "Hello! Main Satya hoon. Tumhara sawal poochho, main jawab dunga!",
    "hllo": "Hello! Main Satya hoon. Tumhara sawal poochho, main jawab dunga!",
    "helo": "Hello! Main Satya hoon. Tumhara sawal poochho, main jawab dunga!",
    "heloo": "Hello! Main Satya hoon. Tumhara sawal poochho, main jawab dunga!",
    "helloo": "Hello! Main Satya hoon. Tumhara sawal poochho, main jawab dunga!",
    "helllo": "Hello! Main Satya hoon. Tumhara sawal poochho, main jawab dunga!",
    "hallo": "Hello! Main Satya hoon. Tumhara sawal poochho, main jawab dunga!",
    "hy": "Hi! Main Satya hoon. Kya poochhna hai?",
    "hyy": "Hi! Main Satya hoon. Kya poochhna hai?",
    "hi": "Hi! Main Satya hoon. Kya poochhna hai?",
    "hii": "Hii! Main Satya hoon. Kya poochhna hai?",
    "hiii": "Hii! Main Satya hoon. Kya poochhna hai?",
    "hey": "Hey! Main Satya hoon. Kya poochhna hai?",
    "heyy": "Hey! Main Satya hoon. Kya poochhna hai?",
    "heyyy": "Hey! Main Satya hoon. Kya poochhna hai?",
    "namaste": "Namaste! Main Satya hoon. Kya poochhna hai?",
    "namaskar": "Namaskar! Main Satya hoon. Kya poochhna hai?",
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

# Greeting wale sawal (Settings me "Greeting reply" off ho to ye FAQ se nahi,
# model se jawab denge). Sirf greeting — pehchan (who are you) hamesha FAQ se.
GREETING_KEYS = {"hello", "hlo", "hllo", "helo", "heloo", "helloo", "helllo",
                 "hallo", "hy", "hyy", "hi", "hii", "hiii", "hey", "heyy",
                 "heyyy", "namaste", "namaskar", "good morning", "good afternoon",
                 "good evening", "ram ram", "salaam", "adaab", "how are you",
                 "how are you?", "kaise ho", "kaise ho?", "kya haal hai",
                 "kya haal hai?", "bye", "goodbye", "alvida"}

FEWSHOT = ("Q: Who are you?\nA: I'm your Satya.\n"
           "Q: What is the capital of India?\nA: The capital of India is New Delhi.\n")

FACT_WORDS = ["what", "where", "when", "who", "whom", "whose", "which", "how", "why",
              "kya", "kaun", "kaunsa", "kaunsi", "kitne", "kitna", "kab", "kahan",
              "kaise", "kaisa", "kaisi", "kyun", "kisne", "kisko", "konsa", "konsi",
              "mausam", "weather", "temperature", "tapman", "aaj", "kal", "parso",
              "today", "tomorrow", "barish", "garmi", "thand", "hawa"]


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
    # Greeting reply (Settings se on/off) — off ho to greeting bhi model se jawab de
    _greet_on = True
    try:
        _greet_on = st.session_state.get("greet_on", True)
    except Exception:
        _greet_on = True
    if clean_q in FAQ and not (_greet_on is False and clean_q in GREETING_KEYS):
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
        # Web pe bhi nahi mila — gibberish ya khaali jawab user ko mat dikhao.
        # Saaf-saaf bata do ki nahi pata, bakwas likhne se behtar hai.
        answer = "Mujhe iska sahi jawab nahi pata. Dobara poochho ya kuch aur poochho!"
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


def run_team(task, history, model, tok, freq="low", time="less"):
    """Ek kaam ko MULTIPLE agents MIL KE solve karein — team work.
    🔎 Search Agent facts dhoondhta hai, 🧮 Math Agent hisaab karta hai,
    💬 Chat Agent jawab banata hai, ✍️ Writer Agent use saaf likhta hai.
    freq (Frequency): low = model + trained Q&A, medium = + web sources,
    high = sabhi sources + Gemini verify (poora process hidden).
    time (Time): less = fast (~1s), medium = ~3s, high = 3-5s, sabse powerful.
    Dono me se jo zyada thorough ho, wahi level chalta hai.
    Returns (final_answer, source, team_list, steps)."""
    _lvl = {"low": 1, "medium": 2, "high": 3}
    _tlvl = {"less": 1, "medium": 2, "high": 3}
    level = max(_lvl.get(freq, 1), _tlvl.get(time, 1))
    team = []
    steps = []
    facts = None
    source = None
    calc = None
    expr = None

    # Pehle FAQ — pehchan/greeting jaise sawal seedha Chat Agent se.
    # Greeting reply off ho to greeting model se jawab dega (FAQ se nahi).
    clean = task.strip().lower()
    _greet_on = True
    try:
        _greet_on = st.session_state.get("greet_on", True)
    except Exception:
        _greet_on = True
    if clean in FAQ and not (_greet_on is False and clean in GREETING_KEYS):
        answer = FAQ[clean]
        save_learned_qa(task, answer)
        return answer, None, ["chat"], ["💬 Chat Agent: ye to mujhe pehle se pata hai!"]

    # Level 1 — sirf model (fast, ~1s)
    if level == 1:
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
        # Fast + sahi: factual sawal ho to Google se check karke galat jawab sahi karo
        answer, _gsrc = google_correct(task, answer)
        if _gsrc and not source:
            source = _gsrc
        save_learned_qa(task, answer)
        return answer, source, team, steps

    # 1. Search Agent — facts dhoondho (agar sawal factual hai)
    # Poore shabd milao — "kahani" me "kahan" na mile, isliye word-boundary check.
    low = task.lower()
    _words = set(re.findall(r"[a-zA-Z\u0900-\u097F]+", low))
    if (_words & set(FACT_WORDS)) or "?" in task:
        team.append("search")
        if level >= 3:
            steps.append("🔎 Search Agent: SABHI sources se dhoondh raha hoon...")
            all_src = web_search_all(task)
            if all_src:
                facts = all_src[0][0]
                source = ", ".join(s for _, s in all_src)
                steps.append("🔎 Search Agent: " + str(len(all_src)) + " sources mil gaye!")
                # Level 3: sabhi sources + model ka jawab jod ke ek complete jawab
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
                # Gemini se chup-chap verify karo (High) — screen pe kuch nahi
                _verified = verify_with_gemini(task, answer)
                if _verified:
                    answer = _verified
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
    # Level 3: Gemini se chup-chap verify karo — screen pe kuch nahi
    if level >= 3:
        _verified = verify_with_gemini(task, answer)
        if _verified:
            answer = _verified
    save_learned_qa(task, answer)
    return answer, source, team, steps


def run_agent(task, history, model, tok, forced=None):
    """Ek task ko uske agent se karwao — tool use karke, steps ke saath.
    Returns (answer, source, agent_key, steps)."""
    # Pehle FAQ — pehchan/greeting jaise sawal seedha yahin se (koi agent nahi).
    # Greeting reply off ho to greeting model se jawab dega (FAQ se nahi).
    clean = task.strip().lower()
    _greet_on = True
    try:
        _greet_on = st.session_state.get("greet_on", True)
    except Exception:
        _greet_on = True
    if clean in FAQ and not (_greet_on is False and clean in GREETING_KEYS):
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


_LANG_WORDS = {
    # Roman Hindi (Hinglish) — pehle check karo, ye user ki main language hai
    "hi": {"tum", "kaun", "kya", "hai", "ho", "mera", "tera", "kaise", "kab",
           "kahan", "kitne", "batao", "karo", "mujhe", "tumhara", "aap", "kyun",
           "suno", "dekho", "chahiye", "raha", "rahi", "kaisa", "kaisi", "naam",
           "bhai", "yaar", "acha", "accha", "theek", "matlab", "abhi", "kal"},
    "es": {"hola", "que", "como", "cómo", "estas", "estás", "gracias", "por",
           "favor", "el", "la", "los", "las", "es", "son", "una", "con", "para",
           "estoy", "eres", "muy", "buenos", "dias"},
    "fr": {"bonjour", "comment", "merci", "je", "tu", "il", "est", "les", "des",
           "une", "avec", "pour", "vous", "allez", "suis", "etes", "êtes", "bien"},
    "pt": {"ola", "olá", "como", "voce", "você", "obrigado", "por", "favor", "eu",
           "sou", "esta", "está", "estou", "muito", "bom", "dia", "tudo", "bem"},
    "it": {"ciao", "come", "stai", "grazie", "che", "per", "sono", "sei",
           "buongiorno", "prego", "bene", "tutto", "questo", "sei"},
    "de": {"hallo", "wie", "geht", "danke", "ich", "bin", "und", "die", "der",
           "ist", "nicht", "gut", "sehr", "bitte", "schon", "gehts"},
    "nl": {"hallo", "hoe", "gaat", "dank", "ik", "ben", "goed", "heel", "met",
           "van", "een", "het", "niet", "graag"},
    "tr": {"merhaba", "nasil", "nasıl", "nasılsın", "tesekkur", "teşekkür", "ben",
           "sen", "iyi", "cok", "çok", "lutfen", "lütfen", "gunaydin", "evet",
           "hayir", "hayır", "neden"},
    "id": {"halo", "apa", "kabar", "terima", "kasih", "saya", "kamu", "baik",
           "selamat", "pagi", "bagaimana", "tolong"},
    "ru": {"privet", "kak", "dela", "spasibo", "pozhaluysta", "ya", "ty", "eto"},
    "ar": {"marhaba", "kayfa", "haluka", "shukran", "ana", "anta", "ma", "hada"},
}


_EN_COMMON = {"the", "is", "a", "an", "in", "of", "to", "and", "or", "it",
              "this", "that", "i", "you", "he", "she", "we", "they", "my",
              "your", "his", "her", "our", "their", "am", "are", "was", "were",
              "have", "has", "had", "do", "does", "did", "will", "would", "can",
              "could", "should", "what", "when", "where", "who", "which", "how",
              "why", "hello", "hi", "how", "are", "you", "name", "capital",
              "india", "largest", "biggest", "planet", "city", "water", "good",
              "bad", "big", "small", "great", "best", "new", "old", "time",
              "people", "today", "year", "month", "week", "day", "night"}


def detect_language(text):
    """User ki language pehchano — script + common words se (koi API nahi, fast).
    Devanagari = hi, Cyrillic = ru, Arabic script = ar,
    Hiragana/Katakana = ja (kanji se pehle check — Japanese me dono hote hain),
    Hangul = ko, CJK = zh, Thai = th,
    Latin script = common words se (Roman Hindi pehle, phir es/fr/pt/it/de/nl/tr/id),
    warna en."""
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    if re.search(r"[\u0400-\u04FF]", text):
        return "ru"
    if re.search(r"[\u0600-\u06FF]", text):
        return "ar"
    if re.search(r"[\u3040-\u30FF]", text):
        return "ja"
    if re.search(r"[\uAC00-\uD7AF]", text):
        return "ko"
    if re.search(r"[\u4E00-\u9FFF]", text):
        return "zh"
    if re.search(r"[\u0E00-\u0E7F]", text):
        return "th"
    # Latin script — saare Unicode letters pakdo (accent wale bhi: ó, á, ı, ç)
    _words = set(re.findall(r"[^\W\d_]+", text.lower()))
    if not _words:
        return "en"
    for _lang, _vocab in _LANG_WORDS.items():
        if len(_words & _vocab) >= 2:
            return _lang
    if len(_words & _EN_COMMON) / len(_words) >= 0.5:
        return "en"
    return "en"


def translate_text(text, target, source="auto"):
    """Free MyMemory API se translate karo (koi key nahi chahiye).
    source='auto' ho to autodetect use karo.
    Returns (translated_text, source_lang) ya (None, None)."""
    try:
        src = source if source != "auto" else "autodetect"
        url = ("https://api.mymemory.translated.net/get?q="
               + urllib.parse.quote(text) + "&langpair=" + src + "|" + target)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode("utf-8"))
        rd = data.get("responseData", {})
        out = (rd.get("translatedText") or "").strip()
        if not out or data.get("responseStatus") != 200:
            return None, None
        return out, source
    except Exception as e:
        print("Translate error:", e)
        return None, None


def google_correct(task, answer):
    """Agar jawab galat ho to Google (web) se sahi karo — fast check.
    Sirf factual sawalon pe: web se sahi jawab mil jaye to use hi final mano.
    Returns (corrected_answer, source) ya (answer, None) agar web na mile."""
    try:
        low = task.lower()
        _words = set(re.findall(r"[a-zA-Z\u0900-\u097F]+", low))
        if not ((_words & set(FACT_WORDS)) or "?" in task):
            return answer, None
        facts, source = web_search(task)
        if not facts:
            return answer, None
        # Web ka jawab hi sahi mano — model ka jawab galat ho sakta hai.
        # Sirf CONTENT words milao (is/the/of jaise aam words nahi) — warna
        # galat jawab bhi "milta-julta" lagta hai (jaise Mumbai vs New Delhi).
        _STOP = {"is", "the", "of", "a", "an", "in", "on", "to", "and", "or",
                 "ka", "ki", "ke", "hai", "kya", "what", "who", "where", "when",
                 "why", "how", "yeh", "yah", "is", "mein", "se", "ko", "ne"}
        _fw = set(re.findall(r"[a-zA-Z\u0900-\u097F]+", facts.lower())) - _STOP
        _aw = set(re.findall(r"[a-zA-Z\u0900-\u097F]+", answer.lower())) - _STOP
        _common = _fw & _aw
        if len(_common) >= 3:
            return answer, source
        return facts, source
    except Exception as e:
        print("Google correct error:", e)
        return answer, None


def verify_with_gemini(question, answer):
    """High mode: jawab ko Gemini se chup-chap verify karo.
    Poora process hidden rehta hai — screen pe kuch nahi dikhta.
    Key Streamlit secrets se (GEMINI_API_KEY, free: aistudio.google.com).
    Bina key ke None — tab multi-source jawab hi final hota hai.
    Returns (verified_answer ya None)."""
    try:
        key = None
        try:
            key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            key = None
        if not key:
            return None
        prompt = ("Check this answer for the question and correct it if wrong. "
                  "Give ONLY the final correct answer, in the same language as the question.\n"
                  "Question: " + question + "\nAnswer: " + answer)
        body = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 300},
        }).encode("utf-8")
        url = ("https://generativelanguage.googleapis.com/v1beta/models/"
               "gemini-2.0-flash:generateContent?key=" + key)
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
        cands = data.get("candidates") or []
        if cands:
            parts = cands[0].get("content", {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts).strip()
            if text:
                return text
        return None
    except Exception as e:
        print("Gemini verify error:", e)
        return None


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


# ---------- UI (WhatsApp jaisa — BILKUL SAAF) ----------
# Main screen pe SIRF: header (logo + SAIME + online), chat bubbles,
# message input, aur neeche-left ek chhota Info button.
# Baaki SAB kuch Info popup ke andar hai — Frequency, Time, Font,
# Greeting reply, Photo upload.
# Har AI jawab pe: "SAIME Responding" label, typing animation (live),
# 🔊 speaker button, 📋 copy button, 👍 like button.
# Background listening aur wake word web app me possible nahi hai
# (browser mic sirf app khula ho tabhi chalta hai, background me nahi) —
# isliye ye dono is version me nahi hain.

import html as _html
import time as _time
import streamlit.components.v1 as _components

model, tok, step = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "freq" not in st.session_state:
    st.session_state.freq = "low"
if "time" not in st.session_state:
    st.session_state.time = "less"
if "greet_on" not in st.session_state:
    st.session_state.greet_on = True
if "font" not in st.session_state:
    st.session_state.font = "Default"
if "liked" not in st.session_state:
    st.session_state.liked = set()
if "play_idx" not in st.session_state:
    st.session_state.play_idx = None

# Font designs — Info popup se chuno, bubbles usi font me dikhenge
FONTS = {
    "Default": "inherit",
    "Serif": "Georgia, 'Times New Roman', serif",
    "Monospace": "'Courier New', monospace",
    "Rounded": "'Trebuchet MS', Verdana, sans-serif",
}
_font_family = FONTS.get(st.session_state.font, "inherit")

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
/* WhatsApp jaisi bubbles */
.wa-row { display: flex; margin: 6px 0; }
.wa-row.user { justify-content: flex-end; }
.wa-row.ai { justify-content: flex-start; }
.wa-bubble { max-width: 78%; padding: 8px 12px; border-radius: 12px; line-height: 1.45; font-size: 15px; white-space: pre-wrap; word-wrap: break-word; font-family: """ + _font_family + """; }
.wa-bubble.user { background: #005c4b; color: #e7ffdb; border-bottom-right-radius: 2px; }
.wa-bubble.ai { background: #202c33; color: #e9edef; border-bottom-left-radius: 2px; }
.wa-label { font-size: 11px; color: #8696a0; margin-bottom: 2px; }
.wa-label.ai { color: #53bdeb; }
.wa-label.user { text-align: right; }
.wa-typing { color: #8696a0; font-style: italic; font-size: 14px; padding: 6px 4px; }
.wa-thanks { font-size: 12px; color: #25d366; margin: 2px 0 6px 4px; }
/* Info button — medium size, left side (mobile pe columns stack ho jate hain,
   isliye popover ki width CSS se fix karo) */
div[data-testid="stPopover"] { width: 150px !important; }
div[data-testid="stPopover"] > button { width: 150px !important; }
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


def bubble_html(text, who):
    """Bubble ka HTML banao — user pe 'You' label, AI pe 'SAIME Responding'."""
    safe = _html.escape(text)
    if who == "user":
        label = "<div class='wa-label user'>You</div>"
    else:
        label = "<div class='wa-label ai'>SAIME Responding</div>"
    return ("<div class='wa-row " + who + "'><div style='max-width:78%;'>"
            + label + "<div class='wa-bubble " + who + "'>" + safe
            + "</div></div></div>")


def wa_bubble(text, who):
    """WhatsApp jaisi bubble dikhao — label ke saath."""
    st.markdown(bubble_html(text, who), unsafe_allow_html=True)


def typewriter_bubble(answer):
    """AI jawab ko typing animation se dikhao — shabd-dar-shabd,
    jaise AI real time me type kar raha ho. Cursor (▍) dikhta hai."""
    ph = st.empty()
    words = answer.split(" ")
    partial = ""
    for w in words:
        partial += w + " "
        ph.markdown(bubble_html(partial.rstrip() + " ▍", "ai"),
                    unsafe_allow_html=True)
        _time.sleep(0.025)
    ph.markdown(bubble_html(answer, "ai"), unsafe_allow_html=True)


def copy_button(text, key):
    """Chhota 📋 Copy button — click karne pe pura jawab clipboard me copy.
    (Streamlit ka components HTML use karta hai — clipboard JS.)
    FIX: pehle onclick="..." ke andar JSON string daala tha — usme jo
    double-quote aata tha wo attribute ko tod deta tha aur JS text screen
    pe dikhne lagta tha. Ab button + alag <script> me event listener hai,
    koi quote-tod nahi."""
    _js = json.dumps(text).replace("</", "<\\/")
    _bid = "cpy_" + str(key)
    _html_code = (
        "<button id='" + _bid + "' style='background:transparent;"
        "border:1px solid #2a2f32;color:#8696a0;border-radius:8px;"
        "padding:2px 8px;font-size:12px;cursor:pointer;'>📋 Copy</button>"
        "<script>"
        "document.getElementById('" + _bid + "').addEventListener('click', function(){"
        "var b=this;"
        "navigator.clipboard.writeText(" + _js + ").then(function(){"
        "b.innerText='✓ Copied';"
        "setTimeout(function(){b.innerText='📋 Copy';},1500);});});"
        "</script>")
    _components.html(_html_code, height=32)


def ai_actions(idx, text):
    """Har AI jawab ke neeche: 🔊 speaker, 📋 copy, 👍 like buttons."""
    _c1, _c2, _c3, _c4 = st.columns([1, 1, 1, 9])
    with _c1:
        if st.button("🔊", key="spk_" + str(idx), help="Sunno"):
            st.session_state.play_idx = idx
            st.rerun()
    with _c2:
        copy_button(text, "cpy_" + str(idx))
    with _c3:
        if st.button("👍", key="like_" + str(idx), help="Like"):
            st.session_state.liked.add(idx)
            st.rerun()
    # Speaker dabane pe us jawab ki awaaz bajao (sirf tabhi, auto nahi)
    if st.session_state.play_idx == idx:
        _audio = speak(text)
        if _audio:
            st.audio(_audio, format="audio/mp3")
    # Like karne pe friendly message
    if idx in st.session_state.liked:
        st.markdown("<div class='wa-thanks'>Thank you for your feedback, dude! 💚</div>",
                    unsafe_allow_html=True)


# Pichla chat dikhao — labels + action buttons ke saath
for _i, m in enumerate(st.session_state.messages):
    if m.get("photo"):
        _pc1, _pc2 = st.columns([1, 2])
        with _pc2:
            st.image(m["photo"], use_container_width=True)
        if m.get("content"):
            wa_bubble(m["content"], m["role"])
            if m["role"] == "assistant":
                ai_actions(_i, m["content"])
    else:
        wa_bubble(m["content"], m["role"])
        if m["role"] == "assistant":
            ai_actions(_i, m["content"])

prompt = st.chat_input("Message likho...")

# ℹ️ Info button — neeche-left, medium size. Click karne pe chhota pop-up:
# Frequency + Time + Font + Greeting reply + Photo upload.
_info_l, _info_r = st.columns([1, 4])
with _info_l:
    with st.popover("ℹ️ Info", use_container_width=True):
        st.markdown("**Frequency**")
        _freq = st.radio(
            "Frequency",
            ["Low", "Medium", "High"],
            index=["low", "medium", "high"].index(st.session_state.freq),
            horizontal=True, label_visibility="collapsed",
            key="freq_radio")
        st.session_state.freq = _freq.lower()
        st.caption("Low: sources + trained Q&A se meaningful jawab. "
                   "Medium: kai sources se accurate jawab + related info. "
                   "High: sabhi sources jod ke + Gemini verify (hidden).")
        st.markdown("**Time**")
        _time_choice = st.radio(
            "Time",
            ["Less", "Medium", "High"],
            index=["less", "medium", "high"].index(st.session_state.time),
            horizontal=True, label_visibility="collapsed",
            key="time_radio")
        st.session_state.time = _time_choice.lower()
        st.caption("Less: ~1 second. Medium: ~3 seconds. "
                   "High: 3–5 seconds, sabse powerful jawab.")
        st.markdown("**Font**")
        _font = st.selectbox(
            "Font",
            ["Default", "Serif", "Monospace", "Rounded"],
            index=["Default", "Serif", "Monospace", "Rounded"].index(
                st.session_state.font),
            label_visibility="collapsed", key="font_sel")
        if _font != st.session_state.font:
            st.session_state.font = _font
            st.rerun()
        st.markdown("**Greeting reply**")
        _greet = st.checkbox(
            "👋 Greeting pe jawab do",
            value=st.session_state.greet_on, key="greet_chk")
        st.session_state.greet_on = _greet
        st.caption("Off ho to 'hello'/'hlo' pe bhi model se jawab aayega, "
                   "ready-made greeting nahi.")
        st.markdown("**Photo**")
        _photo = st.file_uploader(
            "📷 Photo bhejo — SAIME dekh ke jawab dega",
            type=["jpg", "jpeg", "png", "webp"],
            key="photo_upload")
        if _photo is not None:
            _pkey = _photo.name + "_" + str(_photo.size)
            if st.session_state.get("last_photo_key") != _pkey:
                st.session_state.last_photo_key = _pkey
                _bytes = _photo.getvalue()
                st.session_state.messages.append(
                    {"role": "user", "content": "📷 Photo bheji", "photo": _bytes})
                caption, err = describe_photo(_bytes)
                if caption:
                    st.session_state.photo_caption = caption
                    _ans = ("Photo me ye dikh raha hai: " + caption +
                            ". Is photo ke baare me kuch bhi poochho!")
                elif err == "no_token":
                    _ans = ("Photo mil gayi! 📷 Lekin photo padhne ke liye "
                            "Streamlit me HF_TOKEN add karna hoga (App settings → "
                            "Secrets). Tab SAIME photo dekh ke sare jawab dega!")
                else:
                    _ans = ("Photo mil gayi! 📷 Lekin abhi photo padh nahi paya. "
                            "Dobara try karo!")
                st.session_state.messages.append(
                    {"role": "assistant", "content": _ans})
                st.rerun()

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    wa_bubble(prompt, "user")
    save_idea(prompt)
    # Translation (all-rounder): user ki language pehchano — AI usi language
    # me jawab dega (Hindi, English, Spanish, French...). Ye pehle se built hai.
    _lang = detect_language(prompt)
    _needs_translate = _lang not in ("hi", "en")
    _prompt_en = prompt
    if _needs_translate:
        _tr, _ = translate_text(prompt, "en")
        if _tr:
            _prompt_en = _tr
    tasks = split_tasks(_prompt_en)
    for task in tasks:
        # Photo ka context — agar photo bheji hai to SAIME use dekh ke jawab de
        _task = task
        if st.session_state.get("photo_caption"):
            _task = ("Photo me ye hai: " + st.session_state.photo_caption
                     + ". Sawal: " + task)
        _typing = st.empty()
        _typing.markdown("<div class='wa-typing'>Typing...</div>",
                         unsafe_allow_html=True)
        # TEAM: ek kaam ko kai agents mil ke solve karein
        # (Frequency + Time — Info popup se)
        answer, source, team, steps = run_team(
            _task, st.session_state.messages[:-1], model, tok,
            freq=st.session_state.get("freq", "low"),
            time=st.session_state.get("time", "less"))
        # Jawab wapas user ki language me translate karo
        if _needs_translate:
            _tr_ans, _ = translate_text(answer, _lang)
            if _tr_ans:
                answer = _tr_ans
        _typing.empty()
        # Typing animation se jawab dikhao — jaise AI real time me likh raha ho
        typewriter_bubble(answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer})
    st.rerun()
