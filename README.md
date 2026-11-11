# SAIME — Streamlit App

SAIME — tumhara khud ka personal AI (SATYA V1 model, 49.6M params, Hindi + English + Hinglish).

**Live app:** Streamlit Community Cloud par deploy hoti hai (free).

**Model:** Hugging Face Hub se load hota hai — [`satyamkumar78131/saime-model`](https://huggingface.co/satyamkumar78131/saime-model) (step-22000 checkpoint, 567 MB).

**Features:**
- Chat (memory ke saath), FAQ lookup, few-shot prompt
- Web search FIRST for factual questions (30+ sources, source ke saath)
- Gibberish detection — model bakwas de to web se jawab
- Voice (gTTS) — jawab ko awaaz me suno
- Ideas collection — chat me idea aaye to save hota hai

**Honest limits (free tier):**
- Photo padhna (BLIP) is version me nahi hai — Streamlit free tier me 1 GB RAM milti hai.
- Chat history / learned data sirf app chalte hue save hoti hai (server restart par khatm).
- Koi use na kare to app so jata hai, kholte hi 1-2 min me jaag jata hai.
