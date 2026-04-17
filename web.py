import re
import os
import torch
import warnings
import logging
import streamlit as st
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_stopwords

# --- Silence logs ---
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- CONFIG ---
BEST_MODEL_DIR = "best_model"
BASE_MODEL = "airesearch/wangchanberta-base-att-spm-uncased"
MAX_LEN = 256

# --- Page config ---
st.set_page_config(
    page_title="Thai Fake News Detector",
    page_icon="🛡️",
    layout="wide", # ปรับเป็น wide เพื่อให้มีพื้นที่วางคอลัมน์
)

# --- Custom CSS (ปรับปรุงใหม่) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans Thai', sans-serif; }
.stApp { background: #0e1117; color: #e2e8f0; }

/* Hero Section */
.hero { text-align: center; padding: 3rem 0 2rem; background: linear-gradient(180deg, #161b22 0%, #0e1117 100%); border-radius: 0 0 30px 30px; margin-bottom: 2rem; }
.hero h1 { font-size: 3rem; font-weight: 700; color: #ffffff; letter-spacing: -0.02em; }
.hero h1 span { color: #00d1b2; text-shadow: 0 0 20px rgba(0,209,178,0.3); }

/* Card Styling */
.card { 
    background: #1d232d; 
    border: 1px solid rgba(255,255,255,0.05); 
    border-radius: 16px; 
    padding: 1.5rem; 
    height: 100%;
    transition: transform 0.2s ease;
}
.card:hover { border-color: rgba(0,209,178,0.3); }
.card-title { 
    font-family: 'IBM Plex Mono', monospace; 
    font-size: 0.8rem; 
    color: #8892a4; 
    text-transform: uppercase; 
    margin-bottom: 1rem; 
    display: flex; 
    align-items: center; 
    gap: 8px;
}

/* Verdict Box */
.verdict-box {
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
    border: 1px solid transparent;
}
.verdict-fake { background: rgba(255, 77, 77, 0.1); border-color: #ff4d4d; color: #ff4d4d; }
.verdict-real { background: rgba(0, 209, 178, 0.1); border-color: #00d1b2; color: #00d1b2; }

/* Custom Button */
.stButton > button {
    background: linear-gradient(90deg, #00d1b2 0%, #009eeb 100%) !important;
    color: white !important;
    border: none !important;
    padding: 0.75rem 2rem !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    width: 100%;
    box-shadow: 0 4px 15px rgba(0,209,178,0.2) !important;
}

/* Chips */
.token-chip { background: #2d3748; padding: 4px 10px; border-radius: 6px; font-size: 0.85rem; margin: 2px; display: inline-block; border: 1px solid rgba(255,255,255,0.1); }
</style>
""", unsafe_allow_html=True)

# --- Functions (เหมือนเดิมแต่ปรับปรุง NER) ---
@st.cache_resource(show_spinner=False)
def load_all_models():
    # โหลดพร้อมกันในฟังก์ชันเดียวเพื่อความเร็ว
    base_tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_mdl = AutoModel.from_pretrained(BASE_MODEL)
    fake_tok = AutoTokenizer.from_pretrained(BEST_MODEL_DIR)
    fake_mdl = AutoModelForSequenceClassification.from_pretrained(BEST_MODEL_DIR)
    from pythainlp.tag import NER
    ner_mdl = NER("wangchanberta")
    return base_tok, base_mdl, fake_tok, fake_mdl, ner_mdl

# --- UI Header ---
st.markdown("""
<div class="hero">
    <h1>🍰 Real or Cake 🍰</h1>
    <p style="color: #8892a4; font-size: 1.1rem;">Thai Fake News<span>Detector</span></p>
</div>
""", unsafe_allow_html=True)

# --- Layout ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="card"><div class="card-title">📝 INPUT SECTION</div>', unsafe_allow_html=True)
    text_input = st.text_area("", placeholder="วางข่าวหรือข้อความที่นี่เพื่อเริ่มการวิเคราะห์...", height=250)
    if st.button("🚀 เริ่มวิเคราะห์เดี๋ยวนี้"):
        run_analysis = True
    else:
        run_analysis = False
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("⚙️ Advanced Settings"):
        model_path = st.text_input("Model Path", value=BEST_MODEL_DIR)
        st.caption("ปรับแต่งค่า Model Path กรณีรันบน Colab หรือ Path อื่น")

# --- Logic & Results ---
if run_analysis:
    if not text_input.strip():
        st.error("❌ กรุณาใส่ข้อความก่อน")
    else:
        with st.spinner("🧠 ระบบกำลังประมวลผลด้วย AI..."):
            # โหลดโมเดล
            base_tok, base_mdl, fake_tok, fake_mdl, ner_model = load_all_models()
            
            # Predict
            enc = fake_tok(text_input, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
            with torch.no_grad():
                logits = fake_mdl(**enc).logits
                probs = torch.softmax(logits, dim=1).squeeze().tolist()
            
            is_fake = torch.argmax(logits).item() == 1
            conf = probs[1 if is_fake else 0] * 100

        with col2:
            # ส่วนแสดงผลลัพธ์ (Verdict)
            v_class = "verdict-fake" if is_fake else "verdict-real"
            v_icon = "🚨" if is_fake else "✅"
            v_text = "FAKE NEWS DETECTED" if is_fake else "REAL NEWS VERIFIED"
            
            st.markdown(f"""
            <div class="verdict-box {v_class}">
                <h2 style="margin:0;">{v_icon} {v_text}</h2>
                <p style="margin:0; opacity:0.8;">ความเชื่อมั่นของระบบ: {conf:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

            # Tabs สำหรับข้อมูลเชิงลึก
            tab1, tab2, tab3 = st.tabs(["📊 Statistics", "🏷️ Named Entities", "🔍 Tokens"])
            
            with tab1:
                st.markdown("### Probability Distribution")
                st.write("Real News")
                st.progress(probs[0])
                st.write("Fake News")
                st.progress(probs[1])

            with tab2:
                st.markdown("### Entities found in text")
                entities = ner_model.tag(text_input)
                named = [(w, t) for w, t in entities if t != "O"]
                if named:
                    for w, t in named:
                        st.markdown(f"<span class='token-chip' style='border-left: 3px solid #00d1b2;'><b>{t}</b>: {w}</span>", unsafe_allow_html=True)
                else:
                    st.write("No entities found.")

            with tab3:
                st.markdown("### Tokenization Output")
                tokens = word_tokenize(text_input, engine="newmm")
                chips = "".join([f"<span class='token-chip'>{t}</span>" for t in tokens if t.strip()])
                st.markdown(chips, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #555;'>AISE Prince of Songkla University | NLP Project 2026</p>", unsafe_allow_html=True)