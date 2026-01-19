import streamlit as st
import requests
import json
import time

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="TruthLens - Fake News Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================
# CONFIG & CONSTANTS
# ============================================
MODEL_ID = "hamzab/roberta-fake-news-classification"




HF_BASE = "https://router.huggingface.co/hf-inference/models"
HF_URL = f"{HF_BASE}/{MODEL_ID}"


# Get token from Streamlit Secrets (NOT hardcoded)
HF_TOKEN = st.secrets.get("HF_TOKEN", "")

# ============================================
# HELPER FUNCTIONS
# ============================================

def safe_json(resp):
    """Safely parse JSON from response."""
    try:
        return resp.json()
    except Exception:
        return resp.text


def call_hf_model(text: str) -> dict:
    """
    Call Hugging Face model API.
    
    Returns:
      {
        "ok": bool,
        "real": float (0.0-1.0),
        "fake": float (0.0-1.0),
        "status": int,
        "msg": str,
        "raw": any
      }
    """
    if not HF_TOKEN:
        return {
            "ok": False,
            "status": 0,
            "msg": "❌ Missing HF_TOKEN in Streamlit Secrets. Contact admin.",
            "raw": None,
            "real": 0,
            "fake": 0
        }

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {"inputs": f"<title>{text[:256]}<content>{text[:512]}<end>"}


    try:
        r = requests.post(HF_URL, headers=headers, json=payload, timeout=60)
    except requests.RequestException as e:
        return {
            "ok": False,
            "status": 0,
            "msg": f"❌ Network Error: {str(e)}",
            "raw": None,
            "real": 0,
            "fake": 0
        }

    # Handle model warming (503 = model loading)
    if r.status_code == 503:
        return {
            "ok": False,
            "status": 503,
            "msg": "⏳ Model is warming up on Hugging Face. Wait 30-60s and try again.",
            "raw": safe_json(r),
            "real": 0,
            "fake": 0
        }

    # Handle auth errors (401 = invalid token)
    if r.status_code == 401:
        return {
            "ok": False,
            "status": 401,
            "msg": "❌ Unauthorized (401). HF token is missing/invalid. Update Streamlit Secrets.",
            "raw": safe_json(r),
            "real": 0,
            "fake": 0
        }

    # Handle other errors
    if not r.ok:
        error_msg = safe_json(r)
        return {
            "ok": False,
            "status": r.status_code,
            "msg": f"❌ API Error ({r.status_code}): {str(error_msg)[:200]}",
            "raw": error_msg,
            "real": 0,
            "fake": 0
        }

    # Parse successful response
    data = safe_json(r)
    st.write("DEBUG raw HF response:", data)


    # Expected format: [[{"label":"REAL","score":0.9},{"label":"FAKE","score":0.1}]]
    preds = []
        # If HF returns an error JSON, stop early
    if isinstance(data, dict) and "error" in data:
        return {
            "ok": False,
            "status": r.status_code,
            "msg": f"HF error: {data.get('error')}",
            "raw": data,
            "real": 0.0,
            "fake": 0.0,
        }

    # Accept both formats:
    # A) [[{...},{...}]]  (nested)
    # B) [{...},{...}]    (flat)
    preds = []
    if isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], list):
            preds = data[0]
        elif isinstance(data[0], dict):
            preds = data

    real = 0.0 
    fake = 0.0
    for p in preds:
        lab = str(p.get("label", "")).upper()
        sc = float(p.get("score", 0.0))
    
        if any(x in lab for x in ["REAL", "LABEL_1", "TRUE", "LEGIT"]):
            real = max(real, sc)
        elif any(x in lab for x in ["FAKE", "LABEL_0", "FALSE"]):
            fake = max(fake, sc)

    return {"ok": True, "status": 200, "real": real, "fake": fake, "raw": data, "msg": "ok"}




def get_verdict(real: float, fake: float) -> tuple:
    """
    Get verdict text and emoji based on scores.
    Returns: (verdict_text, emoji, class_name)
    """
    if real >= 0.65:
        return (
            "✅ <strong>LIKELY AUTHENTIC</strong><br>This news appears to be genuine based on AI analysis.",
            "authentic"
        )
    elif fake >= 0.65:
        return (
            "🚨 <strong>LIKELY FAKE</strong><br>This news shows characteristics of misinformation.",
            "fake"
        )
    else:
        return (
            "⚠️ <strong>UNCERTAIN</strong><br>Analysis inconclusive. Verify with multiple sources.",
            "uncertain"
        )


# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }

        .main {
            background: transparent;
            padding: 0 !important;
        }

        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px !important;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .stTabs [role="tablist"] button {
            display: none;
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
        }

        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }

        .content {
            padding: 40px;
        }

        .info-box {
            background: #e7f3ff;
            color: #004085;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 5px solid #0066cc;
            font-size: 0.95em;
            line-height: 1.6;
        }

        .result-card {
            background: #f8f9fa;
            border-left: 5px solid #667eea;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 20px;
        }

        .result-label {
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
            font-weight: 600;
        }

        .result-value {
            font-size: 2.5em;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 15px;
        }

        .progress-bar {
            height: 10px;
            background: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 10px;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }

        .stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 20px;
        }

        .stat-item {
            background: white;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #ddd;
            text-align: center;
        }

        .stat-label {
            font-size: 0.9em;
            color: #666;
            font-weight: 600;
        }

        .stat-value {
            font-size: 1.8em;
            font-weight: 700;
            color: #667eea;
            margin-top: 5px;
        }

        .verdict {
            margin-top: 20px;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 1.2em;
            font-weight: 600;
        }

        .verdict.authentic {
            background: #d4edda;
            color: #155724;
            border-left: 5px solid #28a745;
        }

        .verdict.fake {
            background: #f8d7da;
            color: #721c24;
            border-left: 5px solid #dc3545;
        }

        .verdict.uncertain {
            background: #fff3cd;
            color: #856404;
            border-left: 5px solid #ffc107;
        }

        .footer {
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
            border-top: 1px solid #ddd;
        }

        @media (max-width: 768px) {
            .header h1 {
                font-size: 1.8em;
            }

            .result-value {
                font-size: 2em;
            }

            .stats {
                grid-template-columns: 1fr;
            }
        }
    </style>
""", unsafe_allow_html=True)

# ============================================
# UI STRUCTURE
# ============================================

# Header
with st.container():
    st.markdown("""
        <div class="container">
            <div class="header">
                <h1>🛡️ TruthLens</h1>
                <p>AI-Powered Fake News Detection</p>
            </div>
            <div class="content">
                <div class="info-box">
                    <strong>ℹ️ How it works:</strong> Paste any news article below and our AI will analyze it to detect if it's likely authentic or fake. 
                    Results are based on advanced machine learning models trained on thousands of articles.
                </div>
    """, unsafe_allow_html=True)

# Session state for storing results
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.real_score = 0
    st.session_state.fake_score = 0

# Input section
col1, col2 = st.columns([1, 0.2])

with col1:
    news_text = st.text_area(
        label="Paste News Article:",
        placeholder="Paste the news article text here...",
        height=150,
        key="news_input"
    )

# Buttons
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    analyze_clicked = st.button("🔍 Analyze News", use_container_width=True, key="analyze_btn")

with col_btn2:
    clear_clicked = st.button("Clear", use_container_width=True, key="clear_btn")

# Handle clear button
if clear_clicked:
    st.session_state.analysis_done = False
    st.session_state.real_score = 0
    st.session_state.fake_score = 0
    st.rerun()

# Handle analyze button
if analyze_clicked:
    # Validation
    if not news_text or len(news_text.strip()) == 0:
        st.error("❌ Please enter some news text to analyze")
    elif len(news_text.strip()) < 20:
        st.error("❌ Please enter at least 20 characters")
    else:
        with st.spinner("⏳ Analyzing... This may take 10-20 seconds"):
            result = call_hf_model(news_text)

            if result["ok"]:
                st.session_state.analysis_done = True
                st.session_state.real_score = result["real"]
                st.session_state.fake_score = result["fake"]
            else:
                st.error(result["msg"])

# Display results if analysis is done
if st.session_state.analysis_done:
    real = st.session_state.real_score
    fake = st.session_state.fake_score
    score_percent = int(real * 100)

    verdict_text, verdict_class = get_verdict(real, fake)

    st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Authenticity Score</div>
            <div class="result-value">{score_percent}%</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {score_percent}%"></div>
            </div>
        </div>

        <div class="stats">
            <div class="stat-item">
                <div class="stat-label">Real Probability</div>
                <div class="stat-value">{int(real * 100)}%</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Fake Probability</div>
                <div class="stat-value">{int(fake * 100)}%</div>
            </div>
        </div>

        <div class="verdict {verdict_class}">
            {verdict_text}
        </div>
    """, unsafe_allow_html=True)

# Close container and add footer
st.markdown("""
            </div>
        </div>

        <div style="background: #f8f9fa; padding: 20px; text-align: center; color: #666; font-size: 0.9em; border-top: 1px solid #ddd; margin-top: 0; border-radius: 0 0 20px 20px; max-width: 900px; margin-left: auto; margin-right: auto; margin-bottom: 0;">
            <strong>⚠️ Disclaimer:</strong> This tool provides AI-based analysis for informational purposes. Always verify with multiple reliable sources. AI predictions are not 100% accurate.
        </div>
    """, unsafe_allow_html=True)
