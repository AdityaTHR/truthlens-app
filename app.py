import streamlit as st
import requests
import json

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="TruthLens", page_icon="🛡️", layout="centered")

# Use the correct HF Fake News model
MODEL_ID = "michelecafagna26/bert-fake-news-detection"

# Get token from Streamlit Secrets
HF_TOKEN = st.secrets.get("HF_TOKEN", "")

# HF Inference API endpoint
HF_BASE = "https://api-inference.huggingface.co/models"
HF_URL = f"{HF_BASE}/{MODEL_ID}"

# ----------------------------
# HELPERS
# ----------------------------
def call_hf_fake_news(text: str) -> dict:
    """
    Call HF Inference API for fake news detection.
    Returns: {"ok": bool, "real": float, "fake": float, "status": int, "msg": str, "raw": any}
    """
    if not HF_TOKEN:
        return {
            "ok": False,
            "status": 0,
            "msg": "❌ Missing HF_TOKEN. Add it to Streamlit Secrets.",
            "raw": None,
            "real": 0.0,
            "fake": 0.0,
        }

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {"inputs": text[:512]}

    try:
        r = requests.post(HF_URL, headers=headers, json=payload, timeout=60)
    except requests.RequestException as e:
        return {
            "ok": False,
            "status": 0,
            "msg": f"❌ Network error: {str(e)}",
            "raw": None,
            "real": 0.0,
            "fake": 0.0,
        }

    # Model warming / cold start
    if r.status_code == 503:
        return {
            "ok": False,
            "status": 503,
            "msg": "⏳ Model is loading. Wait 30-60 seconds and try again.",
            "raw": safe_json(r),
            "real": 0.0,
            "fake": 0.0,
        }

    # Auth problems
    if r.status_code == 401:
        return {
            "ok": False,
            "status": 401,
            "msg": "❌ Unauthorized (401). Check your HF token in Streamlit Secrets.",
            "raw": safe_json(r),
            "real": 0.0,
            "fake": 0.0,
        }

    if not r.ok:
        return {
            "ok": False,
            "status": r.status_code,
            "msg": f"❌ API Error {r.status_code}: {r.text[:200]}",
            "raw": safe_json(r),
            "real": 0.0,
            "fake": 0.0,
        }

    data = safe_json(r)

    # Check for HF error response
    if isinstance(data, dict) and "error" in data:
        return {
            "ok": False,
            "status": r.status_code,
            "msg": f"❌ HF Error: {data.get('error')}",
            "raw": data,
            "real": 0.0,
            "fake": 0.0,
        }

    # Parse predictions (handle both nested [[...]] and flat [...] formats)
    preds = []
    if isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], list):
            preds = data[0]
        elif isinstance(data[0], dict):
            preds = data

    real = 0.0
    fake = 0.0

    for p in preds:
        if not isinstance(p, dict):
            continue

        lab = str(p.get("label", "")).upper()
        sc = float(p.get("score", 0.0))

        # Handle multiple label conventions
        if ("REAL" in lab) or (lab in {"LABEL_1", "POSITIVE"}):
            real = max(real, sc)
        if ("FAKE" in lab) or (lab in {"LABEL_0", "NEGATIVE"}):
            fake = max(fake, sc)

    # Normalize scores
    s = real + fake
    if s > 0:
        real, fake = real / s, fake / s

    return {
        "ok": True,
        "status": 200,
        "real": real,
        "fake": fake,
        "raw": data,
        "msg": "✅ Analysis complete",
    }


def safe_json(resp):
    """Safely extract JSON from response."""
    try:
        return resp.json()
    except Exception:
        return {"error": resp.text[:200]}


def verdict_class(real: float, fake: float) -> tuple[str, str]:
    """Determine verdict class and message."""
    if real >= 0.65:
        return (
            "authentic",
            "✅ <strong>LIKELY AUTHENTIC</strong><br>This news appears genuine based on AI analysis.",
        )
    if fake >= 0.65:
        return (
            "fake",
            "🚨 <strong>LIKELY FAKE</strong><br>This news shows characteristics of misinformation.",
        )
    return (
        "uncertain",
        "⚠️ <strong>UNCERTAIN</strong><br>Analysis inconclusive. Verify with multiple sources.",
    )


# ----------------------------
# UI STYLING
# ----------------------------
st.markdown(
    """
<style>
    /* Page background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    [data-testid="stToolbar"] { right: 2rem; }

    .container {
        max-width: 900px;
        margin: 20px auto;
        background: white;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
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
        margin: 0 0 10px 0;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 15px;
    }
    
    .header p {
        font-size: 1.1em;
        opacity: 0.9;
        margin: 0;
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

    /* Streamlit text area styling */
    .stTextArea textarea {
        border-radius: 10px !important;
        border: 2px solid #ddd !important;
        font-family: 'Segoe UI', sans-serif !important;
    }

    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.2) !important;
    }

    /* Button styling */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 15px 30px !important;
        font-size: 1.1em !important;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
    }

    .stButton > button[kind="secondary"] {
        color: #667eea !important;
        border: 2px solid #667eea !important;
        background: white !important;
    }

    .stButton > button[kind="secondary"]:hover {
        background: #f5f7ff !important;
    }

    /* Result cards */
    .result-card {
        background: #f8f9fa;
        border-left: 5px solid #667eea;
        padding: 25px;
        border-radius: 10px;
        margin-top: 20px;
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
        transition: width 0.5s ease-out;
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

<div class="container">
  <div class="header">
    <h1><span>🛡️</span>TruthLens</h1>
    <p>AI-Powered Fake News Detection</p>
  </div>
  <div class="content">
    <div class="info-box">
      <strong>ℹ️ How it works:</strong> Paste any news article below and our AI will analyze it to detect if it's likely authentic or fake. Results are based on advanced machine learning models.
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# MAIN APP LOGIC
# ----------------------------

# Initialize session state for results
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# Create container for better layout
with st.container():
    st.markdown('<div class="content">', unsafe_allow_html=True)

    # Input textarea
    news_text = st.text_area(
        "Paste News Article:",
        height=150,
        placeholder="Paste the news article text here...",
        key="news_input",
    )

    # Buttons
    col1, col2 = st.columns([1, 1], gap="small")

    with col1:
        analyze_btn = st.button(
            "🔍 Analyze News", type="primary", use_container_width=True, key="analyze"
        )

    with col2:
        clear_btn = st.button("Clear", use_container_width=True, key="clear")

    # Handle clear button
    if clear_btn:
        st.session_state.last_result = None
        st.session_state.news_input = ""
        st.rerun()

    # Handle analyze button
    if analyze_btn:
        text = news_text.strip()

        if len(text) == 0:
            st.error("❌ Please enter some news text to analyze")
        elif len(text) < 20:
            st.error("❌ Please enter at least 20 characters")
        else:
            with st.spinner("🔄 Analyzing... This may take 10-20 seconds"):
                result = call_hf_fake_news(text)
                st.session_state.last_result = result

    # Display results
    if st.session_state.last_result:
        res = st.session_state.last_result

        if not res.get("ok"):
            st.error(res.get("msg", "Unknown error occurred"))
        else:
            real = float(res["real"])
            fake = float(res["fake"])
            score = round(real * 100)
            cls, verdict_html = verdict_class(real, fake)

            st.markdown(
                f"""
<div class="result-card">
  <div class="result-label">Authenticity Score</div>
  <div class="result-value">{score}%</div>
  <div class="progress-bar">
    <div class="progress-fill" style="width: {score}%"></div>
  </div>

  <div class="stats">
    <div class="stat-item">
      <div class="stat-label">Real Probability</div>
      <div class="stat-value">{round(real * 100)}%</div>
    </div>
    <div class="stat-item">
      <div class="stat-label">Fake Probability</div>
      <div class="stat-value">{round(fake * 100)}%</div>
    </div>
  </div>

  <div class="verdict {cls}">
    {verdict_html}
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(
    """
<div class="container">
  <div class="footer">
    <strong>⚠️ Disclaimer:</strong> This tool provides AI-based analysis for informational purposes.
    Always verify with multiple reliable sources. AI predictions are not 100% accurate.
  </div>
</div>
""",
    unsafe_allow_html=True,
)
