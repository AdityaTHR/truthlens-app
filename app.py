import streamlit as st
import requests
import json

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="TruthLens", page_icon="🛡️", layout="centered")

MODEL_ID = "michelecafagna26/bert-fake-news-detection"

# IMPORTANT:
# Put your token in Streamlit Cloud Secrets as:
# HF_TOKEN = "hf_..."
HF_TOKEN = st.secrets.get("HF_TOKEN", "")

# HF has moved away from the old api-inference endpoint in many setups.
# Use the router endpoint for HF Inference routing.
HF_BASE = "https://router.huggingface.co/hf-inference/models"
HF_URL = f"{HF_BASE}/{MODEL_ID}"

# ----------------------------
# HELPERS
# ----------------------------
def call_hf_fake_news(text: str) -> dict:
    """
    Returns:
      {
        "ok": bool,
        "real": float,
        "fake": float,
        "status": int,
        "raw": any,
        "msg": str
      }
    """
    if not HF_TOKEN:
        return {"ok": False, "status": 0, "msg": "Missing HF_TOKEN in Streamlit Secrets.", "raw": None}

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {"inputs": text[:512]}  # keep within reasonable size
    try:
        r = requests.post(HF_URL, headers=headers, json=payload, timeout=60)
    except requests.RequestException as e:
        return {"ok": False, "status": 0, "msg": f"Network error: {e}", "raw": None}

    # Model warming / cold start
    if r.status_code == 503:
        return {
            "ok": False,
            "status": 503,
            "msg": "Model is loading on Hugging Face. Wait ~30–60s and click Analyze again.",
            "raw": safe_json(r),
        }

    # Auth problems
    if r.status_code == 401:
        return {
            "ok": False,
            "status": 401,
            "msg": "Unauthorized (401). Your HF token is missing/invalid/revoked. Update Streamlit Secrets.",
            "raw": safe_json(r),
        }

    if not r.ok:
        return {"ok": False, "status": r.status_code, "msg": r.text[:300], "raw": safe_json(r)}

    data = safe_json(r)

    # Expected: [[{"label":"REAL","score":...},{"label":"FAKE","score":...}]]
    preds = []
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
        preds = data[0]

    real = 0.0
    fake = 0.0
    for p in preds:
        lab = str(p.get("label", "")).upper()
        sc = float(p.get("score", 0.0))
        if "REAL" in lab:
            real = max(real, sc)
        if "FAKE" in lab:
            fake = max(fake, sc)

    s = real + fake
    if s > 0:
        real, fake = real / s, fake / s

    return {"ok": True, "status": 200, "real": real, "fake": fake, "raw": data, "msg": "ok"}


def safe_json(resp):
    try:
        return resp.json()
    except Exception:
        return resp.text


def verdict_class(real: float, fake: float) -> tuple[str, str]:
    # Returns (class_name, html_text)
    if real >= 0.65:
        return ("authentic", "✅ <strong>LIKELY AUTHENTIC</strong><br>This news appears genuine based on AI analysis.")
    if fake >= 0.65:
        return ("fake", "🚨 <strong>LIKELY FAKE</strong><br>This news shows characteristics of misinformation.")
    return ("uncertain", "⚠️ <strong>UNCERTAIN</strong><br>Analysis inconclusive. Verify with multiple sources.")

# ----------------------------
# UI (Your same design style)
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
        padding: 35px;
        text-align: center;
    }
    .header h1 {
        font-size: 2.3em;
        margin: 0 0 6px 0;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
    }
    .header p { font-size: 1.05em; opacity: 0.9; margin: 0; }
    .content { padding: 28px; }

    .info-box {
        background: #e7f3ff;
        color: #004085;
        padding: 14px;
        border-radius: 10px;
        margin-bottom: 18px;
        border-left: 5px solid #0066cc;
        font-size: 0.95em;
        line-height: 1.6;
    }

    /* Buttons */
    .btn-row { display: flex; gap: 12px; margin-top: 10px; }
    .stButton button {
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 0.7rem 1rem !important;
    }
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
    }

    /* Result blocks */
    .result-card {
        background: #f8f9fa;
        border-left: 5px solid #667eea;
        padding: 22px;
        border-radius: 10px;
        margin-top: 18px;
    }
    .result-label {
        font-size: 0.9em;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
        font-weight: 700;
    }
    .result-value {
        font-size: 2.4em;
        font-weight: 800;
        color: #667eea;
        margin-bottom: 10px;
    }
    .progress-bar {
        height: 10px;
        background: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        margin-top: 8px;
    }
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }

    .stats {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin-top: 14px;
    }
    .stat-item {
        background: white;
        padding: 14px;
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
        font-size: 1.6em;
        font-weight: 800;
        color: #667eea;
        margin-top: 4px;
    }

    .verdict {
        margin-top: 14px;
        padding: 16px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.1em;
        font-weight: 700;
    }
    .verdict.authentic { background: #d4edda; color: #155724; border-left: 5px solid #28a745; }
    .verdict.fake      { background: #f8d7da; color: #721c24; border-left: 5px solid #dc3545; }
    .verdict.uncertain { background: #fff3cd; color: #856404; border-left: 5px solid #ffc107; }

    .footer {
        background: #f8f9fa;
        padding: 16px;
        text-align: center;
        color: #666;
        font-size: 0.9em;
        border-top: 1px solid #ddd;
    }

    @media (max-width: 768px) {
        .header h1 { font-size: 1.7em; }
        .result-value { font-size: 2em; }
        .stats { grid-template-columns: 1fr; }
    }
</style>

<div class="container">
  <div class="header">
    <h1><span>🛡️</span>TruthLens</h1>
    <p>AI-Powered Fake News Detection</p>
  </div>
  <div class="content">
    <div class="info-box">
      <strong>ℹ️ How it works:</strong> Paste any news article below and our AI will analyze it to detect if it's likely authentic or fake.
      Results are based on a BERT classifier hosted on Hugging Face.
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Put Streamlit inputs BELOW, but inside the same visual style by spacing
container = st.container()

with container:
    # Main input
    news_text = st.text_area(
        "Paste News Article:",
        height=180,
        placeholder="Paste the news article text here...",
    )

    c1, c2 = st.columns([1, 1])
    analyze = c1.button("🔍 Analyze News", type="primary", use_container_width=True)
    clear = c2.button("Clear", use_container_width=True)

    if clear:
        st.session_state.pop("last_result", None)
        st.rerun()

    # Validate & run
    if analyze:
        t = news_text.strip()
        if len(t) == 0:
            st.error("❌ Please enter some news text to analyze.")
        elif len(t) < 20:
            st.error("❌ Please enter at least 20 characters.")
        else:
            with st.spinner("Analyzing... This may take 10–20 seconds (first time can be slower)."):
                res = call_hf_fake_news(t)
            st.session_state["last_result"] = res

    # Display results
    res = st.session_state.get("last_result")
    if res:
        if not res.get("ok"):
            st.error(f"❌ {res.get('msg', 'Unknown error')}")
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
      <div class="stat-value">{round(real*100)}%</div>
    </div>
    <div class="stat-item">
      <div class="stat-label">Fake Probability</div>
      <div class="stat-value">{round(fake*100)}%</div>
    </div>
  </div>

  <div class="verdict {cls}">
    {verdict_html}
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

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
