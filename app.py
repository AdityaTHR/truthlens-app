import streamlit as st
import requests
import json

st.set_page_config(page_title="TruthLens", page_icon="🛡️", layout="centered")

# ✅ WORKING MODEL + ENDPOINT COMBO (tested)
MODEL_ID = "Pulk17/Fake-News-Detection"
HF_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

HF_TOKEN = st.secrets.get("HF_TOKEN", "")

def call_hf_fake_news(text: str) -> dict:
    if not HF_TOKEN:
        return {"ok": False, "msg": "❌ Add HF_TOKEN to Streamlit Secrets", "real": 0.0, "fake": 0.0}

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {"inputs": text[:512]}

    try:
        r = requests.post(HF_URL, headers=headers, json=payload, timeout=60)
    except:
        return {"ok": False, "msg": "❌ Network error", "real": 0.0, "fake": 0.0}

    if r.status_code == 503:
        return {"ok": False, "msg": "⏳ Model loading... Wait 30-60s & retry", "real": 0.0, "fake": 0.0}
    if r.status_code == 401:
        return {"ok": False, "msg": "❌ Invalid HF token", "real": 0.0, "fake": 0.0}
    if not r.ok:
        return {"ok": False, "msg": f"❌ Error {r.status_code}", "real": 0.0, "fake": 0.0}

    try:
        data = r.json()
    except:
        return {"ok": False, "msg": "❌ Invalid response", "real": 0.0, "fake": 0.0}

    preds = data[0] if isinstance(data, list) and data else []
    real, fake = 0.0, 0.0

    for p in preds:
        label = str(p.get("label", "")).upper()
        score = float(p.get("score", 0.0))
        
        if "REAL" in label or "LABEL_1" in label:
            real = max(real, score)
        if "FAKE" in label or "LABEL_0" in label:
            fake = max(fake, score)

    total = real + fake
    if total > 0:
        real, fake = real/total, fake/total

    return {"ok": True, "real": real, "fake": fake}

def verdict(real, fake):
    if real >= 0.65:
        return "authentic", "✅ <strong>LIKELY AUTHENTIC</strong><br>Genuine news based on AI analysis."
    if fake >= 0.65:
        return "fake", "🚨 <strong>LIKELY FAKE</strong><br>Shows misinformation characteristics."
    return "uncertain", "⚠️ <strong>UNCERTAIN</strong><br>Verify with multiple sources."

# UI
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);}
[data-testid="stHeader"] {background: transparent;}
.container {max-width:900px;margin:20px auto;background:white;border-radius:20px;box-shadow:0 20px 60px rgba(0,0,0,0.3);}
.header {background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:40px;text-align:center;}
.header h1 {font-size:2.5em;display:flex;align-items:center;justify-content:center;gap:15px;}
.content {padding:40px;}
.info-box {background:#e7f3ff;color:#004085;padding:15px;border-radius:10px;margin-bottom:20px;border-left:5px solid #0066cc;}
.stTextArea textarea {border-radius:10px !important;border:2px solid #ddd !important;}
.stTextArea textarea:focus {border-color:#667eea !important;box-shadow:0 0 10px rgba(102,126,234,0.2) !important;}
.stButton>button {border-radius:10px !important;font-weight:700 !important;padding:15px 30px !important;font-size:1.1em !important;}
.stButton>button[kind="primary"] {background:linear-gradient(135deg,#667eea 0%,#764ba2 100%) !important;border:none !important;}
.stButton>button[kind="secondary"] {color:#667eea !important;border:2px solid #667eea !important;background:white !important;}
.result-card {background:#f8f9fa;border-left:5px solid #667eea;padding:25px;border-radius:10px;margin-top:20px;}
.result-label {font-size:0.9em;color:#666;text-transform:uppercase;letter-spacing:1px;font-weight:600;}
.result-value {font-size:2.5em;font-weight:700;color:#667eea;margin-bottom:15px;}
.progress-bar {height:10px;background:#e0e0e0;border-radius:10px;overflow:hidden;margin-top:10px;}
.progress-fill {height:100%;background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);border-radius:10px;}
.stats {display:grid;grid-template-columns:1fr 1fr;gap:15px;margin-top:20px;}
.stat-item {background:white;padding:15px;border-radius:10px;border:1px solid #ddd;text-align:center;}
.stat-label {font-size:0.9em;color:#666;font-weight:600;}
.stat-value {font-size:1.8em;font-weight:700;color:#667eea;margin-top:5px;}
.verdict {margin-top:20px;padding:20px;border-radius:10px;text-align:center;font-size:1.2em;font-weight:600;}
.verdict.authentic {background:#d4edda;color:#155724;border-left:5px solid #28a745;}
.verdict.fake {background:#f8d7da;color:#721c24;border-left:5px solid #dc3545;}
.verdict.uncertain {background:#fff3cd;color:#856404;border-left:5px solid #ffc107;}
.footer {background:#f8f9fa;padding:20px;text-align:center;color:#666;font-size:0.9em;border-top:1px solid #ddd;}
</style>
<div class="container">
<div class="header"><h1><span>🛡️</span>TruthLens</h1><p>AI-Powered Fake News Detection</p></div>
<div class="content">
<div class="info-box"><strong>ℹ️ Model:</strong> Pulk17/Fake-News-Detection (BERT-based)<br>Paste news → Get authenticity score instantly!</div>
""", unsafe_allow_html=True)

if "result" not in st.session_state:
    st.session_state.result = None

news_text = st.text_area("Paste News Article:", height=150, placeholder="Paste news article here...", key="input")
col1, col2 = st.columns([1,1])
if col1.button("🔍 Analyze", type="primary", use_container_width=True):
    if len(news_text.strip()) < 20:
        st.error("❌ Enter 20+ characters")
    else:
        with st.spinner("Analyzing..."):
            st.session_state.result = call_hf_fake_news(news_text)

if col2.button("Clear", use_container_width=True):
    st.session_state.result = None
    st.rerun()

if st.session_state.result:
    res = st.session_state.result
    st.markdown('<div class="content">', unsafe_allow_html=True)
    
    if not res["ok"]:
        st.error(res["msg"])
    else:
        real, fake = res["real"], res["fake"]
        score = round(real * 100)
        cls, html = verdict(real, fake)
        
        st.markdown(f'''
<div class="result-card">
<div class="result-label">Authenticity Score</div>
<div class="result-value">{score}%</div>
<div class="progress-bar"><div class="progress-fill" style="width:{score}%"></div></div>
<div class="stats">
<div class="stat-item"><div class="stat-label">Real</div><div class="stat-value">{round(real*100)}%</div></div>
<div class="stat-item"><div class="stat-label">Fake</div><div class="stat-value">{round(fake*100)}%</div></div>
</div>
<div class="verdict {cls}">{html}</div>
</div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div></div>', unsafe_allow_html=True)

st.markdown("""
<div class="container"><div class="footer">
<strong>⚠️ Disclaimer:</strong> AI analysis only. Verify with trusted sources.
</div></div>
""", unsafe_allow_html=True)
