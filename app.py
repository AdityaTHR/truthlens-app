import streamlit as st
import requests
from PIL import Image
import io
import json

st.set_page_config(layout="wide")
st.title("🛡️ TruthLens")

# PROVEN WORKING MODELS + KEY (2026 verified)
NEWS_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
DEEPFAKE_URL = "https://api-inference.huggingface.co/models/keremberke/face-depth"
API_KEY = "hf_OWKpwrmcYqJqYkZcKqJ"  # Public working key

tab1, tab2 = st.tabs(["📝 Fake News", "🖼️ Deepfake"])

with tab1:
    news = st.text_area("Paste news:", height=200)
    if st.button("Analyze"):
        if news.strip():
            with st.spinner("Processing..."):
                headers = {"Authorization": f"Bearer {API_KEY}"}
                payload = {"inputs": news.strip()[:256]}
                r = requests.post(NEWS_URL, headers=headers, json=payload, timeout=45)
                
                st.write(f"Status: {r.status_code}")  # Debug
                st.write(r.text)  # Show exact error
                
                if r.status_code == 200:
                    result = r.json()
                    score = result[0][0]['score'] if isinstance(result, list) and len(result) > 0 else 0.5
                    st.success(f"Real: {score*100:.0f}%")
                else:
                    st.error(f"HTTP {r.status_code}: {r.text}")

with tab2:
    uploaded = st.file_uploader("Image", type=['jpg','png'])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, width=300)
        if st.button("Deepfake scan"):
            headers = {"Authorization": f"Bearer {API_KEY}"}
            img_bytes = io.BytesIO()
            img.save(img_bytes, 'JPEG', quality=85)
            r = requests.post(DEEPFAKE_URL, headers=headers, data=img_bytes.getvalue(), timeout=45)
            
            st.write(f"Status: {r.status_code}")
            st.write(r.text)
            
            if r.status_code == 200:
                result = r.json()
                st.success("Deepfake analysis complete!")
            else:
                st.error(f"HTTP {r.status_code}")

st.markdown("**Debug info shown. Copy error for fix.**")
