import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(layout="wide")
st.title("🛡️ TruthLens - Fake News & Deepfake Detector")

API_KEY = "hf_RZqGfVvYkZcKqJqYkZcKqJ"

col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Fake News")
    text = st.text_area("Paste news article:", height=200)
    if st.button("🔍 Analyze", type="primary"):
        url = "https://api-inference.huggingface.co/models/michelecafagna26/bert-fake-news-detection"
        r = requests.post(url, headers={"Authorization": f"Bearer {API_KEY}"}, json={"inputs": text[:500]})
        if r.status_code == 200 and r.json():
            result = r.json()[0][0]
            st.metric("✅ Real Chance", f"{result['score']*100:.0f}%")
            st.balloons()

with col2:
    st.subheader("🖼️ Deepfake")
    uploaded = st.file_uploader("Upload image")
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, width=250)
        if st.button("🤖 Scan"):
            url = "https://api-inference.huggingface.co/models/prithivMLmods/Deep-Fake-Detector-v2-Model"
            img_bytes = io.BytesIO()
            img.save(img_bytes, 'PNG')
            r = requests.post(url, headers={"Authorization": f"Bearer {API_KEY}"}, data=img_bytes.getvalue())
            if r.status_code == 200:
                score = r.json()[0]['score']
                st.metric("🟢 Real Chance", f"{score*100:.0f}%")
                st.progress(score)

st.markdown("**Click & Use Instantly! No Login!** 🤗")
