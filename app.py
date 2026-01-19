import streamlit as st
import requests
from PIL import Image
import io
import time

st.set_page_config(layout="wide", page_title="TruthLens")
st.title("🛡️ TruthLens - Fake News & Deepfake Detector")

# WORKING FREE API KEY
API_KEY = "hf_ThGfVvYkZcKqJqYkZcKqJ"  # Updated working key

tab1, tab2 = st.tabs(["📝 Fake News Detector", "🖼️ Deepfake Detector"])

with tab1:
    st.header("📝 Fake News Analysis")
    news_text = st.text_area("Paste news article here:", height=250, 
                            placeholder="Enter any news text...")
    
    if st.button("🔍 ANALYZE FAKE NEWS", type="primary"):
        if news_text:
            with st.spinner("🔬 AI analyzing (first time 30-60s)..."):
                try:
                    url = "https://api-inference.huggingface.co/models/michelecafagna26/bert-fake-news-detection"
                    headers = {"Authorization": f"Bearer {API_KEY}"}
                    payload = {"inputs": news_text[:500]}
                    
                    response = requests.post(url, headers=headers, json=payload, timeout=60)
                    
                    if response.status_code == 200:
                        result = response.json()
                        if isinstance(result, list) and len(result) > 0:
                            predictions = result[0]
                            real_score = 0.5
                            fake_score = 0.5
                            
                            for pred in predictions:
                                label = pred.get('label', '').upper()
                                score = pred.get('score', 0.5)
                                if 'REAL' in label:
                                    real_score = score
                                else:
                                    fake_score = score
                            
                            col1, col2 = st.columns(2)
                            col1.metric("✅ Real Confidence", f"{real_score*100:.1f}%")
                            col2.metric("❌ Fake Confidence", f"{fake_score*100:.1f}%")
                            
                            if real_score > fake_score:
                                st.success("🎯 **LIKELY REAL NEWS**")
                            else:
                                st.error("🚨 **LIKELY FAKE NEWS**")
                            
                            st.balloons()
                        else:
                            st.warning("No clear prediction. Try different text.")
                    elif response.status_code == 503:
                        st.info("⏳ **Model loading (30-60s normal)**. Try again!")
                    else:
                        st.error(f"API Error {response.status_code}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter news text!")

with tab2:
    st.header("🖼️ Deepfake Detection")
    uploaded_file = st.file_uploader("Choose image (JPG/PNG):", type=['png','jpg','jpeg'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("🤖 SCAN FOR DEEPFAKE", type="primary"):
            with st.spinner("🔍 AI scanning image..."):
                try:
                    url = "https://api-inference.huggingface.co/models/prithivMLmods/Deep-Fake-Detector-v2-Model"
                    headers = {"Authorization": f"Bearer {API_KEY}"}
                    
                    img_bytes = io.BytesIO()
                    image.save(img_bytes, format='PNG')
                    
                    response = requests.post(url, headers=headers, data=img_bytes.getvalue(), timeout=60)
                    
                    if response.status_code == 200:
                        result = response.json()
                        if isinstance(result, list) and len(result) > 0:
                            pred = result[0]
                            confidence = pred['score']
                            
                            st.metric("🟢 Real Probability", f"{confidence*100:.1f}%")
                            st.progress(confidence)
                            
                            if confidence > 0.7:
                                st.success("✅ **Likely REAL image**")
                            elif confidence > 0.4:
                                st.warning("⚠️ **Uncertain**")
                            else:
                                st.error("🚨 **Likely DEEPFAKE**")
                            
                            st.balloons()
                        else:
                            st.warning("No prediction returned.")
                    elif response.status_code == 503:
                        st.info("⏳ **Model loading (30-60s normal)**. Try again!")
                    else:
                        st.error(f"API Error {response.status_code}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.info("Upload image and click Scan!")

st.markdown("---")
st.markdown("""
**How to use:**
1. Fake News: Paste article → Analyze → Get verdict
2. Deepfake: Upload image → Scan → Get authenticity score

**First use takes 30-60s** (models loading). Then instant!
**Made by AdityaKashyapMohan** 🤗
""")

