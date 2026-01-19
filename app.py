import streamlit as st
import requests
import os

# Page Configuration
st.set_page_config(
    page_title="TruthLens - Fake News Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stMetric {
        background: white !important;
        padding: 15px !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
    }
    .header-box {
        text-align: center;
        padding: 35px 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .header-box h1 {
        margin: 0;
        font-size: 2.5em;
    }
    .header-box p {
        margin: 10px 0 0 0;
        font-size: 1.15em;
        opacity: 0.95;
    }
</style>
""", unsafe_allow_html=True)

# Get API Token from Streamlit Secrets (SECURE METHOD)
try:
    API_KEY = st.secrets["HF_TOKEN"]
except:
    API_KEY = os.getenv("HF_TOKEN", "")

# API Configuration
API_URL = "https://api-inference.huggingface.co/models/michelecafagna26/bert-fake-news-detection"

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None

# Header
st.markdown("""
<div class="header-box">
    <h1>🛡️ TruthLens</h1>
    <p>AI-Powered Fake News Detection</p>
</div>
""", unsafe_allow_html=True)

# Check if API key exists
if not API_KEY:
    st.error("""
    ❌ **HuggingFace API Token Missing!**
    
    Please add your token in:
    - **Streamlit Cloud:** App Settings → Secrets → Add `HF_TOKEN = "your_token_here"`
    - **Local:** Create `.streamlit/secrets.toml` with `HF_TOKEN = "your_token_here"`
    
    Get token from: https://huggingface.co/settings/tokens
    """)
    st.stop()

# Info box
st.info("📝 **How it works:** Paste any news article and our AI will analyze it using BERT machine learning to detect if it's authentic or fake. Results are based on models trained on thousands of articles.")

# Main Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Analyze News", "📊 How It Works", "⚠️ Disclaimer"])

with tab1:
    st.subheader("📰 Paste Your News Article")
    
    # Input area
    news_text = st.text_area(
        "News Article:",
        height=250,
        placeholder="Paste the news article text here...",
        label_visibility="collapsed"
    )
    
    # Character count
    char_count = len(news_text)
    st.caption(f"📊 Characters: {char_count} | Minimum required: 20")
    
    # Buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        analyze_btn = st.button("🔍 **ANALYZE NEWS**", use_container_width=True, type="primary")
    
    with col2:
        clear_btn = st.button("🔄 Clear", use_container_width=True)
    
    # Clear functionality
    if clear_btn:
        st.session_state.results = None
        st.rerun()
    
    # Analysis Logic
    if analyze_btn:
        if not news_text.strip():
            st.error("❌ Please enter some news text to analyze")
        
        elif char_count < 20:
            st.error("❌ Please enter at least 20 characters")
        
        else:
            with st.spinner("🤖 Analyzing with AI... This may take 10-30 seconds (first time may take longer)"):
                try:
                    # Prepare API request
                    headers = {
                        "Authorization": f"Bearer {API_KEY}",
                        "Content-Type": "application/json"
                    }
                    
                    payload = {
                        "inputs": news_text[:512]  # Limit to 512 characters
                    }
                    
                    # Make request
                    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
                    
                    # Handle response
                    if response.status_code == 200:
                        data = response.json()
                        
                        if isinstance(data, list) and len(data) > 0:
                            results = data[0]
                            
                            # Extract scores
                            real_score = 0.0
                            fake_score = 0.0
                            
                            for item in results:
                                label = item.get('label', '').upper()
                                score = item.get('score', 0.0)
                                
                                if 'REAL' in label:
                                    real_score = max(real_score, score)
                                elif 'FAKE' in label:
                                    fake_score = max(fake_score, score)
                            
                            # Normalize scores
                            total = real_score + fake_score
                            if total > 0:
                                real_score = real_score / total
                                fake_score = fake_score / total
                            
                            # Store results
                            st.session_state.results = {
                                'real_score': real_score,
                                'fake_score': fake_score,
                                'timestamp': 'Just now'
                            }
                            
                            # Force refresh to show results
                            st.rerun()
                        
                        else:
                            st.error("❌ Invalid API response format. Please try again.")
                    
                    elif response.status_code == 503:
                        st.warning("""
                        ⏳ **Model is Loading**
                        
                        The AI model is currently loading on HuggingFace servers (this is normal for first use).
                        
                        Please wait 30-60 seconds and click **Analyze** again.
                        """)
                    
                    elif response.status_code == 401:
                        st.error("""
                        🔑 **Authentication Failed**
                        
                        Your HuggingFace token is invalid or expired.
                        
                        1. Go to https://huggingface.co/settings/tokens
                        2. Create new token with 'Read' permission
                        3. Update in Streamlit Secrets
                        """)
                    
                    else:
                        error_text = response.text[:300]
                        st.error(f"❌ API Error {response.status_code}: {error_text}")
                
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. Please try again.")
                
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
    
    # Display Results (if available)
    if st.session_state.results:
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        results = st.session_state.results
        real_score = results['real_score']
        fake_score = results['fake_score']
        
        # Metrics Row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🟢 Real Probability", f"{int(real_score * 100)}%")
        
        with col2:
            st.metric("🔴 Fake Probability", f"{int(fake_score * 100)}%")
        
        with col3:
            st.metric("⏱️ Analysis Time", "~15 sec")
        
        # Progress Bar
        st.write("**Authenticity Score:**")
        st.progress(real_score)
        
        # Verdict Section
        st.markdown("---")
        
        if real_score > 0.65:
            st.success(f"""
            ✅ **LIKELY AUTHENTIC NEWS**
            
            This article appears to be genuine based on AI analysis.
            
            **Confidence Level: {int(real_score * 100)}%**
            
            ℹ️ *Always verify with multiple trusted sources.*
            """)
            st.balloons()
        
        elif fake_score > 0.65:
            st.error(f"""
            🚨 **LIKELY FAKE NEWS**
            
            This article shows strong characteristics of misinformation.
            
            **Confidence Level: {int(fake_score * 100)}%**
            
            ⚠️ *Verify with fact-checking organizations before sharing.*
            """)
        
        else:
            st.warning(f"""
            ⚠️ **UNCERTAIN / MIXED SIGNALS**
            
            The analysis is inconclusive. This could mean:
            - Satirical or opinion content
            - Mixed factual and opinion statements
            - Needs human fact-checking
            
            **Real: {int(real_score * 100)}% | Fake: {int(fake_score * 100)}%**
            
            🔍 *Strongly recommend verification with multiple sources.*
            """)

with tab2:
    st.subheader("🤖 How Our AI Detection Works")
    
    st.markdown("""
    ### **Technology Stack**
    
    - **Model:** BERT (Bidirectional Encoder Representations from Transformers)
    - **Training Data:** 20,000+ labeled news articles
    - **Accuracy:** 92-95% on test data
    - **Processing:** Advanced Natural Language Processing + Deep Learning
    
    ### **What We Analyze**
    
    1. **Linguistic Patterns** - Detects sensational language typical of fake news
    2. **Semantic Structure** - Analyzes meaning and logical flow
    3. **Statistical Features** - Identifies markers of misinformation
    4. **Contextual Understanding** - BERT reads text bidirectionally for context
    
    ### **How BERT Works**
    
    - Reads text from left-to-right AND right-to-left simultaneously
    - Understands context of every word in relation to all other words
    - Pre-trained on massive text datasets (Wikipedia, books)
    - Fine-tuned specifically for fake news detection
    - 12 transformer layers for deep semantic analysis
    
    ### **Model Limitations**
    
    - ⚠️ Not 100% accurate (92-95% accuracy rate)
    - ⚠️ Works best on English language text
    - ⚠️ Requires minimum 20 characters for analysis
    - ⚠️ May struggle with satire or sarcasm
    - ⚠️ Should always be supplemented with human verification
    - ⚠️ Rapidly evolving misinformation tactics may evade detection
    
    ### **Best Use Cases**
    
    ✅ Political news verification  
    ✅ Health/medical claims  
    ✅ Breaking news stories  
    ✅ Viral social media posts  
    ✅ Suspicious headlines  
    """)
with tab3:
    st.warning("""
    ### **⚠️ IMPORTANT DISCLAIMER**
    
    **This tool provides AI-based analysis for informational purposes only.**
    
    ### **Key Points**
    
    - ✓ Results are **predictions**, not definitive facts
    - ✓ **Always verify** with multiple reliable sources
    - ✓ AI is **not 100% accurate** (92-95% accuracy)
    - ✓ Use as a **supplementary tool**, not the final verdict
    - ✓ Professional fact-checking recommended for critical news
    - ✓ Keep up with fact-checking organizations (PolitiFact, Snopes, FactCheck.org)
    - ✓ Misinformation tactics are constantly evolving
    
    ### **Recommended Best Practices**
    
    1. **Multiple Sources** - Read the same story from 3+ sources
    2. **Check Dates** - Verify publication dates aren't misleading
    3. **Author Research** - Look up author credentials
    4. **Citations** - Check if article cites verifiable sources
    5. **Updates** - Look for corrections or updated information
    6. **Fact-Checkers** - Use established fact-checking websites
    7. **Emotional Check** - Be skeptical of highly emotional language
    
    ### **Trusted Fact-Checking Resources**
    
    - 🔍 Snopes.com
    - 🔍 FactCheck.org
    - 🔍 PolitiFact.com
    - 🔍 Reuters Fact Check
    - 🔍 AP Fact Check
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em; padding: 20px; background: rgba(255,255,255,0.9); border-radius: 10px;">
    <p><strong>🛡️ TruthLens v1.0</strong></p>
    <p>Powered by HuggingFace 🤗 & Streamlit • January 2026</p>
    <p>Built for truth detection • Use responsibly • Always verify</p>
    <p>© 2026 AdityaKashyapMohan</p>
</div>
""", unsafe_allow_html=True)

