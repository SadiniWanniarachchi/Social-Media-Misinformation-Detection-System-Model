"""
Social Media Misinformation Detection System - Streamlit App
Built with ❤️ by Sadini Wanniarachchi
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Misinformation Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .fake-alert {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        animation: pulse 2s infinite;
    }
    .real-alert {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid #e0e0e0;
        background: white;
    }
    .info-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize NLTK resources
@st.cache_resource
def init_nltk():
    """Download required NLTK data"""
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        with st.spinner("Downloading language resources..."):
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('omw-1.4', quiet=True)

init_nltk()

# Load models
@st.cache_resource
def load_models():
    """Load the trained model and vectorizer"""
    try:
        model = joblib.load('models/best_misinfo_detection_model.joblib')
        vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
        metadata = joblib.load('models/model_metadata.joblib')
        return model, vectorizer, metadata
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# Text preprocessing
@st.cache_data
def get_stopwords():
    """Get English stopwords"""
    return set(stopwords.words('english'))

def advanced_text_cleaning(text):
    """Clean and preprocess text"""
    stop_words = get_stopwords()
    lemmatizer = WordNetLemmatizer()
    
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove social media mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtag symbols but keep words
    text = re.sub(r'#', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords and short words
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    # Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)

def extract_text_features(text):
    """Extract statistical features from text"""
    if not isinstance(text, str):
        return {
            'text_length': 0,
            'word_count': 0,
            'avg_word_length': 0,
            'punctuation_count': 0,
            'capital_count': 0,
            'question_marks': 0,
            'exclamation_marks': 0
        }
    
    words = text.split()
    
    return {
        'text_length': len(text),
        'word_count': len(words),
        'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
        'punctuation_count': sum(1 for char in text if char in string.punctuation),
        'capital_count': sum(1 for char in text if char.isupper()),
        'question_marks': text.count('?'),
        'exclamation_marks': text.count('!')
    }

def predict_misinformation(text, model, vectorizer, threshold=0.55):
    """Predict if text is misinformation"""
    # Clean the text
    cleaned = advanced_text_cleaning(text)
    
    if not cleaned or len(cleaned.strip()) == 0:
        return None, "No meaningful content found after cleaning."
    
    # Extract features
    features = extract_text_features(text)
    
    # Vectorize
    text_tfidf = vectorizer.transform([cleaned])
    
    # Get probabilities
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(text_tfidf)[0]
        prob_real = probabilities[0]
        prob_fake = probabilities[1]
        
        # Apply threshold
        if prob_fake >= threshold:
            prediction = 1
            confidence = prob_fake
        else:
            prediction = 0
            confidence = prob_real
    else:
        prediction = model.predict(text_tfidf)[0]
        confidence = 1.0
        prob_real = 1.0 if prediction == 0 else 0.0
        prob_fake = 1.0 if prediction == 1 else 0.0
    
    # Determine verdict
    if prediction == 1:
        if confidence >= 0.75:
            verdict = "FAKE NEWS"
            warning_level = "HIGH"
            color = "red"
        elif confidence >= threshold:
            verdict = "LIKELY FAKE"
            warning_level = "MEDIUM"
            color = "orange"
        else:
            verdict = "UNCERTAIN (Leaning Fake)"
            warning_level = "LOW-MEDIUM"
            color = "yellow"
    else:
        if confidence >= 0.75:
            verdict = "REAL NEWS"
            warning_level = "LOW"
            color = "green"
        elif confidence >= threshold:
            verdict = "LIKELY REAL"
            warning_level = "LOW-MEDIUM"
            color = "blue"
        else:
            verdict = "UNCERTAIN (Leaning Real)"
            warning_level = "MEDIUM"
            color = "cyan"
    
    return {
        'prediction': int(prediction),
        'verdict': verdict,
        'confidence': float(confidence),
        'probability_real': float(prob_real),
        'probability_fake': float(prob_fake),
        'warning_level': warning_level,
        'color': color,
        'cleaned_text': cleaned,
        'features': features,
        'threshold_used': threshold
    }, None

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Misinformation Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Fake News Detector using Advanced Machine Learning</p>', unsafe_allow_html=True)
    
    # Load models
    model, vectorizer, metadata = load_models()
    
    if model is None:
        st.error("⚠️ Could not load models. Please ensure model files are in the 'models/' directory.")
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
        st.title("⚙️ Settings")
        
        # Model info
        with st.expander("📊 Model Information", expanded=False):
            if metadata:
                st.write(f"**Model:** {metadata.get('model_name', 'Ensemble Model')}")
                st.write(f"**Accuracy:** {metadata.get('performance_metrics', {}).get('accuracy', 0)*100:.2f}%")
                st.write(f"**F1-Score:** {metadata.get('performance_metrics', {}).get('f1_score', 0):.4f}")
                st.write(f"**Training Samples:** {metadata.get('dataset_info', {}).get('training_samples', 'N/A'):,}")
        
        # Threshold slider
        st.subheader("🎚️ Detection Threshold")
        threshold = st.slider(
            "Adjust sensitivity",
            min_value=0.3,
            max_value=0.9,
            value=0.55,
            step=0.05,
            help="Higher = More conservative (fewer false positives)"
        )
        
        st.info(f"""
        **Current: {threshold:.2f}**
        
        - 0.50: Balanced
        - 0.55: Recommended
        - 0.60+: Conservative
        """)
        
        # Sample texts
        st.subheader("📝 Sample Texts")
        sample_choice = st.selectbox(
            "Try a sample",
            ["Custom Input", "Sample Fake News", "Sample Real News", "Sample Uncertain"]
        )
        
        samples = {
            "Sample Fake News": "BREAKING: Scientists discover chocolate cures all diseases! Click here for miracle cure! 🍫💊 #FakeNews #Clickbait",
            "Sample Real News": "The World Health Organization announced new guidelines for vaccine distribution in developing countries, focusing on equitable access and proper cold chain management.",
            "Sample Uncertain": "Local politician makes controversial statement about education funding during town hall meeting yesterday."
        }
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter Text to Analyze")
        
        # Text input
        if sample_choice != "Custom Input":
            default_text = samples[sample_choice]
        else:
            default_text = ""
        
        user_input = st.text_area(
            "Paste news article, social media post, or any text:",
            value=default_text,
            height=200,
            placeholder="Enter or paste text here... (e.g., news articles, tweets, social media posts)"
        )
        
        # Character count
        if user_input:
            st.caption(f"📊 Character count: {len(user_input)} | Word count: {len(user_input.split())}")
        
        # Analyze button
        analyze_button = st.button("🔍 Analyze Text", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("ℹ️ How It Works")
        st.markdown("""
        <div class="info-card">
        <b>1. Text Preprocessing</b><br>
        Cleans and normalizes input text
        <br><br>
        <b>2. Feature Extraction</b><br>
        Extracts TF-IDF and statistical features
        <br><br>
        <b>3. AI Analysis</b><br>
        Ensemble model predicts authenticity
        <br><br>
        <b>4. Confidence Score</b><br>
        Returns probability-based verdict
        </div>
        """, unsafe_allow_html=True)
    
    # Analysis results
    if analyze_button and user_input:
        with st.spinner("🔄 Analyzing text..."):
            # Simulate processing time for effect
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            result, error = predict_misinformation(user_input, model, vectorizer, threshold)
            
            if error:
                st.error(f"❌ {error}")
                return
            
            # Main verdict display
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            if result['prediction'] == 1:
                st.markdown(f"""
                <div class="fake-alert">
                    <h1 style="margin:0; font-size: 3rem;">⚠️ {result['verdict']}</h1>
                    <p style="font-size: 1.5rem; margin-top: 1rem;">Confidence: {result['confidence']*100:.1f}%</p>
                    <p style="margin-top: 0.5rem;">Warning Level: {result['warning_level']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="real-alert">
                    <h1 style="margin:0; font-size: 3rem;">✅ {result['verdict']}</h1>
                    <p style="font-size: 1.5rem; margin-top: 1rem;">Confidence: {result['confidence']*100:.1f}%</p>
                    <p style="margin-top: 0.5rem;">Warning Level: {result['warning_level']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed metrics
            st.markdown("### 📈 Detailed Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Real Probability",
                    f"{result['probability_real']*100:.1f}%",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Fake Probability",
                    f"{result['probability_fake']*100:.1f}%",
                    delta=None
                )
            
            with col3:
                st.metric(
                    "Confidence",
                    f"{result['confidence']*100:.1f}%",
                    delta=None
                )
            
            with col4:
                st.metric(
                    "Classification",
                    "Fake" if result['prediction'] == 1 else "Real",
                    delta=None
                )
            
            # Probability gauge chart
            st.markdown("### 🎯 Probability Distribution")
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = result['probability_fake'] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Fake News Probability", 'font': {'size': 24}},
                delta = {'reference': 50, 'increasing': {'color': "red"}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkred" if result['probability_fake'] > 0.5 else "darkgreen"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': '#90EE90'},
                        {'range': [30, 50], 'color': '#FFFFE0'},
                        {'range': [50, 70], 'color': '#FFD700'},
                        {'range': [70, 100], 'color': '#FF6B6B'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold * 100
                    }
                }
            ))
            
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # Bar chart comparison
            col1, col2 = st.columns(2)
            
            with col1:
                fig_bar = go.Figure(data=[
                    go.Bar(
                        x=['Real', 'Fake'],
                        y=[result['probability_real']*100, result['probability_fake']*100],
                        marker_color=['#4CAF50', '#F44336'],
                        text=[f"{result['probability_real']*100:.1f}%", f"{result['probability_fake']*100:.1f}%"],
                        textposition='auto',
                    )
                ])
                fig_bar.update_layout(
                    title="Probability Comparison",
                    yaxis_title="Probability (%)",
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                # Text features
                features = result['features']
                feature_names = list(features.keys())
                feature_values = list(features.values())
                
                fig_features = go.Figure(data=[
                    go.Bar(
                        y=feature_names,
                        x=feature_values,
                        orientation='h',
                        marker_color='#667eea'
                    )
                ])
                fig_features.update_layout(
                    title="Text Features",
                    xaxis_title="Value",
                    height=300
                )
                st.plotly_chart(fig_features, use_container_width=True)
            
            # Additional insights
            st.markdown("### 💡 Insights")
            
            prob_diff = abs(result['probability_real'] - result['probability_fake'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="info-card">
                <b>🎯 Interpretation</b><br><br>
                """, unsafe_allow_html=True)
                
                if prob_diff < 0.15:
                    st.write(f"⚠️ **Close probabilities** ({prob_diff*100:.1f}% difference)")
                    st.write("The model is uncertain. Consider additional verification from trusted sources.")
                elif result['confidence'] >= 0.85:
                    st.write(f"✅ **High confidence** ({result['confidence']*100:.0f}% certain)")
                    st.write("The model is very confident in this prediction.")
                else:
                    st.write(f"📊 **Moderate confidence** ({result['confidence']*100:.0f}%)")
                    st.write("The model has reasonable certainty in this prediction.")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="info-card">
                <b>📝 Text Statistics</b><br><br>
                """, unsafe_allow_html=True)
                
                features = result['features']
                st.write(f"📏 **Length:** {features['text_length']} characters")
                st.write(f"📖 **Words:** {features['word_count']} words")
                st.write(f"📊 **Avg Word Length:** {features['avg_word_length']:.1f}")
                st.write(f"❗ **Exclamation Marks:** {features['exclamation_marks']}")
                st.write(f"❓ **Question Marks:** {features['question_marks']}")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Cleaned text preview
            with st.expander("🔍 View Cleaned Text", expanded=False):
                st.text_area("Preprocessed text used for analysis:", result['cleaned_text'], height=100)
            
            # Download results
            st.markdown("### 💾 Export Results")
            
            result_df = pd.DataFrame([{
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Verdict': result['verdict'],
                'Confidence': f"{result['confidence']*100:.2f}%",
                'Real Probability': f"{result['probability_real']*100:.2f}%",
                'Fake Probability': f"{result['probability_fake']*100:.2f}%",
                'Warning Level': result['warning_level'],
                'Original Text': user_input[:100] + "..." if len(user_input) > 100 else user_input
            }])
            
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name=f"misinformation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    elif analyze_button:
        st.warning("⚠️ Please enter some text to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p><b>Social Media Misinformation Detection System</b></p>
        <p>Built with Streamlit • Powered by Machine Learning</p>
        <p>Model Accuracy: 86.95% | Ensemble Learning with Calibrated Classifiers</p>
        <p>© 2025 Sadini Wanniarachchi | 
        <a href="https://github.com/SadiniWanniarachchi/Social-Media-Misinformation-Detection-System-Model" target="_blank">GitHub</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
