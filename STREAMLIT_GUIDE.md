# 🚀 Streamlit Web Application

## Running the Misinformation Detection App

### Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

3. **Open your browser:**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in the terminal

### Features

#### 🎯 Real-Time Detection
- Paste any text (news articles, social media posts, tweets)
- Get instant AI-powered analysis
- Adjustable detection threshold

#### 📊 Comprehensive Analysis
- **Verdict Display**: Clear FAKE/REAL classification
- **Confidence Scores**: Probability-based predictions
- **Interactive Gauges**: Visual probability indicators
- **Text Statistics**: Character count, word analysis, punctuation patterns
- **Feature Extraction**: Detailed text feature breakdown

#### 🎨 Beautiful Visualizations
- **Gauge Chart**: Fake news probability meter
- **Bar Charts**: Probability comparison
- **Feature Charts**: Text statistics visualization
- **Color-coded Alerts**: Red for fake, blue/green for real

#### ⚙️ Customization
- **Threshold Slider**: Adjust detection sensitivity
- **Sample Texts**: Pre-loaded examples to test
- **Model Info**: View training metrics and performance

#### 💾 Export Results
- Download analysis results as CSV
- Includes timestamp, verdict, probabilities, and text

### Usage Tips

1. **For Best Results:**
   - Enter complete sentences or paragraphs
   - Include context and details
   - Minimum 20-30 words recommended

2. **Threshold Settings:**
   - **0.50**: Balanced (equal treatment of real/fake)
   - **0.55**: Recommended (slightly conservative)
   - **0.60+**: Conservative (reduces false positives)

3. **Interpreting Results:**
   - **High Confidence (>85%)**: Strong prediction
   - **Medium Confidence (55-85%)**: Moderate certainty
   - **Low Confidence (<55%)**: Uncertain, verify elsewhere

### Screenshots

The app includes:
- 🎨 Modern gradient UI with custom styling
- 📱 Responsive layout for all screen sizes
- 🌈 Color-coded alerts (red=fake, green=real)
- 📊 Interactive Plotly charts
- ⚡ Fast predictions (<1 second)

### Technical Details

**Frontend:** Streamlit  
**Backend:** Ensemble ML Model (86.95% accuracy)  
**Features:** TF-IDF vectorization + statistical features  
**Processing:** Real-time text cleaning and lemmatization  

### Troubleshooting

**Models not loading?**
- Ensure `models/` directory contains `.joblib` files
- Check file paths are correct

**NLTK data errors?**
- The app auto-downloads required data
- If issues persist, manually run: `python -c "import nltk; nltk.download('all')"`

**Port already in use?**
- Run on different port: `streamlit run app.py --server.port 8502`

### Deployment Options

#### Deploy to Streamlit Cloud (Free)
1. Push your code to GitHub
2. Visit [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect repository and deploy

#### Deploy to Heroku
```bash
# Create Procfile
echo "web: streamlit run app.py --server.port $PORT" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

#### Deploy to AWS/Azure/GCP
- Use Docker container with Streamlit
- See official Streamlit deployment docs

### Advanced Features

**Batch Processing:** Modify `app.py` to accept CSV uploads  
**API Integration:** Add REST API endpoints for external access  
**Multi-language:** Extend with translation support  
**Model Comparison:** Add multiple model selection  

### Performance

- **Average Response Time:** <1 second
- **Concurrent Users:** Supports multiple simultaneous analyses
- **Model Size:** ~25 MB (loads in <2 seconds)
- **Memory Usage:** ~200-300 MB

### Updates and Maintenance

To update the model:
1. Replace files in `models/` directory
2. Restart the Streamlit app
3. No code changes needed

---

**Built with ❤️ using Streamlit and Machine Learning**

For issues or suggestions, open an issue on [GitHub](https://github.com/SadiniWanniarachchi/Social-Media-Misinformation-Detection-System-Model).
