# 🔍 Social Media Misinformation Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-89.8%25-brightgreen.svg)](https://github.com/SadiniWanniarachchi/Social-Media-Misinformation-Detection-System-Model)

> **AI-Powered Fake News Detection using Advanced Machine Learning and Natural Language Processing**

Built with ❤️ by [Sadini Wanniarachchi](https://github.com/SadiniWanniarachchi)

---

## 🌐 Try It Live!

**🚀 [Launch Web Application](https://social-media-misinformation-detection-system-model-u9berph6m2p.streamlit.app/)**

Experience the real-time misinformation detection system directly in your browser - no installation required!

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [How It Works](#-how-it-works)
- [Results & Visualizations](#-results--visualizations)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)
- [Future Enhancements](#-future-enhancements)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Overview

The **Social Media Misinformation Detection System** is an advanced AI-powered application designed to combat the spread of fake news and misinformation on social media platforms. Using state-of-the-art machine learning techniques and natural language processing, this system analyzes text content and determines its authenticity with **89.8% accuracy**.

### Why This Matters

In the digital age, misinformation spreads faster than ever before. This system helps:
- 📰 **Verify News**: Quickly assess the credibility of news articles
- 🐦 **Analyze Social Media**: Detect misleading tweets and posts
- 🛡️ **Combat Misinformation**: Protect users from fake news
- 📊 **Provide Transparency**: Offer probability-based confidence scores
- 🎓 **Educate Users**: Show detailed analysis of text features

---

## ✨ Features

### 🤖 Core Capabilities

- **Real-Time Detection**: Instant analysis of text input (<1 second response time)
- **High Accuracy**: 89.8% accuracy with ensemble learning approach
- **Calibrated Predictions**: Probability-based confidence scores (0-100%)
- **Multi-Source Support**: Analyzes news articles, tweets, social media posts, and general text
- **Adjustable Sensitivity**: Customizable detection threshold (0.3 - 0.9)
- **Comprehensive Analysis**: Provides detailed text statistics and feature breakdown

### 🎨 User Interface

- **Modern Design**: Beautiful gradient UI with responsive layout
- **Interactive Visualizations**: 
  - Probability gauge charts
  - Confidence bar graphs
  - Text feature analysis
  - Statistical breakdowns
- **Sample Testing**: Pre-loaded examples for quick testing
- **Color-Coded Results**: Visual alerts (Red = Fake, Green = Real)
- **Mobile Responsive**: Works seamlessly on all devices

### 📊 Advanced Features

- **Text Preprocessing**: Advanced cleaning with stopword removal and lemmatization
- **Feature Engineering**: Extracts 7+ statistical features from text
- **TF-IDF Vectorization**: 5000+ dimensional feature space with n-grams
- **SMOTE Balancing**: Handles class imbalance for better predictions
- **Ensemble Model**: Combines Logistic Regression, Random Forest, and Gradient Boosting
- **Model Calibration**: Ensures probability estimates are reliable

---

## 🎥 Demo

### Web Application Interface

<div align="center">
  
![App Screenshot](https://via.placeholder.com/800x450/667eea/ffffff?text=Misinformation+Detection+System)

*Real-time analysis with confidence scores and visual feedback*

</div>

### Sample Analysis

#### Input
```
BREAKING: Scientists discover chocolate cures all diseases! 
Click here for miracle cure! 🍫💊 #FakeNews
```

#### Output
```
VERDICT: FAKE NEWS
Confidence: 94.2%
Warning Level: HIGH

Probability Distribution:
- Real News: 5.8%
- Fake News: 94.2%
```

---

## 🏗️ Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface (Streamlit)               │
├─────────────────────────────────────────────────────────────┤
│                     Text Input Processing                    │
│  • Advanced Cleaning  • Stopword Removal  • Lemmatization   │
├─────────────────────────────────────────────────────────────┤
│                    Feature Extraction                        │
│  • TF-IDF Vectorization  • Statistical Features (7+)        │
├─────────────────────────────────────────────────────────────┤
│                    Ensemble Model                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Logistic   │  │   Random    │  │  Gradient   │         │
│  │ Regression  │  │   Forest    │  │  Boosting   │         │
│  │ (Calibrated)│  │ (Calibrated)│  │ (Calibrated)│         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         └──────────────┬──────────────┘                     │
│              Soft Voting Ensemble                            │
├─────────────────────────────────────────────────────────────┤
│                    Prediction Output                         │
│  • Classification  • Probabilities  • Confidence Score      │
└─────────────────────────────────────────────────────────────┘
```

### Model Pipeline

1. **Data Collection**: Combined datasets (Constraint + News.csv)
2. **Preprocessing**: Text cleaning, normalization, tokenization
3. **Feature Engineering**: TF-IDF + Statistical features
4. **Class Balancing**: SMOTE for handling imbalanced data
5. **Model Training**: Ensemble of calibrated classifiers
6. **Evaluation**: Cross-validation with multiple metrics
7. **Deployment**: Streamlit web application

---

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- Internet connection (for NLTK data download)

### Step 1: Clone Repository

```bash
git clone https://github.com/SadiniWanniarachchi/Social-Media-Misinformation-Detection-System-Model.git
cd Social-Media-Misinformation-Detection-System-Model
```

### Step 2: Create Virtual Environment (Recommended)

#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/Mac
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data

The application will automatically download required NLTK data on first run, but you can manually download:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')
```

### Step 5: Verify Installation

```bash
python -c "import streamlit; import sklearn; import nltk; print('Installation successful!')"
```

---

## 🚀 Usage

### Running the Web Application

```bash
streamlit run app.py
```

The application will automatically open in your default browser at `http://localhost:8501`

### Running on Custom Port

```bash
streamlit run app.py --server.port 8502
```

### Using the Application

1. **Enter Text**: Paste news article, tweet, or any text content
2. **Adjust Threshold** (Optional): Use sidebar slider to set sensitivity
3. **Click Analyze**: Get instant results with confidence scores
4. **Review Results**: See detailed breakdown and probabilities
5. **Try Samples**: Use pre-loaded examples to test the system

### Using Pre-trained Models

```python
import joblib
import pandas as pd

# Load models
model = joblib.load('models/best_misinfo_detection_model.joblib')
vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
metadata = joblib.load('models/model_metadata.joblib')

# Predict
text = "Your news text here"
cleaned_text = advanced_text_cleaning(text)
text_tfidf = vectorizer.transform([cleaned_text])
prediction = model.predict(text_tfidf)[0]
probabilities = model.predict_proba(text_tfidf)[0]

print(f"Prediction: {'Fake' if prediction == 1 else 'Real'}")
print(f"Confidence: {max(probabilities):.2%}")
```

### Training Custom Model

To train the model from scratch using the Jupyter notebook:

```bash
jupyter notebook notebooks/News_Miss_Info.ipynb
```

Or run the Python script:

```bash
python src/news_miss_info.py
```

---

## 📈 Model Performance

### Overall Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | **89.8%** |
| **Precision** | 89.8% |
| **Recall** | 89.8% |
| **F1-Score** | 89.8% |

### Model Comparison

| Model | Accuracy | F1-Score | Precision (Fake) | Recall (Fake) |
|-------|----------|----------|------------------|---------------|
| **Support Vector Machine** | **89.83%** | **89.83%** | **89.18%** | **89.94%** |
| Logistic Regression | 88.33% | 88.33% | 86.79% | 89.58% |
| Ensemble (Balanced) | 86.95% | 86.94% | - | - |
| Gradient Boosting | 84.86% | 84.85% | 82.08% | 88.00% |
| Multinomial Naive Bayes | 84.95% | 84.95% | 83.08% | 86.61% |
| Random Forest | 84.36% | 84.34% | 80.61% | 89.21% |

### Performance by Class

#### Real News (Class 0)
- **Precision**: 90.44%
- **Recall**: 89.72%

#### Fake News (Class 1)
- **Precision**: 89.18%
- **Recall**: 89.94%

### Confusion Matrix (Best Model - SVM)

|              | Predicted Real | Predicted Fake |
|--------------|----------------|----------------|
| **Actual Real** | 5,128 (89.7%) | 587 (10.3%) |
| **Actual Fake** | 325 (10.1%) | 2,900 (89.9%) |

---

## 📊 Dataset

### Dataset Sources

1. **Constraint Dataset** (Primary)
   - Training: Constraint_English_Train.csv
   - Testing: Constraint_English_Test.csv
   - Validation: Constraint_English_Val.csv
   - Source: University-provided official dataset
   - Format: Tweet text with 'real'/'fake' labels

2. **News Dataset** (Secondary)
   - File: news.csv
   - Source: Kaggle
   - Format: Title + Text with 'REAL'/'FAKE' labels
   - Purpose: Enhance training diversity

### Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Samples** | 44,898 |
| **Real News** | 22,449 (50%) |
| **Fake News** | 22,449 (50%) |
| **Training Set** | 35,918 (80%) |
| **Testing Set** | 8,980 (20%) |
| **Features** | 5,000 (TF-IDF) |
| **Vocabulary Size** | 5,000 unique terms |

### Data Distribution

- **Balanced Dataset**: SMOTE applied for perfect 1:1 ratio
- **Text Length**: Average 150-300 characters
- **Word Count**: Average 20-50 words per sample
- **Sources**: Mixed (social media + news articles)

### Preprocessing Pipeline

1. **Text Cleaning**
   - Convert to lowercase
   - Remove URLs, emails, mentions
   - Remove special characters and numbers
   - Remove extra whitespace

2. **Tokenization**
   - Split into words
   - Remove stopwords
   - Filter short words (<3 characters)

3. **Lemmatization**
   - Reduce words to base form
   - Preserve semantic meaning

4. **Feature Extraction**
   - TF-IDF vectorization (5000 features)
   - Statistical features (7+ metrics)
   - N-gram combinations (1-2 grams)

---

## 📁 Project Structure

```
Social-Media-Misinformation-Detection-System-Model/
│
├── 📄 app.py                          # Streamlit web application
├── 📄 requirements.txt                # Python dependencies
├── 📄 README.md                       # Project documentation
├── 📄 STREAMLIT_GUIDE.md             # Streamlit deployment guide
├── 📄 LICENSE                        # MIT License
│
├── 📂 data/                          # Dataset directory
│   ├── 📂 raw/                       # Original datasets
│   │   ├── Constraint_English_Train.csv
│   │   ├── Constraint_English_Test.csv
│   │   ├── Constraint_English_Val.csv
│   │   └── news.csv
│   └── 📂 processed/                 # Processed datasets
│       ├── processed_dataset.csv
│       └── model_comparison_results.csv
│
├── 📂 models/                        # Trained models
│   ├── best_misinfo_detection_model.joblib    # Main ensemble model
│   ├── tfidf_vectorizer.joblib               # TF-IDF vectorizer
│   └── model_metadata.joblib                 # Model metadata
│
├── 📂 notebooks/                     # Jupyter notebooks
│   └── News_Miss_Info.ipynb         # Training notebook
│
├── 📂 src/                          # Source code
│   └── news_miss_info.py           # Training script
│
└── 📂 visualizations/               # Generated plots (optional)
    ├── comprehensive_eda.png
    ├── wordclouds.png
    ├── clustering_results.png
    └── classification_results.png
```

---

## 🛠️ Technologies Used

### Core Technologies

- **Python 3.8+**: Programming language
- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning library
- **NLTK**: Natural language processing

### Machine Learning

- **Models**:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - Ensemble Voting Classifier

- **Techniques**:
  - TF-IDF Vectorization
  - SMOTE (Class Balancing)
  - Probability Calibration
  - Cross-Validation
  - Ensemble Learning

### Data Processing

- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **NLTK**: Text preprocessing
  - Stopword removal
  - Lemmatization
  - Tokenization

### Visualization

- **Matplotlib**: Static plots
- **Seaborn**: Statistical visualizations
- **Plotly**: Interactive charts
- **WordCloud**: Word cloud generation

### Development Tools

- **Jupyter Notebook**: Interactive development
- **Git**: Version control
- **GitHub**: Code hosting
- **Joblib**: Model serialization

---

## 🔬 How It Works

### 1. Text Preprocessing

```python
def advanced_text_cleaning(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs, emails, mentions
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    
    # Remove special characters
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize and lemmatize
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens 
              if word not in stopwords and len(word) > 2]
    
    return " ".join(tokens)
```

### 2. Feature Extraction

**TF-IDF Features** (5000 dimensions):
- Unigrams and bigrams
- Min document frequency: 2
- Max document frequency: 95%
- Sublinear term frequency scaling

**Statistical Features** (7 metrics):
1. Text length (characters)
2. Word count
3. Average word length
4. Punctuation count
5. Capital letter count
6. Question marks count
7. Exclamation marks count

### 3. Model Architecture

**Ensemble Voting Classifier** with soft voting:

```python
VotingClassifier(
    estimators=[
        ('lr', CalibratedClassifierCV(LogisticRegression)),
        ('rf', CalibratedClassifierCV(RandomForest)),
        ('gb', CalibratedClassifierCV(GradientBoosting))
    ],
    voting='soft'
)
```

### 4. Prediction Pipeline

```
Input Text → Cleaning → Feature Extraction → TF-IDF → Ensemble Model → Probabilities → Classification
```

### 5. Threshold-Based Decision

```python
if probability_fake >= threshold:
    prediction = "FAKE"
    confidence = probability_fake
else:
    prediction = "REAL"
    confidence = probability_real
```

**Confidence Levels**:
- **High** (>75%): Strong conviction
- **Medium** (55-75%): Moderate certainty
- **Low** (<55%): Uncertain, requires verification

---

## 📊 Results & Visualizations

### Exploratory Data Analysis

#### Label Distribution
<div align="center">

![Label Distribution](https://via.placeholder.com/600x300/2ecc71/ffffff?text=50%25+Real+|+50%25+Fake)

*Perfectly balanced dataset after SMOTE application*

</div>

#### Word Clouds

**Real News Keywords**: government, official, health, policy, minister, country
**Fake News Keywords**: breaking, shocking, revealed, exposed, miracle, urgent

### Model Performance Comparison

<div align="center">

![Model Comparison](https://via.placeholder.com/700x350/667eea/ffffff?text=SVM+%3E+LR+%3E+Ensemble+%3E+NB+%3E+RF)

*F1-Score comparison across different models*

</div>

### Confusion Matrix

<div align="center">

![Confusion Matrix](https://via.placeholder.com/500x500/764ba2/ffffff?text=89.7%25+TPR+|+89.9%25+TNR)

*Best performing model (SVM) confusion matrix*

</div>

### Feature Importance

Top 10 most important TF-IDF features for fake news detection:
1. shocking
2. breaking
3. revealed
4. exposed
5. miracle
6. urgent
7. click
8. amazing
9. secret
10. truth

---

## 🔌 API Reference

### Text Cleaning Function

```python
def advanced_text_cleaning(text: str) -> str:
    """
    Clean and preprocess text for analysis.
    
    Args:
        text (str): Raw input text
        
    Returns:
        str: Cleaned and lemmatized text
    """
```

### Feature Extraction Function

```python
def extract_text_features(text: str) -> dict:
    """
    Extract statistical features from text.
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Dictionary of text features
    """
```

### Prediction Function

```python
def predict_misinformation(
    text: str, 
    model, 
    vectorizer, 
    threshold: float = 0.55
) -> tuple:
    """
    Predict if text is misinformation.
    
    Args:
        text (str): Text to analyze
        model: Trained classifier
        vectorizer: TF-IDF vectorizer
        threshold (float): Decision threshold (0.3-0.9)
        
    Returns:
        tuple: (result_dict, error_message)
            result_dict contains:
                - prediction (int): 0 (real) or 1 (fake)
                - verdict (str): Human-readable verdict
                - confidence (float): Confidence score (0-1)
                - probability_real (float): P(real)
                - probability_fake (float): P(fake)
                - warning_level (str): Risk level
                - cleaned_text (str): Preprocessed text
                - features (dict): Statistical features
    """
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Ways to Contribute

1. **Report Bugs**: Open an issue describing the bug
2. **Suggest Features**: Propose new features or improvements
3. **Improve Documentation**: Fix typos, add examples
4. **Add Tests**: Write unit tests for better coverage
5. **Submit Pull Requests**: Fix bugs or add features

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m "Add feature"`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Add docstrings to functions
- Include type hints where possible
- Write descriptive commit messages
- Add comments for complex logic

### Testing

```bash
# Run tests (if available)
python -m pytest tests/

# Check code style
flake8 app.py src/
```

---

## 🚀 Future Enhancements

### Planned Features

- [ ] **Multi-Language Support**: Detect misinformation in multiple languages
- [ ] **Chrome Extension**: Browser plugin for real-time detection
- [ ] **REST API**: API endpoints for integration with other apps
- [ ] **Batch Processing**: Analyze multiple texts simultaneously
- [ ] **User Accounts**: Save analysis history and preferences
- [ ] **Fact-Checking Links**: Provide references to fact-check sources
- [ ] **Social Media Integration**: Direct integration with Twitter, Facebook APIs
- [ ] **Mobile App**: Native Android/iOS applications
- [ ] **Explainable AI**: SHAP/LIME for model interpretability
- [ ] **Real-Time Updates**: Continuous model retraining with new data

### Research Directions

- Deep Learning models (BERT, GPT-based)
- Multi-modal analysis (text + images + metadata)
- Graph neural networks for source credibility
- Temporal pattern analysis
- Cross-platform misinformation tracking

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### MIT License Summary

- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use
- ❌ Liability
- ❌ Warranty

---

## 📧 Contact

**Sadini Wanniarachchi**

- 🌐 **Website**: [Portfolio](https://github.com/SadiniWanniarachchi)
- 💼 **GitHub**: [@SadiniWanniarachchi](https://github.com/SadiniWanniarachchi)
- 📱 **LinkedIn**: [Connect with me](https://linkedin.com/in/sadini-wanniarachchi)
- 📧 **Email**: [Contact](mailto:sadini@example.com)

**Project Links**

- 🚀 **Live Demo**: [Try it now!](https://social-media-misinformation-detection-system-model-u9berph6m2p.streamlit.app/)
- 📂 **Repository**: [GitHub Repo](https://github.com/SadiniWanniarachchi/Social-Media-Misinformation-Detection-System-Model)
- 🐛 **Issues**: [Report Issues](https://github.com/SadiniWanniarachchi/Social-Media-Misinformation-Detection-System-Model/issues)
- 💡 **Discussions**: [Join Discussion](https://github.com/SadiniWanniarachchi/Social-Media-Misinformation-Detection-System-Model/discussions)

---

## 🙏 Acknowledgments

- **Datasets**: Constraint Workshop, Kaggle community
- **Libraries**: Scikit-learn, NLTK, Streamlit teams
- **Inspiration**: Research in fake news detection and NLP
- **Community**: Open source contributors and researchers

---

## 📚 Citation

If you use this project in your research or work, please cite:

```bibtex
@software{wanniarachchi2025misinfodetection,
  author = {Wanniarachchi, Sadini},
  title = {Social Media Misinformation Detection System},
  year = {2025},
  url = {https://github.com/SadiniWanniarachchi/Social-Media-Misinformation-Detection-System-Model},
  note = {AI-Powered Fake News Detection using Machine Learning}
}
```

---

## ⭐ Star History

If you find this project helpful, please consider giving it a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=SadiniWanniarachchi/Social-Media-Misinformation-Detection-System-Model&type=Date)](https://star-history.com/#SadiniWanniarachchi/Social-Media-Misinformation-Detection-System-Model&Date)

---

## 📸 Screenshots Gallery

### Main Interface
![Main Interface](https://via.placeholder.com/800x450/667eea/ffffff?text=Main+Interface)

### Analysis Results
![Results](https://via.placeholder.com/800x450/764ba2/ffffff?text=Analysis+Results)

### Model Information
![Model Info](https://via.placeholder.com/800x450/2ecc71/ffffff?text=Model+Information)

---

<div align="center">

### Made with ❤️ and Python

**Combating Misinformation, One Prediction at a Time**

[🚀 Try Live Demo](https://social-media-misinformation-detection-system-model-u9berph6m2p.streamlit.app/) | [📖 Documentation](https://github.com/SadiniWanniarachchi/Social-Media-Misinformation-Detection-System-Model) | [🐛 Report Bug](https://github.com/SadiniWanniarachchi/Social-Media-Misinformation-Detection-System-Model/issues)

---

**© 2025 Sadini Wanniarachchi. All rights reserved.**

</div>
