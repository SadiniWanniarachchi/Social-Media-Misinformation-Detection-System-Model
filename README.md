# 🛡️ Social Media Misinformation Detection System

A machine learning-powered system that analyzes text content to detect misinformation and fake news with 89.8% accuracy.

[![Live Demo](https://img.shields.io/badge/Demo-Live-success)](https://social-media-misinformation-detection-system-model-u9berph6m2p.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 📋 Overview

This system leverages Natural Language Processing (NLP) and Machine Learning to identify fake news and misinformation in real-time. The model achieves 89.8% accuracy using an ensemble approach combining multiple classification algorithms.

**[🚀 Try the Live Demo](https://social-media-misinformation-detection-system-model-u9berph6m2p.streamlit.app/)**

## ✨ Key Features

- ⚡ **Real-time Analysis**: Instant predictions with confidence scores
- 🎯 **High Accuracy**: 89.8% classification accuracy
- 🤖 **Ensemble Learning**: Combines Logistic Regression, Random Forest, and Gradient Boosting
- 🌐 **Interactive Web Interface**: Built with Streamlit for easy accessibility
- 📊 **Confidence Scoring**: Provides probability distribution for predictions
- 📦 **Batch Processing**: Analyze multiple texts efficiently

## 🏗️ System Architecture

### Complete Processing Pipeline

```mermaid
graph TB
    Start([👤 User Input Text]) --> Input[📝 Raw Text Data]
    
    Input --> PreProcess{🧹 Text Preprocessing}
    PreProcess --> Clean[Remove URLs & Special Chars]
    PreProcess --> Lower[Convert to Lowercase]
    PreProcess --> Token[Tokenization]
    
    Clean --> Combine1[⚙️ Cleaned Text]
    Lower --> Combine1
    Token --> Combine1
    
    Combine1 --> NLP{🔤 NLP Pipeline}
    NLP --> Stop[Remove Stopwords]
    NLP --> Lemma[Lemmatization]
    NLP --> Norm[Text Normalization]
    
    Stop --> Combine2[📄 Processed Text]
    Lemma --> Combine2
    Norm --> Combine2
    
    Combine2 --> Feature{🎯 Feature Engineering}
    Feature --> TFIDF[TF-IDF Vectorization<br/>5000 Features]
    Feature --> Stats[Statistical Features<br/>Length, Word Count, etc.]
    
    TFIDF --> Combine3[🔢 Feature Vector]
    Stats --> Combine3
    
    Combine3 --> Ensemble{🤖 Ensemble Model}
    
    Ensemble --> LR[Logistic Regression<br/>Weight: 0.33]
    Ensemble --> RF[Random Forest<br/>Weight: 0.33]
    Ensemble --> GB[Gradient Boosting<br/>Weight: 0.33]
    
    LR --> Vote[🗳️ Soft Voting]
    RF --> Vote
    GB --> Vote
    
    Vote --> Calib[⚖️ Calibration Layer<br/>Sigmoid Function]
    
    Calib --> Predict{📊 Prediction}
    
    Predict --> Real[✅ REAL NEWS<br/>Probability Score]
    Predict --> Fake[❌ FAKE NEWS<br/>Probability Score]
    
    Real --> Output[📈 Final Output]
    Fake --> Output
    
    Output --> Display([🖥️ Display Results<br/>Classification + Confidence])
    
    style Start fill:#667eea,stroke:#333,stroke-width:3px,color:#fff
    style Display fill:#2ecc71,stroke:#333,stroke-width:3px,color:#fff
    style Ensemble fill:#e74c3c,stroke:#333,stroke-width:2px,color:#fff
    style Vote fill:#f39c12,stroke:#333,stroke-width:2px,color:#fff
    style Real fill:#27ae60,stroke:#333,stroke-width:2px,color:#fff
    style Fake fill:#c0392b,stroke:#333,stroke-width:2px,color:#fff
```

### Ensemble Model Architecture

```mermaid
graph LR
    A[📊 Feature Vector<br/>5000+ Dimensions] --> B[🤖 Ensemble Classifier]
    
    B --> C[Model 1:<br/>Logistic Regression<br/>C=0.5, balanced]
    B --> D[Model 2:<br/>Random Forest<br/>n_estimators=200]
    B --> E[Model 3:<br/>Gradient Boosting<br/>n_estimators=150]
    
    C --> F[Probability: P1]
    D --> G[Probability: P2]
    E --> H[Probability: P3]
    
    F --> I[🗳️ Soft Voting<br/>Average Probabilities]
    G --> I
    H --> I
    
    I --> J[⚖️ Calibration<br/>CV=5, Sigmoid]
    
    J --> K{Decision<br/>Threshold: 0.5}
    
    K -->|P ≥ 0.5| L[❌ Fake News]
    K -->|P < 0.5| M[✅ Real News]
    
    L --> N[📈 Output with<br/>Confidence Score]
    M --> N
    
    style A fill:#3498db,stroke:#333,stroke-width:2px,color:#fff
    style B fill:#9b59b6,stroke:#333,stroke-width:2px,color:#fff
    style I fill:#f39c12,stroke:#333,stroke-width:2px,color:#fff
    style J fill:#1abc9c,stroke:#333,stroke-width:2px,color:#fff
    style L fill:#e74c3c,stroke:#333,stroke-width:2px,color:#fff
    style M fill:#2ecc71,stroke:#333,stroke-width:2px,color:#fff
    style N fill:#34495e,stroke:#333,stroke-width:3px,color:#fff
```

### Data Flow Diagram

```mermaid
flowchart TD
    A[(📚 Dataset<br/>44,898 Samples)] --> B[🔄 Data Split]
    
    B --> C[📖 Training Set<br/>35,918 samples<br/>80%]
    B --> D[🧪 Test Set<br/>8,980 samples<br/>20%]
    
    C --> E[⚖️ SMOTE Balancing<br/>Equal Class Distribution]
    
    E --> F[🎯 Feature Extraction<br/>TF-IDF + Statistical]
    
    F --> G[🤖 Model Training<br/>3 Base Classifiers]
    
    G --> H[📊 Cross-Validation<br/>5-Fold CV]
    
    H --> I[⚙️ Hyperparameter Tuning<br/>Grid Search]
    
    I --> J[🎓 Trained Ensemble Model]
    
    D --> K[🧪 Model Evaluation]
    J --> K
    
    K --> L{📈 Performance Check}
    
    L -->|Accuracy < 85%| M[🔧 Re-tune Parameters]
    M --> I
    
    L -->|Accuracy ≥ 85%| N[✅ Final Model<br/>89.8% Accuracy]
    
    N --> O[💾 Model Serialization<br/>Joblib Save]
    
    O --> P[🚀 Deployment<br/>Streamlit App]
    
    style A fill:#3498db,stroke:#333,stroke-width:2px,color:#fff
    style N fill:#2ecc71,stroke:#333,stroke-width:3px,color:#fff
    style P fill:#e74c3c,stroke:#333,stroke-width:3px,color:#fff
    style J fill:#9b59b6,stroke:#333,stroke-width:2px,color:#fff
```

## 📈 Performance Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 89.8% |
| Precision | 89.8% |
| Recall | 89.8% |
| F1-Score | 89.8% |

### 🏆 Model Comparison

| Model | Accuracy |
|-------|----------|
| Support Vector Machine | 89.83% 🥇 |
| Logistic Regression | 88.33% 🥈 |
| Ensemble (Deployed) | 86.95% 🥉 |
| Multinomial Naive Bayes | 84.95% |
| Gradient Boosting | 84.86% |
| Random Forest | 84.36% |

## 🔧 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/SadiniWanniarachchi/Social-Media-Misinformation-Detection-System-Model.git
cd Social-Media-Misinformation-Detection-System-Model
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data (if not automatically downloaded):**
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

## 🚀 Usage

### Running the Web Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Using Pre-trained Models

```python
import joblib

# Load models
model = joblib.load('models/best_misinfo_detection_model.joblib')
vectorizer = joblib.load('models/tfidf_vectorizer.joblib')

# Make prediction
text = "Your news text here"
text_tfidf = vectorizer.transform([text])
prediction = model.predict(text_tfidf)[0]
probabilities = model.predict_proba(text_tfidf)[0]

print(f"Prediction: {'Fake' if prediction == 1 else 'Real'}")
print(f"Confidence: {max(probabilities):.2%}")
```

## 📊 Dataset

- **Total Samples**: 44,898
- **Real News**: 22,449 (50%)
- **Fake News**: 22,449 (50%)
- **Training Set**: 35,918 (80%)
- **Testing Set**: 8,980 (20%)

**Sources**:
- Constraint Dataset (Primary)
- Kaggle News Dataset (Secondary)

## 🛠️ Technologies Used

- **Machine Learning**: Scikit-learn, NLTK
- **Web Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib
- **Feature Engineering**: TF-IDF Vectorization
- **Class Balancing**: SMOTE

## 📁 Project Structure

```
Social-Media-Misinformation-Detection-System-Model/
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
├── README.md                 # Documentation
├── data/                     # Dataset directory
│   ├── Constraint_English_Train.csv
│   ├── Constraint_English_Test.csv
│   └── news.csv
├── models/                   # Trained models
│   ├── best_misinfo_detection_model.joblib
│   ├── tfidf_vectorizer.joblib
│   └── model_metadata.joblib
├── notebooks/                # Jupyter notebooks
│   └── News_Miss_Info.ipynb
└── src/                      # Source code
    └── news_miss_info.py
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## 📧 Contact

**Sadini Wanniarachchi**
- GitHub: [@SadiniWanniarachchi](https://github.com/SadiniWanniarachchi)
- Project Link: [https://github.com/SadiniWanniarachchi/Social-Media-Misinformation-Detection-System-Model](https://github.com/SadiniWanniarachchi/Social-Media-Misinformation-Detection-System-Model)

---

**🌐 Live Demo**: [Launch Application](https://social-media-misinformation-detection-system-model-u9berph6m2p.streamlit.app/)
