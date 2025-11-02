# Social Media Misinformation Detection System

A comprehensive machine learning system for detecting misinformation in social media content using advanced NLP techniques, ensemble learning, and calibrated classifiers.

## 🎯 Overview

This project implements a sophisticated misinformation detection system that combines multiple machine learning algorithms with text preprocessing, feature engineering, and probability calibration to accurately identify fake news and misinformation in social media posts.

### Key Features

- **Advanced Text Preprocessing**: Comprehensive text cleaning, lemmatization, and feature extraction
- **Multiple ML Models**: Logistic Regression, Naive Bayes, SVM, Random Forest, and Gradient Boosting
- **Ensemble Learning**: Calibrated voting classifier combining the best models
- **SMOTE**: Handles class imbalance for improved accuracy
- **Interactive Prediction**: Real-time misinformation detection with adjustable thresholds
- **Comprehensive EDA**: Detailed exploratory data analysis with visualizations

## 📊 Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Ensemble (Balanced)** | **86.95%** | **87.04%** | **87.03%** | **86.94%** |
| Support Vector Machine | 89.83% | 89.83% | 89.83% | 89.83% |
| Logistic Regression | 88.33% | 88.37% | 88.33% | 88.33% |

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip or conda package manager
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/SadiniWanniarachchi/Social-Media-Misinformation-Detection-System-Model.git
cd Social-Media-Misinformation-Detection-System-Model
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')
```

### Usage

#### Using the Jupyter Notebook

```bash
jupyter notebook notebooks/News_Miss_Info.ipynb
```

#### Using the Python Script

```bash
python src/news_miss_info.py
```

#### Loading Pre-trained Model

```python
import joblib

# Load the trained model and vectorizer
model = joblib.load('models/best_misinfo_detection_model.joblib')
vectorizer = joblib.load('models/tfidf_vectorizer.joblib')

# Make predictions
text = "Your news text here"
prediction = model.predict(vectorizer.transform([text]))
```

## 📁 Project Structure

```
Social-Media-Misinformation-Detection-System-Model/
│
├── data/                           # Dataset directory
│   ├── raw/                        # Original unprocessed datasets
│   │   ├── Constraint_English_Train.csv
│   │   ├── Constraint_English_Test.csv
│   │   ├── Constraint_English_Val.csv
│   │   └── news.csv
│   └── processed/                  # Processed datasets
│       ├── processed_dataset.csv
│       └── model_comparison_results.csv
│
├── models/                         # Trained models and artifacts
│   ├── best_misinfo_detection_model.joblib
│   ├── tfidf_vectorizer.joblib
│   └── model_metadata.joblib
│
├── notebooks/                      # Jupyter notebooks
│   └── News_Miss_Info.ipynb       # Main analysis notebook
│
├── src/                           # Source code
│   └── news_miss_info.py          # Main Python script
│
├── visualizations/                # Generated plots and figures
│   ├── comprehensive_eda.png
│   ├── wordclouds.png
│   ├── smote_effect.png
│   ├── classification_results.png
│   └── clustering_results.png
│
├── docs/                          # Documentation
│   └── methodology.md             # Detailed methodology
│
├── .gitignore                     # Git ignore file
├── requirements.txt               # Python dependencies
├── LICENSE                        # License file
└── README.md                      # This file
```

## 🔬 Methodology

### 1. Data Preprocessing
- Text cleaning and normalization
- URL, email, and mention removal
- Stopword removal and lemmatization
- Feature extraction (length, word count, punctuation, etc.)

### 2. Feature Engineering
- **TF-IDF Vectorization**: 5000 max features with bigrams
- **Statistical Features**: Text length, word count, capital letters, punctuation
- **PCA**: Dimensionality reduction for clustering

### 3. Class Imbalance Handling
- **SMOTE** (Synthetic Minority Over-sampling Technique)
- Balanced class distribution for improved model performance

### 4. Model Training
- 5 calibrated classification models
- Stratified train-test split (80/20)
- Probability calibration using sigmoid method
- Ensemble voting classifier

### 5. Evaluation
- Confusion matrices
- Precision, Recall, F1-Score
- Cross-validation
- Clustering analysis (K-Means)

## 📈 Datasets

### Primary Dataset: Constraint
- Source: University-provided dataset
- Split: Train, Test, Validation
- Format: Tweet-based labeled data

### Secondary Dataset: News
- Source: Kaggle
- Enhanced training with news articles
- Combined title and text content

### Data Statistics
- Total samples after preprocessing: ~50K+
- Features: 5000 TF-IDF features + statistical features
- Classes: Real (0) vs Fake (1)

## 🛠️ Technologies Used

- **Python 3.x**: Core programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms
- **NLTK**: Natural language processing
- **Imbalanced-learn**: SMOTE implementation
- **Matplotlib & Seaborn**: Visualization
- **WordCloud**: Text visualization
- **Joblib**: Model persistence

## 📊 Visualizations

The project includes comprehensive visualizations:
- Label distribution analysis
- Text length and word count distributions
- Word clouds for real vs fake news
- Model performance comparisons
- Confusion matrices
- Clustering analysis
- Feature importance plots

## 🎯 Key Insights

1. **Text Length**: Fake news tends to have different text length patterns
2. **Punctuation**: Higher exclamation marks in fake news
3. **Word Patterns**: Distinct vocabulary differences between real and fake news
4. **Ensemble Performance**: Combining multiple models improves reliability
5. **Calibration**: Probability calibration reduces false positives

## 🔮 Future Enhancements

- [ ] Deep learning models (LSTM, BERT, Transformers)
- [ ] Real-time API for predictions
- [ ] Web interface for user interaction
- [ ] Multi-language support
- [ ] Social media integration (Twitter API)
- [ ] Explainability features (LIME, SHAP)
- [ ] Mobile application

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Sadini Wanniarachchi**

- GitHub: [@SadiniWanniarachchi](https://github.com/SadiniWanniarachchi)
- Repository: [Social-Media-Misinformation-Detection-System-Model](https://github.com/SadiniWanniarachchi/Social-Media-Misinformation-Detection-System-Model)

## 🙏 Acknowledgments

- Constraint dataset providers
- Kaggle for news dataset
- Open-source ML community
- Scikit-learn and NLTK contributors

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue on GitHub or contact through my GitHub profile.

## ⭐ Star This Repository

If you find this project helpful, please consider giving it a star! ⭐
