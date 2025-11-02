# 🔍 Social Media Misinformation Detection System# 🔍 Social Media Misinformation Detection System



<div align="center"><div align="center">



### 🚀 AI-Powered Fake News Detector using Advanced Machine Learning### 🚀 AI-Powered Fake News Detector using Advanced Machine Learning



[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[![Streamlit](https://img.shields.io/badge/Streamlit-Live-red.svg)](https://social-media-misinformation-detection-system-model-u9berph6m2p.streamlit.app/)[![Streamlit](https://img.shields.io/badge/Streamlit-Live-red.svg)](https://social-media-misinformation-detection-system-model-u9berph6m2p.streamlit.app/)

[![Accuracy](https://img.shields.io/badge/Accuracy-86.95%25-success.svg)](README.md)[![Accuracy](https://img.shields.io/badge/Accuracy-86.95%25-success.svg)](README.md)



</div></div>



------



## ✨ Live Demo## ✨ Live Demo



<div align="center"><div align="center">



### 🎯 **[Try the Interactive Web App Now!](https://social-media-misinformation-detection-system-model-u9berph6m2p.streamlit.app/)** 🎯### 🎯 **[Try the Interactive Web App Now!](https://social-media-misinformation-detection-system-model-u9berph6m2p.streamlit.app/)** 🎯



[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://social-media-misinformation-detection-system-model-u9berph6m2p.streamlit.app/)[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://social-media-misinformation-detection-system-model-u9berph6m2p.streamlit.app/)



**Experience real-time misinformation detection with instant confidence scores and detailed analysis****Experience real-time misinformation detection with instant confidence scores and detailed analysis**



📝 Paste any news article or social media post → 🤖 Get AI-powered predictions → 📊 View detailed insights📝 Paste any news article or social media post → 🤖 Get AI-powered predictions → 📊 View detailed insights



</div></div>



------



## 🎯 Overview## � Overview



<div align="center">A comprehensive machine learning system for detecting misinformation in social media content using advanced NLP techniques, ensemble learning, and calibrated classifiers.



**A comprehensive machine learning system for detecting misinformation in social media content**  This project implements a sophisticated misinformation detection system that combines multiple machine learning algorithms with text preprocessing, feature engineering, and probability calibration to accurately identify fake news and misinformation in social media posts.

*Using advanced NLP techniques, ensemble learning, and calibrated classifiers*

### Key Features

</div>

- **Advanced Text Preprocessing**: Comprehensive text cleaning, lemmatization, and feature extraction

This project implements a sophisticated misinformation detection system that combines multiple machine learning algorithms with text preprocessing, feature engineering, and probability calibration to accurately identify fake news and misinformation in social media posts.- **Multiple ML Models**: Logistic Regression, Naive Bayes, SVM, Random Forest, and Gradient Boosting

- **Ensemble Learning**: Calibrated voting classifier combining the best models

### ✨ Key Features- **SMOTE**: Handles class imbalance for improved accuracy

- **Interactive Prediction**: Real-time misinformation detection with adjustable thresholds

- 🧹 **Advanced Text Preprocessing**: Comprehensive text cleaning, lemmatization, and feature extraction- **Comprehensive EDA**: Detailed exploratory data analysis with visualizations

- 🤖 **Multiple ML Models**: Logistic Regression, Naive Bayes, SVM, Random Forest, and Gradient Boosting

- 🎯 **Ensemble Learning**: Calibrated voting classifier combining the best models## 📊 Performance Metrics

- ⚖️ **SMOTE**: Handles class imbalance for improved accuracy

- 🔮 **Interactive Prediction**: Real-time misinformation detection with adjustable thresholds| Model | Accuracy | Precision | Recall | F1-Score |

- 📊 **Comprehensive EDA**: Detailed exploratory data analysis with visualizations|-------|----------|-----------|--------|----------|

- 🌐 **Web Interface**: Beautiful Streamlit dashboard for easy interaction| **Ensemble (Balanced)** | **86.95%** | **87.04%** | **87.03%** | **86.94%** |

| Support Vector Machine | 89.83% | 89.83% | 89.83% | 89.83% |

## 🏆 Performance Metrics| Logistic Regression | 88.33% | 88.37% | 88.33% | 88.33% |



<div align="center">## 🚀 Quick Start



| Model | Accuracy | Precision | Recall | F1-Score |### Prerequisites

|:------|:--------:|:---------:|:------:|:--------:|

| **🏅 Ensemble (Balanced)** | **86.95%** | **87.04%** | **87.03%** | **86.94%** |```bash

| Support Vector Machine | 89.83% | 89.83% | 89.83% | 89.83% |Python 3.8+

| Logistic Regression | 88.33% | 88.37% | 88.33% | 88.33% |pip or conda package manager

```

</div>

### Installation

## 🚀 Quick Start

1. Clone the repository:

### Prerequisites```bash

git clone https://github.com/SadiniWanniarachchi/Social-Media-Misinformation-Detection-System-Model.git

```bashcd Social-Media-Misinformation-Detection-System-Model

Python 3.8+```

pip or conda package manager

```2. Install required packages:

```bash

### Installationpip install -r requirements.txt

```

1. **Clone the repository:**

```bash3. Download NLTK data:

git clone https://github.com/SadiniWanniarachchi/Social-Media-Misinformation-Detection-System-Model.git```python

cd Social-Media-Misinformation-Detection-System-Modelimport nltk

```nltk.download('stopwords')

nltk.download('wordnet')

2. **Install required packages:**nltk.download('punkt')

```bashnltk.download('omw-1.4')

pip install -r requirements.txt```

```

### Usage

3. **Download NLTK data:**

```python#### Using the Jupyter Notebook

import nltk

nltk.download('stopwords')```bash

nltk.download('wordnet')jupyter notebook notebooks/News_Miss_Info.ipynb

nltk.download('punkt')```

nltk.download('omw-1.4')

```#### Using the Python Script



### 🎮 Usage Options```bash

python src/news_miss_info.py

#### 🌐 Option 1: Web Interface (Recommended)```



```bash#### Loading Pre-trained Model

streamlit run app.py

``````python

Then open your browser to `http://localhost:8501`import joblib



#### 📓 Option 2: Jupyter Notebook# Load the trained model and vectorizer

model = joblib.load('models/best_misinfo_detection_model.joblib')

```bashvectorizer = joblib.load('models/tfidf_vectorizer.joblib')

jupyter notebook notebooks/News_Miss_Info.ipynb

```# Make predictions

text = "Your news text here"

#### 🐍 Option 3: Python Scriptprediction = model.predict(vectorizer.transform([text]))

```

```bash

python src/news_miss_info.py## 📁 Project Structure

```

```

#### 💻 Option 4: Load Pre-trained ModelSocial-Media-Misinformation-Detection-System-Model/

│

```python├── data/                           # Dataset directory

import joblib│   ├── raw/                        # Original unprocessed datasets

│   │   ├── Constraint_English_Train.csv

# Load the trained model and vectorizer│   │   ├── Constraint_English_Test.csv

model = joblib.load('models/best_misinfo_detection_model.joblib')│   │   ├── Constraint_English_Val.csv

vectorizer = joblib.load('models/tfidf_vectorizer.joblib')│   │   └── news.csv

│   └── processed/                  # Processed datasets

# Make predictions│       ├── processed_dataset.csv

text = "Your news text here"│       └── model_comparison_results.csv

prediction = model.predict(vectorizer.transform([text]))│

```├── models/                         # Trained models and artifacts

│   ├── best_misinfo_detection_model.joblib

## 📁 Project Structure│   ├── tfidf_vectorizer.joblib

│   └── model_metadata.joblib

```│

Social-Media-Misinformation-Detection-System-Model/├── notebooks/                      # Jupyter notebooks

││   └── News_Miss_Info.ipynb       # Main analysis notebook

├── 📱 app.py                       # Streamlit web application│

├── 📄 README.md                    # Project documentation├── src/                           # Source code

├── 📋 requirements.txt             # Python dependencies│   └── news_miss_info.py          # Main Python script

├── 📜 LICENSE                      # MIT License│

│├── visualizations/                # Generated plots and figures

├── 📊 data/                        # Dataset directory│   ├── comprehensive_eda.png

│   ├── raw/                        # Original datasets│   ├── wordclouds.png

│   │   ├── Constraint_English_Train.csv│   ├── smote_effect.png

│   │   ├── Constraint_English_Test.csv│   ├── classification_results.png

│   │   ├── Constraint_English_Val.csv│   └── clustering_results.png

│   │   └── news.csv│

│   └── processed/                  # Processed data├── docs/                          # Documentation

│       ├── processed_dataset.csv│   └── methodology.md             # Detailed methodology

│       └── model_comparison_results.csv│

│├── .gitignore                     # Git ignore file

├── 🤖 models/                      # Trained models├── requirements.txt               # Python dependencies

│   ├── best_misinfo_detection_model.joblib├── LICENSE                        # License file

│   ├── tfidf_vectorizer.joblib└── README.md                      # This file

│   └── model_metadata.joblib```

│

├── 📓 notebooks/                   # Jupyter notebooks## 🔬 Methodology

│   └── News_Miss_Info.ipynb

│### 1. Data Preprocessing

├── 💻 src/                         # Source code- Text cleaning and normalization

│   └── news_miss_info.py- URL, email, and mention removal

│- Stopword removal and lemmatization

├── 📈 visualizations/              # Generated plots- Feature extraction (length, word count, punctuation, etc.)

│   ├── comprehensive_eda.png

│   ├── wordclouds.png### 2. Feature Engineering

│   ├── smote_effect.png- **TF-IDF Vectorization**: 5000 max features with bigrams

│   ├── classification_results.png- **Statistical Features**: Text length, word count, capital letters, punctuation

│   └── clustering_results.png- **PCA**: Dimensionality reduction for clustering

│

└── 📚 docs/                        # Documentation### 3. Class Imbalance Handling

    └── methodology.md- **SMOTE** (Synthetic Minority Over-sampling Technique)

```- Balanced class distribution for improved model performance



## 🔬 Methodology### 4. Model Training

- 5 calibrated classification models

<details>- Stratified train-test split (80/20)

<summary><b>Click to expand methodology details</b></summary>- Probability calibration using sigmoid method

- Ensemble voting classifier

### 1. Data Preprocessing

- Text cleaning and normalization### 5. Evaluation

- URL, email, and mention removal- Confusion matrices

- Stopword removal and lemmatization- Precision, Recall, F1-Score

- Feature extraction (length, word count, punctuation, etc.)- Cross-validation

- Clustering analysis (K-Means)

### 2. Feature Engineering

- **TF-IDF Vectorization**: 5000 max features with bigrams## 📈 Datasets

- **Statistical Features**: Text length, word count, capital letters, punctuation

- **PCA**: Dimensionality reduction for clustering### Primary Dataset: Constraint

- Source: University-provided dataset

### 3. Class Imbalance Handling- Split: Train, Test, Validation

- **SMOTE** (Synthetic Minority Over-sampling Technique)- Format: Tweet-based labeled data

- Balanced class distribution for improved model performance

### Secondary Dataset: News

### 4. Model Training- Source: Kaggle

- 5 calibrated classification models- Enhanced training with news articles

- Stratified train-test split (80/20)- Combined title and text content

- Probability calibration using sigmoid method

- Ensemble voting classifier### Data Statistics

- Total samples after preprocessing: ~50K+

### 5. Evaluation- Features: 5000 TF-IDF features + statistical features

- Confusion matrices- Classes: Real (0) vs Fake (1)

- Precision, Recall, F1-Score

- Cross-validation## 🛠️ Technologies Used

- Clustering analysis (K-Means)

- **Python 3.x**: Core programming language

</details>- **Pandas & NumPy**: Data manipulation and analysis

- **Scikit-learn**: Machine learning algorithms

## 📈 Datasets- **NLTK**: Natural language processing

- **Imbalanced-learn**: SMOTE implementation

| Dataset | Source | Size | Description |- **Matplotlib & Seaborn**: Visualization

|---------|--------|------|-------------|- **WordCloud**: Text visualization

| **Constraint** | University | ~10K tweets | Official labeled social media posts |- **Joblib**: Model persistence

| **News** | Kaggle | ~40K articles | News articles with title and content |

| **Combined** | Both sources | ~50K+ samples | Cleaned and preprocessed data |## 📊 Visualizations



**Classes:** Real (0) vs Fake (1)  The project includes comprehensive visualizations:

**Features:** 5000 TF-IDF features + statistical features- Label distribution analysis

- Text length and word count distributions

## 🛠️ Technologies Used- Word clouds for real vs fake news

- Model performance comparisons

<div align="center">- Confusion matrices

- Clustering analysis

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)- Feature importance plots

![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)## 🎯 Key Insights

![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)1. **Text Length**: Fake news tends to have different text length patterns

2. **Punctuation**: Higher exclamation marks in fake news

</div>3. **Word Patterns**: Distinct vocabulary differences between real and fake news

4. **Ensemble Performance**: Combining multiple models improves reliability

- **Python 3.x**: Core programming language5. **Calibration**: Probability calibration reduces false positives

- **Pandas & NumPy**: Data manipulation and analysis

- **Scikit-learn**: Machine learning algorithms## 🔮 Future Enhancements

- **NLTK**: Natural language processing

- **Imbalanced-learn**: SMOTE implementation- [ ] Deep learning models (LSTM, BERT, Transformers)

- **Matplotlib & Seaborn**: Visualization- [ ] Real-time API for predictions

- **Streamlit**: Interactive web interface- [ ] Web interface for user interaction

- **WordCloud**: Text visualization- [ ] Multi-language support

- **Joblib**: Model persistence- [ ] Social media integration (Twitter API)

- [ ] Explainability features (LIME, SHAP)

## 📊 Visualizations- [ ] Mobile application



The project includes comprehensive visualizations:## 📝 License



- 📊 Label distribution analysisThis project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

- 📏 Text length and word count distributions

- ☁️ Word clouds for real vs fake news## 👤 Author

- 📈 Model performance comparisons

- 🎯 Confusion matrices**Sadini Wanniarachchi**

- 🔍 Clustering analysis

- ⭐ Feature importance plots- GitHub: [@SadiniWanniarachchi](https://github.com/SadiniWanniarachchi)

- Repository: [Social-Media-Misinformation-Detection-System-Model](https://github.com/SadiniWanniarachchi/Social-Media-Misinformation-Detection-System-Model)

## 🎯 Key Insights

## 🙏 Acknowledgments

| Insight | Finding |

|---------|---------|- Constraint dataset providers

| **Text Length** | Fake news tends to have different text length patterns |- Kaggle for news dataset

| **Punctuation** | Higher exclamation marks in fake news |- Open-source ML community

| **Word Patterns** | Distinct vocabulary differences between real and fake news |- Scikit-learn and NLTK contributors

| **Ensemble Performance** | Combining multiple models improves reliability |

| **Calibration** | Probability calibration reduces false positives |## 📧 Contact



## 🔮 Future EnhancementsFor questions, suggestions, or collaboration opportunities, please open an issue on GitHub or contact through my GitHub profile.



- [ ] 🧠 Deep learning models (LSTM, BERT, Transformers)## ⭐ Star This Repository

- [ ] 🌐 RESTful API for predictions

- [ ] 🌍 Multi-language supportIf you find this project helpful, please consider giving it a star! ⭐

- [ ] 📱 Mobile application
- [ ] 🔗 Social media integration (Twitter API)
- [ ] 🔍 Explainability features (LIME, SHAP)
- [ ] ⚡ Real-time streaming detection
- [ ] 📊 Advanced analytics dashboard

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Sadini Wanniarachchi**

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/SadiniWanniarachchi)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/sadini-wanniarachchi)

</div>

## 🙏 Acknowledgments

- 🎓 Constraint dataset providers
- 📊 Kaggle for news dataset
- 💻 Open-source ML community
- 🔧 Scikit-learn and NLTK contributors
- 🌟 Streamlit team for the amazing framework

## 📧 Contact

For questions, suggestions, or collaboration opportunities:

- 📝 Open an issue on GitHub
- 💬 Connect via GitHub profile
- ⭐ Star the repository if you find it helpful!

---

<div align="center">

### ⭐ Star This Repository

**If you find this project helpful, please consider giving it a star!** ⭐

Made with ❤️ by Sadini Wanniarachchi

</div>
