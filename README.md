# Wine Quality Prediction

A machine learning project that predicts wine quality using various classification algorithms. The project includes comprehensive exploratory data analysis (EDA), feature engineering, dimensionality reduction, and model comparison.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Methodology](#methodology)
- [Project Flow](#project-flow)
- [Features](#features)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## 🎯 Project Overview

This project aims to predict wine quality (binary classification: good/bad) based on various physicochemical properties. The solution employs multiple machine learning algorithms and compares their performance to identify the best model for wine quality prediction.

## 🛠 Tech Stack

- **Programming Language**: Python 3.6+
- **Data Manipulation**: 
  - `pandas` - Data handling and preprocessing
  - `numpy` - Numerical computations
- **Data Visualization**: 
  - `matplotlib` - Plotting and visualization
  - `seaborn` - Statistical visualization
- **Machine Learning**: 
  - `scikit-learn` - ML algorithms and utilities
    - `RandomForestClassifier` - Feature importance analysis
    - `LogisticRegression` - Binary classification
    - `SVC` - Support Vector Machine (Linear & RBF kernels)
    - `StandardScaler` - Feature scaling
    - `PCA` - Principal Component Analysis
    - `train_test_split` - Data splitting
    - `LabelEncoder` - Label encoding
- **Metrics**: 
  - Accuracy, Precision, Recall, F1 Score
  - Confusion Matrix

## 📊 Dataset

The dataset (`winequality.csv`) contains wine samples with the following features:

### Input Features (11):
1. **fixed acidity** - Non-volatile acids in wine
2. **volatile acidity** - Acetic acid content
3. **citric acid** - Citric acid content
4. **residual sugar** - Sugar remaining after fermentation
5. **chlorides** - Salt content
6. **free sulfur dioxide** - Free form of SO2
7. **total sulfur dioxide** - Total SO2 content
8. **density** - Wine density
9. **pH** - Acidity level
10. **sulphates** - Potassium sulphate content
11. **alcohol** - Alcohol percentage

### Target Variable:
- **quality** - Wine quality rating (converted to binary: bad/good)

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Loading & Preprocessing              │
│  - Load CSV dataset                                          │
│  - Check for missing values                                  │
│  - Convert quality to binary classification                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Exploratory Data Analysis (EDA)                 │
│  - Statistical summaries                                     │
│  - Data visualizations (histograms, pairplots)              │
│  - Correlation analysis                                      │
│  - Feature importance analysis                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Feature Engineering                       │
│  - Feature selection (drop target variable)                  │
│  - Train-test split (80:20)                                  │
│  - Feature scaling (StandardScaler)                          │
│  - Dimensionality reduction (PCA, 4 components)              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      Model Training                          │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │ Logistic         │  │ SVM (Linear)     │                 │
│  │ Regression       │  │                  │                 │
│  └──────────────────┘  └──────────────────┘                 │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │ SVM (RBF)        │  │ Random Forest    │                 │
│  │                  │  │                  │                 │
│  └──────────────────┘  └──────────────────┘                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Model Evaluation                          │
│  - Accuracy, Precision, Recall, F1 Score                     │
│  - Confusion Matrix                                          │
│  - Model Comparison                                          │
└─────────────────────────────────────────────────────────────┘
```

## 📈 Methodology

### 1. Data Preprocessing
- **Binary Classification**: Converted quality ratings (2-8) into binary categories:
  - Bad: Quality ≤ 6.5
  - Good: Quality > 6.5
- **Label Encoding**: Transformed categorical labels to numerical format

### 2. Exploratory Data Analysis
- **Statistical Analysis**: Computed descriptive statistics for all features
- **Visualization**: 
  - Histograms for feature distribution
  - Pair plots for feature relationships
  - Correlation heatmap to identify feature relationships
  - Bar chart showing correlation with target variable
- **Feature Importance**: Used Random Forest to rank feature importance

### 3. Feature Engineering
- **Train-Test Split**: 80% training, 20% testing with random state for reproducibility
- **Standardization**: Applied StandardScaler to normalize features
- **Dimensionality Reduction**: PCA with 4 components explaining ~71% variance:
  - Component 1: 28.17%
  - Component 2: 17.15%
  - Component 3: 14.32%
  - Component 4: 11.48%

### 4. Model Training & Evaluation
Four classification models were trained and compared:
- **Logistic Regression**: Baseline linear classifier
- **SVM (Linear Kernel)**: Linear support vector machine
- **SVM (RBF Kernel)**: Non-linear support vector machine
- **Random Forest**: Ensemble method with 100 trees

## 🔄 Project Flow

1. **Import Libraries** → Load necessary Python packages
2. **Load Dataset** → Read `winequality.csv`
3. **Data Preprocessing** → Convert quality to binary classification
4. **EDA** → Explore data distributions, correlations, and relationships
5. **Feature Engineering** → Split data, scale features, apply PCA
6. **Feature Selection** → Analyze feature importance using Random Forest
7. **Model Training** → Train multiple classification models
8. **Model Evaluation** → Compare performance metrics
9. **Results Analysis** → Identify best-performing model

## 🎨 Features

- **Comprehensive EDA**: Visual and statistical analysis of the dataset
- **Feature Importance Analysis**: Identifies most influential features
- **Dimensionality Reduction**: PCA for efficient feature representation
- **Multiple Model Comparison**: Evaluates 4 different algorithms
- **Performance Metrics**: Accuracy, Precision, Recall, F1 Score
- **Reproducible Results**: Fixed random seeds for consistency

## 📊 Model Performance

Based on the evaluation metrics:

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **Random Forest (n=100)** | **90.31%** | **74.19%** | **50.00%** | **59.74%** |
| SVM (RBF) | 87.50% | 71.43% | 21.74% | 33.33% |
| Logistic Regression | 86.88% | 60.00% | 26.09% | 36.36% |
| SVM (Linear) | 85.63% | 0.00% | 0.00% | 0.00% |

**Best Model**: Random Forest Classifier with 100 estimators

### Feature Importance Ranking:
1. Fixed acidity (17.87%)
2. Volatile acidity (12.87%)
3. Citric acid (10.65%)
4. Residual sugar (9.57%)
5. Chlorides (8.93%)
6. Free sulfur dioxide (8.35%)
7. Total sulfur dioxide (7.07%)
8. Density (7.01%)
9. pH (6.05%)
10. Sulphates (6.00%)
11. Alcohol (5.63%)

## 💻 Installation

### Prerequisites
- Python 3.6 or higher
- Jupyter Notebook

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd "Wine Quality prediction"
```

2. Install required packages:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

Or use requirements.txt (if available):
```bash
pip install -r requirements.txt
```

## 🚀 Usage

1. **Open Jupyter Notebook**:
```bash
jupyter notebook Wine.ipynb
```

2. **Run the cells sequentially**:
   - Execute cells in order to perform EDA, preprocessing, and model training
   - Review visualizations and metrics generated at each step

3. **Customize**:
   - Adjust PCA components if needed
   - Modify model hyperparameters
   - Change train-test split ratio
   - Add additional models for comparison

## 📈 Results

- The Random Forest model achieved the highest accuracy (90.31%)
- Fixed acidity and volatile acidity are the most important features
- PCA reduced dimensionality from 11 features to 4 components while retaining ~71% variance
- The binary classification approach effectively separates good and bad quality wines

## 🔮 Future Enhancements

- Hyperparameter tuning for better model performance
- Cross-validation for more robust evaluation
- Additional models (XGBoost, Neural Networks)
- Feature engineering improvements
- Model deployment pipeline
- Real-time prediction API

## 📝 License

This project is open source and available for educational purposes.

## 👤 Author

Developed as part of a machine learning project for wine quality prediction.

---

**Note**: This project is for educational and demonstration purposes. Results may vary based on dataset characteristics and model configurations.

