# 💧 Water Quality Classification System
### Machine Learning–Based Water Quality Prediction for Maharashtra, India

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A machine learning–powered web application that predicts **water quality classes** using physicochemical and biological parameters collected from **222+ monitoring stations across Maharashtra, India**.  
The system supports **environmental monitoring, public health assessment, and policy decision-making**.

---

## 📌 Problem Statement

### Background
Water quality monitoring is essential for public health, agriculture, and environmental sustainability.  
The Maharashtra Pollution Control Board (MPCB) collects large-scale water quality data, but manual analysis is slow and difficult to scale.

### Problem
**How can physicochemical and biological parameters be used to automatically classify water quality across Maharashtra?**

### Objectives
- Analyze water quality data from 222+ monitoring stations  
- Identify key factors affecting water quality  
- Build a robust machine learning classifier  
- Deploy a user-friendly web application for real-time prediction  

### Water Quality Classes

| Class | Description | Primary Use |
|------|------------|------------|
| A | Drinking Water Source | After disinfection only |
| B | Outdoor Bathing | Organized bathing |
| C | Drinking Water Source | After conventional treatment |
| E | Irrigation & Industrial | Cooling and waste disposal |

---

## 📊 Dataset

### Source
- **Provider:** Maharashtra Pollution Control Board (MPCB)  
- **Period:** August 2025  
- **Coverage:** Maharashtra, India  
- **File:** `NWMP_August2025_MPCB_0.csv`

### Dataset Overview

| Attribute | Value |
|---------|------|
| Total Samples | 222 |
| Total Features | 54 |
| Features Used | 15 |
| Target Variable | Water Quality Class |

### Key Parameters Used
- pH  
- BOD (Biochemical Oxygen Demand)  
- COD (Chemical Oxygen Demand)  
- Dissolved Oxygen  
- Conductivity  
- Total Dissolved Solids (TDS)  
- Nitrates, Phosphates, Chlorides, Sulphates  
- Fecal Coliform  
- Total Coliform  
- Fecal Streptococci  
- Temperature  
- Turbidity  

### Data Preprocessing
- Missing values handled using **median imputation**  
- Special values (BDL, NA, ND) cleaned using custom logic  
- Outliers capped using **IQR method**  
- Class imbalance addressed using **SMOTE**  

---

## 🔍 Key Findings

### Exploratory Data Analysis
- Class **A (Drinking Water)** dominates the dataset (~62%)  
- Severe class imbalance handled using SMOTE  
- Strong inverse relationship between **BOD and Dissolved Oxygen**  

### Correlation Insights
- Conductivity ↔ TDS (r = 0.98)  
- BOD ↔ COD (r = 0.76)  
- Dissolved Oxygen ↔ BOD (r = -0.64)  

### Feature Importance (Random Forest)
Top predictors:
1. Fecal Coliform  
2. Temperature  
3. Fecal Streptococci  
4. pH  
5. Chlorides  
6. Total Dissolved Solids  

**Insight:** Biological contamination indicators are the strongest drivers of water quality classification.

### Geographic Patterns
- Most polluted districts: Mumbai, Thane, Pune  
- Cleanest districts: Ratnagiri, Satara, Sindhudurg  
- Best water quality observed in Krishna River Basin  

---

## 📈 Model Performance

### Models Evaluated

| Model | Accuracy | F1-Score |
|------|----------|---------|
| **Random Forest ⭐** | **85%** | **0.84** |
| XGBoost | 79% | 0.77 |
| SVM (RBF) | 73% | 0.72 |
| Logistic Regression | 50% | 0.51 |

### Selected Model
**Random Forest Classifier**

```python
**Why Random Forest?**
- ✅ Highest F1-Score (0.84)
- ✅ Best balance between precision and recall
- ✅ Robust to outliers
- ✅ Handles non-linear relationships
- ✅ Provides feature importance insights
- ✅ No extensive hyperparameter tuning needed

### Model Configuration
```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    class_weight='balanced',
    random_state=42
)
---
##✨ Features
**🎯 Core Functionality***
1. Real-time Prediction

- Input water quality parameters
- Instant classification results
- Confidence scores for each class
2. Interactive Dashboard

- 4 pages: Home, Predict, EDA, Models
- Responsive design
- Mobile-friendly interface
- Exploratory Data Analysis

3. Class distribution visualization
- Parameter distributions (pH, BOD, etc.)
- Correlation heatmaps
Geographic analysis
Model Comparison

Performance metrics table
Visual model comparison
Feature importance charts
🔧 Technical Features
Automatic Data Cleaning: Handles missing values, outliers
Median Imputation: Fills missing data intelligently
Feature Engineering: Creates derived features (oxygen stress, organic load)
SMOTE Oversampling: Balances imbalanced classes
Robust Validation: Stratified train-test split
📱 Usage
Running the Application
Local Deployment
bash
# Navigate to project directory
cd water-quality-classification

# Start Streamlit app
streamlit run app.py
The app will open in your browser at http://localhost:8501

Cloud Deployment (Streamlit Cloud)
Push code to GitHub
Go to share.streamlit.io
Connect your GitHub repository
Deploy with one click
Using ngrok (Colab/Remote)
bash
# Install ngrok
pip install pyngrok

# Run in background
streamlit run app.py &

# Create tunnel
ngrok http 8501
Application Workflow
1. Home Page
Read project overview
Understand water quality classes
Learn about the model
2. Prediction Page
Python
# Input Parameters (Example: Clean River Water)
pH = 7.5
BOD = 3.2
Dissolved O2 = 8.4
COD = 15.0
Conductivity = 320
# ... (12 more parameters)

# Click "Predict Water Quality"
# Result: Class A - Drinking Water Source
3. EDA Page
Explore data distributions
View correlation heatmaps
Analyze geographic patterns
