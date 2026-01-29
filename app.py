import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------
# Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="Water Quality Classification",
    layout="wide"
)

# ----------------------------------
# Load Model & Dataset
# ----------------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_water_quality_model.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("NWMP_August2025_MPCB_0.csv", encoding="latin1")
    # Clean the data
    for col in RAW_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

model = load_model()

# ----------------------------------
# Raw Features (MUST MATCH TRAINING ORDER)
# ----------------------------------
RAW_FEATURES = [
    "pH", "BOD", "Dissolved O2", "COD",
    "Conductivity", "Total Dissolved Solids",
    "Nitrate N", "Phosphate", "Chlorides", "Sulphate",
    "Fecal Coliform", "Total Coliform", "Fecal Streptococci",
    "Temperature", "Turbidity"
]

df = load_data()

# ----------------------------------
# Class Mapping
# ----------------------------------
CLASS_MAP = {
    0: "A – Drinking Water (After Disinfection)",
    1: "B – Outdoor Bathing",
    2: "C – Drinking Water (With Treatment)",
    3: "E – Irrigation / Industrial Use",
    4: "No Information"
}

# ----------------------------------
# Helper Functions
# ----------------------------------
def clean_input(df_input, reference_df):
    """Handle NaN, inf using median imputation"""
    df_input = df_input.replace([np.inf, -np.inf], np.nan)
    
    for col in RAW_FEATURES:
        if pd.isna(df_input[col].iloc[0]):
            median_val = reference_df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            df_input[col] = median_val
    
    return df_input

# ----------------------------------
# Sidebar Navigation
# ----------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Predict", "EDA", "Models"]
)

# ==================================
# HOME PAGE
# ==================================
if page == "Home":
    st.title("💧 Water Quality Classification System")
    
    st.markdown("""
    ## Project Overview
    This application predicts **water quality classes (A, B, C, E)** using
    physicochemical and biological parameters collected from water bodies
    across Maharashtra.
    
    ## Water Quality Classes
    - **A** – Drinking water source (after disinfection)
    - **B** – Outdoor bathing
    - **C** – Drinking water (with treatment)
    - **E** – Irrigation / industrial use
    
    ---
    ✅ Final Model Used: **Random Forest Classifier**
    """)

# ==================================
# PREDICT PAGE
# ==================================
elif page == "Predict":
    st.title("🔮 Water Quality Prediction")
    
    st.info("ℹ️ Missing / invalid values are handled automatically using median imputation.")
    
    user_input = {}
    cols = st.columns(3)
    
    # Get default values from dataset
    default_values = df[RAW_FEATURES].median().to_dict()
    
    for i, feature in enumerate(RAW_FEATURES):
        default_val = default_values.get(feature, 0.0)
        user_input[feature] = cols[i % 3].number_input(
            feature,
            value=float(default_val) if not pd.isna(default_val) else 0.0,
            format="%.3f",
            help=f"Default: {default_val:.3f}"
        )
    
    if st.button("Predict Water Quality", type="primary"):
        try:
            # Create input dataframe with exact feature order
            input_df = pd.DataFrame([user_input])
            
            # Ensure columns are in the correct order
            input_df = input_df[RAW_FEATURES]
            
            # Validation
            if (input_df < 0).any().any():
                st.error("❌ Invalid input: values must be non-negative.")
            else:
                # Clean input
                input_df = clean_input(input_df, df)
                
                # Make prediction
                prediction = model.predict(input_df)[0]
                prediction_proba = model.predict_proba(input_df)[0]
                
                # Display result
                st.success(
                    f"✅ Predicted Water Quality Class: **{CLASS_MAP[prediction]}**"
                )
                
                # Show confidence
                st.subheader("Prediction Confidence")
                conf_df = pd.DataFrame({
                    'Class': [CLASS_MAP[i] for i in range(len(prediction_proba))],
                    'Probability': prediction_proba
                }).sort_values('Probability', ascending=False)
                
                st.dataframe(conf_df, use_container_width=True)
                
                # Show input summary
                with st.expander("📊 Input Parameters Summary"):
                    st.dataframe(input_df.T, use_container_width=True)
                    
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            st.error("Please check your input values and try again.")

# ==================================
# EDA PAGE
# ==================================
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")
    
    st.subheader("Water Quality Class Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    df_clean = df.dropna(subset=['Use Based Class'])
    sns.countplot(y="Use Based Class", data=df_clean, ax=ax, palette="viridis")
    ax.set_title("Distribution of Water Quality Classes")
    st.pyplot(fig)
    plt.close()
    
    st.subheader("pH Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df["pH"].dropna(), kde=True, ax=ax, color="skyblue", bins=30)
    ax.set_title("Distribution of pH Values")
    st.pyplot(fig)
    plt.close()
    
    st.subheader("BOD vs Dissolved Oxygen")
    fig, ax = plt.subplots(figsize=(10, 6))
    df_plot = df.dropna(subset=["BOD", "Dissolved O2", "Use Based Class"])
    sns.scatterplot(
        x="BOD",
        y="Dissolved O2",
        hue="Use Based Class",
        data=df_plot,
        ax=ax,
        palette="Set2",
        alpha=0.6
    )
    ax.set_title("BOD vs Dissolved Oxygen by Water Quality Class")
    st.pyplot(fig)
    plt.close()
    
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 10))
    correlation_matrix = df[RAW_FEATURES].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap="coolwarm", 
                center=0, ax=ax, square=True, linewidths=0.5)
    ax.set_title("Correlation Matrix of Water Quality Parameters")
    st.pyplot(fig)
    plt.close()
    
    # Additional statistics
    st.subheader("Summary Statistics")
    st.dataframe(df[RAW_FEATURES].describe(), use_container_width=True)

# ==================================
# MODELS PAGE
# ==================================
elif page == "Models":
    st.title("🤖 Model Evaluation & Selection")
    
    st.markdown("""
    ### Model Performance Comparison
    Three different models were trained and evaluated on the water quality dataset:
    """)
    
    score_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "Weighted F1-Score": [0.51, 0.84, 0.77]
    })
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(score_df, use_container_width=True)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = sns.barplot(x="Model", y="Weighted F1-Score", data=score_df, 
                          ax=ax, palette="viridis")
        ax.set_ylim([0, 1])
        ax.set_title("Model Performance Comparison")
        ax.set_ylabel("Weighted F1-Score")
        
        # Add value labels on bars
        for i, bar in enumerate(bars.patches):
            bars.text(bar.get_x() + bar.get_width()/2., 
                     bar.get_height() + 0.02,
                     f'{score_df.iloc[i]["Weighted F1-Score"]:.2f}',
                     ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig)
        plt.close()
    
    st.success("✅ **Final Model Selected: Random Forest (F1-Score = 0.84)**")
    
    st.markdown("""
    ### Why Random Forest?
    - **Best Performance**: Achieved the highest F1-score (0.84) among all models
    - **Handles Non-linearity**: Effectively captures complex relationships in water quality data
    - **Feature Importance**: Provides insights into which parameters most influence water quality
    - **Robust**: Less prone to overfitting compared to single decision trees
    - **Handles Missing Data**: Can work with datasets containing some missing values
    """)
    
    # Feature Importance (if available)
    try:
        feature_importance = pd.DataFrame({
            'Feature': RAW_FEATURES,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        st.subheader("Feature Importance")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance, 
                   ax=ax, palette="rocket")
        ax.set_title("Top Features Influencing Water Quality Classification")
        st.pyplot(fig)
        plt.close()
        
    except AttributeError:
        st.info("Feature importance visualization not available for this model type.")
