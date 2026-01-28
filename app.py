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
model = joblib.load("best_water_quality_model.pkl")
df = pd.read_csv("NWMP_August2025_MPCB_0.csv", encoding="latin1")

# ----------------------------------
# Raw & Engineered Features (MUST MATCH TRAINING)
# ----------------------------------
RAW_FEATURES = [
    "pH", "BOD", "Dissolved O2", "COD",
    "Conductivity", "Total Dissolved Solids",
    "Nitrate N", "Phosphate", "Chlorides", "Sulphate",
    "Fecal Coliform", "Total Coliform", "Fecal Streptococci",
    "Temperature", "Turbidity"
]

ENGINEERED_FEATURES = [
    "oxygen_stress",
    "organic_load",
    "nutrient_load",
    "salinity_index",
    "hardness_ratio"
]

ALL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES

# ----------------------------------
# Class Mapping
# ----------------------------------
CLASS_MAP = {
    0: "A – Drinking Water (After Disinfection)",
    1: "B – Outdoor Bathing",
    2: "C – Drinking Water (With Treatment)",
    3: "E – Irrigation / Industrial Use"
}

# ----------------------------------
# Pre-clean reference dataset (numeric coercion)
# ----------------------------------
for col in RAW_FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ----------------------------------
# Helper Functions
# ----------------------------------
def clean_input(df_input, reference_df):
    """Handle NaN, inf, BDL, NA using median imputation"""
    df_input = df_input.replace([np.inf, -np.inf], np.nan)

    for col in RAW_FEATURES:
        median_val = reference_df[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        df_input[col] = df_input[col].fillna(median_val)

    return df_input


def engineer_features(df):
    """Recreate training-time engineered features"""
    eps = 1e-6

    df["oxygen_stress"] = df["BOD"] / (df["Dissolved O2"] + eps)
    df["organic_load"] = df["BOD"] + df["COD"]
    df["nutrient_load"] = df["Nitrate N"] + df["Phosphate"]
    df["salinity_index"] = df["Chlorides"] + df["Total Dissolved Solids"]

    # Safe hardness approximation
    df["hardness_ratio"] = (df["Sulphate"] + df["Total Dissolved Solids"]) / 2

    return df

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

    st.info("ℹ️ Missing / invalid values are handled automatically.")

    user_input = {}
    cols = st.columns(3)

    for i, feature in enumerate(RAW_FEATURES):
        user_input[feature] = cols[i % 3].number_input(
            feature,
            value=None,
            format="%.3f"
        )

    if st.button("Predict Water Quality"):
        input_df = pd.DataFrame([user_input])

        # Validation
        if (input_df < 0).any().any():
            st.error("❌ Invalid input: values must be non-negative.")
        else:
            input_df = clean_input(input_df, df)
            input_df = engineer_features(input_df)
            input_df = input_df[ALL_FEATURES]   # enforce correct order

            prediction = model.predict(input_df)[0]

            st.success(
                f"✅ Predicted Water Quality Class: **{CLASS_MAP[prediction]}**"
            )

# ==================================
# EDA PAGE
# ==================================
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Water Quality Class Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Use Based Class", data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("pH Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["pH"], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("BOD vs Dissolved Oxygen")
    fig, ax = plt.subplots()
    sns.scatterplot(
        x="BOD",
        y="Dissolved O2",
        hue="Use Based Class",
        data=df,
        ax=ax
    )
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[RAW_FEATURES].corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ==================================
# MODELS PAGE
# ==================================
elif page == "Models":
    st.title("🤖 Model Evaluation & Selection")

    score_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "Weighted F1-Score": [0.54, 0.84, 0.73]
    })

    st.dataframe(score_df)

    fig, ax = plt.subplots()
    sns.barplot(x="Model", y="Weighted F1-Score", data=score_df, ax=ax)
    st.pyplot(fig)

    st.success("✅ Final Model Selected: **Random Forest (F1 = 0.84)**")
