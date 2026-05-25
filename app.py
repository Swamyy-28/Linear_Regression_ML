# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(page_title="Linear Regression App", layout="wide")

st.title("📈 Linear Regression")



# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # Load dataset
    df = pd.read_csv(uploaded_file)

    st.success("Dataset Uploaded Successfully ✅")

    # ---------------------------------------------------
    # DATA PREVIEW
    # ---------------------------------------------------

    st.header("📌 Dataset Preview")

    st.dataframe(df.head())

    st.write("### Shape of Dataset")
    st.write(df.shape)

    # ---------------------------------------------------
    # DATA TYPES
    # ---------------------------------------------------

    st.header("📌 Data Types")

    st.write(df.dtypes)

    # ---------------------------------------------------
    # MISSING VALUES
    # ---------------------------------------------------

    st.header("📌 Missing Values")

    st.write(df.isnull().sum())

    # ---------------------------------------------------
    # DESCRIPTIVE STATISTICS
    # ---------------------------------------------------

    st.header("📌 Statistical Summary")

    st.write(df.describe())

    # ---------------------------------------------------
    # EDA SECTION
    # ---------------------------------------------------

    st.header("📊 Exploratory Data Analysis")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Correlation Heatmap
    st.subheader("🔥 Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        df[numeric_cols].corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

    # Histogram
    st.subheader("📉 Distribution Plot")

    selected_col = st.selectbox(
        "Select Column",
        numeric_cols
    )

    fig2, ax2 = plt.subplots()

    sns.histplot(df[selected_col], kde=True, ax=ax2)

    st.pyplot(fig2)

    # Scatter Plot
    st.subheader("📍 Scatter Plot")

    x_scatter = st.selectbox(
        "Select X Column",
        numeric_cols,
        key="x"
    )

    y_scatter = st.selectbox(
        "Select Y Column",
        numeric_cols,
        key="y"
    )

    fig3, ax3 = plt.subplots()

    sns.scatterplot(
        x=df[x_scatter],
        y=df[y_scatter],
        ax=ax3
    )

    st.pyplot(fig3)

    # ---------------------------------------------------
    # FEATURE & TARGET SELECTION
    # ---------------------------------------------------

    st.header("🎯 Model Building")

    target_column = st.selectbox(
        "Select Target Variable",
        numeric_cols
    )

    feature_columns = st.multiselect(
        "Select Feature Variables",
        [col for col in numeric_cols if col != target_column]
    )

    if len(feature_columns) > 0:

        X = df[feature_columns]
        y = df[target_column]

        # ---------------------------------------------------
        # TRAIN TEST SPLIT
        # ---------------------------------------------------

        test_size = st.slider(
            "Select Test Size",
            0.1,
            0.5,
            0.2
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42
        )

        # ---------------------------------------------------
        # MODEL TRAINING
        # ---------------------------------------------------

        model = LinearRegression()

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        st.success("Model Trained Successfully ✅")

        # ---------------------------------------------------
        # COEFFICIENTS
        # ---------------------------------------------------

        st.header("📌 Model Coefficients")

        coef_df = pd.DataFrame({
            "Feature": feature_columns,
            "Coefficient": model.coef_
        })

        st.dataframe(coef_df)

        st.write("Intercept:", model.intercept_)

        # ---------------------------------------------------
        # MODEL EVALUATION
        # ---------------------------------------------------

        st.header("📊 Model Evaluation")

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("MSE", round(mse, 4))
            st.metric("RMSE", round(rmse, 4))

        with col2:
            st.metric("MAE", round(mae, 4))
            st.metric("R² Score", round(r2, 4))

        # ---------------------------------------------------
        # ACTUAL VS PREDICTED
        # ---------------------------------------------------

        st.header("📈 Actual vs Predicted")

        fig4, ax4 = plt.subplots(figsize=(8, 6))

        ax4.scatter(y_test, y_pred)

        ax4.set_xlabel("Actual Values")
        ax4.set_ylabel("Predicted Values")
        ax4.set_title("Actual vs Predicted")

        st.pyplot(fig4)

        # ---------------------------------------------------
        # RESIDUAL PLOT
        # ---------------------------------------------------

        st.header("📉 Residual Plot")

        residuals = y_test - y_pred

        fig5, ax5 = plt.subplots(figsize=(8, 6))

        sns.histplot(residuals, kde=True, ax=ax5)

        ax5.set_title("Residual Distribution")

        st.pyplot(fig5)

        # ---------------------------------------------------
        # DOWNLOAD PREDICTIONS
        # ---------------------------------------------------

        st.header("⬇ Download Predictions")

        pred_df = pd.DataFrame({
            "Actual": y_test,
            "Predicted": y_pred
        })

        csv = pred_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Predictions CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

else:
    st.info("Please upload a CSV file to continue.")
