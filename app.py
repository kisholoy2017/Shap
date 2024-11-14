import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Feature Importance Analysis App

Upload your dataset to analyze feature importance using SHAP values.
""")
st.write('---')

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"])
if uploaded_file is not None:
    # Load the uploaded CSV file
    data = pd.read_csv(uploaded_file)
    
    # Display the dataset
    st.header("Uploaded Dataset")
    st.write(data.head())
    
    # Step 2: Feature and Target Selection
    st.subheader("Feature and Target Selection")
    features = st.multiselect("Select feature columns", options=data.columns, default=data.columns[:-1])
    target = st.selectbox("Select the target column (column to predict)", options=data.columns)
    
    if features and target:
        X = data[features]
        Y = data[target]
        
        # Step 3: Train the Model
        model = RandomForestRegressor()
        model.fit(X, Y)

        # Step 4: SHAP Feature Importance Analysis
        # Explaining the model's predictions using SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Display SHAP summary plot
        st.header("Feature Importance")
        
        # SHAP summary plot
        st.subheader("Feature Importance based on SHAP values")
        fig_summary, ax_summary = plt.subplots()
        shap.summary_plot(shap_values, X, show=False)
        st.pyplot(fig_summary)  # Display SHAP summary plot in Streamlit
        st.write('---')

        # SHAP bar plot
        st.subheader("Feature Importance based on SHAP values (Bar)")
        fig_bar, ax_bar = plt.subplots()
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        st.pyplot(fig_bar)  # Display SHAP bar plot in Streamlit
else:
    st.write("Please upload a CSV file to proceed.")
