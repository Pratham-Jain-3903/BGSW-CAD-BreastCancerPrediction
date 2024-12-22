import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_and_split_data
from feature_selection import select_best_features
from model_training_evaluation import train_and_evaluate_model

# Setting the page configuration
st.set_page_config(page_title="Breast Cancer Prediction App", page_icon=":bar_chart:", layout="wide")

# Sidebar navigation
def sidebar():
    st.sidebar.image("assets/logo.png", use_column_width=True)
    st.sidebar.title("Navigation")
    pages = ["Home", "Data Preprocessing", "Feature Selection", "Model Training & Evaluation", "Prediction"]
    return st.sidebar.radio("Choose a page", pages)

# Main app function
def main():
    page = sidebar()

    if page == "Home":
        st.title("Welcome to the Breast Cancer Prediction App")
        st.write("""
        This app allows you to:
        - Upload a dataset
        - Preprocess the data
        - Select the best features
        - Train and evaluate models
        - Make predictions
        """)
        st.image("assets/logo.png", use_column_width=True)

    elif page == "Data Preprocessing":
        st.header("Data Preprocessing")

        # File upload section
        dataset_file = st.file_uploader("Upload Dataset (.csv)", type=["csv"])
        if dataset_file:
            target_variable = st.text_input("Enter the Target Variable")

            if st.button("Preprocess Data") and target_variable:
                X_train, X_test, y_train, y_test = load_and_split_data(dataset_file, target_variable)

                # Store the target variable and preprocessed data in session state
                st.session_state['target_variable'] = target_variable
                st.session_state['preprocessed'] = True

                st.success("Data preprocessed successfully!")
                st.write("**Training Data Preview**", X_train.head())
                st.write("**Testing Data Preview**", X_test.head())
                st.write(f"**Target Variable**: {target_variable}")

    elif page == "Feature Selection":
        st.header("Feature Selection")

        # Check if the data was preprocessed
        if 'preprocessed' not in st.session_state:
            st.warning("Please preprocess the data first in the 'Data Preprocessing' section.")
        else:
            target_variable = st.session_state['target_variable']

            if st.button("Select Best Features"):
                selected_features = select_best_features('train_data.csv', target_variable)
                st.write(f"**Selected Features**: {selected_features}")

                # Visualize feature importance
                st.subheader("Feature Importance")
                plt.figure(figsize=(10, 5))
                sns.barplot(x=selected_features, y=range(len(selected_features)))
                st.pyplot(plt)

    elif page == "Model Training & Evaluation":
        st.header("Model Training & Evaluation")
        if st.button("Train Model"):
            selected_features = pd.read_csv('selected_features_info.csv')['selected_features'].tolist()
            accuracy, cv_scores = train_and_evaluate_model(selected_features)
            st.success("Model trained successfully!")
            st.write(f"Test set accuracy: {accuracy}")
            st.write("Cross-validation scores", cv_scores)

            # Plot Cross-validation scores
            st.subheader("Cross-validation Score Distribution")
            plt.figure(figsize=(8, 4))
            sns.boxplot(cv_scores)
            st.pyplot(plt)

    elif page == "Prediction":
        st.header("Prediction")

        model = joblib.load('models/trained_ensemble_model.pkl')
        selected_features = pd.read_csv('selected_features_info.csv')['selected_features'].tolist()

        st.write("Enter values for the following features:")
        user_input = {}
        for feature in selected_features:
            user_input[feature] = st.number_input(f"Enter {feature}", step=0.01)

        if st.button("Predict"):
            user_data = pd.DataFrame([user_input])
            prediction = model.predict(user_data)
            prediction_proba = model.predict_proba(user_data)
            st.write(f"**Predicted class**: {prediction[0]}")
            st.write(f"**Prediction probabilities**: {prediction_proba[0]}")

            # Plot Prediction probabilities
            st.subheader("Prediction Probability Distribution")
            plt.bar(range(len(prediction_proba[0])), prediction_proba[0])
            st.pyplot(plt)

if __name__ == "__main__":
    main()
