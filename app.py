import streamlit as st
import joblib

# Load model
model = joblib.load("iris_model.pkl")

# Page config
st.set_page_config(page_title="Iris Prediction App")

# Styling (background + text + button color)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7fa;
    }

    /* Force text to be visible */
    h1, h2, h3, h4, h5, h6, p, label {
        color: #000000 !important;
    }

    /* Button styling */
    div.stButton > button {
        background-color: #87CEFA;  /* light blue */
        color: black;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #6bbbe8;  /* slightly darker on hover */
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.title("🌸 Iris Prediction")
st.write(
    "Provide the measurements of an iris flower below and click the predict button "
    "to determine its species. This application uses a trained machine learning model "
    "to make accurate predictions based on your inputs."
)

# Inputs
sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, step=0.1)

# Prediction
if st.button("Predict"):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(features)

    species = ["🌼 Iris-setosa", "🌸 Iris-versicolor", "🌺 Iris-virginica"]

    st.success(f"Predicted Species: {species[prediction[0]]}")