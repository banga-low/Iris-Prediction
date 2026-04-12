import streamlit as st
import joblib
 
#load model in a variable
model = joblib.load("iris_model.pkl")
 
 
features = [[3.5, 4.5, 5, 2.5]]
y = model.predict(features)
print(y)
 
#Streamlit part
st.title("Iris Classification")
st.write("This webpage allows a user to enter iris flower petal lengths and widths and sepal lengths and widths and returns the iris class to him")
sepal_length = st.number_input("Enter sepal length", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("Enter sepal width", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("Enter petal length", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input("Enter petal width", min_value=0.0, max_value=10.0, step=0.1)
 
if st.button("Predict"):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    y = model.predict(features)
    if y == 0:
        y = "Iris-setosa"
    elif y == 1:
        y = "Iris-versicolor"
    else:
        y = "Iris-virginica"
    st.write(f"The predicted iris species is: {y}")