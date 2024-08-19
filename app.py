import streamlit as st
import numpy as np
import pickle
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Load the trained model from the pickle file
with open("iris_model.pkl", "rb") as file:
    model = pickle.load(file)

# Define a function to make predictions
def predict_species(features):
    prediction = model.predict(features)
    return iris.target_names[prediction]

# Streamlit app
def main():
    st.title("Iris Species Predictor")
    st.write("This app predicts the species of an Iris flower based on its features.")
    
    # User input form
    with st.form("iris_form"):
        st.header("Enter Iris Flower Features")
        # Input fields for flower features as text without default values
        sepal_length = st.text_input("Sepal Length (cm)")

        sepal_width = st.text_input("Sepal Width (cm)")

        petal_length = st.text_input("Petal Length (cm)")

        petal_width = st.text_input("Petal Width (cm)")
        # sepal_length = st.number_input(
        #     "Sepal Length (cm)",
        #     min_value=0.0,
        #     max_value=10.0,
        #     value=5.1,
        #     step=0.1
        # )
        
        # sepal_width = st.number_input(
        #     "Sepal Width (cm)",
        #     min_value=0.0,
        #     max_value=10.0,
        #     value=3.5,
        #     step=0.1
        # )
        
        # petal_length = st.number_input(
        #     "Petal Length (cm)",
        #     min_value=0.0,
        #     max_value=10.0,
        #     value=1.4,
        #     step=0.1
        # )
        
        # petal_width = st.number_input(
        #     "Petal Width (cm)",
        #     min_value=0.0,
        #     max_value=10.0,
        #     value=0.2,
        #     step=0.1
        # )
        
        # Submit button
        submitted = st.form_submit_button("Predict Species")

        # When the form is submitted
        if submitted:
            # Prepare the feature array
            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            
            # Predict species
            species = predict_species(features)
            
            # Display the predicted species
            st.success(f"The predicted species is: {species[0]}")
            st.balloons()

if __name__ == "__main__":
    main()
