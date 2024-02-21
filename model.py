import streamlit as st
import pandas as pd
import subprocess
import os
import spacy
def main():
    st.title("Trend Analyzer")

    # File uploader
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        # Save uploaded file to temporary directory
        temp_file_path = save_uploaded_file(uploaded_file)

        # Call preprocessing script (M1.py)
        subprocess.run(["python", "M1.py", temp_file_path])

        # Load preprocessed file
        preprocessed_file_path = "output1.csv"
        preprocessed_df = pd.read_csv(preprocessed_file_path)

        #M2 preprocessing
        subprocess.run(["python", "M2.py", preprocessed_file_path])

        #load preprocessed file
        preprocessed_file_path2 = "output2.csv"
        preprocessed_df = pd.read_csv(preprocessed_file_path2)

        # Display the preprocessed DataFrame
        # st.subheader("Preprocessed DataFrame:")
        # st.dataframe(preprocessed_df)

        #regression model
        subprocess.run(["python", "m3.py", preprocessed_file_path])


        #display the graph
        st.subheader("Generated Graph:")
        st.image("output_graph.png")

def save_uploaded_file(uploaded_file):
    # Create temp directory if it doesn't exist
    if not os.path.exists("temp"):
        os.makedirs("temp")

    # Save uploaded file to temporary directory
    with open(os.path.join("temp", "input.csv"), "wb") as f:
        f.write(uploaded_file.getbuffer())
    return os.path.join("temp", "input.csv")
if __name__ == "__main__":
    main()