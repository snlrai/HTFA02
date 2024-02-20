import spacy
import pandas as pd
import pickle
# Load SpaCy's English model with word vectors
nlp = spacy.load("en_core_web_lg")

# with open('dataframe.pkl', 'rb') as f:
#     df = pickle.load(f)

df = pd.read_csv('./output1.csv')

# Define the columns
purchase_date_titles_lower = [
    "purchase date"
]

product_sales_titles_lower = [
    "quantity sold",
    "units sold",
    "number of items sold",
    "sales volume",
    "total sales",
    "product sales",
    "sales quantity",
    "sales count",
    "product units sold",
    "sales quantity",
    "sold units",
    "sales amount",
    "product count",
    "items sold",
    "sales quantity"
]



# Function to check similarity between input text and each column
def check_similarity(input_text,check_data):
    input_text = input_text.lower()
    similarity_scores = {}
    input_doc = nlp(input_text)
    for col in check_data:
        column_doc = nlp(col)
        if input_doc.vector_norm and column_doc.vector_norm:  # Check if both documents have non-empty vectors
            similarity_scores[col] = input_doc.similarity(column_doc)
        else:
            similarity_scores[col] = 0  # Set similarity score to 0 if one of the documents has an empty vector
    return similarity_scores

# Example input text


column_names = df.columns.tolist()
df.columns = df.columns.str.lower()
column_names = df.columns.tolist()

for column in column_names:
    similarity_scores = check_similarity(column,purchase_date_titles_lower)
    for col, score in similarity_scores.items():
        if score > 0.8:
            print(f"Similarity with {column} '{col}': {score}")
            df = df.rename(columns={column:'purchase date'})

for column in column_names:
    similarity_scores = check_similarity(column, product_sales_titles_lower)
    for col, score in similarity_scores.items():
        if score > 0.8:
            print(f"Similarity with {column} '{col}': {score}")
            df = df.rename(columns={column:'quantity'})

df.to_csv('output2.csv', index=False)
