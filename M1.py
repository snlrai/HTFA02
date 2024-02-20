import spacy
import pandas as pd
# Load SpaCy's English model with word vectors
nlp = spacy.load("en_core_web_lg")

# Define the columns
columns = [
    'name' , 'nickname' , 'full name', 'first name', 'last name' , 'customer name', 'email', 'phone number', 'address', 'city' ,
    'state', 'postal_code', 'country', 'gender', 'age' , 'customer age' , 'birthdate', 'customer_id',
    'social_security_number', 'passport_number', 'driver_license_number',
    'bank_account_number', 'credit_card_number', 'medical_record_number', 'health_insurance_number'
]

# Function to check similarity between input text and each column
def check_similarity(input_text):
    input_text = input_text.lower()
    similarity_scores = {}
    input_doc = nlp(input_text)
    for col in columns:
        column_doc = nlp(col)
        if input_doc.vector_norm and column_doc.vector_norm:  # Check if both documents have non-empty vectors
            similarity_scores[col] = input_doc.similarity(column_doc)
        else:
            similarity_scores[col] = 0  # Set similarity score to 0 if one of the documents has an empty vector
    return similarity_scores

# Example input text
df = pd.read_csv('./ecommerce_customer_data_large.csv')
column_names = df.columns.tolist()
df.columns = df.columns.str.lower()
column_names = df.columns.tolist()
# print(column_names)
for colu in column_names:
    similarity_scores = check_similarity(colu)
    for col, score in similarity_scores.items():
        if score>0.7:
            print(f"Similarity with {colu} '{col}': {score}")
            if colu == col:
                df = df.drop(columns=[colu])
                column_names = df.columns.tolist()
                print(column_names)
# df.to_csv('output1.csv', index=False)

