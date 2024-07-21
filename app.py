import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone and model
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define index name and parameters
index_name = "locobo"
if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)

pc.create_index(
    name=index_name,
    dimension=384, 
    metric="cosine", 
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)

index = pc.Index(index_name)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the FAQ data with the correct encoding
csv_file_path = 'faq.csv'
faqs = pd.read_csv(csv_file_path, encoding='windows-1252')

# Prepare data for indexing
data_to_upsert = []
for idx, row in faqs.iterrows():
    question = row['question']
    answer = row['answer']
    embedding = model.encode(question).tolist()  # Convert question to embedding vector
    data_to_upsert.append((f'id-{idx}', embedding, {'question': question, 'answer': answer}))

# Upsert data to Pinecone
index.upsert(vectors=data_to_upsert)

print("Data uploaded to Pinecone successfully.")
