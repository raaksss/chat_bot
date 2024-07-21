import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

csv_file_path = 'faq.csv'
faqs = pd.read_csv(csv_file_path)


faqs = faqs.dropna()

model = SentenceTransformer('all-MiniLM-L6-v2')
pc = Pinecone(api_key="a79d5f47-9065-4a62-81fd-34b335170e2a")
index_name = "locobo"
index = pc.Index(index_name)

# Prepare data for indexing
data = []
for idx, row in faqs.iterrows():
    question = str(row['question'])  
    answer = str(row['answer'])      
    embedding = model.encode(question).tolist()
    data.append((f'id-{idx}', embedding, {'question': question, 'answer': answer}))


index.upsert(vectors=data)

print("Data uploaded to Pinecone successfully.")
print(index.describe_index_stats())
