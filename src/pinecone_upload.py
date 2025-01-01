from chunks_extractor import chunker_extractor
import os
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
print("Imported libraries and loaded environment variables")

# Initialize Pinecone
index_name = "simple-rag-app"
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create the index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric='dotproduct',
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"Index '{index_name}' created")
else:
    print(f"Index '{index_name}' already exists")

# Initialize the Pinecone index object
index = pc.Index(index_name)
print("Pinecone index initialized")

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
print("Google Generative AI configured")

# Initialize and fit BM25 encoder
bm25_encoder = BM25Encoder().default()
print("BM25 encoder initialized")

# Extract chunks from the PDF
pdf_name = "knowledge.pdf"
chunks = chunker_extractor(pdf_name)
print(f"Extracted {len(chunks)} chunks from the PDF")

# Fit BM25 encoder on the chunks
bm25_encoder.fit(chunks)
bm25_encoder.dump("bm25_vals.json")
print("BM25 encoder fitted and dumped")

# Initialize the PineconeHybridSearchRetriever
retriever = PineconeHybridSearchRetriever(
    index=index,  # Pass the Pinecone index object, not the name
    embeddings=embeddings,
    sparse_encoder=bm25_encoder
)
print("PineconeHybridSearchRetriever initialized")

# Add texts to the retriever
retriever.add_texts(chunks)
print("Texts added to retriever")

# Example query
query = "Which city did I visit in 2022?"
results = retriever.invoke(query)
print("Query results:", results)