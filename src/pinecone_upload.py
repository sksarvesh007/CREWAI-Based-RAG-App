#script for extracting the text from the pdfs and uploading to pinecone vector database 
import PyPDF2
import google.generativeai as genai
import pinecone
from typing import List

def upload_pdf_to_pinecone(pdf_path: str, google_api_key: str, pinecone_api_key: str, pinecone_environment: str, pinecone_index_name: str, chunk_size: int = 1000):
    """
    Uploads a PDF file's text chunks to Pinecone VectorDB using Google Generative AI embeddings.

    Args:
        pdf_path (str): Path to the PDF file.
        google_api_key (str): Google Generative AI API key.
        pinecone_api_key (str): Pinecone API key.
        pinecone_environment (str): Pinecone environment.
        pinecone_index_name (str): Pinecone index name.
        chunk_size (int): Size of each text chunk (default: 1000 characters).
    """
    # Initialize Google Generative AI
    genai.configure(api_key=google_api_key)
    embedding_model = genai.embed_content

    # Initialize Pinecone
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)

    # Function to extract text from PDF
    def extract_text_from_pdf(pdf_path: str) -> str:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text

    # Function to split text into chunks
    def split_text_into_chunks(text: str, chunk_size: int) -> List[str]:
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        return chunks

    # Function to generate embeddings using Google Generative AI
    def generate_embeddings(chunks: List[str]) -> List[List[float]]:
        embeddings = []
        for chunk in chunks:
            response = embedding_model(model="models/embedding-001", content=chunk)
            embeddings.append(response["embedding"])
        return embeddings

    # Function to upload embeddings to Pinecone
    def upload_to_pinecone(index_name: str, chunks: List[str], embeddings: List[List[float]]):
        index = pinecone.Index(index_name)
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vectors.append((f"chunk_{i}", embedding, {"text": chunk}))
        index.upsert(vectors=vectors)

    # Step 1: Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    print("Text extracted from PDF.")

    # Step 2: Split text into chunks
    chunks = split_text_into_chunks(text, chunk_size)
    print(f"Text split into {len(chunks)} chunks.")

    # Step 3: Generate embeddings
    embeddings = generate_embeddings(chunks)
    print("Embeddings generated.")

    # Step 4: Upload embeddings to Pinecone
    upload_to_pinecone(pinecone_index_name, chunks, embeddings)
    print("Embeddings uploaded to Pinecone.")