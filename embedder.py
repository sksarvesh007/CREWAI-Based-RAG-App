from langchain_community.retrievers import PineconeHybridSearchRetriever
import os 
from pinecone import Pinecone , ServerlessSpec
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
from langchain_google_genai import GoogleGenerativeAIEmbeddings
index_name = "simple-rag-app"
from dotenv import load_dotenv
import pinecone
load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name,dimension=768 , metric='dotproduct' , spec=ServerlessSpec(cloud="aws", region="us-east-1"))

embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

from pinecone_text.sparse import BM25Encoder
bm25_encoder = BM25Encoder().default()

pdf_name = "knowledge.pdf"
import PyPDF2
pdf_file = open(pdf_name, 'rb')
pdf_reader = PyPDF2.PdfFileReader(pdf_file)
text = ""
for page_num in range(pdf_reader.numPages):
    page = pdf_reader.getPage(page_num)
    text += page.extract_text()
pdf_file.close()
