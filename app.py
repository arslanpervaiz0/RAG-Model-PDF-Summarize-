import os
import gradio as gr
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import faiss
from groq import Groq
from dotenv import load_dotenv

# Load .env if available
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Initialize embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')


# Step 1: Extract text from PDFs
def extract_text_from_pdfs(files):
    full_text = ""
    for file in files:
        reader = PdfReader(file.name)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text


# Step 2: Split text into semantic chunks
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)


# Step 3: Embed chunks using sentence-transformers
def embed_chunks(chunks):
    embeddings = embed_model.encode(chunks)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))
    return index, embeddings


# Step 4: Retrieve top relevant chunks
def get_top_chunks(question, chunks, index):
    q_embedding = embed_model.encode([question])
    D, I = index.search(np.array(q_embedding), k=5)
    return [chunks[i] for i in I[0]]


# Step 5: Send to Groq LLM
def query_llm(context, question):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# Step 6: Combine all into one Gradio app function
def rag_chatbot(files, question):
    if not files:
        return "Please upload at least one PDF file."

    # Extract, chunk, embed, retrieve
    text = extract_text_from_pdfs(files)
    if not text.strip():
        return "No readable text found in the PDFs."

    chunks = split_text(text)
    index, _ = embed_chunks(chunks)
    top_chunks = get_top_chunks(question, chunks, index)

    context = "\n".join(top_chunks)
    answer = query_llm(context, question)
    return answer


# Step 7: Gradio UI
iface = gr.Interface(
    fn=rag_chatbot,
    inputs=[
        gr.File(file_types=[".pdf"], file_count="multiple", label="Upload PDF files"),
        gr.Textbox(label="Ask your question")
    ],
    outputs="text",
    title="📚 RAG Chatbot with PDF Upload",
    description="Upload PDFs and ask questions. Uses sentence-transformers, FAISS, and Groq's LLM (llama3)."
)

if __name__ == "__main__":
    iface.launch()
