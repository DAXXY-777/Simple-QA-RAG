import os
import tempfile
import re
import chromadb
import ollama
import streamlit as st
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile

# System prompt for the LLM
system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

Context will be passed as "Context:"
User question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""

# Document processing
def process_document(uploaded_file: UploadedFile) -> list[Document]:
    """Processes an uploaded PDF file into text chunks."""
    with tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    loader = PyMuPDFLoader(temp_file_path)
    docs = loader.load()

    try:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
    except Exception as e:
        st.error(f"Error deleting temp file: {e}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return text_splitter.split_documents(docs)

# ChromaDB vector collection
def get_vector_collection() -> chromadb.Collection:
    """Gets or creates a ChromaDB collection for vector storage."""
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="snowflake-arctic-embed2:latest",
    )

    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )

# Add documents to vector store
def add_to_vector_collection(all_splits: list[Document], file_name: str):
    """Adds document splits to a vector collection."""
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    st.success("Data added to the vector store!")

# Query ChromaDB
def query_collection(prompt: str, n_results: int = 10):
    """Queries the vector collection with a given prompt."""
    collection = get_vector_collection()
    return collection.query(query_texts=[prompt], n_results=n_results)
# LLM response generation
def call_llm(context: str, prompt: str):
    """Calls the language model with context and prompt."""
    response = ollama.chat(
        model="phi4",
        stream=False,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}\nQuestion: {prompt}"},
        ],
    )
    return response["message"]["content"]
    
# CrossEncoder re-ranking
def rerank_documents(query: str, documents: list[str], top_k: int = 3):
    """Re-ranks documents using a CrossEncoder model."""
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    cleaned_docs = [str(doc).strip() for doc in documents]  # Force string conversion
    
    # Create query-doc pairs
    sentence_pairs = [[query, doc] for doc in cleaned_docs]
    
    # Get scores (batch processing for efficiency)
    scores = cross_encoder.predict(sentence_pairs, batch_size=32)
    
    # Rank documents
    ranked_results = sorted(
        zip(cleaned_docs, scores), 
        key=lambda x: x[1], 
        reverse=True
    )[:top_k]
    
    return [doc for doc, _ in ranked_results], [score for _, score in ranked_results]

# Streamlit UI
if __name__ == "__main__":
    st.set_page_config(page_title="QA BOT - Financial Data")
    
    # Custom CSS Styling
    st.markdown(
    """
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #1e1e2f;
            color: #f5f5f5;
        }
        .stApp {
            background: linear-gradient(to bottom, #1e1e2f, #2c3e50);
        }
        .title {
            font-size: 2.5rem;
            font-weight: bold;
            color: #f1c40f;
            text-align: center;
            margin-bottom: 1rem;
        }
        .subtitle {
            font-size: 1rem;
            color: #bdc3c7;
            text-align: center;
            margin-bottom: 2rem;
        }
        .upload-box {
            border: 2px dashed #3498db;
            background-color: #34495e;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: #ecf0f1;
            font-size: 1.1rem;
        }
        .question-box {
            background-color: #2c3e50;
            border: 2px solid #1abc9c;
            padding: 10px;
            border-radius: 10px;
            color: #ecf0f1;
        }
        .answer-box {
            background-color: #34495e;
            padding: 15px;
            border-radius: 10px;
            margin-top: 10px;
            color: #f5f5f5;
            font-family: 'Courier New', monospace;
        }
        .footer {
            margin-top: 4rem;
            font-size: 0.9rem;
            color: #7f8c8d;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
    # Sidebar for document upload
    with st.sidebar:
        st.header("Document Upload")
        uploaded_file = st.file_uploader(
            "Upload PDF files", type=["pdf"], accept_multiple_files=False
        )
        if st.button("Process Document") and uploaded_file:
            normalized_name = uploaded_file.name.translate(
                str.maketrans({"-": "_", ".": "_", " ": "_"})
            )
            all_splits = process_document(uploaded_file)
            add_to_vector_collection(all_splits, normalized_name)

    # Main interface
    st.title("RAG-QA-FINANCIAL-DATA")
    st.markdown("---")
    
    user_query = st.text_area("Ask a question about the document:")
    
    if st.button("Get Answer") and user_query:
        with st.spinner("Searching documents..."):
            results = query_collection(user_query)
            
        if results and results.get("documents") and len(results["documents"][0]) > 0:
            context_documents = results["documents"][0]
            
            with st.spinner("Re-ranking results..."):
                reranked_docs, scores = rerank_documents(user_query, context_documents)
                text1 = "\n".join(reranked_docs)
                text2 = "\n".join(context_documents)
                text = str(text1) + " " + str(text2)
                relevant_text = re.sub(r"\s+", " ", text).strip()
                # print(relevant_text)
            
            with st.spinner("Generating answer..."):
                full_answer = call_llm(relevant_text, user_query)
                # print(relevant_text)
                answer_box = st.empty()
                answer_box.markdown(f"""
                <div class="answer-box">
                <h4>Answer:</h4>
                {full_answer}
                </div>
                """, unsafe_allow_html=True)

            # Display retrieval information
            st.markdown("### Retrieval Details")
            
            with st.expander("View Retrieved Documents"):
                for idx, doc in enumerate(context_documents):
                    st.markdown(f"**Document {idx+1}:**")
                    st.text(doc[:500] + "..." if len(doc) > 500 else doc)
                    st.markdown("---")
            
            with st.expander("View Most Relevant Sections"):
                for idx, doc in enumerate(reranked_docs):
                    st.markdown(f"**Relevant Section {idx+1} (Score: {scores[idx]:.2f}):**")
                    st.text(doc[:500] + "..." if len(doc) > 500 else doc)
                    st.markdown("---")
        else:
            st.warning("No relevant documents found in the knowledge base.")

    st.markdown("---")
