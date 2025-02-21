# RAG

## Overview
This Streamlit application enables users to upload PDF documents and perform question-answering using advanced retrieval and language model techniques.

## Key Components

### 1. Document Processing
- **Function**: `process_document()`
- **Purpose**: Converts uploaded PDF files into text chunks
- **Features**:
  - Uses PyMuPDFLoader for PDF parsing
  - Applies RecursiveCharacterTextSplitter for text segmentation
  - Handles temporary file management
  - Chunk size: 400 characters
  - Chunk overlap: 100 characters

### 2. Vector Storage
- **Database**: ChromaDB
- **Embedding Model**: Snowflake Arctic Embed2
- **Features**:
  - Persistent vector storage
  - Cosine similarity search
  - Metadata tracking for document chunks

### 3. Retrieval Process
- **Retrieval Steps**:
  1. Query vector collection
  2. Cross-encoder re-ranking
  3. Select top 3 most relevant document chunks

### 4. Language Model Interaction
- **Model**: LLaMA 3.2
- **Prompt Engineering**:
  - System prompt guides structured, context-based responses
  - Ensures answer generation based solely on provided context

## Key Functions

### `get_vector_collection()`
- Initializes ChromaDB collection
- Configures Ollama embedding function
- Sets up cosine similarity space

### `add_to_vector_collection()`
- Adds document chunks to vector store
- Generates unique IDs based on filename
- Stores document metadata

### `query_collection()`
- Performs semantic search in vector collection
- Retrieves most relevant documents

### `re_rank_cross_encoders()`
- Uses MS MARCO MiniLM cross-encoder
- Re-ranks retrieved documents
- Selects top 3 most relevant chunks

### `call_llm()`
- Streams responses from LLaMA 3.2
- Applies system prompt for structured answering

## User Interface
- Sidebar for PDF upload
- Main area for question input
- Streaming response display
- Expandable sections for retrieved documents

## Dependencies
- Streamlit
- ChromaDB
- Ollama
- PyMuPDFLoader
- Sentence Transformers
- LangChain

## Setup Requirements
1. Install dependencies
2. Ensure Ollama is running locally
3. Download required embedding and language models

## Potential Improvements
- Add error handling
- Implement multi-document support
- Create model configuration options
- Add citation/source tracking

## Security Considerations
- Temporary file management
- Handling file upload permissions
- Secure embedding and model usage

## Performance Optimization
- Chunk size and overlap tuning
- Model selection
- Caching mechanisms

## Usage Example
1. Upload a PDF document
2. Click "Process"
3. Ask questions about the document
4. Receive AI-generated answers based on document context
