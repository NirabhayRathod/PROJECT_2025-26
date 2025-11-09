# RAG App for Legal Consultancy

## Overview
The **RAG App for Legal Consultancy** is an intelligent question-answering application built using **Retrieval-Augmented Generation (RAG)** techniques.  
It allows users to query large legal documents, such as Indian laws, and receive precise, context-based answers.  
The app combines **document embeddings**, **vector search**, and a **large language model (LLM)** to deliver accurate and detailed legal information efficiently.

---

## Features
- **Query large legal documents:** Works with PDFs up to several hundred pages.  
- **Full-document retrieval:** Retrieves context from any part of the document, including beginning, middle, and end.  
- **Detailed answers:** Uses multiple document chunks to provide comprehensive responses.  
- **Chat-style interface:** Simple and interactive UI for asking questions.  
- **Document similarity search:** View which sections of the document contributed to the answer.  

---

## How It Works
1. **Document Loading**  
   - Reads PDF documents using `PyPDFLoader`.

2. **Text Splitting**  
   - Splits document into overlapping chunks (`RecursiveCharacterTextSplitter`) for better context continuity.

3. **Embedding Creation**  
   - Converts each chunk into vector embeddings using `HuggingFaceEmbeddings`.  
   - Stores embeddings in a **FAISS vector database** for similarity search.

4. **Retrieval and Generation**  
   - Fetches the top relevant chunks when a user asks a question.  
   - Passes chunks to LLM (`ChatGroq Gemma2-9b-It`) to generate detailed answers.

5. **Result Display**  
   - Displays the generated answer.  
   - Optionally shows the **document chunks** used to generate the answer.

---

## Technologies Used
- **Python 3.10+**  
- **Streamlit:** Interactive web interface  
- **LangChain:** RAG and LLM orchestration  
- **FAISS:** Vector database for document retrieval  
- **HuggingFace Embeddings:** Converts text chunks into embeddings  
- **ChatGroq (llama-3.1-8b-instant):** Large language model for generating answers  

---
## Author
**Name:** Nirabhay Singh Rathod , Aman Bansal , Ahmad Raja Khan  
**Email:** nirbhay105633016@example.com 