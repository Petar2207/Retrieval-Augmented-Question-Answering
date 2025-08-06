# **Retrieval-Augmented Question Answering with LangChain & Google Gemini**

This project demonstrates a **Retrieval-Augmented Generation (RAG)** approach for building an advanced question-answering system. By leveraging LangChain and Google’s Gemini (Generative AI) model, the system retrieves relevant information from a knowledge base (PDF documents), processes it using embeddings, and generates highly accurate, context-aware answers. This method enhances the model’s ability to provide precise, insightful responses based on real-time document retrieval.

🎯 **Project Goal**

**Primary Objective**: To develop a sophisticated question-answering application using LangChain, LangGraph, and Google Gemini's generative AI capabilities, built on the **RAG** paradigm. The system is designed to intelligently retrieve relevant information from a document, combine it with generative AI, and provide answers to user queries based on the content of the document.

✅ **Workflow Summary**

1. **Retrieval-Augmented Generation (RAG) Setup**
    - **Retrieval**: The system first retrieves relevant chunks of text from a document stored in an in-memory vector store. It does this by performing a similarity search based on the user's question and the document’s content.
    - **Generation**: Once relevant document chunks are retrieved, the system uses these as context to prompt the Google Gemini model, generating an answer that is both contextually accurate and informative.

2. **Document Loading & Preprocessing**
    - PDF documents are loaded using LangChain’s `PyPDFLoader` and split into smaller, manageable chunks using `RecursiveCharacterTextSplitter`. This ensures that each chunk of text is meaningful and provides focused context during retrieval.
    - The chunked text is then embedded using `GoogleGenerativeAIEmbeddings`, transforming it into a vector representation for efficient retrieval.

3. **Embedding & Vector Search**
    - With `GoogleGenerativeAIEmbeddings`, document chunks are transformed into vector space, allowing the system to search for the most contextually relevant information based on the user’s query.
    - The vector store (`InMemoryVectorStore`) indexes these embeddings to enable fast similarity search and context retrieval.

4. **Google Gemini Model & Custom Prompting**
    - The `gemini-2.5-flash` model is initialized to generate answers from the retrieved document context. A custom `ChatPromptTemplate` defines the behavior of the model to ensure that it generates accurate, relevant responses while following predefined conversational guidelines.
    - The model responds intelligently, providing answers based on the context retrieved by the RAG approach, and falls back on a default response if no relevant context is found.

5. **State-Driven Question-Answering Flow**
    - A state machine using `LangChain` and `LangGraph` orchestrates the retrieval and generation process. The state manages the flow of user questions, retrieves the relevant context from the vector store, and generates an appropriate response.
    - The process is repeated for each user query, ensuring a smooth and interactive experience.

6. **Interactive CLI for Real-Time Question-Answering**
    - The application runs in an interactive loop, allowing users to ask questions continuously. It retrieves the most relevant information from the document and generates responses dynamically, ensuring that each answer is contextually relevant and informative.

7. **Fallback & Termination**
    - If the model cannot generate an answer, it returns a fallback message, asking if the user needs further assistance. Additionally, if the response is "bye," the session gracefully ends.

8. **Model & Environment Configuration**
    - The Google Gemini API key is required for model initialization. The application will prompt users to enter their API key at runtime.
    - This system is designed to run in Jupyter/Colab environments, making it easy to upload documents and interact with the question-answering model.

🛠️ **Tech Stack**

- **Languages**: Python
- **Libraries**: LangChain, LangGraph, Google Gemini, LangChain-Community, LangChain-Text-Splitters, PyPDF, Google API
- **Frameworks**: LangChain for model interaction and vector store, LangGraph for state management

📜 **License**

This project is licensed under the MIT License — free to use and modify.

👤 **Author**

Petar Rajic

### To install the required libraries, use:

```bash
pip install --quiet --upgrade langchain-text-splitters langchain-community langgraph
pip install -qU "langchain[google-genai]"
pip install -qU langchain-core
pip install pypdf
