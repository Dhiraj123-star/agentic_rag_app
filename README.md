
# 🤖 Agentic RAG Demo App

Build an **Agentic Retrieval-Augmented Generation (RAG)** pipeline using **CrewAI**, **LangChain**, and **Gradio** — inspired by the blog *“Hands-on demo with building Agentic RAG pipeline”* by Ajay Arunachalam.

---

## 🚀 Overview

This project demonstrates how to build an **LLM-powered Agentic RAG pipeline** where an **agent** decides which data source to query before generating an answer — improving accuracy and contextual relevance.

---

## 🧩 Features

- 🔍 Multi-source document retrieval  
- 🧠 Agentic decision-making using LLMs  
- 💬 Context-aware response generation  
- 🌐 Simple and interactive Gradio UI  
- 🧰 Uses LangChain, CrewAI, and Tavily for retrieval & reasoning  

---

## ⚙️ Tech Stack

- **CrewAI**  
- **CrewAI Tools**  
- **LangChain Community**  
- **LangChain Groq**  
- **LangChain HuggingFace**  
- **Sentence Transformers**  
- **Gradio**  
- **Tavily Python**

---

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd <repo-directory>
````

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🧰 Requirements

Your `requirements.txt` should look like this:

```
crewai
crewai_tools
langchain_community
sentence-transformers
langchain-groq
langchain_huggingface
gradio
tavily-python
```

---

## 🚦 How to Run

1. Add your API keys in a `.env` file (for OpenAI, Tavily, etc.)
2. Run the application:

   ```bash
   python agentic_rag_app_demo.py
   ```
3. Open the **Gradio UI** link shown in your terminal.

---

## 🧠 How It Works

1. User asks a question via the Gradio interface.
2. The **CrewAI agent** decides which data source or retriever to use.
3. Relevant context is fetched and passed to the LLM.
4. The **LLM** generates a detailed, contextually accurate response.

---



