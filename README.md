
# 🤖 Agentic RAG Assistant

Build an **Agentic Retrieval-Augmented Generation (RAG)** pipeline using **CrewAI**, **LangChain**, and **Gradio** — an intelligent system that routes queries to the best information sources and generates comprehensive answers.

---

## 🚀 Overview

This project demonstrates a sophisticated **LLM-powered Agentic RAG pipeline** where intelligent agents collaborate to:

1. **Route** incoming queries to the most relevant data source (PDF or Web)
2. **Retrieve** information from the selected source
3. **Synthesize** comprehensive, well-structured answers

The system improves accuracy and contextual relevance by using intelligent decision-making rather than searching all sources blindly.

---

## 🧩 Features

- 🎯 **Intelligent Query Routing** - Automatically decides between PDF and web search based on query content
- 📄 **PDF Document Search** - Semantic search over the "Attention is All You Need" paper with embeddings
- 🌐 **Real-time Web Search** - Integrates Tavily API for current information retrieval
- 🧠 **Multi-Agent Orchestration** - Three specialized agents work together (Router, Retriever, Answer)
- 💬 **Context-Aware Response Generation** - Synthesizes comprehensive, well-structured answers
- 🔐 **Secure API Management** - Environment variables for secure credential handling
- 🎨 **Interactive Gradio UI** - User-friendly web interface for easy interaction
- ⚡ **Fast & Efficient** - Powered by OpenAI GPT-3.5-turbo for quick responses
- 📊 **Intelligent Source Selection** - Routes Transformer/Attention queries to PDFs and general queries to web
- 🔍 **Multi-Source Retrieval** - Seamlessly pulls from both local documents and live web data

---

## 🏗️ Architecture

### Three-Agent System

**Router Agent**
- Analyzes incoming questions
- Determines optimal data source (PDF or Web)
- Routes queries intelligently based on content

**Retriever Agent**
- Executes targeted searches
- Uses PDF semantic search for academic papers
- Performs real-time web searches for current information
- Returns detailed, relevant results

**Answer Agent**
- Synthesizes retrieved information
- Structures responses clearly with explanations
- Cites sources and provides comprehensive coverage
- Performs supplementary searches if needed

---

## ⚙️ Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM Framework** | LangChain + OpenAI |
| **Agent Orchestration** | CrewAI |
| **Web Search** | Tavily API |
| **PDF Processing** | CrewAI Tools |
| **Embeddings** | HuggingFace (BAAI/bge-small-en-v1.5) |
| **UI Framework** | Gradio |
| **LLM Model** | OpenAI GPT-3.5-turbo |
| **Language** | Python 3.9+ |

---

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- OpenAI API key
- Tavily API key

### Setup Steps
1. Clone the repository
2. Create a virtual environment
3. Install dependencies from requirements
4. Create `.env` file with API keys
5. Run the application

---

## 🚀 Usage

### Interactive Queries
- Ask about AI and Machine Learning topics
- Query about Transformers and attention mechanisms
- Request current information from the web
- Get answers from the "Attention is All You Need" paper

### Example Query Types
- "What is a Recurrent Neural Network?"
- "Explain self-attention mechanisms"
- "What are Transformers in deep learning?"
- "How do attention mechanisms work?"
- "What is the latest in AI research?"

---

## 📋 How It Works

**Query Processing Flow:**
1. User submits a question via Gradio UI
2. Router Agent analyzes the query
3. Router decides: PDF search or Web search?
4. Retriever Agent executes the search
5. Answer Agent synthesizes the results
6. Comprehensive answer displayed to user

**Intelligent Routing:**
- PDF queries: Self-attention, Transformers, Attention mechanisms
- Web queries: Current events, news, general topics

---

## 🔧 Configuration Options

- **LLM Model Selection** - Switch between gpt-3.5-turbo and gpt-4
- **Temperature Control** - Adjust response creativity (0.0 = deterministic)
- **Token Limits** - Configure max response length
- **Search Results** - Customize number of web search results
- **Embedder Selection** - Change embedding model for PDF search

---

## 📁 Project Structure

- Main application file
- Environment configuration
- Dependencies management
- Auto-downloaded research papers
- Project documentation

---

## 🎯 Key Advantages

✅ Intelligent source selection for accurate retrieval  
✅ Semantic search over academic papers  
✅ Real-time web information access  
✅ Well-structured, comprehensive answers  
✅ Cost-effective with GPT-3.5-turbo  
✅ Simple, intuitive user interface  
✅ Extensible architecture for additional sources  
✅ Secure credential management  
✅ Fast response times  
✅ Multi-agent collaboration for better results  

---

## 🐛 Troubleshooting

### Common Issues
- API key validation and quota checks
- PDF download verification
- Internet connectivity
- Response formatting

### Support
- Verify API keys are valid
- Check internet connection
- Ensure write permissions
- Review error messages

---

## 📚 References

- CrewAI Documentation
- LangChain Documentation
- OpenAI API Reference
- Tavily Search API
- Gradio UI Framework
- Attention is All You Need Paper

---

## 📄 License

Open source project available under MIT License

---

## 🤝 Contributing

Contributions welcome - open issues or submit pull requests

---

## 👤 Author

Built as a demonstration of modern Agentic RAG systems combining multiple data sources and intelligent agent orchestration.
```