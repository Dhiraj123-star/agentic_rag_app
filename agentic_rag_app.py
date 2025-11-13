import os
import gradio as gr
import requests
from langchain_openai import ChatOpenAI
from crewai_tools import PDFSearchTool
from langchain_community.tools.tavily_search import TavilySearchResults
from crewai_tools import tools
from crewai import Crew, Task, Agent
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY=os.getenv("GROQ_API_KEY")
TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")

# download PDF if not already present
def download_pdf():
    pdf_url="https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf"
    if not os.path.exists('attention_is_all_you_need.pdf'):
        response = requests.get(pdf_url)
        with open('attention_is_all_you_need.pdf', 'wb') as f:
            f.write(response.content)

# LLM configuration
llm= ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=GROQ_API_KEY,
    model_name ="llama3-8b-8192",
    temperature=0.1,
    max_tokens=1000,

)
print("code works!!!!")