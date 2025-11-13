import os
import gradio as gr
import requests
from langchain_openai import ChatOpenAI
from crewai_tools import PDFSearchTool
from langchain_community.tools.tavily_search import TavilySearchResults
from crewai_tools import tools
from crewai import Crew, Task, Agent

