import os
import gradio as gr
import requests
from langchain_openai import ChatOpenAI
from crewai_tools.tools import PDFSearchTool
from langchain_tavily import TavilySearch
from crewai.tools import tool
from crewai import Crew, Task, Agent
from dotenv import load_dotenv
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Download PDF if not already present
def download_pdf():
    pdf_url = "https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf"
    pdf_path = 'attention_is_all_you_need.pdf'
    if not os.path.exists(pdf_path):
        try:
            response = requests.get(pdf_url)
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            print(f"PDF downloaded successfully to {pdf_path}")
        except Exception as e:
            print(f"Error downloading PDF: {e}")

# LLM configuration - Using OpenAI GPT-3.5-turbo
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=OPENAI_API_KEY,
    temperature=0.2,
    max_tokens=1024,
)

# PDF Search Tool
pdf_search_tool = PDFSearchTool(
    pdf="attention_is_all_you_need.pdf",
    config=dict(
        llm=dict(
            provider="openai",
            config=dict(
                model="gpt-3.5-turbo",
                api_key=OPENAI_API_KEY,
            ),
        ),
        embedder=dict(
            provider="huggingface",
            config=dict(
                model="BAAI/bge-small-en-v1.5",
            ),
        ),
    )
)

# Web Search Tool - Wrapped as CrewAI BaseTool
class TavilySearchInput(BaseModel):
    """Input schema for TavilySearchTool."""
    query: str = Field(..., description="The search query to look up on the web.")

class TavilySearchTool(BaseTool):
    name: str = "Tavily Web Search"
    description: str = "Search the web for current information using Tavily."
    args_schema: Type[BaseModel] = TavilySearchInput

    def _run(self, query: str) -> str:
        try:
            tavily = TavilySearch(max_results=3, api_key=TAVILY_API_KEY)
            results = tavily.invoke(query)
            print(f"Web search results for '{query}': {results}")
            return str(results)
        except Exception as e:
            print(f"Error in web search: {e}")
            return f"Web search error: {str(e)}"

web_search_tool = TavilySearchTool()

# Router Tool
@tool
def router_tool(question: str) -> str:
    """Router Function to decide between vectorstore and web search."""
    keywords = ['self-attention', 'transformer', 'attention', 'language model', 'attention is all you need']
    if any(keyword in question.lower() for keyword in keywords):
        return "vectorstore"
    else:
        return "web_search"

# Agent definition
def create_agents():
    Router_Agent = Agent(
        role='Router',
        goal="Route user question to a vectorstore or web search based on the content.",
        backstory=(
            "You are an expert at routing a user question to a vectorstore or web search. "
            "Use the vectorstore for questions related to the 'Attention is All You Need' paper, "
            "Transformers, self-attention, or language models. "
            "Use web_search for all other topics."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[router_tool],
    )
    
    Retriever_Agent = Agent(
        role="Retriever",
        goal="Retrieve relevant information using the appropriate tool based on routing decision.",
        backstory=(
            "You are an expert retrieval specialist. "
            "You use PDFSearchTool for Transformer/Attention related questions and Tavily Web Search for general queries. "
            "Always provide accurate, well-sourced information."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[pdf_search_tool, web_search_tool],
    )
    
    Grader_Agent = Agent(
        role="Answer Grader",
        goal="Evaluate the relevance and quality of retrieved information.",
        backstory=(
            "You are a quality assurance expert. "
            "You assess whether retrieved documents are relevant to the user's question. "
            "Grade based on keyword presence and semantic relevance."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )
    
    Hallucination_Grader = Agent(
        role="Hallucination Grader",
        goal="Verify that answers are factually grounded and not fabricated.",
        backstory=(
            "You are a fact-checking expert. "
            "You verify that answers are based on retrieved information and don't contain made-up facts. "
            "You ensure factual accuracy above all."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )
    
    Final_Answer_Agent = Agent(
        role="Final Answer Agent",
        goal="Synthesize all information into a clear, comprehensive final answer.",
        backstory=(
            "You are an expert synthesizer. "
            "You combine information from retrieval, grading, and verification steps "
            "to create clear, concise, and informative answers to user questions."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[web_search_tool],
    )
    
    return Router_Agent, Retriever_Agent, Grader_Agent, Hallucination_Grader, Final_Answer_Agent

# Task definitions
def create_tasks(agents, tools):
    Router_Agent, Retriever_Agent, Grader_Agent, Hallucination_Grader, Final_Answer_Agent = agents

    router_task = Task(
        description=(
            "Analyze the user's question: {question}\n"
            "Determine if it relates to Transformers, self-attention, language models, or the 'Attention is All You Need' paper.\n"
            "Return ONLY 'vectorstore' if yes, otherwise return ONLY 'web_search'. No explanation needed."
        ),
        expected_output="Either 'vectorstore' or 'web_search'",
        agent=Router_Agent,
        tools=[router_tool],
    )

    retriever_task = Task(
        description=(
            "Retrieve relevant information for the user's question: {question}\n"
            "If the router indicated 'vectorstore', use PDFSearchTool to search the Attention paper.\n"
            "If the router indicated 'web_search', use Tavily Web Search to find current information.\n"
            "Provide comprehensive, relevant results."
        ),
        expected_output="Detailed information retrieved from the appropriate source",
        agent=Retriever_Agent,
        context=[router_task],
    )

    grader_task = Task(
        description=(
            "Evaluate the relevance of retrieved information for the question: {question}\n"
            "Check if the retrieved content directly addresses the question.\n"
            "Return 'RELEVANT' if the information is useful and addresses the question, or 'NOT RELEVANT' if it doesn't."
        ),
        expected_output="Either 'RELEVANT' or 'NOT RELEVANT'",
        agent=Grader_Agent,
        context=[retriever_task],
    )

    hallucination_task = Task(
        description=(
            "Verify that the retrieved information is factually grounded for question: {question}\n"
            "Check if information is from the retrieved sources and not fabricated.\n"
            "Return 'FACTUAL' if the information is grounded in sources, or 'HALLUCINATED' if it contains made-up information."
        ),
        expected_output="Either 'FACTUAL' or 'HALLUCINATED'",
        agent=Hallucination_Grader,
        context=[retriever_task],
    )

    answer_task = Task(
        description=(
            "Generate a comprehensive final answer to: {question}\n"
            "Use information from the retrieval step.\n"
            "Only include information that was marked as RELEVANT and FACTUAL in previous steps.\n"
            "If information is incomplete, use web search to fill gaps.\n"
            "Provide a clear, well-structured answer."
        ),
        expected_output="A clear, accurate, and comprehensive final answer to the user's question",
        agent=Final_Answer_Agent,
    )

    return [router_task, retriever_task, grader_task, hallucination_task, answer_task]

# Gradio interface
def gradio_interface(query):
    if not query or query.strip() == "":
        return "Please enter a valid question."
    try:
        return run_rag_pipeline(query)
    except Exception as e:
        return f"Error processing your question: {str(e)}"

# Create Gradio App
def create_gradio_app():
    iface = gr.Interface(
        fn=gradio_interface,
        inputs=gr.Textbox(
            label="Enter your question",
            placeholder="Ask about AI, Language Models, Transformers, Neural Networks, etc."
        ),
        outputs=gr.Textbox(label="Response", lines=10),
        title="Agentic RAG Demo: AI Knowledge Assistant",
        description=(
            "Ask questions about AI, Language Models, Transformers, and Neural Networks. "
            "The system uses a multi-agent approach with RAG over the 'Attention is All You Need' paper "
            "and web search capability."
        ),
        theme="soft",
        flagging_mode="never"
    )
    return iface

# Main RAG Function
def run_rag_pipeline(question):
    print(f"\n{'='*80}")
    print(f"Processing question: {question}")
    print(f"{'='*80}\n")
    
    # Download PDF if not exists
    download_pdf()
    
    # Create agents
    agents = create_agents()
    
    # Create tasks
    tasks = create_tasks(agents, None)
    
    # Create Crew with simpler configuration
    rag_crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=True,
    )
    
    # Run the pipeline
    try:
        print("Starting crew execution...")
        result = rag_crew.kickoff(inputs={"question": question})
        print(f"\nFinal Result:\n{result}\n")
        return str(result)
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        return f"An error occurred: {str(e)}"

# Launch the Gradio App
if __name__ == "__main__":
    print("Launching Gradio app...")
    app = create_gradio_app()
    app.launch(share=True)