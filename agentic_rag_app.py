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
# tool configuration
def setup_tools():
    # RAG Tool
    rag_tool = PDFSearchTool(
        pdf="attention_is_all_you_need.pdf",
        config=dict(
            llm=dict(
                provider="groq",
                config=dict(
                    model="llama3-8b-8192",
            ),
        ),
        embedder=dict(
            provider="huggingface",
            config=dict(
                model="BAAI/bge-small-en-v2",
        ),
        ),

        )
    )
    # Web Search Tool
    web_search_tool = TavilySearchResults(k=3)

    return [rag_tool, web_search_tool]

# router tool
@tools.tool
def router_tool(question):
    """Router Function"""
    keywords=['self-attention',
            'transformer',
            'attention',
            'language model',
    ] 
    if any(keyword in question.lower() for keyword in keywords):
        return "vectotstore"
    else:
        return "web_search"

# agent definition

def create_agents():
    Router_Agent = Agent(
        role='Router',
        goal= "Route user question to a vectorstore or web search based on the content.",
        backstory=(
            "You are an expert at routing a user question to a vectorstore or web search"
            "use the vectorstore for questions related to Retrieval-Augment Generation"
            "Be flexible in interpreting keywords related to these topics."
        ),
        verbose=True,
        allow_delegation =False,
        llm=llm,

    )
    Retriever_Agent = Agent(
        role="Retriever",
        goal="Use retrieved information to answer the question",
        backstory=(
            "You are an assistant for question-answering tasks."
            "Provide clear,concise answers using retrieved context."
        ),
        verbose =True,
        allow_delegation=False,
        llm=llm,
    
    )
    Grader_Agent = Agent(
        role="Answer Grader",
        goal="Filter out irrelevant retrievals",
        backstory=(
            "You are grader assessing the relevance of retrived documents."
            "Evaluate if the document contains keywords related to the user question"
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,

    )
    Hallunication_Grader = Agent(
        role ="Hallunication Grader",
        goal="Verify answer factuality",
        backstory=(
            "You are responsible for ensuring the answer is grounded in facts"
            "and directly address the user's question"
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )
    Final_Answer_Agent = Agent(
        role = "Final Answer Agent",
        goal="Provide a comprehensive and accurate response",
        backstory=(
            "You synthesis information from various source to create"
            "a clear,concise and informative answer to user's question."
        ),
        verbose = True,
        allow_delegation=False,
        llm=llm,

    )
    return Router_Agent, Retriever_Agent,Grader_Agent,Hallunication_Grader,Final_Answer_Agent

# task definitions
def create_tasks(agents,tools):
    rag_tool,web_search_tool=tools
    Router_Agent, Retriever_Agent,Grader_Agent,Hallunication_Grader,Final_Answer_Agent = agents

    router_task= Task(
        description=(
            "Analyse the keywords in the question {question}. "
            "Decide whether it requires a vectorstore search or web search."
        ),
        expected_output= "Return 'websearch' or 'vectorstore'",
        agent = Router_Agent,
        tools=[router_tool],
    )

    retriever_task= Task(
        description=(
            "Retrieve information for the question {question}."
            "Using either web search or vectorstore based on router task."
        ),
        expected_output="Provide retrieved information",
        agent = Retriever_Agent,
        context=[router_task],
        tools = [rag_tool,web_search_tool],
    )

    grader_task = Task(
        description = "Evaluate the relevance of retrieved content for the question {question}.",
        expected_output = "Return 'yes' or 'no' for relevance",
        agent = Grader_Agent ,
        context = [retriever_task],

    )

    hallunication_task = Task(
        description = "Verify if the retrieved answer is factually grounded.",
        expected_output=" Return 'yes' or 'no' for factuality",
        agent = Hallunication_Grader,
        context = [grader_task],

    )
    answer_task= Task(
        description = (
            "Generate a final answer based on retrieved and verified information."
            "Perform additional search if needed"
        ),
        expected_output = "Provide a clear, concise answer",
        agent = Final_Answer_Agent,
        context=[hallunication_task],
        tools = [web_search_tool],

    )
    return [router_task, retriever_task,grader_task,hallunication_task, answer_task]

# create agents
agents = create_agents ()

# create tasks
tasks = create_tasks(agents,tools)

# Create Crew
rag_crew = Crew(
    agents = agents,
    tasks= tasks,
    verbose=True,
)

# gradio interface

def gradio_interface(query):
    if not query:
        return "Please enter a question."
    return run_rag_pipeline(query)

# create Gradio App
def create_gradio_app():
    iface = gr.Interface(
        fn=gradio_interface,
        inputs = gr.Textbox(label="Enter your question about AI,Language Models, or Self-Attention"),
        outputs = gr.Textbox(label="Response"),
        title = "Agentic RAG Demo: AI Knowledge Assistant",
        description = "Ask questions about AI, Language Models, Transformers, and Self-Attention mechanisms. The System uses a multi-agent approach to retrieve and verify information.",
        theme="soft",
        allow_flagging="never"
    )
    return iface

# Main RAG Function
def run_rag_pipeline(question):
    # Download PDF if not exists
    download_pdf()

    # Setup tools
    rag_tool, web_search_tool = setup_tools()
    tools = (rag_tool, web_search_tool)

    # Create agents
    agents = create_agents()

    # Create tasks
    tasks = create_tasks(agents, tools)

    # Create Crew
    rag_crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=True,
    )

    # Run the pipeline
    try:
        result = rag_crew.kickoff(inputs={"question": question})
        return result
    except Exception as e:
        return f"An error occurred: {str(e)}"