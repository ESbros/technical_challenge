import re
from urllib.parse import urlparse

from typing import List, Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.types import Command
from langgraph.graph import MessagesState, END
from langchain_core.tools import tool

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader

from langchain_core.language_models.chat_models import BaseChatModel
from typing_extensions import TypedDict


""" Tools """
tavily_tool = TavilySearchResults(max_results=5)

llm = ChatOpenAI(model="gpt-4o", temperature=0)


@tool
def scrape_webpages(urls: List[str]) -> str:
    """Use requests and bs4 to scrape the provided web pages for detailed information."""
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return "\n\n".join(
        [
            f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )


@tool
def is_valid_url(url: str) -> str:
    """Use to validate that the input URL is able for processing"""
    if not url:  # Check for empty string
        return False

    # Regex Check (Quick initial filter)
    regex = r"^(?:http(s)?:\/\/)?[\w.-]+(?:\.[\w\.-]+)+[\w\-\._~:/?#[\]@!\$&'\(\)\*\+,;=.]+$"
    if not re.match(regex, url):
        return 'Invalid URL'

    try:
        parsed_url = urlparse(url)

        # Check if scheme contains http/https
        if not parsed_url.scheme:
          return 'Invalid URL'

        # Check if network location is present
        if not parsed_url.netloc:
            return 'Invalid URL'
        
        return 'Valid URL'

    except Exception:
        return 'Invalid URL'
    

def get_next_node(last_message: BaseMessage, goto: str):
    if "Invalid URL" in last_message.content or "invalid URL" in last_message.content:
        return END
    return goto

def get_next_node_based_on_text_existence(last_message: BaseMessage, goto: str):
    if len(last_message.content) <= 1000:
        return END
    return goto

def get_next_node_based_on_super_graph(last_message: BaseMessage, goto: str):
    if "Invalid URL" in last_message.content or "invalid URL" in last_message.content:
        return END
    return goto


""" Helper Functions """
class State(MessagesState):
    next: str


def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> str:
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

    def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
        """An LLM-based router."""
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto})

    return supervisor_node


""" The following solution propose a hierarchical structure composed of 2 teams: Preprocessing and Analysis. """

""" Preprocessing Team """
# Input Agent: Validate URL
input_agent = create_react_agent(
    llm, 
    tools=[is_valid_url],
    prompt=(
        "You are only in charge of validating the URL, do not modify the URL in any way."
        "If you identify that the URL is invalid, please stop by Invalid URL identification."
    )
)

def input_node(state: MessagesState) -> Command[Literal["web_scraper", END]]:
    result = input_agent.invoke(state)
    print('state inside:', state)
    goto = get_next_node(result["messages"][-1], "web_scraper")
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="input")
            ]
        },
        goto=goto,
    )


# Web Scraper Agent: Extract content from the web page using custom tools
web_scraper_agent = create_react_agent(
    llm, 
    tools=[scrape_webpages],
)

def web_scraper_node(state: MessagesState) -> Command[Literal["content_formatting", END]]:
    result = web_scraper_agent.invoke(state)
    goto = get_next_node_based_on_text_existence(result["messages"][-1], "content_formatting")
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="web_scraper")
            ]
        },
        goto=goto,
    )


# Content Formatting Agent: Normalize and structure the extracted content
content_formatting_agent = create_react_agent(
    llm,
    tools=[],
    prompt=(
        "You are an agent tasked to improve the formatting of the content scraped from the a web page."
        "Structure the extracted content into a readable format."
        "Organize the text using headings, bullet points, or summary sections."
        "Normalize and removes unwanted characters or redundant data."
    )
)

def content_formatting_node(state: MessagesState) -> Command[Literal[END]]:
    result = content_formatting_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="content_formatting")
            ]
        },
        goto=END,
    )


""" Analyzer Team """
# Analyzer Agent: Analyze the content and identify biases, missing perspectives, or additional context
analizer_agent = create_react_agent(
    llm,
    tools=[],
    prompt=(
        "Improve the following text by considering"
        "cross-references information to identify biases, missing perspectives, or additional context."
        "Ask for help if there is any information that you need to clarify."
    ),
)

def analyzer_node(state: State) -> Command[Literal["supervisor"]]:
    result = analizer_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="analyzer")
            ]
        },
        goto="supervisor",
    )

# Research Agent: Find additional information on the topic usin Tavily Search
research_agent = create_react_agent(
    llm,
    tools=[tavily_tool],
    prompt=(
        "You are a researcher tasked with finding additional information on the topic."
    ),
)

def research_node(state: State) -> Command[Literal["supervisor"]]:
    result = research_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="researcher")
            ]
        },
        goto="supervisor",
    )

# Summarizer Agent: Generate a structured analytical report based on key findings
summarizer_agent = create_react_agent(
    llm, 
    tools=[],
    prompt=(
        "You are a summarizer agent in charge of generating a structured analyticasl report based on key findings."
    )
)

def summarizer_node(state: State) -> Command[Literal["supervisor"]]:
    result = summarizer_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=result["messages"][-1].content, name="summarizer"
                )
            ]
        },
        goto="supervisor",
    )

# Supervisor Node: Manage the conversation between the workers in the Analysis Team
doc_analyzer_node = make_supervisor_node(
    llm, ["analyzer", "researcher", "summarizer"]
)


def main_workflow(url: str):
    # Preprocessing Team: Input, Web Scraper, Content Formatting
    workflow_preprocessing = StateGraph(MessagesState)
    workflow_preprocessing.add_node("input", input_node)
    workflow_preprocessing.add_node("web_scraper", web_scraper_node)
    workflow_preprocessing.add_node("content_formatting", content_formatting_node)

    workflow_preprocessing.add_edge(START, "input")
    preprocessing_graph = workflow_preprocessing.compile()

    # Analysis Team: Analyzer, Researcher, Summarizer
    analyzer_builder = StateGraph(State)
    analyzer_builder.add_node("supervisor", doc_analyzer_node)
    analyzer_builder.add_node("analyzer", analyzer_node)
    analyzer_builder.add_node("researcher", research_node)
    analyzer_builder.add_node("summarizer", summarizer_node)

    analyzer_builder.add_edge(START, "supervisor")
    analyzer_graph = analyzer_builder.compile()

    # Main Graph: Preprocessing Team -> Analysis Team
    def call_preprocessing_team(state: State) -> Command[Literal["analyzer_team", END]]:
        response = preprocessing_graph.invoke({"messages": state["messages"][-1]})
        goto = get_next_node_based_on_super_graph(response["messages"][-1], "analyzer_team")
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=response["messages"][-1].content, name="preprocessing_team"
                    )
                ]
            },
            goto=goto,
        )


    def call_analyzing_team(state: State) -> Command[Literal[END]]:
        response = analyzer_graph.invoke({"messages": state["messages"][-1]})
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=response["messages"][-1].content, name="analyzing_team"
                    )
                ]
            },
            goto=END,
        )

    # Main Graph Definition
    super_builder = StateGraph(State)
    super_builder.add_node("preprocessing_team", call_preprocessing_team)
    super_builder.add_node("analyzer_team", call_analyzing_team)

    super_builder.add_edge(START, "preprocessing_team")
    super_graph = super_builder.compile()


    for response in super_graph.stream(
        {
            "messages": [
                ("user", f"Scrap the following URL: {url}",)
            ],
        },
        {"recursion_limit": 150},
    ):
        print(response)
        print("---")

    return response