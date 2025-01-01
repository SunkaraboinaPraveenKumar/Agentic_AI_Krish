from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv

load_dotenv()

# web search agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tools_calls=True,
    markdown=True
)

# Financial Agent
financial_agent = Agent(
    name="Finance AI Agent",
    role="Search the web for the information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)
    ],
    instructions=["Use tables to display data"],
    show_tools_calls=True,
    markdown=True
)

multi_ai_agent = Agent(
    model=Groq(id='llama-3.1-70b-versatile'),
    team=[web_search_agent, financial_agent],
    instructions=["Always include sources", "Use tables to display data"],
    show_tools_calls=True,
    markdown=True
)

multi_ai_agent.print_response("What is the status of India Vs Australia Test Match?", stream=True)