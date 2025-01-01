import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Agents
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tools_calls=True,
    markdown=True
)

financial_agent = Agent(
    name="Finance AI Agent",
    role="Provide financial data and analysis",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[YFinanceTools(
        stock_price=True,
        analyst_recommendations=True,
        stock_fundamentals=True,
        company_news=True
    )],
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

# Streamlit App
st.set_page_config(page_title="Multi-Agent AI Assistant", page_icon="🧪", layout="wide")

# Header
st.title("🧪 Multi-Agent AI Assistant")
st.markdown(
    """<style> 
        .reportview-container .main .block-container {
            padding: 1.5rem 2rem;
            background-color: #f7f8fa;
        }
    </style>""",
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.header("AI Agents")
st.sidebar.markdown("Select an agent to assist you:")
agent_selection = st.sidebar.radio(
    "Choose Agent",
    ["Web Search Agent", "Finance AI Agent", "Multi-Agent AI"]
)

st.sidebar.markdown("---")
st.sidebar.info("This app is powered by Groq models and phi tools.")

# Input Section
st.markdown("### Enter your query below:")
query = st.text_area("What would you like to know?", "What is the status of India Vs Australia Test Match?")

# Process Query
if st.button("Get Response"):
    st.markdown("### AI Response:")
    placeholder = st.empty()
    placeholder.markdown("Waiting for response...")

    # Get response based on agent selection
    response = None
    if agent_selection == "Web Search Agent":
        response = web_search_agent.print_response(query, stream=False)  # Ensure no stream
    elif agent_selection == "Finance AI Agent":
        response = financial_agent.print_response(query, stream=False)  # Ensure no stream
    else:
        response = multi_ai_agent.print_response(query, stream=False)  # Ensure no stream

    # Check if response is available and display
    if response:
        placeholder.markdown(response)
    else:
        placeholder.markdown("No response available. Please try again.")

# Footer
st.markdown("---")
st.markdown(
    """<footer style='text-align: center;'>
    Built with ❤️ using Streamlit and Groq Models
    </footer>""",
    unsafe_allow_html=True
)
