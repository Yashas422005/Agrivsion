from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
import os 
from langchain.agents import create_agent
from dotenv import load_dotenv
load_dotenv()
tavily_api_key = os.getenv("TAVILY_API_KEY")

llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

search_engine = TavilySearchResults(api_key=tavily_api_key, num_results=3)
tools = [search_engine]
agent = create_agent(model = llm, tools=tools)


from main import response

chat_history = []
print("Agent:Farmer Assistant Here!\n Type 'exit' to stop.\n")

chat_history.append(response['messages'][0].content)
chat_history.append(response['messages'][1].content)
print(f"Farmer: \n{chat_history[0]}\n")
print(f"Agent: \n{chat_history[1]}\n")
while True:
    user_input = input("Farmer: ")
    if user_input.lower() in ["exit", "quit", "stop","end"]:
        print("Conversation ended.")
        break

    chat_history.append(HumanMessage(content=user_input))

    response = agent.invoke({"messages": chat_history})
    
    chat_history = response["messages"]
    print("Farmer:", chat_history[-2].content ,"\n")
    print("Agent:", chat_history[-1].content,"\n")