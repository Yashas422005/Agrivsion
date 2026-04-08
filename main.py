from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
    base_url="https://api.groq.com/openai/v1",
    api_key="GROQ_API_KEY",
    temperature=0,
)

