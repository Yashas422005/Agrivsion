from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
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



"""
You are an expert agricultural advisor for sugarcane crops in India.

Input:
- Detected diseases (from: ["Rust", "Red Rot", "Yellow", "Bacterial Blight", "Mosaic", "Healthy"])
- Recommended chemicals with dosage (already provided)

Task:
Convert the given recommendations into a clear and concise treatment summary.

Rules:
- If disease = "Healthy", output exactly:
  "No treatment needed. Crop is healthy."
- Otherwise:
  - Provide ONLY a 3-line summary.
  - Each line must include:
    1. Chemical name
    2. Given dosage (DO NOT modify it)
    3. Simple usage instruction (spray/method)
- Do NOT add new chemicals or change dosage.
- Do NOT include explanations or extra text.
- Keep language simple and practical for farmers.
- Maximum 3 lines only.

Example Input:
Diseases: ["Rust", "Bacterial Blight"]
Chemicals:
- Propiconazole (0.1%)
- Mancozeb (0.25%)
- Streptocycline (100 ppm) + Copper oxychloride (0.3%)

Example Output:
1. Propiconazole (0.1%) spray on leaves to control rust infection.
2. Mancozeb (0.25%) apply as foliar spray at regular intervals.
3. Streptocycline (100 ppm) + Copper oxychloride (0.3%) spray to manage bacterial blight.
"""


from src.image_model import image_model_pipeline

top3_labels = image_model_pipeline()

if "Healthy" in top3_labels:
    top3_labels.remove("Healthy")

input=",".join(top3_labels)
chemicals = '0.002 ppm'

from langchain_core.prompts import PromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """You are an expert agricultural advisor for sugarcane crops in India.

Rules:
- If disease = Healthy → "No treatment needed. Crop is healthy."
- Otherwise → ONLY 3 lines
- Use given dosage only
- No extra text
- Provide chemicals relevant to the diseases mentioned in the input.
"""),

    ("human", 
     """Diseases: {input}
Dosage: {chemicals}
Additionally , Also provide some relevant chemicals for the diseases.
""")
])

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""You are an expert agricultural advisor for sugarcane crops in India.

Rules:
- If disease = Healthy → "No treatment needed. Crop is healthy."
- Otherwise → ONLY 3 lines
- Use given dosage only
- No extra text
"""
)
from langchain.messages import HumanMessage

response = agent.invoke({
    "messages":[HumanMessage(content=f"""Diseases: {input}
Dosage: {chemicals}

Provide general treatment recommendations in general for the diseases mentioned in input.


""")
]
})
print(response['messages'][1].content)



