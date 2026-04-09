from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

# Your ML pipeline
from src.image_model import image_model_pipeline

# Load env
load_dotenv()

app = Flask(__name__)
CORS(app)  # ✅ FIXED: Allow frontend requests

# ------------------ LLM SETUP ------------------ #
llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

tavily_api_key = os.getenv("TAVILY_API_KEY")

search_tool = TavilySearchResults(
    api_key=tavily_api_key,
    num_results=3
)

agent = create_agent(
    model=llm,
    tools=[search_tool],
    system_prompt="""
You are an expert agricultural advisor for sugarcane crops in India.

Rules:
- If disease = Healthy → "No treatment needed. Crop is healthy."
- Otherwise → ONLY 3 lines
- Use given dosage only
- No extra text
"""
)

# ------------------ MEMORY ------------------ #
chat_sessions = {}

# ------------------ ROUTES ------------------ #

@app.route("/")
def home():
    return "🌱 Farmer Assistant API Running"


# -------- CHAT ENDPOINT -------- #
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()

        user_id = data.get("user_id", "default")
        user_message = data.get("message")

        if not user_message:
            return jsonify({"error": "Message required"}), 400

        # Initialize memory
        if user_id not in chat_sessions:
            chat_sessions[user_id] = []

        chat_history = chat_sessions[user_id]

        # Add user message
        chat_history.append(HumanMessage(content=user_message))

        # Call agent
        response = agent.invoke({"messages": chat_history})

        # Save updated history
        chat_sessions[user_id] = response["messages"]

        bot_reply = response["messages"][-1].content

        return jsonify({
            "response": bot_reply
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------- PREDICT ENDPOINT -------- #
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Run ML model
        top3_labels = image_model_pipeline()

        if "Healthy" in top3_labels:
            top3_labels.remove("Healthy")

        diseases = ",".join(top3_labels)
        chemicals = "0.002 ppm"

        response = agent.invoke({
            "messages": [
                HumanMessage(content=f"""
Diseases: {diseases}
Dosage: {chemicals}

Provide treatment recommendations.
""")
            ]
        })

        result = response["messages"][-1].content

        return jsonify({
            "diseases_detected": top3_labels,
            "treatment": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------ RUN ------------------ #
if __name__ == "__main__":
    app.run(debug=True, port=5000)