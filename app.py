import streamlit as st
import pandas as pd
import xgboost as xgb
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# --- 1. AUTHENTICATION & CONFIG ---
st.set_page_config(page_title="IBM Retention Copilot", layout="wide")

# Securely fetch API Key from Streamlit Secrets
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Missing GOOGLE_API_KEY. Please add it to your Streamlit Cloud Secrets.")
    st.stop()

# --- 2. RESOURCE LOADING (CACHED) ---
@st.cache_resource
def load_resources():
    # Load the XGBoost Model
    model = xgb.XGBClassifier()
    model.load_model("attrition_model.json")
    
    # Load the Dataset
    df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    
    # Load HR Policy for RAG
    if os.path.exists("hr_policy.txt"):
        with open("hr_policy.txt", "r") as f:
            policy_text = f.readlines()
    else:
        policy_text = ["Standard retention policy: Engage employee with career coaching."]

    # Setup Vector Database
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts(policy_text, embeddings)
    
    return model, df, vectorstore

model, df, vectorstore = load_resources()

# --- 3. AGENT TOOL DEFINITION ---
@tool
def search_hr_policy(query: str):
    """Searches the internal HR handbook for retention strategies and policies."""
    docs = vectorstore.similarity_search(query, k=1)
    return docs[0].page_content

tools = [search_hr_policy]

# --- 4. INTERFACE ---
st.title("📋 Agentic-RAG Attrition CoPilot")
st.markdown("Predicting attrition risk and generating policy-backed retention plans for IBM HR.")

# Sidebar Selection
st.sidebar.header("Employee Lookup")
emp_idx = st.sidebar.number_input("Enter Employee Index", min_value=0, max_value=len(df)-1, value=200)

if st.sidebar.button("Run Analysis"):
    with st.spinner("🤖 AI is analyzing data and HR policies..."):
        try:
            # Step A: Get Prediction (Logic should match your Phase 1 preprocessing)
            # For demonstration, we simulate the risk from the model's perspective
            risk_prob = 0.88  # Replace with actual model.predict_proba(features)
            
            # Step B: Initialize Agent
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an IBM HR expert. Recommend a plan based on the risk and company policy. Use the search_hr_policy tool."),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ])
            
            agent = create_tool_calling_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
            
            # Step C: Generate Plan
            query = f"Employee {emp_idx} has a {risk_prob:.2%} risk of leaving. What is the retention plan?"
            response = agent_executor.invoke({"input": query})
            
            # Step D: Display
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Attrition Risk Score", f"{risk_prob:.1%}")
            with col2:
                st.subheader("AI-Generated Retention Strategy")
                st.write(response['output'])
                
        except Exception as e:
            st.error(f"❌ Application Error: {e}")
            if "INVALID_ARGUMENT" in str(e):
                st.info("Check your API key in Streamlit Secrets and ensure the model name is correct.")