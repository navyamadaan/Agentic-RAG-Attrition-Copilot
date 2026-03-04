import streamlit as st
import pandas as pd
import xgboost as xgb
import os
import time

# Use Streamlit's built-in secrets manager
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Please set the GOOGLE_API_KEY in Streamlit Secrets.")

# AI & Agent Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="IBM Retention Copilot", layout="wide")
os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY_HERE" # Update this tomorrow!

# --- 2. LOAD DATA & MODEL (CACHED) ---
@st.cache_resource
def load_resources():
    # Load the XGBoost Model from Phase 1
    bst = xgb.XGBClassifier()
    bst.load_model("attrition_model.json")
    
    # Load the Dataset
    df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    
    # Setup the RAG Policy Database
    policy_text = [
        "OVERTIME POLICY: If risk is high due to overtime, offer a 4-day work week.",
        "SALARY POLICY: If income is below average, provide a 5% market adjustment.",
        "SATISFACTION POLICY: If job satisfaction is < 2, schedule career coaching."
    ]
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts(policy_text, embeddings)
    
    return bst, df, vectorstore

model, df, vectorstore = load_resources()

# --- 3. DEFINE AGENT TOOLS ---
@tool
def search_hr_policy(query: str):
    """Searches the internal HR handbook for retention strategies."""
    docs = vectorstore.similarity_search(query, k=1)
    return docs[0].page_content

tools = [search_hr_policy]

# --- 4. THE INTERFACE ---
st.title("📋 IBM Employee Retention Copilot")
st.markdown("Predicting attrition risk and generating policy-backed retention plans.")

# Sidebar for Employee Selection
st.sidebar.header("Employee Lookup")
emp_idx = st.sidebar.number_input("Enter Employee Index", min_value=0, max_value=len(df)-1, value=200)

if st.sidebar.button("Run Analysis"):
    with st.spinner("🤖 AI is analyzing data and HR policies..."):
        # Calculate Risk Score (XGBoost)
        # Note: In production, you would preprocess df.iloc[emp_idx] here
        risk_prob = 0.99  # Placeholder: use model.predict_proba in full version
        
        # Initialize the Gemini 2.5 Agent
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an IBM HR expert. Recommend a plan based on the risk and company policy."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        # Get Recommendation
        response = agent_executor.invoke({"input": f"Employee {emp_idx} has a {risk_prob:.2%} risk. What is the plan?"})
        
        # Display Results
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Attrition Risk", f"{risk_prob:.1%}")
        with col2:
            st.subheader("Retention Plan")
            st.write(response['output'])