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

# --- 1. AUTHENTICATION ---
st.set_page_config(page_title="Agentic-RAG Attrition CoPilot", layout="wide")

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
        policy_text = ["Standard retention policy: Offer career coaching and flexible hours."]

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
            # Step A: Get Real Prediction from XGBoost
            employee_data = df.iloc[[emp_idx]].copy()
            if 'Attrition' in employee_data.columns:
                employee_data = employee_data.drop(columns=['Attrition'])
            
            # Predict probability (class 1 is 'Yes' for attrition)
            risk_prob = model.predict_proba(employee_data)[0][1]
            
            # Step B: Initialize Agent
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a Senior IBM HR Business Partner. Based on the risk score, use the tool to find a policy and provide a clear, bulleted retention plan. Return only the plan as clean text, no JSON."),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ])
            
            agent = create_tool_calling_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
            
            # Step C: Generate Plan
            query = f"Employee at index {emp_idx} has a predicted {risk_prob:.2%} risk of leaving. What is the recommended retention plan?"
            response = agent_executor.invoke({"input": query})
            
            # Step D: Display Results
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Attrition Risk Score", f"{risk_prob:.1%}")
                st.info(f"Analysis complete for Employee ID: {df.iloc[emp_idx].get('EmployeeNumber', emp_idx)}")
            with col2:
                st.subheader("Personalized Retention Strategy")
                st.success(response['output']) # Extracts the clean text output
                
        except Exception as e:
            st.error(f"❌ Application Error: {e}")
            if "quota" in str(e).lower():
                st.info("💡 Daily API Limit Reached. Please try again tomorrow.")