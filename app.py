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
            # 1. Get the single employee row
            employee_data = df.iloc[[emp_idx]].copy()
            
            # 2. Drop columns that were NOT used in training (based on your error)
            # These columns are in your CSV but the model never saw them
            cols_to_drop = ['Attrition', 'EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
            employee_data = employee_data.drop(columns=[c for c in cols_to_drop if c in employee_data.columns])

            # 3. Apply One-Hot Encoding (pd.get_dummies)
            # This turns 'OverTime' into 'OverTime_Yes', etc.
            employee_encoded = pd.get_dummies(employee_data)

            # 4. ALIGNMENT: The Model expects exactly 44 specific columns
            expected_features = [
                'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 
                'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 
                'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 
                'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 
                'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 
                'YearsSinceLastPromotion', 'YearsWithCurrManager', 'BusinessTravel_Travel_Frequently', 
                'BusinessTravel_Travel_Rarely', 'Department_Research & Development', 'Department_Sales', 
                'EducationField_Life Sciences', 'EducationField_Marketing', 'EducationField_Medical', 
                'EducationField_Other', 'EducationField_Technical Degree', 'Gender_Male', 
                'JobRole_Human Resources', 'JobRole_Laboratory Technician', 'JobRole_Manager', 
                'JobRole_Manufacturing Director', 'JobRole_Research Director', 'JobRole_Research Scientist', 
                'JobRole_Sales Executive', 'JobRole_Sales Representative', 'MaritalStatus_Married', 
                'MaritalStatus_Single', 'OverTime_Yes'
            ]

            # Create any missing columns with 0 (e.g., if this specific employee doesn't work OverTime)
            for col in expected_features:
                if col not in employee_encoded.columns:
                    employee_encoded[col] = 0
            
            # Ensure columns are in the EXACT same order as the list above
            employee_encoded = employee_encoded[expected_features]

            # 5. Predict using the cleaned and aligned data
            risk_prob = model.predict_proba(employee_encoded)[0][1]
            
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