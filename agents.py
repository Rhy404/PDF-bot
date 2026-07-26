import os
from typing import TypedDict
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# llm and rag
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

loader = PyPDFLoader("club_policy.pdf")
docs = loader.load()
chunks = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100).split_documents(docs)
vectorstore = FAISS.from_documents(chunks, HuggingFaceEmbeddings(model_name="all-miniLM-L6-v2"))

retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# shared memory
class AgentState(TypedDict):
    query: str                    # User's initial request
    policy_constraints: str       # Rules found by the Auditor agent
    draft_data: dict              # Generated email JSON
    compliance_score: bool        # Whether the draft meets policy
    iterations: int               # To prevent infinite loops
    is_info_missing: bool         # Returns TRUE if missing info
    missing_info_msg: str         # Msg to user for giving missing info input
    
# auditor agent
def policy_auditor(state: AgentState):
    """
    Searches the policy PDF to find the exact recipients and formatting rules.
    """
    query = state['query']
    
    docs = retriever.invoke(f"Recipients and procedure for: {query}")
    context = "\n".join([d.page_content for d in docs])
    
    prompt = f"""
    You are a Policy Auditor for Club INFERNO.
    Based on the context below, extract the EXACT email addresses for 'To' and 'CC'.
    Also, identify any mandatory fields or formatting constraints.

    Context: {context}
    User Request: {query}

    Output only the identified constraints and email addresses.
    """
    
    response = llm.invoke(prompt)
    
    return {
        "policy_constraints": response.content,
        "iterations": state.get("iterations", 0)
    }
    
# missing info checker agent
def missing_info_checker(state: AgentState):
    """
    Compares the user query against policy constraints to see
    if any mandatory information is missing.
    """
    
    query = state['query']
    constraints = state['policy_constraints']
    
    prompt = f"""
    You are a Strategic Coordinator for the dance club INFERNO.
    
    Policy Constraints for this request:
    {constraints}
    
    User's Request:
    {query}
    
    Task:
    1. Identify if any mandatory information required by the policy is missing from the user's request (e.g., specific dates, room capacity, event type, or budget details).
    2. If information is missing, output a polite message asking the user for those specific details. Start with "MISSING:".
    3. If all necessary information is present to draft the email, output "READY".
    
    Output ONLY "READY" or the "MISSING:" message.
    """
    
    response = llm.invoke(prompt)
    content = response.content.strip()
    
    is_missing = content.startswith("MISSING:")
    
    return {
        "is_info_missing": is_missing,
        "missing_info_msg": content if is_missing else ""
    }
    
    

# drafter agent
def email_drafter(state: AgentState):
    """
    Drafts the email based on user intent and policy constraints.
    """
    query = state['query']
    constraints = state['policy_constraints']
    
    prompt = f"""
    You are the specialized Drafting Agent for Club INFERNO.
    Use the following Policy Constraints to write a formal email.
    
    Policy Constraints:
    {constraints}
    
    User Request:
    {query}
    
    Instructions:
    - You MUST use the 'To' and 'CC' fields exactly as identified in the constraints.
    - If a specific format was mentioned in the constraints, follow it strictly.
    - Use a formal tone.
    - Output ONLY a JSON object with: "to", "cc", "subject", "body".
    """
    
    response = llm.invoke(prompt)
    
    parser = JsonOutputParser()
    draft_json = parser.parse(response.content)
    
    return {
        "draft_data": draft_json,
        "iterations": state.get("iterations", 0) + 1
    }
    
# compliance critic agent
def compliance_critic(state: AgentState):
    """
    Checks if the draft adheres to all policy constraints.
    """
    constraints = state['policy_constraints']
    draft = state['draft_data']
    
    prompt = f"""
    You are the Compliance Officer for the dance club INFERNO.
    Compare the draft below against Policy Constraints.
    
    Policy Constraints:
    {constraints}
    
    Draft to Review:
    {draft}
    
    Instructions:
    - Check if 'To' and 'CC' match the policy exactly.
    - Check if all mandatory information is present.
    - If it is compliant, output ONLY the word "COMPLIANT".
    - If it is NOT compliant, provide a short list of what needs to be fixed.
    """
    
    response = llm.invoke(prompt)
    feedback = response.content.strip()
   
    return {
        "policy_constraints": f"{constraints}\n\nREVISION NEEDED: {feedback}" if "COMPLIANT" not in feedback else constraints,
        "compliance_score": "COMPLIANT" in feedback
    }

