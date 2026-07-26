import os
import base64
from email.message import EmailMessage
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

# Import from agents.py
from agents import AgentState, policy_auditor, email_drafter, compliance_critic, missing_info_checker

# Google API Imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
# gmail setup
load_dotenv()
SCOPES = ['https://www.googleapis.com/auth/gmail.compose']

def get_gmail_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)


# gmail draftin function
def create_gmail_draft(data):
    """Takes the JSON output from the agents and pushes it to Gmail drafts."""
    service = get_gmail_service()
    message = EmailMessage()
    
    message.set_content(data['body'])
    message['To'] = data['to']
    message['Cc'] = data.get('cc', '')
    message['Subject'] = data['subject']
    
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    
    # actual gmail api call
    service.users().drafts().create(userId='me', body={'message': {'raw': raw}}).execute()
    print("\n[System] Draft created successfully in your Gmail.")
    
# langgraph orchestration-----------------------
 
# flow: Start->Auditor->MissingInfoChecker->Drafter->ComplianceCritic->END
workflow = StateGraph(AgentState)

workflow.add_node("auditor", policy_auditor)
workflow.add_node("checker", missing_info_checker)
workflow.add_node("drafter", email_drafter)
workflow.add_node("critic", compliance_critic)


workflow.set_entry_point("auditor")
workflow.add_edge("auditor", "checker")

# condition to check missing info
def check_info_gate(state: AgentState):
    if state.get("is_info_missing"):
        return END
    return "drafter"

workflow.add_conditional_edges("checker", check_info_gate)

workflow.add_edge("drafter", "critic") 

# condition to reach END
def should_continue(state: AgentState):
    if state.get("compliance_score") or state["iterations"] > 3:
        return END
    return "drafter"

workflow.add_conditional_edges("critic", should_continue)

# compile graph into an executable
app = workflow.compile()

# executor
if __name__ == "__main__":
    print("INFERNO Club Email Bot: ACTIVE")
    
    # input from user
    user_query = "We need permission to get a digital standee for our event."

    final_state = app.invoke({"query": user_query, "iterations": 0})
    
    if final_state.get("is_info_missing"):
        print(f"\n[Agent Response]\n{final_state['missing_info_msg']}")
        
    else:
        create_gmail_draft(final_state["draft_data"])