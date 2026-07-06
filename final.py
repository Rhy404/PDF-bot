import os
import base64
from email.message import EmailMessage
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# 1. Setup
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

# 2. Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# 3. RAG Setup
loader = PyPDFLoader("club_policy.pdf")
chunks = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100).split_documents(loader.load())
vectorstore = FAISS.from_documents(chunks, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# 4. Improved Prompt
email_template = """
You are the Email Bot for Club INFERNO. 
Strictly follow the Club Policy provided in the context.

Context: {context}
User Request: {question}

Instructions:
- Identify if the request has multiple procedures (e.g., Hostels vs. General Campus). 
- If the user hasn't specified a location, default to the most formal procedure but include a note in the body asking them to confirm the location.
- Extract the "To" email and the "CC" email exactly as written in the policy.
- Output ONLY a JSON object with: "to", "cc", "subject", "body". 
- If no CC is mentioned, leave the "cc" field as an empty string.
- If you need more information from the user, clarify and ask the user again.
- Keep in mind we are a Cultual Club - INFERNO, and also use our INFERNO 2026-2027 signature which we already made, in every single mail draft.
- Draft mails with content in a formal tone.

JSON Output:"""

email_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | ChatPromptTemplate.from_template(email_template)
    | llm
    | JsonOutputParser()
)

# 5. Updated Draft Function (Handles CC)
def create_gmail_draft(data):
    service = get_gmail_service()
    message = EmailMessage()
    message.set_content(data['body'])
    message['To'] = data['to']
    message['Cc'] = data.get('cc', '')
    message['Subject'] = data['subject']

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    service.users().drafts().create(userId='me', body={'message': {'raw': raw}}).execute()
    print("Draft created with CC successfully.")

if __name__ == "__main__":
    query = "Draft an email for putting up digital standees for our upcoming showcase."
    email_data = email_chain.invoke(query)
    create_gmail_draft(email_data)
