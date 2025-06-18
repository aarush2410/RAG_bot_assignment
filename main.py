import os
import json
import uuid
import sqlite3
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import hashlib

# FastAPI and related imports
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, validator
import uvicorn

# RAG and NLP imports
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Database Models
@dataclass
class Complaint:
    complaint_id: str
    name: str
    phone_number: str
    email: str
    complaint_details: str
    created_at: str

# Pydantic Models for API
class ComplaintCreate(BaseModel):
    name: str
    phone_number: str
    email: EmailStr
    complaint_details: str
    
    @validator('phone_number')
    def validate_phone(cls, v):
        # Simple phone validation - adjust regex as needed
        if not re.match(r'^\+?1?\d{9,15}$', v):
            raise ValueError('Invalid phone number format')
        return v

class ComplaintResponse(BaseModel):
    complaint_id: str
    message: str

class ComplaintDetails(BaseModel):
    complaint_id: str
    name: str
    phone_number: str
    email: str
    complaint_details: str
    created_at: str

# Database Manager
class DatabaseManager:
    def __init__(self, db_path: str = "complaints.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with complaints table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS complaints (
                complaint_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                phone_number TEXT NOT NULL,
                email TEXT NOT NULL,
                complaint_details TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_complaint(self, name: str, phone_number: str, email: str, complaint_details: str) -> str:
        """Create a new complaint and return complaint ID"""
        complaint_id = self.generate_complaint_id()
        created_at = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO complaints (complaint_id, name, phone_number, email, complaint_details, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (complaint_id, name, phone_number, email, complaint_details, created_at))
        
        conn.commit()
        conn.close()
        
        return complaint_id
    
    def get_complaint(self, complaint_id: str) -> Optional[Complaint]:
        """Retrieve complaint by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM complaints WHERE complaint_id = ?', (complaint_id,))
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            return Complaint(
                complaint_id=result[0],
                name=result[1],
                phone_number=result[2],
                email=result[3],
                complaint_details=result[4],
                created_at=result[5]
            )
        return None
    
    def generate_complaint_id(self) -> str:
        """Generate unique complaint ID"""
        return f"CMP{uuid.uuid4().hex[:8].upper()}"

# RAG Knowledge Base Manager
class RAGKnowledgeBase:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge_base = []
        self.embeddings = []
        self.load_knowledge_base()
    
    def load_knowledge_base(self):
        """Load sample knowledge base for customer service"""
        sample_knowledge = [
            "Customer complaints should be handled with empathy and professionalism at all times.",
            "When collecting complaint information, always ask for name, phone number, email, and detailed description.",
            "Delivery complaints typically involve delays, damaged packages, or wrong items delivered.",
            "Refund requests require verification of purchase details and complaint severity assessment.",
            "Product quality issues should be documented with specific details about the defect or problem.",
            "Customer service representatives should provide complaint ID for tracking purposes.",
            "Follow-up communication should occur within 24-48 hours of complaint registration.",
            "Escalation procedures apply when complaints cannot be resolved at first level support.",
            "Privacy policy requires secure handling of customer personal information.",
            "Complaint resolution timeline varies based on issue complexity and investigation requirements."
        ]
        
        self.knowledge_base = sample_knowledge
        self.embeddings = self.model.encode(sample_knowledge)
    
    def retrieve_relevant_context(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve most relevant context for the query"""
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [self.knowledge_base[i] for i in top_indices]

# Conversation State Manager
class ConversationState:
    def __init__(self):
        self.sessions = {}
    
    def get_session(self, session_id: str) -> Dict:
        """Get or create conversation session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'stage': 'initial',
                'data': {},
                'context': []
            }
        return self.sessions[session_id]
    
    def update_session(self, session_id: str, stage: str, data: Dict):
        """Update conversation session"""
        session = self.get_session(session_id)
        session['stage'] = stage
        session['data'].update(data)
    
    def clear_session(self, session_id: str):
        """Clear conversation session"""
        if session_id in self.sessions:
            del self.sessions[session_id]

# RAG Chatbot
class RAGChatbot:
    def __init__(self, db_manager: DatabaseManager, rag_kb: RAGKnowledgeBase):
        self.db_manager = db_manager
        self.rag_kb = rag_kb
        self.conversation_state = ConversationState()
        
        # Fields required for complaint creation
        self.required_fields = ['name', 'phone_number', 'email', 'complaint_details']
        
    def process_message(self, message: str, session_id: str = "default") -> str:
        """Process user message and return appropriate response"""
        message = message.strip().lower()
        session = self.conversation_state.get_session(session_id)
        
        # Check if user wants to retrieve complaint details
        if self.is_complaint_retrieval_request(message):
            return self.handle_complaint_retrieval(message)
        
        # Check if user wants to file a complaint
        if self.is_complaint_request(message) or session['stage'] != 'initial':
            return self.handle_complaint_creation(message, session_id)
        
        # General query - use RAG
        return self.handle_general_query(message)
    
    def is_complaint_request(self, message: str) -> bool:
        """Check if message is a complaint request"""
        complaint_keywords = [
            'complaint', 'complain', 'issue', 'problem', 'file a complaint',
            'report', 'delayed', 'delivery', 'refund', 'quality', 'defective'
        ]
        return any(keyword in message for keyword in complaint_keywords)
    
    def is_complaint_retrieval_request(self, message: str) -> bool:
        """Check if message is requesting complaint details"""
        retrieval_patterns = [
            r'show.*complaint.*cmp\w+',
            r'details.*cmp\w+',
            r'status.*cmp\w+',
            r'check.*cmp\w+'
        ]
        return any(re.search(pattern, message) for pattern in retrieval_patterns)
    
    def handle_complaint_retrieval(self, message: str) -> str:
        """Handle complaint retrieval requests"""
        # Extract complaint ID from message
        complaint_id_match = re.search(r'cmp\w+', message, re.IGNORECASE)
        if not complaint_id_match:
            return "Please provide a valid complaint ID (format: CMPXXXXXXXX)."
        
        complaint_id = complaint_id_match.group().upper()
        complaint = self.db_manager.get_complaint(complaint_id)
        
        if not complaint:
            return f"No complaint found with ID: {complaint_id}"
        
        return f"""Complaint Details:
Complaint ID: {complaint.complaint_id}
Name: {complaint.name}
Phone: {complaint.phone_number}
Email: {complaint.email}
Details: {complaint.complaint_details}
Created At: {complaint.created_at}"""
    
    def handle_complaint_creation(self, message: str, session_id: str) -> str:
        """Handle complaint creation process"""
        session = self.conversation_state.get_session(session_id)
        
        if session['stage'] == 'initial':
            # First time complaint request
            self.conversation_state.update_session(session_id, 'collecting_name', {})
            return "I'm sorry to hear you're experiencing an issue. To help you file a complaint, I'll need to collect some information. Please provide your full name."
        
        elif session['stage'] == 'collecting_name':
            if len(message.strip()) < 2:
                return "Please provide a valid name."
            self.conversation_state.update_session(session_id, 'collecting_phone', {'name': message.strip()})
            return f"Thank you, {message.strip()}. What is your phone number?"
        
        elif session['stage'] == 'collecting_phone':
            if not self.validate_phone(message.strip()):
                return "Please provide a valid phone number (e.g., 1234567890 or +1234567890)."
            session['data']['phone_number'] = message.strip()
            self.conversation_state.update_session(session_id, 'collecting_email', session['data'])
            return "Got it. Please provide your email address."
        
        elif session['stage'] == 'collecting_email':
            if not self.validate_email(message.strip()):
                return "Please provide a valid email address."
            session['data']['email'] = message.strip()
            self.conversation_state.update_session(session_id, 'collecting_details', session['data'])
            return "Thanks. Now, please provide detailed information about your complaint or issue."
        
        elif session['stage'] == 'collecting_details':
            if len(message.strip()) < 10:
                return "Please provide more detailed information about your complaint."
            
            session['data']['complaint_details'] = message.strip()
            
            # Create complaint
            try:
                complaint_id = self.db_manager.create_complaint(
                    name=session['data']['name'],
                    phone_number=session['data']['phone_number'],
                    email=session['data']['email'],
                    complaint_details=session['data']['complaint_details']
                )
                
                # Clear session
                self.conversation_state.clear_session(session_id)
                
                return f"Your complaint has been registered successfully with ID: {complaint_id}. You'll hear back from us soon. You can check the status anytime by asking for details with your complaint ID."
                
            except Exception as e:
                return "Sorry, there was an error processing your complaint. Please try again."
        
        return "I'm here to help with your complaint. Please let me know what issue you're experiencing."
    
    def validate_phone(self, phone: str) -> bool:
        """Validate phone number format"""
        return bool(re.match(r'^\+?1?\d{9,15}$', phone))
    
    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email))
    
    def handle_general_query(self, message: str) -> str:
        """Handle general queries using RAG"""
        relevant_context = self.rag_kb.retrieve_relevant_context(message)
        
        # Simple response generation based on context
        if any('complaint' in ctx.lower() for ctx in relevant_context):
            return "I can help you file a complaint or answer questions about our complaint process. Would you like to file a complaint or need information about complaint handling?"
        
        return "I'm here to help with customer service inquiries and complaints. How can I assist you today?"

# Initialize components
db_manager = DatabaseManager()
rag_kb = RAGKnowledgeBase()
chatbot = RAGChatbot(db_manager, rag_kb)

# FastAPI Application
app = FastAPI(title="RAG Complaint Management System", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints
@app.post("/complaints", response_model=ComplaintResponse)
async def create_complaint(complaint: ComplaintCreate):
    """Create a new complaint"""
    try:
        complaint_id = db_manager.create_complaint(
            name=complaint.name,
            phone_number=complaint.phone_number,
            email=complaint.email,
            complaint_details=complaint.complaint_details
        )
        
        return ComplaintResponse(
            complaint_id=complaint_id,
            message="Complaint created successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to create complaint")

@app.get("/complaints/{complaint_id}", response_model=ComplaintDetails)
async def get_complaint(complaint_id: str):
    """Retrieve complaint details by ID"""
    complaint = db_manager.get_complaint(complaint_id.upper())
    
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found")
    
    return ComplaintDetails(
        complaint_id=complaint.complaint_id,
        name=complaint.name,
        phone_number=complaint.phone_number,
        email=complaint.email,
        complaint_details=complaint.complaint_details,
        created_at=complaint.created_at
    )

@app.post("/chat")
async def chat_endpoint(message: dict):
    """Chat endpoint for the RAG chatbot"""
    user_message = message.get("message", "")
    session_id = message.get("session_id", "default")
    
    if not user_message:
        raise HTTPException(status_code=400, detail="Message is required")
    
    response = chatbot.process_message(user_message, session_id)
    
    return {
        "response": response,
        "session_id": session_id
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG-based Complaint Management System",
        "endpoints": {
            "create_complaint": "POST /complaints",
            "get_complaint": "GET /complaints/{complaint_id}",
            "chat": "POST /chat"
        }
    }

# CLI Interface for Testing
def run_cli_demo():
    """Run CLI demo of the chatbot"""
    print("RAG Complaint Chatbot Demo")
    print("Type 'quit' to exit")
    print("-" * 40)
    
    session_id = str(uuid.uuid4())
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        response = chatbot.process_message(user_input, session_id)
        print(f"Bot: {response}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        # Run CLI demo
        run_cli_demo()
    else:
        # Run FastAPI server
        print("Starting RAG Complaint Management System...")
        print("API Documentation: http://localhost:8000/docs")
        uvicorn.run(app, host="0.0.0.0", port=8000)