# RAG_bot_assignment
Assessment for Cyfuture

# RAG-Based Complaint Management System

A comprehensive customer service solution that combines Retrieval-Augmented Generation (RAG) with conversational AI to automate complaint handling and customer support interactions.

## 🚀 Features

- **Intelligent Chatbot**: RAG-powered conversational AI that handles customer complaints naturally
- **Automated Data Collection**: Guided conversation flow to collect complaint information
- **RESTful API**: Complete API endpoints for complaint management
- **SQLite Database**: Persistent storage for complaint records
- **Conversation State Management**: Multi-turn conversation handling with session management
- **Validation**: Built-in validation for phone numbers and email addresses
- **Retrieval System**: Context-aware responses using sentence transformers
- **CLI Interface**: Command-line demo for testing and development

## 🛠️ Technologies Used

- **Backend**: FastAPI, Python 3.7+
- **Database**: SQLite3
- **AI/ML**: 
  - Sentence Transformers (all-MiniLM-L6-v2)
  - OpenAI API (configured but not actively used in current implementation)
  - scikit-learn for similarity calculations
- **Data Validation**: Pydantic models with custom validators
- **CORS**: Enabled for cross-origin requests

## 📋 Prerequisites

- Python 3.7 or higher
- pip package manager

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/rag-complaint-management.git
   cd rag-complaint-management
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create requirements.txt** (if not already present)
   ```txt
   fastapi==0.104.1
   uvicorn==0.24.0
   pydantic[email]==2.5.0
   sentence-transformers==2.2.2
   scikit-learn==1.3.2
   numpy==1.24.3
   openai==1.3.7
   ```

## 🚀 Usage

### Running the FastAPI Server

```bash
python app.py
```

The API will be available at:
- **API Base URL**: http://localhost:8000
- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc

### Running the CLI Demo

```bash
python app.py cli
```

## 📖 API Endpoints

### 1. Create Complaint
**POST** `/complaints`

Create a new customer complaint.

**Request Body:**
```json
{
  "name": "John Doe",
  "phone_number": "1234567890",
  "email": "john.doe@email.com",
  "complaint_details": "My order was delivered damaged and I need a replacement."
}
```

**Response:**
```json
{
  "complaint_id": "CMPAB123456",
  "message": "Complaint created successfully"
}
```

### 2. Get Complaint Details
**GET** `/complaints/{complaint_id}`

Retrieve complaint details by ID.

**Response:**
```json
{
  "complaint_id": "CMPAB123456",
  "name": "John Doe",
  "phone_number": "1234567890",
  "email": "john.doe@email.com",
  "complaint_details": "My order was delivered damaged and I need a replacement.",
  "created_at": "2024-01-15T10:30:00.123456"
}
```

### 3. Chat Interface
**POST** `/chat`

Interact with the RAG chatbot for complaint handling.

**Request Body:**
```json
{
  "message": "I want to file a complaint",
  "session_id": "optional-session-id"
}
```

**Response:**
```json
{
  "response": "I'm sorry to hear you're experiencing an issue. To help you file a complaint, I'll need to collect some information. Please provide your full name.",
  "session_id": "session-id"
}
```

## 🤖 Chatbot Conversation Flow

The chatbot guides users through a structured complaint filing process:

1. **Initial Request**: User expresses intent to file a complaint
2. **Name Collection**: Bot requests customer's full name
3. **Phone Collection**: Bot requests phone number with validation
4. **Email Collection**: Bot requests email address with validation
5. **Details Collection**: Bot requests detailed complaint information
6. **Confirmation**: Bot creates complaint and provides tracking ID

### Supported Commands

- **File Complaints**: "I want to file a complaint", "I have an issue", "delivery problem"
- **Check Status**: "Show complaint CMP12345678", "Status of CMP12345678"
- **General Queries**: The bot can answer general customer service questions

## 📊 Database Schema

The system uses SQLite with the following schema:

```sql
CREATE TABLE complaints (
    complaint_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    phone_number TEXT NOT NULL,
    email TEXT NOT NULL,
    complaint_details TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

## 🧠 RAG Knowledge Base

The system includes a built-in knowledge base covering:
- Customer service best practices
- Complaint handling procedures
- Delivery and refund policies
- Escalation procedures
- Privacy and data handling guidelines

## 🔒 Validation Rules

- **Phone Numbers**: Must match pattern `^\+?1?\d{9,15}$`
- **Email**: Standard email validation using Pydantic EmailStr
- **Names**: Minimum 2 characters
- **Complaint Details**: Minimum 10 characters

## 🛡️ Error Handling

The system includes comprehensive error handling for:
- Invalid input validation
- Database connection issues
- Malformed requests
- Missing complaint IDs
- Session management errors

## 📁 Project Structure

```
rag-complaint-management/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── complaints.db         # SQLite database (created automatically)
├── README.md            # This file
└── .gitignore          # Git ignore file
```

## 🧪 Testing

### Manual Testing with CLI
```bash
python app.py cli
```

### API Testing with curl
```bash
# Create a complaint
curl -X POST "http://localhost:8000/complaints" \
     -H "Content-Type: application/json" \
     -d '{
       "name": "Test User",
       "phone_number": "1234567890",
       "email": "test@example.com",
       "complaint_details": "Test complaint details"
     }'

# Get complaint details
curl -X GET "http://localhost:8000/complaints/CMPXXXXXXXX"

# Chat with bot
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "I want to file a complaint",
       "session_id": "test-session"
     }'
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) for embedding models
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Pydantic](https://pydantic-docs.helpmanual.io/) for data validation

---

