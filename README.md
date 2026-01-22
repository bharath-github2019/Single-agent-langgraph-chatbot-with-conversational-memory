# 🤖 AI Agent with Persistent Memory (LangGraph + Azure OpenAI)

A command-line AI agent built using LangGraph, LangChain, and Azure OpenAI.  
The agent supports tool-based reasoning, multi-step execution, and persistent
conversation memory across sessions.

---

## ✨ Features

- Stateful AI agent using LangGraph
- Built-in math tools (add, subtract, multiply)
- Persistent conversation memory stored in JSON
- Context-aware responses using past interactions
- Streaming responses in the CLI
- Secure credential management via `.env`

---

## 🧠 Architecture Overview

User input flows through memory, agent reasoning, optional tool calls, and
finally back into persistent storage.

---

## 🛠 Tech Stack

- Python 3.9+
- LangChain
- LangGraph
- Azure OpenAI
- python-dotenv

---

## 📂 Project Structure
├── main.py 
├── conversation_memory.json 
├── .env 
├── .gitignore 
├── README.md

---

## 🔐 Environment Setup

### Create a `.env` file
AZURE_ENDPOINT=https://your-azure-endpoint 
AZURE_VERSION=2025-01-01-preview 
AZURE_CHAT_DEPLOYMENT=azure.gpt-5
AZURE_KEY=your_azure_openai_api_key

