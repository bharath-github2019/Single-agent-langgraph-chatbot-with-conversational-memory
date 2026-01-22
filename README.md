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
