# =====================================================
# Imports
# =====================================================

from typing import Annotated, Sequence, TypedDict
from datetime import datetime
import json
import os

from dotenv import load_dotenv

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import AzureChatOpenAI


# =====================================================
# Load Environment Variables
# =====================================================

load_dotenv(override=True)

AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_VERSION = os.getenv("AZURE_VERSION")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT")
AZURE_KEY = os.getenv("AZURE_KEY")

if not all([AZURE_ENDPOINT, AZURE_VERSION, AZURE_CHAT_DEPLOYMENT, AZURE_KEY]):
    raise EnvironmentError("Missing one or more Azure OpenAI environment variables")


# =====================================================
# Tool Definitions
# =====================================================

@tool
def add(a: int, b: int):
    """Add two numbers"""
    return a + b


@tool
def subtract(a: int, b: int):
    """Subtract two numbers"""
    return a - b


@tool
def multiply(a: int, b: int):
    """Multiply two numbers"""
    return a * b


TOOLS = [add, subtract, multiply]


# =====================================================
# Model Initialization
# =====================================================

model = AzureChatOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    azure_deployment=AZURE_CHAT_DEPLOYMENT,
    api_version=AZURE_VERSION,
    api_key=AZURE_KEY,
    temperature=0,
)

model_with_tools = model.bind_tools(TOOLS)


# =====================================================
# Agent State
# =====================================================

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# =====================================================
# Agent Logic
# =====================================================

def model_call(state: AgentState) -> AgentState:
    system_message = SystemMessage(
        content=(
            "You are my personal AI agent with memory.\n"
            "1. Answer accurately\n"
            "2. Use previous context\n"
            "3. Use tools when required\n"
            "4. Stay conversational"
        )
    )

    messages = [system_message] + state["messages"]
    response = model_with_tools.invoke(messages)

    return {"messages": [response]}


def should_continue(state: AgentState):
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    return "end"


# =====================================================
# LangGraph Construction
# =====================================================

graph = StateGraph(AgentState)

graph.add_node("agent_calling", model_call)
graph.set_entry_point("agent_calling")

tool_node = ToolNode(tools=TOOLS)
graph.add_node("tools", tool_node)

graph.add_conditional_edges(
    "agent_calling",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "agent_calling")

app = graph.compile()


# =====================================================
# Conversation Memory
# =====================================================

class ConversationMemory:
    def __init__(self, memory_file="conversation_memory.json"):
        self.memory_file = memory_file
        self.conversation_history = []
        self.load_memory()

    def load_memory(self):
        if not os.path.exists(self.memory_file):
            return

        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.conversation_history = data.get("conversations", [])
                print(f"📚 Loaded {len(self.conversation_history)} conversations")
        except Exception as e:
            print(f"⚠️ Memory load failed: {e}")

    def save_memory(self):
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "conversations": self.conversation_history,
                        "last_updated": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception as e:
            print(f"⚠️ Memory save failed: {e}")

    def add_conversation(self, user_input, agent_response):
        self.conversation_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "user": user_input,
                "agent": agent_response,
            }
        )
        self.conversation_history = self.conversation_history[-50:]

    def get_context_messages(self, limit=5):
        messages = []
        for conv in self.conversation_history[-limit:]:
            messages.append(HumanMessage(content=conv["user"]))
            messages.append(
                SystemMessage(content=f"Previous response: {conv['agent']}")
            )
        return messages

    def search_memory(self, keyword):
        return [
            conv
            for conv in self.conversation_history
            if keyword.lower() in conv["user"].lower()
            or keyword.lower() in conv["agent"].lower()
        ]

    def show_recent_conversations(self, limit=5):
        for conv in self.conversation_history[-limit:]:
            ts = datetime.fromisoformat(conv["timestamp"]).strftime("%Y-%m-%d %H:%M")
            print(f"[{ts}] You: {conv['user']}")
            print(f"Agent: {conv['agent'][:120]}")
            print("-" * 40)


# =====================================================
# Streaming + Memory Capture
# =====================================================

def print_stream_with_memory(stream, memory, user_input):
    agent_response = ""

    for s in stream:
        message = s["messages"][-1]
        message.pretty_print()

        if hasattr(message, "content"):
            agent_response = message.content

    if agent_response:
        memory.add_conversation(user_input, agent_response)
        memory.save_memory()


# =====================================================
# CLI Application
# =====================================================

def main():
    memory = ConversationMemory()

    print("🤖 AI Agent with Memory")
    print("Commands: memory | search <keyword> | clear memory | help | quit")
    print("=" * 60)

    while True:
        try:
            user_prompt = input("You: ").strip()

            if user_prompt.lower() in {"quit", "exit", "bye"}:
                print("Agent: Goodbye! 👋")
                break

            if user_prompt == "memory":
                memory.show_recent_conversations()
                continue

            if user_prompt.startswith("search "):
                results = memory.search_memory(user_prompt[7:])
                print(results or "No matches found")
                continue

            if user_prompt == "clear memory":
                memory.conversation_history = []
                memory.save_memory()
                print("🗑️ Memory cleared")
                continue

            if not user_prompt:
                continue

            context = memory.get_context_messages()
            inputs = {"messages": context + [HumanMessage(content=user_prompt)]}

            print_stream_with_memory(
                app.stream(inputs, stream_mode="values"),
                memory,
                user_prompt,
            )

        except KeyboardInterrupt:
            print("\nAgent: Goodbye! 👋")
            break

        except Exception as e:
            print(f"Agent Error: {e}")


# =====================================================
# Entry Point
# =====================================================

if __name__ == "__main__":
    main()
