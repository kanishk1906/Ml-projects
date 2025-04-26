# Deep Research AI Agent

This repository implements a multi-agent AI system for deep online research using **Tavily**, **LangGraph**, and **LangChain** frameworks.

---

## 🚀 Project Overview

The system contains two key autonomous agents:
- **Research Agent**: Responsible for gathering information from the internet using Tavily and organizing relevant details.
- **Answer Drafting Agent**: Takes the collected information and generates structured, human-readable answers.

Agents communicate and coordinate using a graph-based flow (LangGraph).

---

## 🛠️ Technologies Used

- Python 3.10+
- [LangChain](https://github.com/langchain-ai/langchain) (for building agent tools)
- [LangGraph](https://github.com/langchain-ai/langgraph) (for graphing multi-agent workflows)
- [Tavily API](https://docs.tavily.com/) (for web search & crawling)
- OpenAI API (for GPT-4 / LLM outputs)
- Streamlit (optional: if you make a UI)

---

## 📚 System Architecture

