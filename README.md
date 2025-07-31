# LLM-Powered Research Agent

A terminal-based AI tool to summarize and query research papers using LLMs and semantic search.

##  Overview

This project is a **minimal working Research Agent** that allows users to:

- Upload and parse research papers (PDFs)
- Automatically generate a **summary** using LLaMA 3 models
- Ask **natural language questions** about the paper
- Receive answers generated using **retrieval-augmented generation (RAG)**

It combines document parsing, semantic chunking, FAISS indexing, and a conversational LLM pipeline into one streamlined terminal-based tool.

---

## Features

- Upload any academic **PDF**
- Automatic **chunking** of text for semantic understanding
- Uses **SentenceTransformers** for vector embeddings
- Fast and efficient similarity search with **FAISS**
- Query answering powered by **Groq’s LLaMA 3** models
- Terminal-based, no GUI required
- Open-source and adaptable for other document types

---

## Tech Stack

- **Python 3.8+**
- `PyMuPDF (fitz)` – PDF parsing
- `SentenceTransformers` – Embedding generation
- `FAISS` – Vector similarity search
- `OpenAI / Groq API` – LLaMA 3 for summarization + QA
- `NumPy`, `os`, `sys` – Utilities

---
