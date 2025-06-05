# FISTBOT: AI-Powered Academic Advisor Chatbot

> FISTBOT is an AI-based academic advising chatbot developed for the Faculty of Information Science and Technology (FIST), Multimedia University. It helps students plan their courses, track prerequisites, and ask academic questions based on the official course structure. The system integrates Retrieval-Augmented Generation (RAG), PDF parsing, hybrid retrieval, and fine-tuned LLMs, providing a fast and intelligent advisory experience.


## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Folder Structure](#folder-structure)
* [Setup](#setup)

## General Info
FISTBOT is designed to automate and enhance student advising through structured document understanding and language model reasoning. It supports natural language queries, prerequisite checks, and personalized term planning.

## Technologies
* Python 3.10++
* Flask
* etc (install all requirements library from requirements.txt)

## Folder Structure

### 📁 `Evaluation on FISTBOT`
Contains experimental results and baseline comparisons:
- PyPDFLoader
- LlamaParse
- Camelot
- Proposed Structured Chunking
- Response time logs and charts

### 📁 `FISTBOT-master`
The main application:
- `app.py`: Main Flask app entry point
- `chatbot.py`: Core RAG logic with hybrid retriever and planning functions
- `db.py`: MongoDB and vector store setup
- `templates/`: HTML files (student login, chatbot interface, admin)
- `upload.py`: PDF course structure loader

### 📁 `mini-FISTBOT`
Lightweight version for testing:
- `chatbot_core.py`: Minimal chatbot logic with short-term memory
- `templates/`: UI for testing interface
- `doc/`: Folder for storing course documents

## Setup
To run this project, install it locally using pip:

```bash
pip install -r requirements.txt
cd FISTBOT-master
python app.py

