# FISTBOT: AI-Powered Academic Advisor Chatbot

> This is the Final Year Project (FYP) developed for the Faculty of Information Science and Technology (FIST) at Multimedia University. FISTBOT is an AI-based academic advising chatbot that helps students plan their courses, track prerequisites, and ask academic questions based on the official course structure. The system integrates Retrieval-Augmented Generation (RAG), PDF parsing, hybrid retrieval, and fine-tuned LLMs, providing a fast and intelligent advisory experience.

---

## 🧰 Technologies

- Python 3.10++
- MongoDB
- LangChain, ChromaDB, FAISS
- Flask (Web Framework)
- PDF Parsers (Camelot, PyPDFLoader, LlamaParse)
- CUDA + cuDNN (for GPU support, optional)
- Install all required libraries via `requirements.txt`

---

## 📁 Folder Structure

### `Evaluation on FISTBOT/`
Contains experimental results and baseline comparisons:
- PyPDFLoader
- LlamaParse
- Camelot
- Proposed Structured Chunking
- Response time logs and charts

### `FISTBOT-master/` (Main App)
- `app.py` – Main Flask app entry point (run this to host locally)
- `chatbot.py` – Core RAG logic with hybrid retriever and planning functions
- `db.py` – MongoDB and vector store setup
- `templates/` – HTML templates for student interface and admin panel
- `upload.py` – PDF course structure loader

### `mini-FISTBOT/` (Lightweight Version)
- `app.py` – Flask app entry point for testing version
- `chatbot_core.py` – Minimal chatbot logic
- `templates/` – Testing UI
- `doc/` – Folder for storing course documents

---

## ⚙️ Setup Instructions

### 1. Install MongoDB (Local Database)
- Download: https://www.mongodb.com/try/download/community  
- Installation guide (video): https://www.youtube.com/watch?v=gB6WLkSrtJk&t=448s

> After installation, open `FISTBOT-master/db.py` and modify this line:
> ```python
> "mongodb://localhost:27017/FISTBOT"
> ```
> Replace with your MongoDB URI if it's different.

---

### 2. Create Python Virtual Environment (Python 3.10+)

```bash
python -m venv venv
source venv/bin/activate   
```

---

### 3. Prepare GPU Support (Optional)

- CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit  
- cuDNN: https://developer.nvidia.com/cudnn  
- PyTorch GPU Setup: https://pytorch.org/get-started/locally/  
- Setup guide video: https://www.youtube.com/watch?v=nATRPPZ5dGE

---

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 5. Run the Project

#### ▶️ Run Full FISTBOT (Main Version)
```bash
cd FISTBOT-master
python app.py
```

#### ▶️ Run Mini-FISTBOT (Testing Version)
```bash
cd mini-FISTBOT
python app.py
```

---

## 👤 Contact

- **Name**: Koay Xin Kuang  
- **Student ID**: 1201102955  
- **Email**: 1201102955@student.mmu.edu.my  
- **Supervisor**: Dr. Ong Lee Yeng  
- **Supervisor Email**: lyong@mmu.edu.my  
- **University**: Multimedia University, Malaysia

---
