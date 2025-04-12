# Volunteer Description Matching - Submission

## About
This project implements a simple volunteer description matching system with a FastAPI entry point to interact via APIs with all features for further integration with other systems. The user is able to select between different hugging-face emebdiing models, select between in-memory and 2 vector databases for storage (FAISS & ChromaDB) and also trigger a model finetunning for better performance.

---

## 🚀 Features

### ✅ Core Functionality

- Upload a CSV of volunteer profiles.
- Input a new query description.
- Return top-K most similar volunteers with scores.

### 🔎 Bonus Features

- Multiple embedding model support (`all-mpnet-base-v2` by default)
- FAISS (in-memory) or ChromaDB (persistent) vector storage
- Language detection and translation to English for other languages
- Text cleaning (Punctuation, stopwords and emoji removal, lemmatization.)
- Fine-tuning pipeline for domain-specific improvements triggered by API call
- Timer utility to track performance

---

## 🧪 Setup Instructions

### 🔹 Step 1: Create virtual env and install requiremments

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt
```

### 🔹 Step 2: Start the App from the FastAPI entry point 

```bash
uvicorn api.main:app --reload
```
Open `http://127.0.0.1:8000/docs` on your browser to see the Swagger UI and interract with the App

---

## 📊 Embedding Models
There are 4 Hugging-Face sentence-transformer models available for use and live comparisions. 
`all-mpnet-base-v2, all-MiniLM-L6-v2, paraphrase-multilingual-MiniLM-L12-v2, all-distilroberta-v1`

You can also Trigger a finetuned model based on `all-mpnet-base-v2` and proceed accordingly. 

### Model Comparison

- ⏱️ **Time Taken**: Time to generate embeddings
- 🎯 **Relevant Results**: Match quality (based on relevance to query at k=3)

| Model Name                               | Avg Time per Query | Relvance & Ranking            |
|------------------------------------------|--------------------|-------------------------------|
| `all-mpnet-base-v2` (default)            | 3.51s              | High  (>=2)                   |
| `all-MiniLM-L6-v2`                       | 2.66s              | Medium (~2)                   |
| `paraphrase-multilingual-MiniLM-L12-v2`  | 7.01s              | Low (~1)                      |
| `all-distilroberta-v1`                   | 3.61s              | Low (~1)                      |
| Fine-tuned `all-mpnet-base-v2`           | 19s                | Very High (>=2)               |

### 📊 How to Run a Comparison

You can test different models by updating the model parameters through the `/upload/` and `/finetuned_uload/` APIs

### 🧪 Finetuning Steps:

A simple fintuning appraoch was explored to improve the embedding model performance by giving it more context on the domain specific terms.

- 1 Create sample pairs of descriptions and attach a similarity score in order to give the model and idea of similarity levels per text.

- 2 Select a base model to fine-tune on. `all-MiniLM-L6-v2` was used.

- 3 Define loss function for the training. Cosine similarity was used

- 4 Fine tune the model by providing epochs

- 5 Save finetune model and use for embedding and querying.

User can trigger the finetune process with the `/finetuned_uload/` API and inputting a model name.
---

## 🧰 Switching Engines

You can choose from 3 options:	
- simple – in-memory cosine similarity
- faiss – fast similarity search in memory
- chromadb – persistent storage (auto deletes existing collection on reload)

---
## 🧠 Ideas for Future
-  	UI with Streamlit or React
- 	LLM/RAG-enhanced volunteer matching
- 	Integration with real-time databases or CRMs

---
## Credits

Built by: Chigozilai Kejeh
Project Goal: Volunteer Description Matching
Contact: kebochig@gmail.com