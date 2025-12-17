# SHL Assessment Recommendation Engine

## Overview
This project implements an intelligent Assessment Recommendation Engine using SHL’s product catalog.
The system recommends the most relevant assessments based on a job description, required skills, and role intent.

It is designed to be:
- Explainable
- Scalable
- Domain-aware (AI, tech, leadership, entry-level roles)

---

## Approach

### 1. Retrieval (Semantic Search)
- SentenceTransformer (`all-MiniLM-L6-v2`) generates embeddings for assessment descriptions.
- FAISS is used for fast nearest-neighbor similarity search.

### 2. Rule-Based Re-ranking
A domain-aware `rule_boost` function adjusts rankings based on:
- Role intent (AI / Tech / Leadership / Language)
- Skill relevance
- Job level alignment
- Penalization of irrelevant assessments

This hybrid approach ensures:
- Strong relevance
- Reduced noise
- Human-interpretable decisions

---

## API Endpoints

### Health Check


### Recommend Assessments

### Structure
shl-reco-engine/
│
├── src/
│   ├── api.py                  
│   ├── build_index.py          
│   └── __pycache__/            
│       └── api.cpython-311
│
├── data/
│   ├── catalog.csv             
│   ├── catalog.pkl             
│   └── catalog.index           
│
├── SHL_Assessment_Recommendation_Engine_Notebook.ipynb
└── README.md
       

#### Sample Input
```json
{
  "job_title": "Research AI Intern",
  "job_description": "Building ML and NLP pipelines",
  "skills": ["Python", "Machine Learning", "NLP", "Statistics"],
  "top_k": 5
}

#### How to Run Locally
pip install fastapi uvicorn sentence-transformers faiss-cpu pandas numpy
python -m uvicorn src.api:app --reload
open:
http://127.0.0.1:8000/docs


Use this input:
```json
{
  "job_title": "Research AI Intern",
  "job_description": "ML, NLP, experimentation",
  "skills": ["Python", "Machine Learning", "NLP", "Deep Learning"],
  "top_k": 5
}
