
# **SHL Assessment Recommendation Engine**

## 📌 Overview

This project implements an **intelligent Assessment Recommendation Engine** using **SHL’s product catalog**.

The system recommends the **most relevant SHL assessments** for a given job role or hiring query by combining:

* **Semantic search (ML-based retrieval)**
* **Domain-aware rule-based re-ranking**

The solution was further used to generate recommendations for the **provided Excel dataset**, as required in the assignment.

Key properties:

* ✅ Explainable
* ✅ Scalable
* ✅ Domain-aware (AI, Tech, Leadership, Entry-level roles)
* ✅ Suitable for real-world hiring workflows

---

## 🧠 What We Built

We built a **hybrid recommendation system** that:

1. Understands job descriptions semantically
2. Matches them against SHL assessments
3. Applies intelligent business rules to improve ranking quality

The system exposes a **FastAPI-based REST API** that can be used interactively or programmatically.

---

## 🧩 High-Level Architecture

**Input (Job Description / Query)**
⬇
**Sentence Embeddings (SentenceTransformer)**
⬇
**FAISS Vector Search (Top-K similar assessments)**
⬇
**Rule-Based Boosting & Penalization**
⬇
**Final Ranked SHL Assessment Recommendations**

---

## ⚙️ Approach

### 1️⃣ Retrieval (Semantic Search)

* Used **SentenceTransformer (`all-MiniLM-L6-v2`)** to generate embeddings for:

  * SHL assessment descriptions
  * Incoming job queries
* Used **FAISS** for fast cosine-similarity-based nearest neighbor search

This ensures:

* Robust semantic matching
* No keyword dependency
* Good performance at scale

---

### 2️⃣ Rule-Based Re-ranking (Domain Intelligence)

A custom `rule_boost` function adjusts rankings based on:

* **Role Intent**

  * AI / ML / Research
  * Software Engineering
  * Leadership / Managerial
  * Language / Communication
* **Skill relevance**
* **Job level alignment**
* **Penalization of irrelevant assessments**

  * Business skills for technical roles
  * Senior-level tests for intern roles
  * Language tests unless explicitly requested

This hybrid ML + rules approach ensures:

* High precision
* Reduced noise
* Human-interpretable decisions

---

## 📊 Excel Dataset Usage (Assignment Requirement)

The provided **Excel dataset** contained hiring queries.

**What we did:**

1. Loaded each query from the Excel file
2. Sent it to the `/recommend` API endpoint
3. Generated **Top-K SHL assessment recommendations**
4. Collected results in tabular form:

   * `Query`
   * `Recommended Assessment URLs`

➡️ The Excel file was used as **input**, while the **recommendation engine was already built and reusable**.

---

## 🚀 API Endpoints

### 🔹 Health Check

```
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

---

### 🔹 Recommend Assessments

```
POST /recommend
```

#### Sample Input

```json
{
  "job_title": "Research AI Intern",
  "job_description": "Building ML and NLP pipelines",
  "skills": ["Python", "Machine Learning", "NLP", "Statistics"],
  "top_k": 5
}
```

#### Sample Output

```json
{
  "query": "Research AI Intern Building ML and NLP pipelines Python Machine Learning NLP Statistics",
  "results": [
    {
      "rank": 1,
      "assessment_id": "A008",
      "name": "Technical Skills Assessments",
      "url": "https://www.shl.com/products/assessments/skills-and-simulations/technical-skills/",
      "score": 1.91
    }
  ]
}
```

---

### 🔹 Pretty Output (Readable Summary)

```
POST /recommend/pretty
```

Returns:

* Structured JSON
* Human-readable recommendation summary

---

## 📁 Project Structure

```
shl-reco-engine/
│
├── src/
│   ├── api.py                  # FastAPI application
│   ├── build_index.py          # FAISS index + embeddings builder
│   └── __pycache__/
│
├── data/
│   ├── catalog.csv             # Raw SHL assessment catalog
│   ├── catalog.pkl             # Metadata used by API
│   └── catalog.index           # FAISS vector index
│
├── SHL_Assessment_Recommendation_Engine_Notebook.ipynb
└── README.md
```

---

## ▶️ How to Run Locally

### 1️⃣ Install Dependencies

```bash
pip install fastapi uvicorn sentence-transformers faiss-cpu pandas numpy
```

### 2️⃣ Start API Server

```bash
python -m uvicorn src.api:app --reload
```

### 3️⃣ Open Swagger UI

```
http://127.0.0.1:8000/docs
```

Use the API interactively to test recommendations.

---

## 🧪 Example Usage

```json
{
  "job_title": "Research AI Intern",
  "job_description": "ML, NLP, experimentation",
  "skills": ["Python", "Machine Learning", "NLP", "Deep Learning"],
  "top_k": 5
}
```

---

## ✅ Key Highlights

* Hybrid ML + rules system
* Explainable ranking decisions
* Real-world hiring relevance
* Excel dataset successfully processed using the same engine
* Easily extensible to new roles and domains

---

## 🏁 Conclusion

This project demonstrates how **semantic search + domain intelligence** can be combined to build a **production-ready recommendation engine** for talent assessment use cases.


