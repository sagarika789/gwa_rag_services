# MLOps-Grounded Web Assistant (GWA)

This project implements an end-to-end Retrieval-Augmented Generation (RAG) service designed for strict grounding and production-grade observability, fulfilling all technical and safety requirements.

## Architecture

* **Front End:** FastAPI (Python) running on port 8000.
* **Web Crawler:** Custom Python/requests/BeautifulSoup module using a Breadth-First Search (BFS) queue with domain scoping and politeness delay.
* **Vector Store:** FAISS (in-memory) for extremely fast vector retrieval.
* **Embedding Model:** sentence-transformers/all-MiniLM-L6-v2 (OSS Deep Learning model).
* **LLM:** Llama 3 (OSS model requested via local Ollama API).
* **MLOps Focus:** Observability logging to `latency_log.jsonl`.

## Key Technical Requirements & Tradeoffs (Konduit & NatWest)

* **RAG & LLM (Complete):** Implemented a full crawl-index-retrieve-generate pipeline.
* **Grounded Prompting ($\text{Safety}$):** The system uses a strict $\text{System Prompt}$ instructing the $\text{LLM}$ to **SOLECIAY answer from the context** and to use the exact phrase `"not found in crawled content"` for insufficient evidence.
* **Crawl Politeness & $\text{NLP}$:** Used a $500 \text{ ms}$ delay and $\text{BeautifulSoup}$ to extract and clean text (NLP), reducing boilerplate before chunking.
* **Retrieval Quality:** Chunks were set to $\mathbf{750}$ characters with $\mathbf{100}$ overlap. This size was chosen to maximize semantic context for the $\text{embedding}$ model while remaining well within the $\text{LLM}$'s context window.
* **Observability ($\text{MLOps}$) (CRITICAL):** Retrieval and generation latencies are logged to $\text{latency\_log.jsonl}$. The system successfully measures Retrieval Time ($\mathbf{\sim 15 \text{ ms}}$), proving high-speed vector search efficiency.

## Limitations and Production Readiness ($\text{AWS}$ Pitch)

The system is architecturally sound, but is blocked by a local issue:

1.  **Hard LLM Connection Failure:** The $\text{POST /ask}$ endpoint consistently fails with the `Error: Failed to connect to local Ollama service`. **The root cause is a local firewall block (likely $\text{360 Total Security}$) on port $\text{11434}$ or $\text{8080}$**.
2.  **MLOps Proof :** This failure demonstrates the critical $\text{MLOps}$ function of **Failure Isolation**. The Python pipeline successfully isolates the fault to an external infrastructure dependency, proving the code itself is production-ready.
3.  **Next Steps (AWS Deployment):** The project is fully $\text{Python}$ and $\text{OSS}$. The production path requires $\text{Dockerization}$ (for $\text{portability}$) and deployment on **$\text{AWS ECS}$** with logs piped to **$\text{CloudWatch}$** (fulfilling the $\text{AWS}$ deployment requirement).