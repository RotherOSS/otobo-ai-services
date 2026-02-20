## Introduction

This project provides a modular, standalone RAG (Retrieval-Augmented Generation) hosting service designed to integrate with the OTOBO ticket system.
It enables easy experimentation and deployment of multiple custom RAG pipelines.
Each RAG module defines its own retrieval/generation logic and API schema, loaded dynamically at runtime.

The system uses **LangGraph**, **LangChain**, and **ChromaDB** to support flexible embedding and LLM-powered generation.
It exposes a REST API via **FastAPI** for ingesting content and interacting with configured RAGs.

---

## Installation and Setup

Clone the repository and set up the environment using Docker Compose:

### 1. Clone the Repository

```bash
git clone git@github.com:RotherOSS/otobo-ai.git
cd otobo-ai
```

### 2. Get a RAG definition

Example RAG definitions are provided under `rag_examples`.
The `simple_rag` is for stand alone development.
If you use this setup with OTOBO, choose the `tfd_rag1`.
It supports Tickets, FAQ and Documentation.

Copy the RAG description to your RAG definition folder.

```bash
cp -r rags_examples/tfd_rag1 rags
```

All RAG definitions placed here are exposed at the web service.
You may tune it to your liking, or create a new one!

### 3. Configure the Environment

Create a `.env` file in the root directory to configure environment variables:

```bash
cp .docker_compose_env_ai .env
```

Edit the `.env` file to set your desired configuration options.

### 4. Start the Containers

Use Docker Compose to build and run the server:

```bash
docker compose up --build --detach
```

The `docker compose.yaml` mounts the local `./rags` directory into the container as `src/rags`, enabling external customization.

---

## API Endpoints

All endpoints are mounted under `/otobo-ai/`, secured with API key authentication via the `get_api_key` dependency.

### Embedding Endpoints

#### `PUT /otobo-ai/embedding/ingest/`

Ingest a **single** data item for embedding.

**Input:**

```json
{
  "type": "documentation",
  "content": [[{ "type": "text", "text": "your content here" }]],
  "embed_content_types": ["text"],
  "store_fulltext": false
}
```

**Response:** `200 OK` or `500 Internal Server Error`

---

#### `PUT /otobo-ai/embedding/ingest-many/`

Ingest a **batch** of data items for embedding.

**Input:**

```json
{
  "type": "faq",
  "content": [
    [
      { "type": "question", "text": "What is OTOBO?" },
      { "type": "answer", "text": "A ticketing system." }
    ],
    [
      { "type": "question", "text": "In what language is OTOBO written?" },
      { "type": "answer", "text": "In Perl." }
    ]
  ],
  "embed_content_types": ["question"],
  "store_fulltext": true,
  "fulltext_types": ["answer"]
}
```

**Response:** `200 OK` or `500 Internal Server Error`

---

### RAG Endpoints

Each registered RAG module is exposed under:

#### `POST /otobo-ai/{rag_name}/invoke`

Run the full RAG process (retrieve and generate).

**Input:**

```json
{
  "input": {
    "question": "What is OTOBO?",
    "do_scoring": true
  },
  "config": {},
  "kwargs": {}
}
```

**Output:**

```json
{
  "output": {
    "question": "What is OTOBO?",
    "generation": "OTOBO is an open-source ticketing system.",
    "score": 0.87
  },
  "metadata": {
    "run_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "feedback_tokens": [
      {
        "key": "score-feedback",
        "token_url": "https://feedback.example.com/token/3fa85f64",
        "expires_at": "2025-04-06T08:05:36.868Z"
      }
    ]
  }
}
```

Each RAG can define its own input/output models, but this is the expected default format.
Note that only the `input` and `output` fields are relevant for usage within OTOBO, the other fields are automatically provided by **LangServe**

---

## Technical Usage: Writing Your Own RAG

Each RAG lives in its own folder under `rags/`, structured as follows:

```
rags/
└── my_custom_rag/
    ├── graph.py
    ├── chains.py           # optional
    ├── io_models.py
    └── prompts/
        └── prompt.txt
```

### Required Files

- **`graph.py`**
  Must define a `graph` object using `StateGraph`, compiled with `graph = workflow.compile()`.

- **`io_models.py`**
  Must define:

  ```python
  class RAGInput(BaseModel): ...
  class RAGOutput(BaseModel): ...
  ```

These models define the request and response schema for your RAG.

### Optional Files

- **`chains.py`**
  Use this to separate LangChain logic or define chains used in your workflow.

- **`prompts/`**
  Store custom prompt templates here. Load them with `Path(__file__).parent / "prompts" / "prompt.txt"`.

### Example: Minimal `graph.py`

```python
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
graph = workflow.compile()
```

### Registration

All RAGs in `rags/` are auto-registered by `register_rags(app)` on startup.
If files or required types are missing, the module will be skipped with a warning.

---

## RAG Configuration and Extension

This project supports dynamic loading of RAG modules at runtime via the `register_rags(app: FastAPI)` function.

### How It Works

At startup, the server scans the `src/rags/` directory for subdirectories containing a `graph.py` file.
Each `graph.py` must define a `graph` object with a `.with_config(config)` method.
These are registered as individual FastAPI routes under `/otobo-ai/{rag_name}`, protected by API key.

Each RAG module must follow this structure:

- A `graph.py` defining the LangGraph workflow (`graph`)
- An `io_models.py` defining the input/output types (`RAGInput` and `RAGOutput`)
- Optionally, `chains.py`, prompt templates, and other helpers

### Repository Structure

Only `rag_examples/` is version-controlled. The `src/rags/` directory is `.gitignore`d so users can safely define custom modules without affecting the repo.

```bash
src/
├── rags/              ← not version controlled
│   ├── simple_rag/    ← copy or create your RAG modules here
│   │   ├── graph.py
│   │   ├── chains.py
│   │   ├── io_models.py
│   │   └── prompts/
│   └── ...
└── ...
rag_examples/          ← reference implementations
├── simple_rag/
└── ...
```

---

## Updating Dependencies

To update the dependencies in a controlled way:

### 1. Unpin Non-Critical Dependencies

In `requirements.txt`, remove version pins from most packages.
Keep pins **only** for critical compatibility fixes.

**Before:**

```txt
uvicorn==0.25.0
fastapi==0.108.0

# critical compatibility fixes
numpy==1.26.4
```

**After:**

```txt
uvicorn
fastapi

# critical compatibility fixes
numpy==1.26.4
```

### 2. Rebuild Containers

```bash
docker compose build
```

### 3. Start the Project

```bash
docker compose up
```

Ensure it runs without version issues.

### 4. Test Everything

Verify ingestion and RAG endpoints still work correctly.

### 5. Check Installed Versions

After confirming stability, inspect the installed versions:

```bash
docker compose run --rm otobo-ai pip list
```

### 6. Pin Final Versions

Use the output to update `requirements.txt` with the final versions used.

**Example:**

```txt
uvicorn==0.34.0
fastapi==0.115.0

# critical compatibility fixes
numpy==1.26.4
```

### 7. Commit the Changes

```bash
git add requirements.txt
git commit -m "Update pinned dependencies"
```
