# AI Usage Disclosure

## Development tools used

The team used OpenAI Codex/ChatGPT and Anthropic Claude as development
assistants during the project. Git history identifies AI-assisted commits where
co-author information was recorded. The production RAG system separately uses
Groq-hosted language models for answer generation and MLflow evaluation; those
runtime models are part of the implemented system rather than development
assistants.

## How development AI was used

Development assistants helped with:

- planning the numbered Milestone 1 and Milestone 2 implementation phases;
- drafting and reviewing Python, FastAPI, Docker, GitHub Actions, Azure shell,
  monitoring, and test changes;
- diagnosing provider errors, retrieval latency, container memory use, Render
  out-of-memory failures, and Azure startup/secret issues;
- proposing unit and integration test cases;
- organizing MLflow experiment comparisons and interpreting tracked metrics;
- drafting README, runbook, handoff, report, and evidence documentation; and
- checking repository consistency, secret-safety rules, and submission
  requirements.

## Human review and verification

AI output was not treated as proof that the system worked. Team members:

- reviewed changes before committing them;
- ran Ruff and pytest locally and through GitHub Actions;
- built and inspected the Docker image;
- started a local container and verified its Docker health status;
- called `/health`, `/docs`, and `/predict` locally and on the deployed service;
- inspected request metrics and privacy-safe structured logs;
- reviewed MLflow runs, nested traces, datasets, judges, and artifacts;
- manually tested recursive and fixed retrieval paths and the Gradio interface;
- measured container memory and cold-start behavior; and
- captured screenshots and run/deployment identifiers as evidence.

The previously verified submission commit `0e636dd` passed the automated lint,
test, and Docker-build jobs. Live-service claims in the repository are backed
by recorded public smoke tests rather than by AI-generated descriptions alone.

## Data, privacy, and credentials

API keys are loaded from local `.env` files or Azure Container Apps secrets.
They are excluded from Git and the Docker build context. Keys and secret values
were not intentionally included in prompts, documentation, screenshots, test
fixtures, or committed logs. Monitoring records route, status, request ID, and
duration only; it does not record legal questions, generated answers, headers,
or credentials.

## Limitations

Development assistants can produce incorrect code, commands, explanations, or
citations. Their suggestions were therefore subject to tests and manual review.
External services such as Pinecone, Groq, GitHub Actions, Docker Hub, and Azure
can also change independently of generated code, so the team verified the
actual services and documented failures and corrective actions. Responsibility
for the submitted work and its claims remains with the student team.
