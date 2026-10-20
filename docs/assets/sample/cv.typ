#set page(paper: "a4", margin: (x: 1.6cm, y: 1.4cm))

#set text(font: ("Inter", "Helvetica Neue", "Arial"), size: 10pt, lang: "pl")

#show heading.where(level: 2): it => block(above: 1.1em, below: .5em)[#text(size: 11pt, weight: "bold", fill: rgb("#1f4e79"))[#upper(it.body)] #line(length: 100%, stroke: .5pt)]

#align(center)[#text(size: 20pt, weight: "bold")[Jan Kowalski] \ #text(size: 11pt)[Python Backend & GenAI Engineer — FastAPI, LangGraph, RAG] \ #text(size: 9pt)[jan.kowalski\@example.com · +48 600 000 000 · Warszawa · https://www.linkedin.com/in/jan-kowalski · https://github.com/jankowalski]]

== Profil

Backend engineer shipping LLM features to production: retrieval pipelines, evaluation harnesses and async APIs. Four years of Python with FastAPI, PostgreSQL and Docker on AWS. Comfortable owning a service end to end, from schema design to on-call.

== Kluczowe technologie

Python, FastAPI, LangGraph, RAG, PostgreSQL, Docker, AWS

#text(fill: luma(90))[Redis, Celery, SQLAlchemy, pytest, Kubernetes, GitHub Actions]

== Projekty

*DocuMind* #h(1fr) #text(size: 8.5pt, style: "italic")[Python, FastAPI, LangGraph, PostgreSQL, pgvector, Redis, Docker, AWS]

- Cut answer latency p95 from 9.1s to 2.4s with hybrid BM25 + pgvector retrieval and response streaming

- Built an offline eval harness (RAGAS + 300 labelled Q/A) that gated every prompt and model change in CI

- Served \~1.2k daily active users at 99.9% uptime over 6 months

*Pricing API* #h(1fr) #text(size: 8.5pt, style: "italic")[Python, FastAPI, PostgreSQL, SQLAlchemy, Celery, Redis, Kubernetes]

- Sustained 1.8k req/s at p99 \< 60 ms after query-plan tuning and Redis read-through caching

- Migrated a Django monolith endpoint to async FastAPI with zero downtime (shadow traffic + feature flags)

== Języki

Polski (ojczysty), Angielski (C1)

#v(1fr) #text(size: 7pt, fill: luma(120))[Wyrażam zgodę na przetwarzanie moich danych osobowych w celu prowadzenia rekrutacji.]
