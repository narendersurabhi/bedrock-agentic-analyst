# Bedrock Agentic Analyst — RAG + Tools (Synthetic)

Problem
Analysts need fast, sourced summaries of potential FWA cases.

Approach
- Retrieve: Bedrock Knowledge Base or OpenSearch index over synthetic evidence.
- Tools: claim lookup, provider profile, policy snippets.
- Agent: LangGraph-style planner → calls tools → writes a brief with citations.

Quickstart
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export BEDROCK_REGION=us-east-1
python data/synthetic_evidence.py
python src/ingest/opensearch_index.py
python src/agent/run.py --query "Pattern on provider 123?"

Outputs
- `out/brief.md` with inline citations.
- `logs/trace.json` tool call trace.

Disclaimer
Synthetic data only. No employer IP.
