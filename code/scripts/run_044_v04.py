"""Fill the cache for schema 0.4: E1 (50 real, dated) and E2 extractor (50 probes, undated/dated).

    python run_044_v04.py gpt-4o gpt-5.5        # API models
    python run_044_v04.py qwen3-30b-a3b         # local model (llama-server must be running)
"""
import sys
import extract_044 as X

models = sys.argv[1:] or X.API_MODELS
real, probes = X.load_jsonl("real_50.jsonl"), X.load_jsonl("synthetic_50.jsonl")
jobs = [(r["headline"], r["date"], m, True) for m in models for r in real]
jobs += [(p["headline"], p["as_of"], m, d) for m in models for d in (False, True) for p in probes]
outs = X.run_many(X.extract_v04, jobs, workers=4 if any(m in X.LOCAL for m in models) else 8)
print(models, len(outs), "calls;", sum("_failed" in o for o in outs), "failed")
