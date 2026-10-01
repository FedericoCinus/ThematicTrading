"""Fill the 0.44 cache for a local model (llama-server must be running): E1 + all three E2 requests."""
import sys
import extract_044 as X

model = sys.argv[1] if len(sys.argv) > 1 else "qwen3-30b-a3b"
real, probes = X.load_jsonl("real_50.jsonl"), X.load_jsonl("synthetic_50.jsonl")
jobs = [(r["headline"], r["date"], model, True) for r in real]
jobs += [(p["headline"], p["as_of"], model, d) for d in (False, True) for p in probes]
outs = X.run_many(X.extract, jobs, workers=4)
naive = X.run_many(X.extract_naive, [(p["headline"], p["as_of"], model, d) for d in (False, True) for p in probes], workers=4)
know = X.run_many(X.knowledge, [(p["headline"], p["as_of"], model) for p in probes], workers=4)
print(model, len(outs), "extractions,", len(naive), "naive,", len(know), "answers;",
      sum("_failed" in o for o in outs + naive), "failed")
