"""E2 controls: direct knowledge question (100 calls) and naive prompt undated/dated (200 calls)."""
import extract_044 as X

probes = X.load_jsonl("synthetic_50.jsonl")
jobs = [(p["headline"], p["as_of"], m) for m in X.MODELS for p in probes]
ans = X.run_many(X.knowledge, jobs, workers=8)
njobs = [(p["headline"], p["as_of"], m, d) for m in X.MODELS for d in (False, True) for p in probes]
outs = X.run_many(X.extract_naive, njobs, workers=8)
print(len(ans), "knowledge answers;", len(outs), "naive extractions;", sum("_failed" in o for o in outs), "failed")
