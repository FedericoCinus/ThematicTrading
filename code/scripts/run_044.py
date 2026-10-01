"""Fill the 0.44 GPT cache: E1 (50 real, dated) and E2 (50 synthetic, undated and dated), both models."""
import extract_044 as X

real = X.load_jsonl("real_50.jsonl")
probes = X.load_jsonl("synthetic_50.jsonl")
jobs = [(r["headline"], r["date"], m, True) for m in X.MODELS for r in real]
jobs += [(p["headline"], p["as_of"], m, d) for m in X.MODELS for d in (False, True) for p in probes]
outs = X.run_many(X.extract, jobs, workers=8)
failed = [(j[2], j[0][:50]) for j, o in zip(jobs, outs) if "_failed" in o]
print(len(outs), "calls;", len(failed), "failed", failed[:5])
