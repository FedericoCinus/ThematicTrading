"""Run the pipeline from a terminal.

    python RUN_FULL_PIPELINE.py --asof 2023-01-02 --until 2025-12-31
    python RUN_FULL_PIPELINE.py --asof 2023-01-02 --until 2025-12-31 --experiment filings_basket
    python RUN_FULL_PIPELINE.py --asof 2023-01-02 --until 2025-12-31 --set monitor.entry.z_threshold=2.0

Everything it does is `pipeline.run`; this only turns flags into the environment variables
`config.py` already reads, so the notebook and the terminal cannot drift apart.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ======================================================================================
# WHAT A RUN IS
#
#   IN    --asof        the day the pipeline pretends it is. Detection and the basket never
#                       read past it: this is what keeps the whole chain point-in-time.
#   IN    --until       run the monitor forward to here and backtest what it says.
#                       Leave it out and the run stops at the portfolio.
#   IN    --experiment  which file in experiments/ defines the pipeline      default: default
#   IN    --set         one override, repeatable:  --set basket.top_n=20
#
#   OUT   data/processed/{themes,signal,weights,backtest}/{run id}.parquet
#                                                        .diag.parquet · .json
#   OUT   one line per theme on stdout: return, benchmark, sharpe, max drawdown
#   OUT   --plot  data/processed/backtest/{run id}.html — the curves, since a
#                 terminal cannot show a chart
#
#   A SWEEP is this script in a shell loop. Nothing to name: the run id carries a digest of the
#   configuration, so each z writes its own files and a repeat replaces only its own.
#       for z in 0.5 1.0 2.0; do
#           python RUN_FULL_PIPELINE.py --asof 2023-01-02 --until 2025-12-31 \
#                         --set monitor.entry.z_threshold=$z
#       done
# ======================================================================================


def main(argv: list[str] | None = None) -> int:
    """Parse the flags, set the environment, run the pipeline.

    INPUT   argv   command-line arguments; None reads sys.argv
    OUTPUT  0 when the run finished, 1 when the environment is incomplete
    """
    parse = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parse.add_argument("--asof", required=True, help="the day the pipeline pretends it is")
    parse.add_argument("--until", help="run the monitor forward to here and backtest")
    parse.add_argument("--experiment", help="a file in experiments/ (without .yaml)")
    parse.add_argument("--set", action="append", default=[], metavar="a.b=c",
                       help="override one value; repeatable")
    parse.add_argument("--no-save", action="store_true", help="write nothing to data/processed")
    parse.add_argument("--plot", action="store_true",
                       help="also write the equity curves as an HTML file")
    args = parse.parse_args(argv)

    if args.experiment:
        os.environ["TT_EXPERIMENT"] = args.experiment
    if args.set:
        os.environ["TT_OVERRIDES"] = " ".join(args.set)

    import config
    import pipeline
    import progress

    config.load_env()
    print(f"experiment {os.environ.get('TT_EXPERIMENT', 'default')}   run {config.run_id()}")
    if not pipeline.check_environment():
        return 1

    started = time.monotonic()
    out = pipeline.run(args.asof, until=args.until, save=not args.no_save)
    print(f"\n⏱  whole run   {progress.clock(time.monotonic() - started)}")

    print("\nframes")
    for name, frame in out.items():
        if hasattr(frame, "shape"):
            print(f"   {name:22} {frame.shape}")

    if len(out.get("performance", [])):
        _performance(out["performance"], params := pipeline.params("backtest"))
        if args.plot:
            print(f"\nplot   {_plot(out['curves'], params['benchmark'], config.run_id())}")
    return 0


def _performance(performance, params: dict) -> None:
    """One line per theme: what it made, what the benchmark made over the same window."""
    bench = params["benchmark"]
    print(f"\n   {'theme':28} {'return':>9} {bench:>9} {'sharpe':>7} {'vs':>7} {'max dd':>8}")
    for r in performance.sort("total_return", descending=True).iter_rows(named=True):
        print(f"   {r['theme'][:28]:28} {r['total_return']:>+9.1%} {r['bench_total_return']:>+9.1%}"
              f" {r['sharpe']:>7.2f} {r['bench_sharpe']:>7.2f} {r['max_dd']:>+8.1%}")


def _plot(curves, benchmark: str, run: str) -> Path:
    """Every theme's equity curve against the benchmark, written as a standalone HTML file.

    INPUT   curves      [theme, date, level, bench]
            benchmark   its name, for the legend
            run         the run id, which names the file
    OUTPUT  the path written — a terminal cannot show a chart, so it leaves one behind
    """
    import plotly.graph_objects as go
    import polars as pl

    import config

    figure = go.Figure()
    for theme in curves["theme"].unique().sort():
        one = curves.filter(pl.col("theme") == theme).sort("date")
        figure.add_scatter(x=one["date"], y=one["level"], name=theme)
        figure.add_scatter(x=one["date"], y=one["bench"], name=f"{theme} · {benchmark}",
                           line=dict(dash="dot"), opacity=.45)
    figure.update_layout(title=f"{run} — level, 1.0 at entry", height=480)

    path = config.PROC / "backtest" / f"{run}.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
