"""Make a whole Bipnik crash video on one computer (all runs one after another).

Example (quick test):  python crash_agent/make_video.py --scenario wall --runs 3 --quality test
Example (real video):  python crash_agent/make_video.py --scenario random --runs 6 --max-kmh 200 --quality final
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bipnik_post as post  # noqa: E402
from render_run import run_one  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="random")
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--max-kmh", type=int, default=200)
    ap.add_argument("--seed", default="")
    ap.add_argument("--quality", default="final", choices=list(post.QUALITY))
    ap.add_argument("--out", default="build")
    a = ap.parse_args()
    p = post.plan(a.scenario, a.runs, a.max_kmh, a.seed, a.quality)
    print(f"Plan: {p['scenario']} | {p['runs']} runs | max {p['max_kmh']} km/h | seed {p['seed']}", flush=True)
    for i in range(p["runs"]):
        run_one(p, i, a.out)
    final, title = post.finish(p, os.path.join(a.out, "segments"), os.path.join(a.out, "final"), os.path.join(a.out, "work_final"))
    print("FINAL:", final)
    print("TITLE:", title)


if __name__ == "__main__":
    main()
