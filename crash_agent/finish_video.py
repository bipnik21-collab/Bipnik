"""Join all rendered runs into the final Bipnik video (+ caption and thumbnail). No Blender needed.

Usage: python crash_agent/finish_video.py --scenario wall --runs 6 --max-kmh 200 --seed 123 --out build
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bipnik_post as post  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="random")
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--max-kmh", type=int, default=200)
    ap.add_argument("--seed", default="")
    ap.add_argument("--quality", default="final")
    ap.add_argument("--out", default="build")
    a = ap.parse_args()
    p = post.plan(a.scenario, a.runs, a.max_kmh, a.seed, a.quality)
    final, title = post.finish(p, os.path.join(a.out, "segments"), os.path.join(a.out, "final"), os.path.join(a.out, "work_final"))
    print("FINAL:", final)
    print("TITLE:", title)


if __name__ == "__main__":
    main()
