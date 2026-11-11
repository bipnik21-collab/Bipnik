"""Render ONE run of a Bipnik crash video (needs bpy).

Usage: python crash_agent/render_run.py --index 0 --scenario wall --runs 6 --max-kmh 200 --seed 123 --quality final --out build
Writes build/segments/seg_<i>.mp4 and build/segments/meta_<i>.json
"""
import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bipnik_post as post  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--scenario", default="random")
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--max-kmh", type=int, default=200)
    ap.add_argument("--seed", default="")
    ap.add_argument("--quality", default="final", choices=list(post.QUALITY))
    ap.add_argument("--out", default="build")
    a = ap.parse_args()
    p = post.plan(a.scenario, a.runs, a.max_kmh, a.seed, a.quality)
    run_one(p, a.index, a.out)


def run_one(p, index, out):
    import crash_scene as cs
    cfg = dict(p["run_cfgs"][index])
    seg_dir = os.path.join(out, "segments")
    frames_dir = os.path.join(out, f"frames_{index}")
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    t = time.time()
    print(f"[run {index + 1}/{p['runs']}] {p['scenario']} {cfg['kmh']} km/h twist={cfg['twist']} style={cfg['style']}", flush=True)
    r = cs.render_run(cfg, os.path.abspath(frames_dir))
    print(f"[run {index + 1}] rendered {r['end'] - r['start'] + 1} frames in {time.time() - t:.0f}s", flush=True)
    meta = dict(r, index=index, runs=p["runs"], kmh=cfg["kmh"], twist=cfg["twist"], scenario=p["scenario"])
    hook, _ = post.hook_and_title(p)
    seg = os.path.join(seg_dir, f"seg_{index}.mp4")
    post.encode_segment(frames_dir, meta, dict(hook=hook), seg, os.path.join(out, f"work_{index}"))
    with open(os.path.join(seg_dir, f"meta_{index}.json"), "w") as f:
        json.dump(meta, f)
    # keep one clean frame (no text) for the thumbnail
    tf = meta["slow_start"] + 22
    src = os.path.join(frames_dir, f"f_{tf:04d}.png")
    if os.path.exists(src):
        shutil.copy(src, os.path.join(seg_dir, f"thumb_{index}.png"))
    shutil.rmtree(frames_dir, ignore_errors=True)
    print(f"[run {index + 1}] done -> {seg}", flush=True)


if __name__ == "__main__":
    main()
