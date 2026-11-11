"""Everything that does not need Blender: run planning, Bipnik overlays, sound design,
final assembly, caption and thumbnail. Uses only numpy, Pillow and ffmpeg."""
import json
import math
import os
import random
import subprocess
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = os.path.join(HERE, "fonts", "LuckiestGuy-Regular.ttf")
W, H = 1080, 1920
FPS = 30
SR = 44100
YELLOW = (255, 210, 52)
WHITE = (255, 255, 255)
NAVY = (30, 42, 58)
RED = (229, 57, 53)
PLATFORMS = "YOUTUBE | TIKTOK | INSTAGRAM | FACEBOOK"

SCENARIOS = ["wall", "bricks", "bollards", "head_on"]
NORMAL_STYLES = ["sedan", "hatch", "pickup", "van", "sports"]
PAINTS = [
    (0.80, 0.05, 0.04), (0.98, 0.70, 0.02), (0.05, 0.25, 0.85), (0.95, 0.35, 0.02),
    (0.10, 0.55, 0.15), (0.92, 0.92, 0.92), (0.45, 0.10, 0.65), (0.02, 0.60, 0.70),
    (0.55, 0.55, 0.58), (0.85, 0.15, 0.45),
]
QUALITY = {
    "test": dict(w=270, h=480, samples=4),
    "draft": dict(w=540, h=960, samples=6),
    "final": dict(w=1080, h=1920, samples=10),
}
TWIST_LABEL = {
    "wall": "TWIST: TINY CAR!",
    "bricks": "TWIST: 3 CARS!",
    "bollards": "TWIST: HIDDEN RAMP!",
    "head_on": "TWIST: TINY VS TRUCK!",
}
OBSTACLE_NAME = {"wall": "Concrete Wall", "bricks": "Brick Wall", "bollards": "Steel Bollards", "head_on": "Head-On Crash"}


# ----------------------------------------------------------------------------- planning
def plan(scenario="random", runs=6, max_kmh=200, seed=None, quality="final"):
    if seed in (None, "", 0, "0"):
        seed = random.randint(1, 999999)
    seed = int(seed)
    rng = random.Random(seed)
    if scenario in (None, "", "random"):
        scenario = rng.choice(SCENARIOS)
    runs = max(3, min(8, int(runs)))
    max_kmh = max(60, min(400, int(max_kmh)))
    n = runs - 1
    styles = NORMAL_STYLES[:]
    rng.shuffle(styles)
    paints = PAINTS[:]
    rng.shuffle(paints)
    theme = rng.choice(["grass", "desert"])
    sun_rot = rng.uniform(0, 2 * math.pi)
    out = []
    for i in range(runs):
        twist = i == runs - 1
        kmh = int(round(max_kmh * (i + 1) / n / 10.0) * 10) if not twist else max_kmh
        out.append(dict(
            index=i, runs=runs, seed=seed * 100 + i, scenario=scenario, twist=twist,
            kmh=kmh, speed_mps=kmh / 3.6,
            style=styles[i % len(styles)], style_b=styles[(i + 2) % len(styles)],
            paint=paints[i % len(paints)], paint_b=paints[(i + 4) % len(paints)],
            theme=theme, sun_rot=sun_rot, quality=QUALITY[quality],
        ))
    return dict(seed=seed, scenario=scenario, runs=runs, max_kmh=max_kmh, quality=quality, run_cfgs=out)


# ----------------------------------------------------------------------------- text images
def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)


def _text(d, xy, txt, size, fill, stroke=8, anchor="mm", stroke_fill=NAVY):
    d.text(xy, txt, font=font(size), fill=fill, stroke_width=stroke, stroke_fill=stroke_fill, anchor=anchor)


def _fit(txt, size, maxw):
    while size > 20 and font(size).getlength(txt) > maxw:
        size -= 4
    return size


def canvas():
    return Image.new("RGBA", (W, H), (0, 0, 0, 0))


def pill(d, cx, cy, txt, size, fill_box, txt_fill, pad=(40, 22)):
    f = font(size)
    tw = f.getlength(txt)
    box = (cx - tw / 2 - pad[0], cy - size / 2 - pad[1], cx + tw / 2 + pad[0], cy + size / 2 + pad[1])
    d.rounded_rectangle(box, radius=28, fill=fill_box, outline=NAVY, width=7)
    d.text((cx, cy + 4), txt, font=f, fill=txt_fill, anchor="mm")


def overlay_watermark(path):
    im = canvas()
    d = ImageDraw.Draw(im)
    _text(d, (40, 150), "@bipnik1", 58, WHITE, stroke=8, anchor="lm")
    _text(d, (42, 206), PLATFORMS, 28, YELLOW, stroke=5, anchor="lm")
    a = np.array(im)
    a[..., 3] = (a[..., 3] * 0.9).astype(np.uint8)
    Image.fromarray(a).save(path)


def overlay_run(path, run_i, runs, kmh):
    im = canvas()
    d = ImageDraw.Draw(im)
    pill(d, W / 2, 330, f"RUN {run_i + 1}/{runs}", 62, RED, WHITE)
    pill(d, W / 2, 1420, f"{kmh} KM/H", 104, NAVY, YELLOW, pad=(50, 26))
    im.save(path)


def overlay_title(path, line1, line2):
    im = canvas()
    d = ImageDraw.Draw(im)
    s1 = _fit(line1, 104, W - 120)
    s2 = _fit(line2, 104, W - 120)
    _text(d, (W / 2, 520), line1, s1, YELLOW, stroke=12)
    _text(d, (W / 2, 650), line2, s2, WHITE, stroke=12)
    im.save(path)


def overlay_label(path, txt, y, size=84, color=YELLOW):
    im = canvas()
    d = ImageDraw.Draw(im)
    _text(d, (W / 2, y), txt, _fit(txt, size, W - 140), color, stroke=11)
    im.save(path)


def end_card(frame_path, path):
    base = Image.open(frame_path).convert("RGB").resize((W, H)).filter(ImageFilter.GaussianBlur(18))
    dark = Image.new("RGB", (W, H), (10, 16, 28))
    im = Image.blend(base, dark, 0.55)
    d = ImageDraw.Draw(im)
    _text(d, (W / 2, 640), "BIPNIK", 230, YELLOW, stroke=16)
    _text(d, (W / 2, 860), "FOLLOW @BIPNIK1", _fit("FOLLOW @BIPNIK1", 100, W - 100), WHITE, stroke=11)
    _text(d, (W / 2, 960), PLATFORMS, _fit(PLATFORMS, 44, W - 100), YELLOW, stroke=7)
    pill(d, W / 2, 1180, "WHICH RUN WAS THE BEST?", _fit("WHICH RUN WAS THE BEST?", 60, W - 220), RED, WHITE)
    _text(d, (W / 2, 1300), "COMMENT BELOW!", 64, WHITE, stroke=9)
    im.save(path)


# ----------------------------------------------------------------------------- ffmpeg helpers
def ff(args):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"] + args
    subprocess.run(cmd, check=True)


ENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(FPS)]


def encode_segment(frames_dir, meta, plan_info, out_mp4, work):
    """PNG frames + run overlays -> 1080x1920 mp4 (no audio)."""
    os.makedirs(work, exist_ok=True)
    i, runs, kmh = meta["index"], meta["runs"], meta["kmh"]
    n = meta["end"] - meta["start"] + 1
    dur = n / FPS
    ov_run = os.path.join(work, f"run{i}.png")
    overlay_run(ov_run, i, runs, kmh)
    ov_slow = os.path.join(work, f"slow{i}.png")
    overlay_label(ov_slow, "SLOW MO", 1240, 76, WHITE)
    s0 = (meta["slow_start"] - meta["start"]) / FPS
    s1 = (meta["slow_end"] - meta["start"]) / FPS
    inputs = ["-framerate", str(FPS), "-start_number", str(meta["start"]), "-i", os.path.join(frames_dir, "f_%04d.png")]
    layers = [(ov_run, 0 if i > 0 else 3.2, dur), (ov_slow, s0 + 0.3, s1)]
    if i == 0:
        ov_t = os.path.join(work, "title.png")
        l1, l2 = plan_info["hook"]
        overlay_title(ov_t, l1, l2)
        layers.append((ov_t, 0, 3.2))
    if meta.get("twist"):
        ov_tw = os.path.join(work, "twist.png")
        overlay_label(ov_tw, TWIST_LABEL[meta["scenario"]], 560, 92, YELLOW)
        layers.append((ov_tw, 0, 2.6))
    for p, _, _ in layers:
        inputs += ["-loop", "1", "-i", p]
    fc = f"[0:v]scale={W}:{H}:flags=lanczos,setsar=1[v0]"
    last = "v0"
    for k, (_, a, b) in enumerate(layers):
        fc += f";[{last}][{k + 1}:v]overlay=0:0:enable='between(t,{a:.3f},{b:.3f})'[v{k + 1}]"
        last = f"v{k + 1}"
    ff(inputs + ["-filter_complex", fc, "-map", f"[{last}]", "-t", f"{dur:.3f}"] + ENC + [out_mp4])


# ----------------------------------------------------------------------------- sound
def _lp(x, n):
    if n <= 1:
        return x
    k = np.ones(n) / n
    return np.convolve(x, k, mode="same")


def synth_audio(metas, end_secs, out_wav, seed=1):
    rng = np.random.default_rng(seed)
    total = sum((m["end"] - m["start"] + 1) for m in metas) / FPS + end_secs
    N = int(total * SR) + SR
    mix = np.zeros(N)
    t_off = 0.0
    vmax = max(m["speed_mps"] for m in metas)
    for m in metas:
        frames = m["frames"]
        nf = len(frames)
        seg_n = int(nf / FPS * SR)
        idx = np.minimum((np.arange(seg_n) * FPS / SR).astype(int), nf - 1)
        v = np.array([fr["v"] for fr in frames])
        ts = np.array([fr["ts"] for fr in frames])
        fnum = np.array([fr["f"] for fr in frames])
        alive = (fnum < m["impact"]).astype(float)
        # engine fades out over 0.4 s (real time) after impact
        env_engine = np.convolve(alive, np.ones(12) / 12, mode="same")
        slow = (ts < 0.9).astype(float)
        vs = _lp(v[idx], 2000)
        sl = _lp(slow[idx], 4000)
        eng_env = _lp(env_engine[idx], 2000)
        freq = (48 + 3.4 * vs) * (1 - 0.55 * sl)
        phase = 2 * np.pi * np.cumsum(freq) / SR
        engine = (0.55 * np.sin(phase) + 0.28 * np.sin(2 * phase + 0.3) + 0.18 * np.sin(3 * phase)
                  + 0.12 * np.sign(np.sin(0.5 * phase)))
        growl = _lp(rng.standard_normal(seg_n), 25) * 0.6
        engine = (engine + growl) * (0.18 + 0.22 * (vs / vmax)) * eng_env * (1 - 0.4 * sl)
        engine = _lp(engine, 3)
        wind = _lp(rng.standard_normal(seg_n), 18) * 0.5 * (vs / vmax) ** 1.5 * (1 - 0.7 * sl) * 0.35
        seg = engine + wind
        # whoosh into slow motion
        ws = int((m["slow_start"] - m["start"]) / FPS * SR)
        wl = int(0.7 * SR)
        if 0 <= ws < seg_n:
            e = np.sin(np.linspace(0, np.pi, wl)) ** 2
            wn = _lp(rng.standard_normal(wl), 9) - _lp(rng.standard_normal(wl), 60)
            end = min(seg_n, ws - wl // 3 + wl)
            a = max(0, ws - wl // 3)
            seg[a:end] += (wn * e)[: end - a] * 0.5
        # impacts
        evs = {}
        for ev in m["events"]:
            key = ev["f"] // 3
            evs[key] = max(evs.get(key, 0), ev["strength"])
        evlist = sorted(evs.items())[:14]
        for key, s in evlist:
            f = key * 3
            k = f - m["start"]
            if k < 0 or k >= nf:
                continue
            is_slow = ts[k] < 0.9
            pos = int(k / FPS * SR)
            L = int((3.0 if is_slow else 1.2) * SR)
            tt = np.arange(L) / SR
            tau = 0.55 if is_slow else 0.16
            thump = np.sin(2 * np.pi * (38 if is_slow else 70) * tt * (1 - 0.3 * tt)) * np.exp(-tt / tau)
            nz = rng.standard_normal(L)
            crunch = (nz - _lp(nz, 6 if not is_slow else 14))
            crunch = _lp(crunch, 2 if not is_slow else 5) * np.exp(-tt / (0.9 if is_slow else 0.25))
            debris = np.zeros(L)
            for _ in range(int(6 + 14 * s)):
                p0 = int(rng.uniform(0.05, 0.9 if not is_slow else 2.4) * SR)
                fr = rng.uniform(1500, 4200) * (0.4 if is_slow else 1.0)
                cl = int(0.05 * SR)
                if p0 + cl < L:
                    tc = np.arange(cl) / SR
                    debris[p0:p0 + cl] += np.sin(2 * np.pi * fr * tc) * np.exp(-tc / 0.012) * rng.uniform(0.2, 0.6)
            hit = (1.0 * thump + 0.75 * crunch + 0.25 * debris) * (0.35 + 0.65 * s)
            e2 = min(seg_n, pos + L)
            seg[pos:e2] += hit[: e2 - pos]
        a = int(t_off * SR)
        mix[a:a + seg_n] += seg[: len(mix) - a]
        t_off += nf / FPS
    # end card pop
    p = int(t_off * SR)
    tt = np.arange(int(0.25 * SR)) / SR
    pop = np.sin(2 * np.pi * (520 + 900 * tt) * tt) * np.exp(-tt / 0.06) * 0.5
    mix[p:p + len(pop)] += pop[: len(mix) - p]
    mix = np.tanh(mix * 1.3)
    mix = mix / (np.max(np.abs(mix)) + 1e-9) * 0.9
    mix = mix[: int(total * SR)]
    left = mix
    right = np.concatenate([np.zeros(12), mix[:-12]])
    st = (np.stack([left, right], axis=1) * 32767).astype(np.int16)
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(st.tobytes())
    return total


# ----------------------------------------------------------------------------- texts
def hook_and_title(p):
    sc, mx = p["scenario"], p["max_kmh"]
    rng = random.Random(p["seed"])
    if sc == "wall":
        hooks = [("CAN THIS CAR", f"SURVIVE {mx} KM/H?"), ("CARS VS WALL", f"UP TO {mx} KM/H!")]
        titles = [f"Can This Car Survive {mx} KM/H?", f"Cars vs Concrete Wall at {mx // (p['runs'] - 1)}-{mx} KM/H"]
    elif sc == "bricks":
        hooks = [("CARS VS", "BRICK WALL!"), ("CAN IT BREAK", "THE BRICKS?")]
        titles = [f"Cars vs Brick Wall at Every Speed", f"Brick Wall vs {mx} KM/H Car"]
    elif sc == "bollards":
        hooks = [("STEEL BOLLARDS", f"VS {mx} KM/H"), ("WILL THE CAR", "SURVIVE THIS?")]
        titles = [f"Steel Bollards vs Every Car Speed", f"Only One Car Beat These Bollards..."]
    else:
        hooks = [("HEAD-ON CRASH", f"AT {mx} KM/H!"), ("WHO WINS", "HEAD-ON?")]
        titles = [f"Head-On Crash at Every Speed", f"Tiny Car vs Giant Truck Head-On"]
    return rng.choice(hooks), rng.choice(titles)


def write_caption(p, title, path):
    desc = {
        "wall": "Every run is faster than the last. Watch what happens to the cars when they hit the wall... and wait for the twist at the end!",
        "bricks": "Cars smash into a brick wall at higher and higher speed. Stay until the last run for the twist!",
        "bollards": "Steel bollards vs cars at increasing speed. The last run is not what you expect!",
        "head_on": "Two cars, one road, higher speed every run. Don't miss the final crash!",
    }[p["scenario"]]
    txt = (f"TITLE: {title}\n\nDESCRIPTION:\n{desc}\n"
           "3D physics crash test animation made in Blender.\n"
           "Follow Bipnik for more crash tests!\n"
           "YouTube: @bipnik1 | TikTok: @bipnik1 | Instagram: @bipnik1 | Facebook: Bipnik\n\n"
           "HASHTAGS: #shorts #carcrash #crashtest #3danimation #blender #satisfying #cars\n\n"
           "NOTE: Made for kids = NO\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)


def make_thumbnail(seg_mp4, meta, hook, path, work, raw_png=None):
    if raw_png and os.path.exists(raw_png):
        im = Image.open(raw_png).convert("RGB").resize((W, H), Image.LANCZOS)
    else:
        t = max(0.0, (meta["slow_start"] - meta["start"] + 22) / FPS)
        fr = os.path.join(work, "thumb_frame.png")
        ff(["-ss", f"{t:.2f}", "-i", seg_mp4, "-frames:v", "1", fr])
        im = Image.open(fr).convert("RGB")
    d = ImageDraw.Draw(im)
    l1, l2 = hook
    _text(d, (W / 2, 420), l1, _fit(l1, 120, W - 100), YELLOW, stroke=14)
    _text(d, (W / 2, 570), l2, _fit(l2, 120, W - 100), WHITE, stroke=14)
    im.save(path, quality=92)


# ----------------------------------------------------------------------------- final assembly
def finish(p, seg_dir, out_dir, work):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    metas = []
    for i in range(p["runs"]):
        with open(os.path.join(seg_dir, f"meta_{i}.json")) as f:
            metas.append(json.load(f))
    hook, title = hook_and_title(p)
    # end card from the last frame of the last run
    last_seg = os.path.join(seg_dir, f"seg_{p['runs'] - 1}.mp4")
    lf = os.path.join(work, "last.png")
    ff(["-sseof", "-0.2", "-i", last_seg, "-frames:v", "1", lf])
    ec = os.path.join(work, "endcard.png")
    end_card(lf, ec)
    END = 3.0
    ec_mp4 = os.path.join(work, "endcard.mp4")
    ff(["-loop", "1", "-i", ec, "-t", f"{END}", "-vf", f"scale={W}:{H},setsar=1,zoompan=z='1+0.0008*on':d=1:s={W}x{H}:fps={FPS}"] + ENC + [ec_mp4])
    lst = os.path.join(work, "list.txt")
    with open(lst, "w") as f:
        for i in range(p["runs"]):
            f.write(f"file '{os.path.abspath(os.path.join(seg_dir, f'seg_{i}.mp4'))}'\n")
        f.write(f"file '{os.path.abspath(ec_mp4)}'\n")
    full = os.path.join(work, "full.mp4")
    ff(["-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", full])
    wav = os.path.join(work, "audio.wav")
    synth_audio(metas, END, wav, seed=p["seed"])
    wm = os.path.join(work, "wm.png")
    overlay_watermark(wm)
    name = f"Bipnik_Crash_{p['scenario']}_{p['seed']}"
    final = os.path.join(out_dir, name + ".mp4")
    ff(["-i", full, "-loop", "1", "-i", wm, "-i", wav,
        "-filter_complex", "[0:v][1:v]overlay=0:0:shortest=1[v]",
        "-map", "[v]", "-map", "2:a",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-maxrate", "9M", "-bufsize", "18M",
        "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
        "-shortest", "-movflags", "+faststart", final])
    write_caption(p, title, os.path.join(out_dir, name + "_caption.txt"))
    best = max(range(p["runs"] - 1), key=lambda i: metas[i]["kmh"])
    make_thumbnail(os.path.join(seg_dir, f"seg_{best}.mp4"), metas[best], hook, os.path.join(out_dir, name + "_thumbnail.jpg"), work,
                   raw_png=os.path.join(seg_dir, f"thumb_{best}.png"))
    return final, title
