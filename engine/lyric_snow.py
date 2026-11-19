"""Lyric videos in the «Игра рукой — лирик снег» look: the render core of lyrics_snippets.

One long hazy shot slowed to 0.9×, milky grade with a glow over the upper third, small Arial
text that smears in word by word and fades to pale. Rendered 4:3 (1440×1080) and boxed into
9:16 (1080×1920) with black bars.

Inputs (see README.md): videos/videos.json (one spec per video), videos/lyrics/<vid>.txt (one sung
line per line), tracks/<track>/{track.json, source.wav, stems/, words.json}, footage/clips/.
Normally driven through ./snip; directly:

  python3 -I engine/lyric_snow.py phrases dora1
  python3 -I engine/lyric_snow.py still dora1 --t=8 [--box=1] [--out=path.jpg]
  python3 -I engine/lyric_snow.py video dora1
"""
import json, os, re, subprocess, sys, difflib
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACKS_DIR = os.path.join(ROOT, 'tracks')
VIDEOS_DIR = os.path.join(ROOT, 'videos')
STOCK = os.path.join(ROOT, 'footage', 'clips')
OUT = os.path.join(ROOT, 'output')
CACHE = os.path.join(ROOT, 'cache')
W, H, FPS = 1440, 1080, 30
SUP = '/System/Library/Fonts/Supplemental/'

TRACKS = {}   # id → (title for file names, Whisper language), from tracks/<id>/track.json
VIDEOS = {}   # vid → spec dict from videos/videos.json
TINT = {
    'cold':  ([0.94, 0.98, 1.04], [0.95, 0.96, 0.98]),     # the original snow
    'cream': ([1.03, 1.00, 0.94], [0.97, 0.96, 0.93]),
    'steel': ([0.92, 0.97, 1.05], [0.90, 0.92, 0.95]),
}
SPEED = 0.9
ONSET = {}   # (vid, word index) → song s, for words Whisper places badly


def load_videos():
    for tid in sorted(os.listdir(TRACKS_DIR)):
        meta = os.path.join(TRACKS_DIR, tid, 'track.json')
        if os.path.exists(meta):
            m = json.load(open(meta))
            TRACKS[tid] = (m['title'], m.get('lang', 'ru'))
    VIDEOS.update(json.load(open(os.path.join(VIDEOS_DIR, 'videos.json'))))
    for vid, v in VIDEOS.items():
        for i, t in v.get('onsets', {}).items():
            ONSET[(vid, int(i))] = t


def track_file(track, *parts):
    return os.path.join(TRACKS_DIR, track, *parts)


def lyrics_file(vid):
    return os.path.join(VIDEOS_DIR, 'lyrics', f'{vid}.txt')


def vocals(track):
    return track_file(track, 'stems', 'vocals.wav')


# ---------------------------------------------------------------- timing
NUM = {'11': 'одиннадцать', '10': 'десять'}


def norm(w):
    w = re.sub(r'[^0-9a-zа-я]', '', w.lower().replace('ё', 'е'))
    return NUM.get(w, w)


_DB = {}


def vocal_db(track):
    """Vocal-stem level in dB, 100 values per second of the song."""
    if track not in _DB:
        r = subprocess.run(['ffmpeg', '-v', 'error', '-i', vocals(track), '-ac', '1', '-ar', '8000',
                            '-f', 'f32le', '-'], capture_output=True, check=True)
        v = np.frombuffer(r.stdout, np.float32)
        v = v[:len(v) // 80 * 80].reshape(-1, 80)
        _DB[track] = 10 * np.log10((v ** 2).mean(1) + 1e-12)
    return _DB[track]


def snap_line_start(track, t):
    """Whisper puts a line's first word up to ~0.5 s early, on the reverb tail of the previous
    line. Lines start right after a silent gap in the vocal stem (< −46 dB for ≥ 80 ms): take the
    gap ending nearest to t (−0.3…+0.9 s) and return the first frame after it above −35 dB."""
    db = vocal_db(track)
    floor = np.percentile(db, 60)                      # quiet records sit lower than −46
    q, on = min(-46, floor - 6), min(-35, floor + 4)
    i0, i1 = max(0, int((t - 0.3) * 100)), int((t + 0.9) * 100)
    quiet = db[i0:i1] < q
    best = None
    k = 0
    while k < len(quiet):
        if quiet[k]:
            e = k
            while e < len(quiet) and quiet[e]:
                e += 1
            if e - k >= 8:
                end = i0 + e
                if best is None or abs(end / 100 - t) < abs(best / 100 - t):
                    best = end
            k = e
        else:
            k += 1
    if best is None:
        return t
    j = best
    while j < min(i1 + 30, len(db)) and db[j] <= on:
        j += 1
    return j / 100


HOLD, LAG = 0.8, 0.45    # min time a line stays after its last word; max delay this puts on the next line


def load_phrases(vid):
    """[[start, end, [(word, t), ...]], ...] in clip time."""
    v = VIDEOS[vid]
    track, t0, t1 = v['track'], v['start'], v['end']
    lines = [l.strip() for l in open(lyrics_file(vid), encoding='utf-8') if l.strip()]
    ref = [(w['w'], w['s']) for seg in json.load(open(track_file(track, 'words.json')))
           for w in seg['words'] if t0 - 0.3 < w['s'] < t1]
    user = [(i, w) for i, l in enumerate(lines) for w in l.split()]
    a = [norm(w) for _, w in user]
    b = [norm(w) for w, _ in ref]
    times = [None] * len(user)
    for blk in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            times[blk.a + k] = ref[blk.b + k][1]
    for (v, k), t in ONSET.items():
        if v == vid:
            times[k] = t
    known = [i for i, t in enumerate(times) if t is not None]
    if not known:
        raise ValueError(f'{vid}: no word of the lyrics matches Whisper in {t0}-{t1} s; '
                         f'check the range or run ./snip retime, or pin words in "onsets"')
    for i in range(len(times)):
        if times[i] is None:
            lo = max([k for k in known if k < i], default=None)
            hi = min([k for k in known if k > i], default=None)
            if lo is None:
                times[i] = times[hi] - 0.3 * (hi - i)
            elif hi is None:
                times[i] = times[lo] + 0.3 * (i - lo)
            else:
                times[i] = times[lo] + (times[hi] - times[lo]) * (i - lo) / (hi - lo)
    lines_t = []
    for li in range(len(lines)):
        ks = [k for k, (i, _) in enumerate(user) if i == li]
        ts = [times[k] for k in ks]
        if (vid, ks[0]) not in ONSET:
            ts[0] = snap_line_start(track, ts[0])
        for j in range(1, len(ts)):                    # keep the words in order after the snap
            ts[j] = max(ts[j], ts[j - 1] + 0.12)
        lines_t.append([(user[k][1], t) for k, t in zip(ks, ts)])
    out = [[ws[0][1] - t0, None, [(w, t - t0) for w, t in ws]] for ws in lines_t]
    for k in range(len(out)):
        last = out[k][2][-1][1]
        if k + 1 == len(out):
            out[k][1] = min(t1 - t0 - 0.2, last + 3.2)
        else:                                          # let the last word be read, delay the next line a little
            nxt = out[k + 1][0]
            out[k][1] = out[k + 1][0] = max(nxt, min(last + HOLD, nxt + LAG))
    return [p for p in out if p[1] - p[0] >= 0.5]     # words at the cut would only flash


# ---------------------------------------------------------------- footage
def clip_reader(clip, t_in, n, zoom0=1.0, zoom1=1.0, speed=1.0):
    """n frames at FPS from t_in, centre-cropped to 4:3, slow push from zoom0 to zoom1."""
    src = os.path.join(STOCK, f'pexels_{clip}.mp4')
    big = (int(W * 1.12) // 2 * 2, int(H * 1.12) // 2 * 2)
    vf = f"setpts=PTS/{speed},fps={FPS},scale=-2:{big[1]},crop={big[0]}:{big[1]}"
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', f'{t_in:.3f}', '-i', src, '-vf', vf,
                          '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL)
    last = None
    for i in range(n):
        buf = p.stdout.read(big[0] * big[1] * 3)
        if len(buf) == big[0] * big[1] * 3:
            last = np.frombuffer(buf, np.uint8).reshape(big[1], big[0], 3)
        z = zoom0 + (zoom1 - zoom0) * i / max(1, n - 1)
        im = Image.fromarray(last)
        cw, ch = big[0] / z, big[1] / z
        x0, y0 = (big[0] - cw) / 2, (big[1] - ch) / 2
        im = im.resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + cw, y0 + ch))
        yield np.asarray(im).astype(np.float32) / 255
    p.kill()


# ---------------------------------------------------------------- grade
YY = np.linspace(0, 1, H, dtype=np.float32)[:, None]
XX = np.linspace(0, 1, W, dtype=np.float32)[None, :]
LUM = np.array([0.30, 0.59, 0.11], np.float32)
GLOW = np.exp(-(((XX - 0.5) / 0.45) ** 2 + ((YY - 0.16) / 0.30) ** 2))


def grain(rng, amt):
    return rng.normal(0, amt, (H // 2, W // 2, 1)).repeat(2, 0).repeat(2, 1).astype(np.float32)


def grade(f, rng, tint):
    """Milky haze: lifted blacks, tinted, white glow over the upper third (lyric3.grade_snow)."""
    mul, haze = (np.array(x, np.float32) for x in TINT[tint])
    lum = f @ LUM
    f = f * 0.5 + lum[..., None] * 0.5
    f = 0.13 + f * 0.82
    f = f * mul
    a = (0.55 * GLOW + 0.15 * np.clip(1 - YY * 1.6, 0, 1))[..., None]
    return np.clip(f * (1 - a) + haze * a + grain(rng, 0.008), 0, 1)


# ---------------------------------------------------------------- text
FONT = ImageFont.truetype(SUP + 'Arial.ttf', 44)
INK = {'cold': ([38, 52, 70], [112, 132, 152]), 'cream': ([60, 48, 36], [140, 126, 108]),
       'steel': ([30, 38, 50], [100, 112, 128])}


def wrap(words, font, maxw):
    lines, cur = [], []
    for w in words:
        test = ' '.join(x for x, _ in cur + [w])
        if cur and font.getlength(test) > maxw:
            lines.append(cur); cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(cur)
    return lines


def smear_word(w, font, col, alpha, sm, pad=40):
    """A word image with a horizontal smear sm∈[0,1] (the @md8kln reveal)."""
    sm, alpha = float(sm), float(alpha)
    wimg = Image.new('RGBA', (int(font.getlength(w)) + 2 * pad, 90), (0, 0, 0, 0))
    ImageDraw.Draw(wimg).text((pad, 12), w, font=font, fill=tuple(col) + (int(255 * alpha),))
    if sm > 0.01:
        wimg = wimg.filter(ImageFilter.GaussianBlur(0.6 + 3 * sm))
        arr = np.asarray(wimg).astype(np.float32)
        acc = np.zeros_like(arr)
        n = int(3 + 14 * sm)
        for j in range(n):
            acc += np.roll(arr, int((j - n / 2) * 4.4 * sm), axis=1)
        wimg = Image.fromarray(np.clip(acc / n, 0, 255).astype(np.uint8))
    return wimg


SNOW_X = [0.62, 0.50, 0.58, 0.44, 0.60, 0.52, 0.64, 0.48]


def draw(t, phrases, tint):
    layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    c0, c1 = (np.array(c) for c in INK[tint])
    for k, (s, e, ws) in enumerate(phrases):
        if not (s - 0.05 <= t <= e + 0.05):
            continue
        out = np.clip((t - (e - 0.30)) / 0.30, 0, 1)
        lines = wrap(ws, FONT, 760)
        cx = W * SNOW_X[k % len(SNOW_X)]
        y = 210 - 27 * (len(lines) - 1)
        for ln in lines:
            full = ' '.join(w for w, _ in ln)
            x = min(max(cx - FONT.getlength(full) / 2, 90), W - 90 - FONT.getlength(full))
            for w, wt in ln:
                wt = max(wt, s)                                     # a line held back shows its early words at once
                a = np.clip((t - wt + 0.05) / 0.28, 0, 1)
                if a > 0:
                    sm = max(1 - a, out)
                    alpha = a if out == 0 else 1 - out * 0.85
                    age = np.clip((t - wt) / 2.2, 0, 1)                 # ink → pale
                    col = (c0 + (c1 - c0) * age).astype(int)
                    layer.alpha_composite(smear_word(w, FONT, col, alpha, sm), (int(x - 40), int(y - 12)))
                x += FONT.getlength(w + ' ')
            y += 54
    return layer


def compose(bg, layer):
    im = Image.fromarray((bg * 255).astype(np.uint8)).convert('RGBA')
    im.alpha_composite(layer)
    return im.convert('RGB')


def still(vid, t):
    v = VIDEOS[vid]
    clip, cin, tint = v['clip'], v['clip_in'], v['tint']
    bg = grade(next(clip_reader(clip, cin + t * SPEED, 1, speed=SPEED)), np.random.default_rng(1), tint)
    return compose(bg, draw(t, load_phrases(vid), tint))


# ---------------------------------------------------------------- render
STEM = {'mix': None, 'inst': 'no_vocals.wav', 'voc': 'vocals.wav'}


def read_pcm(path, t0, dur):
    r = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t0:.3f}', '-t', f'{dur:.3f}', '-i', path,
                        '-ar', '44100', '-ac', '2', '-f', 'f32le', '-'], capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.float32).reshape(-1, 2)


def premix(vid):
    """cache/premix/<vid>.wav: the song from start to end, switching between the mix and the
    Demucs stems at the times in the spec's "audio" (30 ms crossfades), so an intro or an ending
    carries no sound of a neighbouring line; levelled to peak at −1 dBFS, "fade" s at the end."""
    spec = VIDEOS[vid]
    track, t0, t1 = spec['track'], spec['start'], spec['end']
    pieces = spec.get('audio') or [[t0, 'mix']]
    dur = t1 - t0
    src = {k: track_file(track, 'source.wav') if v is None else track_file(track, 'stems', v)
           for k, v in STEM.items()}
    pcm = {k: read_pcm(src[k], t0, dur) for k in {p[1] for p in pieces}}
    n = min(len(v) for v in pcm.values())
    out = np.zeros((n, 2), np.float32)
    xf = int(0.03 * 44100)
    for k, (ts, name) in enumerate(pieces):
        a = int((ts - t0) * 44100)
        b = int((pieces[k + 1][0] - t0) * 44100) if k + 1 < len(pieces) else n
        env = np.zeros(n, np.float32)
        env[a:b] = 1
        if k:
            env[a:a + xf] = np.linspace(0, 1, xf)
        if k + 1 < len(pieces):
            env[b:b + xf] = np.linspace(1, 0, xf)[:max(0, min(xf, n - b))]
        out += pcm[name][:n] * env[:, None]
    gain = min(10 ** (-1 / 20) / float(np.abs(out).max()), 10 ** (12 / 20))
    out *= gain
    i0 = int(0.01 * 44100)
    out[:i0] *= np.linspace(0, 1, i0)[:, None]
    f = int(float(spec.get('fade', 1.2)) * 44100)
    out[n - f:] *= np.linspace(1, 0, f)[:, None] ** 1.5
    os.makedirs(os.path.join(CACHE, 'premix'), exist_ok=True)
    wav = os.path.join(CACHE, 'premix', f'{vid}.wav')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', '44100', '-ac', '2', '-i', '-',
                    '-c:a', 'pcm_s16le', wav], input=out.tobytes(), check=True)
    return wav


BOX = 'scale=1080:810:flags=lanczos,pad=1080:1920:0:555:black'


def out_name(vid, ext):
    """output/<title>/<title> — лирик <n> <ext>; n is the vid's trailing number."""
    title = TRACKS[VIDEOS[vid]['track']][0]
    n = re.search(r'(\d+)$', vid).group(1)
    os.makedirs(os.path.join(OUT, title), exist_ok=True)
    return os.path.join(OUT, title, f'{title} — лирик {n} {ext}')


def video(vid):
    v = VIDEOS[vid]
    t0, t1, clip, cin, tint = v['start'], v['end'], v['clip'], v['clip_in'], v['tint']
    phrases = load_phrases(vid)
    dur = t1 - t0
    n = int(round(dur * FPS))
    out = out_name(vid, '9x16.mp4')
    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                            '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-i', premix(vid),
                            '-vf', f'fade=t=out:st={dur - 0.6:.3f}:d=0.6,{BOX}',
                            '-c:v', 'libx264', '-preset', 'slow', '-crf', '17', '-pix_fmt', 'yuv420p',
                            '-c:a', 'aac', '-b:a', '256k', '-shortest', '-movflags', '+faststart', out],
                           stdin=subprocess.PIPE)
    rng = np.random.default_rng(7)
    for i, bg in enumerate(clip_reader(clip, cin, n, 1.0, 1.10, SPEED)):
        enc.stdin.write(compose(grade(bg, rng, tint), draw(i / FPS, phrases, tint)).tobytes())
    enc.stdin.close(); enc.wait()
    print(out)


if __name__ == '__main__':
    load_videos()
    mode, vid = sys.argv[1], sys.argv[2]
    opt = dict(a[2:].split('=', 1) for a in sys.argv[3:] if a.startswith('--'))
    if mode == 'still':
        im = still(vid, float(opt.get('t', 8)))
        if opt.get('box'):                             # the 9:16 frame, for the cover
            box = Image.new('RGB', (1080, 1920))
            box.paste(im.resize((1080, 810), Image.LANCZOS), (0, 555))
            im = box
        os.makedirs(os.path.join(CACHE, 'stills'), exist_ok=True)
        im.save(opt.get('out', os.path.join(CACHE, 'stills', f'{vid}.jpg')), quality=92)
    elif mode == 'phrases':
        for s, e, ws in load_phrases(vid):
            print(f'{s:6.2f} {e:6.2f}  ' + ' '.join(f'{w}@{t:.2f}' for w, t in ws))
    else:
        video(vid)
