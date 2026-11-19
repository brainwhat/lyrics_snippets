"""GigaAM-v3 e2e CTC (MLX) word times for a Russian vocal stem: python engine/transcribe_gigaam.py <vocals.wav> <out.json>
Writes [{s, e, text, words:[{w, s, e, p}]}] like transcribe.py (Whisper), so ./snip map, phrases and render read it unchanged.
Run through snip.py (GIGAAM): uv with gigaam-mlx pinned to a commit. Weights: HF aystream/GigaAM-v3-e2e-ctc-mlx.

GigaAM is trained on short utterances, so the stem is cut into pieces of at most 20 s at the quietest 0.2 s
of the last 6 s (Silero VAD finds no speech in rap over a beat). A CTC token's time is its encoder frame (40 ms);
a word starts at its first token and ends at its last token + 1 frame, never after the next word starts.
CTC fires a little after the sound: on 670 words of 8 tracks the starts were a median 0.10 s later than
Whisper's, which the existing videos are tuned to, so every time is moved LEAD earlier."""
import json, subprocess, sys
import numpy as np
import mlx.core as mx
from gigaam_mlx import load_model
from gigaam_mlx.audio import compute_mel

SR = 16000
GAP = 0.6                                          # a pause this long starts a new line in words.json
LEAD = 0.1                                         # CTC delay against Whisper word starts
src, out = sys.argv[1:3]
y = np.frombuffer(subprocess.run(['ffmpeg', '-v', 'error', '-i', src, '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'],
                                 capture_output=True, check=True).stdout, np.float32)
model, tok = load_model('ctc')
blank = model.num_classes - 1


def pieces(y, limit=20.0, look=6.0):
    res, a = [], 0
    while len(y) - a > limit * SR:
        lo, hi = a + int((limit - look) * SR), a + int(limit * SR)
        w = int(0.2 * SR)
        e = np.convolve(y[lo:hi] ** 2, np.ones(w), 'valid')
        cut = lo + int(np.argmin(e)) + w // 2
        res.append((a, y[a:cut]))
        a = cut
    return res + [(a, y[a:])]


words = []                                         # [text, start, end, [token probs]]
for a, p in pieces(y):
    if len(p) < SR // 10:
        continue
    enc, n = model.encode(mx.array(compute_mel(p)[np.newaxis]))
    n = int(n.item()) if hasattr(n, 'item') else int(n)
    probs = mx.softmax(model.head(enc)[0, :n, :], axis=-1)
    labels = mx.argmax(probs, axis=-1)
    conf = mx.max(probs, axis=-1)
    mx.eval(labels, conf)
    labels, conf = labels.tolist(), conf.tolist()
    step = len(p) / SR / n
    prev = blank
    for t, k in enumerate(labels):
        if k != blank and k != prev:
            piece, ts = tok.id_to_piece(k), a / SR + t * step
            if piece.startswith('▁') or not words or ts - words[-1][2] > GAP:
                words.append([piece.lstrip('▁'), ts, ts + step, [conf[t]]])
            else:
                words[-1][0] += piece
                words[-1][2] = ts + step
                words[-1][3].append(conf[t])
        prev = k

ws = []
for i, (w, s, e, c) in enumerate(words):
    if not w.strip():
        continue
    if i + 1 < len(words):
        e = min(e, words[i + 1][1])
    s, e = max(0.0, s - LEAD), max(0.0, e - LEAD)
    ws.append(dict(w=w.strip(), s=round(s, 3), e=round(max(e, s + 0.04), 3), p=round(float(np.mean(c)), 3)))

segs = []
for w in ws:
    new = not segs or w['s'] - segs[-1]['words'][-1]['e'] > GAP or segs[-1]['words'][-1]['w'][-1] in '.!?'
    if new:
        segs.append(dict(s=w['s'], e=w['e'], text='', words=[]))
    segs[-1]['words'].append(w)
    segs[-1]['e'] = w['e']
for s in segs:
    s['text'] = ' '.join(w['w'] for w in s['words'])
json.dump(segs, open(out, 'w'), ensure_ascii=False, indent=1)
print(len(segs), 'segments,', len(ws), 'words')
