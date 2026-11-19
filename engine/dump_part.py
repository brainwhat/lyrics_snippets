"""Draft lyric file for one video from the Whisper words: python engine/dump_part.py <vid> [--force]
Takes the Whisper segments of the video's track that fall in [start, end] (videos/videos.json)
and writes them to videos/lyrics/<vid>.txt, one segment per line, for hand editing before the render."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

vid = sys.argv[1]
force = '--force' in sys.argv[2:]
spec = json.loads((ROOT / 'videos' / 'videos.json').read_text())[vid]
track, t0, t1 = spec['track'], spec['start'], spec['end']
out = ROOT / 'videos' / 'lyrics' / f'{vid}.txt'
if out.exists() and not force:
    sys.exit(f'{out.name} already exists, add --force to overwrite')

lines = []
for seg in json.loads((ROOT / 'tracks' / track / 'words.json').read_text()):
    words = [w for w in seg['words'] if t0 - 0.3 <= w['s'] < t1]
    if words:
        lines.append(' '.join(w['w'].strip() for w in words))
out.write_text('\n'.join(lines) + '\n')
print(f'{out.name}: {len(lines)} lines, song {t0}–{t1} s')
