"""Re-time one stretch of a track with Whisper, prompted with the known lyric:
python engine/window_words.py <track> <t0> <t1> <prompt.txt>
Runs Whisper on the vocal stem between t0 and t1 with the text as initial prompt and replaces
the words of tracks/<track>/words.json inside [t0, t1] with the new ones (song time). The first
run keeps the untouched Whisper output as tracks/<track>/words.orig.json."""
import json, sys, subprocess
from pathlib import Path
import numpy as np
import mlx_whisper

ROOT = Path(__file__).resolve().parent.parent
SR = 16000
track, t0, t1, prompt = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), Path(sys.argv[4]).read_text()
src = ROOT / 'tracks' / track / 'stems' / 'vocals.wav'
y = np.frombuffer(subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(t0), '-t', str(t1 - t0), '-i', str(src),
                                  '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'],
                                 capture_output=True, check=True).stdout, np.float32)
res = mlx_whisper.transcribe(y, path_or_hf_repo='mlx-community/whisper-large-v3-turbo', language='ru',
                             word_timestamps=True, condition_on_previous_text=False,
                             initial_prompt=' '.join(prompt.split()), verbose=False)
new = [dict(w=w['word'].strip(), s=round(t0 + w['start'], 3), e=round(t0 + w['end'], 3),
            p=round(w.get('probability', 1), 3))
       for s in res['segments'] for w in s.get('words', []) if w['word'].strip()]
path = ROOT / 'tracks' / track / 'words.json'
orig = path.with_name('words.orig.json')
if not orig.exists():
    orig.write_text(path.read_text())
segs = [dict(s, words=[w for w in s['words'] if not t0 <= w['s'] < t1]) for s in json.loads(path.read_text())]
segs = [s for s in segs if s['words']]
if new:
    segs.append(dict(s=new[0]['s'], e=new[-1]['e'], text=' '.join(w['w'] for w in new), words=new))
segs.sort(key=lambda s: s['s'])
path.write_text(json.dumps(segs, ensure_ascii=False, indent=1))
print(len(new), 'words in', t0, '-', t1)
