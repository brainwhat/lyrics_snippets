"""Whisper on short windows of a vocal stem (better than one long pass on rap):
python tools/whisper_windows.py <vocals.wav> <lang> <t0> <t1> [win] [step]"""
import sys, subprocess
import numpy as np
import mlx_whisper
src, lang = sys.argv[1:3]
t0, t1 = float(sys.argv[3]), float(sys.argv[4])
win = float(sys.argv[5]) if len(sys.argv) > 5 else 8.5
step = float(sys.argv[6]) if len(sys.argv) > 6 else 4.0
SR = 16000
y = np.frombuffer(subprocess.run(['ffmpeg', '-v', 'error', '-i', src, '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'],
                                 capture_output=True, check=True).stdout, np.float32)
t = t0
while t < t1:
    seg = y[int(t * SR):int((t + win) * SR)]
    r = mlx_whisper.transcribe(seg, path_or_hf_repo='mlx-community/whisper-large-v3-turbo', language=lang,
                               word_timestamps=True, condition_on_previous_text=False, verbose=None)
    words = ' '.join(f"{w['word'].strip()}@{w['start'] + t:.2f}" for s in r['segments'] for w in s.get('words', []))
    print(f'[{t:6.1f}] {words}', flush=True)
    t += step
