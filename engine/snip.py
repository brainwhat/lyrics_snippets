"""lyrics_snippets command line. Run through ./snip (uv with numpy + pillow); see README.md.

  ./snip list                                   videos with track, part, times, clip, status
  ./snip tracks                                 tracks, sections, which are used
  ./snip add-track <audio> <id> --title=… [--artist=…] [--lang=ru]
  ./snip map <track> [--from=s] [--to=s]        words.json lines + vocal level every 0.5 s + pauses
  ./snip edge <track> <s>                       vocal level every 0.1 s around s, to place a cut
  ./snip retime <track> <s0> <s1> <lyrics.txt>  re-run Whisper on a stretch, prompted by the text
  ./snip new <vid> <track> <s0> <s1> --clip=<id> [--clip_in=0] [--tint=cold] [--part=…]
  ./snip set <vid> --key=<json> …               change fields in videos.json (audio, onsets, clip_in, …) under a lock
  ./snip check <vid>                            spec, lyrics, footage length, line holds
  ./snip phrases <vid>                          line timings (clip time) word@time
  ./snip still <vid> --t=<s>                    test frame → cache/stills/<vid>_<t>.jpg
  ./snip render <vid>…                          video + cover + phone copy into output/<title>/
  ./snip footage [--free]                       footage catalog (--free: not used in any video)
  ./snip footage-add <id> <mp4 url> --desc=… [--tags=a,b]   download, contact sheet, catalog entry
  ./snip sheet <id>                             contact sheet → footage/sheets/pexels_<id>.jpg
  ./snip index                                  rewrite INDEX.md (tracks, videos, footage) after changes
  ./snip queue                                  who holds the heavy-job queue (add-track, retime, render) right now
  ./snip tiktok-login                           open Chrome (own profile), log in to TikTok once
  ./snip schedule <vid>… [--at="Y-m-d H:M"] [--dry-run] [--no-submit]
                                                schedule on TikTok into the first free slot (posting.json)
  ./snip posts [--sync]                         TikTok schedule by day (--sync: also read Studio)

add-track, retime and render share one queue: one heavy job at a time on this Mac, across all sessions.
The next one waits, printing who is running, so start them in the background. Never run two renders with &:
./snip render a b renders them one after another.
"""
import contextlib, fcntl, io, json, os, re, subprocess, sys, time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lyric_snow as L

ROOT = L.ROOT
WHISPER_PY = '/Users/vasilijfrolov/Desktop/Реп/Конструктор снипеттов/.venv/bin/python'
# Russian words.json: GigaAM-v3 e2e CTC on MLX (more accurate than Whisper on rap and ~20x faster, see README)
GIGAAM = ['uv', 'run', '--quiet', '--python', '3.12', '--with',
          'gigaam-mlx @ git+https://github.com/aystream/gigaam-mlx@20276ddd6173d636b37c6c6e13b4ee8f7b94d1ac',
          'python', '-I', os.path.join(ROOT, 'engine', 'transcribe_gigaam.py')]
DEMUCS = ['uvx', '--python', '3.11', '--from', 'demucs==4.1.0', '--with', 'torchaudio<2.9', '--with', 'soundfile',
          'demucs', '-n', 'htdemucs', '--two-stems=vocals', '-d', 'mps']
CATALOG = os.path.join(ROOT, 'footage', 'catalog.json')
SPECS = os.path.join(L.VIDEOS_DIR, 'videos.json')


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


# ---------------------------------------------------------------- heavy-job queue
# Demucs, GigaAM, Whisper and the render each eat the whole machine; two of them at once make both crawl.
# The queue is an flock on cache/heavy.lock: the OS frees it when the holder exits or is killed, so it never
# goes stale. The file holds a line about the current holder for the ones waiting.
LOCK = os.path.join(L.CACHE, 'heavy.lock')
# heavy processes started outside ./snip (by hand, or by an older run that was cut off)
# Not snip.py itself: a run waiting in the queue would look heavy and block the holder forever.
# «rawvideo -pix_fmt rgb24» are the render's frame pipes, «-maxrate 8M» its phone copy.
HEAVY = (r'demucs -n htdemucs|engine/transcribe_gigaam\.py|engine/transcribe\.py|engine/window_words\.py|mlx_whisper'
         r'|ffmpeg .*rawvideo -pix_fmt rgb24|ffmpeg .*-maxrate 8M')
SHELLS = ('sh', 'bash', 'zsh', 'fish', 'grep', 'rg')    # a shell whose command line merely mentions these


def holder():
    try:
        return open(LOCK).read().strip() or 'unknown'
    except OSError:
        return 'unknown'


def foreign_heavy():
    pids = subprocess.run(['pgrep', '-f', HEAVY], capture_output=True, text=True).stdout.split()
    procs = []
    for pid in pids:
        if pid == str(os.getpid()):
            continue
        comm = subprocess.run(['ps', '-o', 'comm=', '-p', pid], capture_output=True, text=True).stdout.strip()
        if not comm or os.path.basename(comm).lstrip('-') in SHELLS:
            continue
        args = subprocess.run(['ps', '-o', 'args=', '-p', pid], capture_output=True, text=True).stdout
        procs.append(f'{pid} {args.strip().splitlines()[0][:150] if args.strip() else comm}')
    return procs


@contextlib.contextmanager
def heavy(what):
    os.makedirs(L.CACHE, exist_ok=True)
    fd = os.open(LOCK, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            t0 = time.time()
            print(f'queue: busy ({holder()}), waiting...', flush=True)
            fcntl.flock(fd, fcntl.LOCK_EX)
            print(f'queue: free after {time.time() - t0:.0f} s', flush=True)
        t0, said = time.time(), False
        while procs := foreign_heavy():
            if not said:
                print('queue: waiting for heavy processes outside ./snip:\n  ' + '\n  '.join(procs), flush=True)
                said = True
            time.sleep(5)
        if said:
            print(f'queue: they finished after {time.time() - t0:.0f} s', flush=True)
        os.ftruncate(fd, 0)
        os.pwrite(fd, f'{what}, pid {os.getpid()}, since {time.strftime("%H:%M:%S")}\n'.encode(), 0)
        yield
    finally:
        os.close(fd)                               # closing releases the flock


def cmd_queue():
    fd = os.open(LOCK, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print('queue: free')
    except BlockingIOError:
        print(f'queue: busy - {holder()}')
    finally:
        os.close(fd)
    for p in foreign_heavy():
        print('  heavy process:', p)


def duration(path):
    return float(sh(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', path],
                    capture_output=True, text=True).stdout)


def load_json(p, default=None):
    return json.load(open(p)) if os.path.exists(p) else default


def save_json(p, d):
    tmp = f'{p}.{os.getpid()}.tmp'                 # write whole, then swap: a reader never sees half a file
    json.dump(d, open(tmp, 'w'), ensure_ascii=False, indent=1)
    os.replace(tmp, p)


@contextlib.contextmanager
def edit_json(p, default=None):
    """Read-change-write a shared json (videos.json, catalog.json) under a lock, from a fresh read.
    Several sessions edit these files; a copy read earlier and saved later wipes their changes."""
    with open(p + '.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        d = load_json(p, default)
        yield d
        save_json(p, d)


def level(track, step):
    """Vocal-stem level in dB, one value per `step` seconds."""
    db = L.vocal_db(track)                         # 100 per second
    k = int(round(step * 100))
    n = len(db) // k * k
    return 10 * np.log10((10 ** (db[:n].reshape(-1, k) / 10)).mean(1))


# ---------------------------------------------------------------- tracks
def cmd_add_track(audio, tid, title, artist='', lang='ru'):
    d = L.track_file(tid)
    os.makedirs(os.path.join(d, 'stems'), exist_ok=True)
    src = os.path.join(d, 'source.wav')
    vocals, words = os.path.join(d, 'stems', 'vocals.wav'), os.path.join(d, 'words.json')
    with heavy(f'add-track {tid}'):
        sh(['ffmpeg', '-v', 'error', '-y', '-i', audio, '-ar', '44100', '-c:a', 'pcm_s16le', src])
        tmp = os.path.join(L.CACHE, 'demucs', tid)     # per track: every input is named source.wav
        sh(DEMUCS + ['-o', tmp, src])
        for stem in ('vocals.wav', 'no_vocals.wav'):
            os.replace(os.path.join(tmp, 'htdemucs', 'source', stem), os.path.join(d, 'stems', stem))
        if lang == 'ru':
            sh(GIGAAM + [vocals, words])
        else:                                      # GigaAM knows only Russian
            sh([WHISPER_PY, os.path.join(ROOT, 'engine', 'transcribe.py'), vocals, lang, words])
    save_json(os.path.join(d, 'track.json'),
              dict(title=title, artist=artist, lang=lang, source=audio,
                   duration=round(duration(src), 2), sections=[], notes='', videos=[]))
    print(f'tracks/{tid}: source.wav, stems/, words.json, track.json - now fill "sections" from ./snip map {tid}')


def cmd_tracks():
    L.load_videos()
    for tid in sorted(L.TRACKS):
        m = load_json(L.track_file(tid, 'track.json'))
        used = [v for v, s in L.VIDEOS.items() if s['track'] == tid]
        print(f"\n{tid}  «{m['title']}»  {m.get('artist', '')}"
              f"  {m.get('duration', 0):.0f} с" + (f"  {m['bpm']} BPM" if m.get('bpm') else ''))
        for s in m.get('sections', []):
            hit = [v for v in used if L.VIDEOS[v]['start'] < s['end'] - 1 and L.VIDEOS[v]['end'] > s['start'] + 1]
            print(f"   {s['start']:7.2f}-{s['end']:7.2f}  {s['name']:<32} {', '.join(hit) if hit else '- свободно'}")
        if m.get('notes'):
            print('   ' + m['notes'])


def cmd_map(track, frm=None, to=None):
    words = json.load(open(L.track_file(track, 'words.json')))
    frm = float(frm) if frm is not None else 0
    to = float(to) if to is not None else 1e9
    print('words.json (vocal stem):')
    for seg in words:
        if frm <= seg['s'] <= to:
            print(f"  {seg['s']:7.2f}-{seg['e']:7.2f}  {seg['text']}")
    lv = level(track, 0.5)
    print('\nvocal level dB every 0.5 s (below -45 = silence):')
    rows = [f'{i * 0.5:6.1f}:{v:4.0f}' for i, v in enumerate(lv) if frm <= i * 0.5 <= to]
    for k in range(0, len(rows), 12):
        print('  ' + '  '.join(rows[k:k + 12]))
    db = L.vocal_db(track)
    quiet = db < -45
    print('\npauses ≥ 0.3 s:')
    i = 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            if j - i >= 30 and frm <= i / 100 <= to:
                print(f'  {i / 100:7.2f}-{j / 100:7.2f}')
            i = j
        else:
            i += 1


def cmd_edge(track, t):
    t = float(t)
    lv = level(track, 0.1)
    rows = [f'{i / 10:7.2f}:{lv[i]:4.0f}' for i in range(max(0, int((t - 1.2) * 10)), min(len(lv), int((t + 1.2) * 10)))]
    for k in range(0, len(rows), 12):
        print('  '.join(rows[k:k + 12]))


def cmd_retime(track, s0, s1, txt):
    with heavy(f'retime {track} {s0}-{s1}'):
        sh([WHISPER_PY, os.path.join(ROOT, 'engine', 'window_words.py'), track, s0, s1, txt])


# ---------------------------------------------------------------- videos
def cmd_list():
    L.load_videos()
    for vid, v in L.VIDEOS.items():
        print(f"{vid:<12} {v['track']:<11} {v['part']:<24} {v['start']:7.2f}-{v['end']:7.2f} "
              f"{v['end'] - v['start']:5.1f} с  clip {v['clip']:<9} {v['tint']:<5} {v['status']}")


def cmd_new(vid, track, s0, s1, clip, clip_in=0, tint='cold', part=''):
    if not re.search(r'\d+$', vid):
        sys.exit('vid must end in a number: it becomes «лирик N» in the file name')
    s0, s1 = float(s0), float(s1)
    with edit_json(SPECS, {}) as specs:
        if vid in specs:
            sys.exit(f'{vid} already exists in videos/videos.json')
        specs[vid] = dict(track=track, part=part, start=s0, end=s1, clip=str(clip), clip_in=float(clip_in),
                          tint=tint, audio=[[s0, 'mix']], fade=0.8, onsets={}, status='draft')
    lf = L.lyrics_file(vid)
    if not os.path.exists(lf):
        open(lf, 'w').close()
    print(f'videos/videos.json: {vid} added; write the lines into videos/lyrics/{vid}.txt')
    if others := clip_users(vid):
        print(f'WARNING: clip {clip} is already in {", ".join(others)}')


def clip_users(vid):
    """Other videos on the same clip (another session may take a «free» clip at the same moment)."""
    specs = load_json(SPECS, {})
    return [k for k, v in specs.items() if k != vid and v['clip'] == specs[vid]['clip']]


def cmd_set(vid, **fields):
    """./snip set <vid> --audio='[[108.95,"inst"],[109.12,"mix"]]' --clip_in=23.7: values are json, else text."""
    with edit_json(SPECS) as specs:
        if vid not in specs:
            sys.exit(f'{vid} is not in videos/videos.json')
        for k, val in fields.items():
            try:
                val = json.loads(val)
            except ValueError:
                pass
            specs[vid][k] = str(val) if k == 'clip' else val
        print(vid, json.dumps(specs[vid], ensure_ascii=False))


def cmd_check(vid):
    L.load_videos()
    v = L.VIDEOS[vid]
    bad = []
    for f in ('source.wav', 'stems/vocals.wav', 'stems/no_vocals.wav', 'words.json'):
        if not os.path.exists(L.track_file(v['track'], f)):
            bad.append(f'tracks/{v["track"]}/{f} missing')
    lf = L.lyrics_file(vid)
    text = open(lf, encoding='utf-8').read() if os.path.exists(lf) else ''
    if not text.strip():
        bad.append(f'videos/lyrics/{vid}.txt is empty')
    if '—' in text:
        bad.append('lyrics contain «—»: the user wants «-»')
    if re.search(r'\(.*?\)', text):
        bad.append('lyrics contain (ad-libs) in brackets: they are not captioned')
    clip = os.path.join(L.STOCK, f"pexels_{v['clip']}.mp4")
    if not os.path.exists(clip):
        bad.append(f'footage/clips/pexels_{v["clip"]}.mp4 missing')
    else:
        need = v['clip_in'] + (v['end'] - v['start']) * L.SPEED
        have = duration(clip)
        if need > have:
            bad.append(f'clip too short: needs {need:.1f} s from clip_in, has {have:.1f} s')
    if others := clip_users(vid):
        bad.append(f'clip {v["clip"]} is also in {", ".join(others)}: fine only if meant, else pick another')
    a = v.get('audio') or []
    if not a or abs(a[0][0] - v['start']) > 0.01:
        bad.append('audio[0] must start at "start"')
    if text.strip():
        try:
            for s, e, ws in L.load_phrases(vid):
                if e - ws[-1][1] < 0.6:
                    bad.append(f'line at {s:.2f} holds only {e - ws[-1][1]:.2f} s after its last word')
        except ValueError as err:
            bad.append(str(err))
    print('\n'.join(bad) if bad else f'{vid}: ok')


def cover(vid):
    ph = L.load_phrases(vid)
    s, e, ws = max(ph[1:-1] or ph, key=lambda p: min(p[1] - p[0], 6))
    t = min(ws[-1][1] + 1.2, e - 0.3)
    box = Image.new('RGB', (1080, 1920))
    box.paste(L.still(vid, t).resize((1080, 810), Image.LANCZOS), (0, 555))
    name = L.out_name(vid, 'обложка.jpg')
    box.save(name, quality=92)
    return name


def phone(vid):
    src = L.out_name(vid, '9x16.mp4')
    d = os.path.join(os.path.dirname(src), 'phone')
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, f'{vid}_phone.mp4')
    sh(['ffmpeg', '-v', 'error', '-y', '-i', src, '-c:v', 'libx264', '-preset', 'slow', '-crf', '20',
        '-maxrate', '8M', '-bufsize', '16M', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', dst])
    return dst


def cmd_render(*vids):
    with heavy(f'render {" ".join(vids)}'):
        for vid in vids:
            L.load_videos()                        # re-read: other sessions edit videos.json while we wait
            L.video(vid)
            print(cover(vid))
            p = phone(vid)
            print(p, f'{os.path.getsize(p) / 1e6:.1f} MB', flush=True)
            with edit_json(SPECS) as specs:
                if specs[vid]['status'] == 'draft':
                    specs[vid]['status'] = 'rendered'
    if load_json(POSTING, {}).get('auto') and os.path.isdir(TIKTOK_PROFILE):
        try:
            cmd_schedule(*vids)
        except SystemExit as e:     # a failed upload must not look like a failed render
            print(f'schedule failed: {e}; the video is rendered, run ./snip schedule {" ".join(vids)} later')


def cmd_still(vid, t):
    L.load_videos()
    os.makedirs(os.path.join(L.CACHE, 'stills'), exist_ok=True)
    out = os.path.join(L.CACHE, 'stills', f'{vid}_{float(t):.1f}.jpg')
    L.still(vid, float(t)).save(out, quality=92)
    print(out)


def cmd_phrases(vid):
    L.load_videos()
    for s, e, ws in L.load_phrases(vid):
        print(f'{s:6.2f} {e:6.2f}  ' + ' '.join(f'{w}@{t:.2f}' for w, t in ws))


# ---------------------------------------------------------------- footage
def cmd_sheet(cid):
    src = os.path.join(L.STOCK, f'pexels_{cid}.mp4')
    os.makedirs(os.path.join(ROOT, 'footage', 'sheets'), exist_ok=True)
    out = os.path.join(ROOT, 'footage', 'sheets', f'pexels_{cid}.jpg')
    d = duration(src)
    sh(['ffmpeg', '-v', 'error', '-y', '-i', src, '-vf', f'fps=6/{d:.2f},scale=320:180,tile=6x1', '-frames:v', '1',
        '-q:v', '3', out])
    return out


def cmd_footage(free=False):
    L.load_videos()
    cat = load_json(CATALOG, {})
    for cid, c in sorted(cat.items(), key=lambda kv: kv[1].get('duration', 0)):
        used = [v for v, s in L.VIDEOS.items() if s['clip'] == cid]
        if free and used:
            continue
        flag = '  ВНИМАНИЕ: ' + c['avoid'] if c.get('avoid') else ''
        print(f"{cid:<9} {c.get('duration', 0):5.1f} с  {c.get('desc', '')}  [{', '.join(c.get('tags', []))}]"
              f"  {'→ ' + ', '.join(used) if used else ''}{flag}")


def cmd_footage_add(cid, url, desc='', tags=''):
    dst = os.path.join(L.STOCK, f'pexels_{cid}.mp4')
    for _ in range(10):
        if subprocess.run(['curl', '-sL', '-C', '-', '-A', 'Mozilla/5.0', '-o', dst, url]).returncode == 0:
            break
    with edit_json(CATALOG, {}) as cat:
        cat[str(cid)] = dict(file=f'clips/pexels_{cid}.mp4', url=url, duration=round(duration(dst), 1), desc=desc,
                             tags=[t for t in tags.split(',') if t], people='not checked', avoid='')
    print(cmd_sheet(cid), '- look at it: no people, no hands, no faces (see README)')


def cmd_index():
    parts = []
    for title, fn in (('Треки (секции, что уже снято)', cmd_tracks), ('Видео', cmd_list),
                      ('Футаж (по длине; → где использован)', cmd_footage)):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fn()
        parts.append(f'## {title}\n\n```\n{buf.getvalue().strip()}\n```\n')
    open(os.path.join(ROOT, 'INDEX.md'), 'w').write(
        '# lyrics_snippets: что есть сейчас\n\nСгенерировано `./snip index`, руками не править.\n\n' + '\n'.join(parts))
    print('INDEX.md')


# ---------------------------------------------------------------- TikTok
# Uploading is engine/tiktok.py (Playwright in real Chrome, own profile); here: slots, captions, bookkeeping.
# Each scheduled video gets "tiktok": {"at": "Y-m-d H:M", "caption": …, "done": iso time} and status "scheduled".
POSTING = os.path.join(ROOT, 'posting.json')
TIKTOK_PROFILE = os.path.expanduser('~/.local/share/lyrics_snippets/tiktok-chrome')
TIKTOK = ['uv', 'run', '--quiet', '--python', '3.12', '--with', 'playwright==1.48.0', 'python', '-I',
          os.path.join(ROOT, 'engine', 'tiktok.py')]
TIKTOK_LOCK = os.path.join(L.CACHE, 'tiktok.lock')
FMT = '%Y-%m-%d %H:%M'


def tiktok(*args, check=True):
    """Run engine/tiktok.py, pass its output through, return the JSON from its last line."""
    p = subprocess.run(TIKTOK + list(args), capture_output=True, text=True)
    sys.stderr.write(p.stderr[-2000:] if p.returncode else '')
    lines = [l for l in p.stdout.strip().splitlines() if l.startswith('{')]
    res = json.loads(lines[-1]) if lines else {'ok': False, 'error': p.stdout[-500:] or p.stderr[-500:]}
    if check and p.returncode:
        sys.exit(f'tiktok.py {args[0]}: {res}')
    return res


def caption(vid):
    import datetime as dt
    cfg = load_json(POSTING)
    v = L.VIDEOS[vid]
    tr = load_json(L.track_file(v['track'], 'track.json'), {})
    title, artist = tr.get('title', v['track']), tr.get('artist', '')
    line = next((l.strip() for l in open(L.lyrics_file(vid)) if l.strip()), '').rstrip(',;:')
    tag = '#' + re.sub(r'\W', '', title.lower())
    tags = ' '.join(t.replace('#{titletag}', tag) for t in cfg['tags'])
    text = cfg['caption'].format(line=line, title=title, by=f' - {artist}' if artist else '', tags=tags)
    return text.replace('—', '-').strip()


def taken_slots(sync=False):
    """Scheduled times: ours from videos.json, plus (sync) everything Studio lists, incl. posts added by hand."""
    specs = load_json(SPECS)
    times = {s['tiktok']['at']: vid for vid, s in specs.items() if s.get('tiktok')}
    if sync:
        for t in tiktok('scheduled')['scheduled']:
            times.setdefault(t, '(в Studio)')
    return times


def free_slot(taken):
    import datetime as dt
    cfg = load_json(POSTING)
    now = dt.datetime.now()
    busy = [dt.datetime.strptime(t, FMT) for t in taken]
    for d in range(cfg['max_days'] + 1):
        day = now.date() + dt.timedelta(days=d)
        if sum(b.date() == day for b in busy) >= cfg['per_day']:
            continue
        for s in cfg['slots']:
            t = dt.datetime.combine(day, dt.time(*map(int, s.split(':'))))
            if t >= now + dt.timedelta(minutes=cfg['lead_min']) and all(abs(t - b) >= dt.timedelta(minutes=30)
                                                                         for b in busy):
                return t.strftime(FMT)
    sys.exit(f"no free slot in the next {cfg['max_days']} days (TikTok schedules at most ~10 days ahead)")


def cmd_tiktok_login():
    res = tiktok('login', check=False)
    print('logged in' if res.get('logged_in') else f'not logged in: {res}')


def cmd_schedule(*vids, at=None, dry_run=False, no_submit=False):
    """Schedule each vid on TikTok into the first free slot. One at a time on the Mac (lock), so two renders
    finishing together don't grab the same slot or open the Chrome profile twice."""
    import datetime as dt
    L.load_videos()
    with open(TIKTOK_LOCK, 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        taken = taken_slots(sync=not dry_run)
        for vid in vids:
            specs = load_json(SPECS)
            if specs[vid].get('tiktok'):
                print(f"{vid}: already scheduled for {specs[vid]['tiktok']['at']}, skipped")
                continue
            mp4 = L.out_name(vid, '9x16.mp4')
            if not os.path.exists(mp4):
                sys.exit(f'{vid}: no {mp4}; render it first')
            when = at or free_slot(taken)
            text = caption(vid)
            print(f'{vid} → {when}\n  ' + text.replace('\n', '\n  '))
            if dry_run:
                taken[when] = vid
                continue
            cap = os.path.join(L.CACHE, 'tiktok', f'{vid}_caption.txt')
            os.makedirs(os.path.dirname(cap), exist_ok=True)
            open(cap, 'w').write(text)
            res = tiktok('upload', mp4, cap, when, *(['--no-submit'] if no_submit else []))
            print(f"  {res.get('shot', '')}")
            if no_submit:
                continue
            taken[when] = vid
            with edit_json(SPECS) as specs:
                specs[vid]['tiktok'] = dict(at=when, caption=text,
                                            done=dt.datetime.now().isoformat(timespec='seconds'))
                specs[vid]['status'] = 'scheduled'


def cmd_posts(sync=False):
    import datetime as dt
    now = dt.datetime.now().strftime(FMT)
    for t, vid in sorted(taken_slots(sync=bool(sync)).items()):
        print(f"{t}  {vid:<12} {'вышло' if t < now else 'ждёт'}")


# ---------------------------------------------------------------- main
def main(argv):
    if not argv or argv[0] in ('-h', '--help', 'help'):
        print(__doc__)
        return
    cmd, pos = argv[0], [a for a in argv[1:] if not a.startswith('--')]
    opt = dict(a[2:].split('=', 1) if '=' in a else (a[2:], '1') for a in argv[1:] if a.startswith('--'))
    opt = {k.replace('-', '_'): v for k, v in opt.items()}
    if 'from' in opt:
        opt['frm'] = opt.pop('from')
    if cmd == 'still':
        return cmd_still(pos[0], opt['t'])
    if cmd == 'footage':
        return cmd_footage('free' in opt)
    fn = globals().get('cmd_' + cmd.replace('-', '_'))
    if not fn:
        sys.exit(f'unknown command {cmd}; ./snip --help')
    fn(*pos, **opt)


if __name__ == '__main__':
    main(sys.argv[1:])
