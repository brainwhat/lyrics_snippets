"""TikTok Studio in a real Chrome window (Playwright), for ./snip tiktok-login / schedule / posts.

Uses its own Chrome profile (PROFILE below, outside the project so cookies never land in iCloud): log in once
with `./snip tiktok-login`, after that the session is reused. Open the profile only through Playwright: it encrypts
cookies with a mock keychain, so plain Chrome on this profile can't read them and TikTok logs the session out. TikTok's web uploader has a native scheduler, so a
scheduled post goes out from TikTok's side: the Mac does not need to be on at publish time.

  tiktok.py login                                   open the login page, wait until logged in
  tiktok.py scheduled                               JSON list of times of posts already scheduled in Studio
  tiktok.py upload <mp4> <caption file> <YYYY-MM-DD HH:MM> [--no-submit]
  tiktok.py dump <url>                              screenshot + html into cache/tiktok/ (for fixing selectors)

Every step leaves a screenshot in cache/tiktok/, so when TikTok changes the page it is easy to see where it broke.
The last stdout line of each command is JSON.
"""
import datetime as dt, json, os, re, sys, time
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOTS = os.path.join(ROOT, 'cache', 'tiktok')
PROFILE = os.path.expanduser('~/.local/share/lyrics_snippets/tiktok-chrome')
UPLOAD = 'https://www.tiktok.com/tiktokstudio/upload?from=webapp'
CONTENT = 'https://www.tiktok.com/tiktokstudio/content'


def browser(p):
    os.makedirs(PROFILE, exist_ok=True)
    os.makedirs(SHOTS, exist_ok=True)
    ctx = p.chromium.launch_persistent_context(
        PROFILE, channel='chrome', headless=False, viewport={'width': 1400, 'height': 1000},
        args=['--disable-blink-features=AutomationControlled'], ignore_default_args=['--enable-automation'])
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    page.set_default_timeout(30000)
    return ctx, page


def shot(page, name):
    path = os.path.join(SHOTS, f'{time.strftime("%H%M%S")}_{name}.png')
    page.screenshot(path=path)
    return path


def logged_in(ctx):
    return any(c['name'] == 'sessionid' and c['value'] for c in ctx.cookies('https://www.tiktok.com'))


def out(**kw):
    print(json.dumps(kw, ensure_ascii=False))


def cmd_login():
    with sync_playwright() as p:
        ctx, page = browser(p)
        if not logged_in(ctx):
            page.goto('https://www.tiktok.com/login')
            print('log in to TikTok in the Chrome window that opened (waiting up to 15 min)...', flush=True)
            for _ in range(900):
                if logged_in(ctx):
                    break
                time.sleep(1)
        ok = logged_in(ctx)
        if ok:
            page.goto(UPLOAD)
            page.wait_for_timeout(4000)
            shot(page, 'login_upload_page')
        ctx.close()
        out(logged_in=ok)
        sys.exit(0 if ok else 1)


def cmd_dump(url):
    with sync_playwright() as p:
        ctx, page = browser(p)
        page.goto(url)
        page.wait_for_timeout(8000)
        png = shot(page, 'dump')
        html = png[:-4] + '.html'
        open(html, 'w').write(page.content())
        ctx.close()
        out(png=png, html=html)


def need_login(ctx):
    if not logged_in(ctx):
        ctx.close()
        out(error='not logged in: run ./snip tiktok-login')
        sys.exit(2)


# ---------------------------------------------------------------- scheduled posts already in Studio
WHEN = re.compile(r'(\d{1,2})[./](\d{1,2})(?:[./](\d{2,4}))?,?\s+(\d{1,2}):(\d{2})')


def cmd_scheduled():
    """Times of posts that Studio lists as scheduled. Rows are read as text, so this survives class renames."""
    with sync_playwright() as p:
        ctx, page = browser(p)
        need_login(ctx)
        page.goto(CONTENT)
        page.wait_for_timeout(6000)
        shot(page, 'content')
        rows = page.evaluate("""() => [...document.querySelectorAll('[data-tt*="row"], [class*="row"], tr')]
            .map(e => e.innerText).filter(t => /schedul|заплан/i.test(t))""")
        ctx.close()
    times = set()
    year = dt.date.today().year
    for r in rows:
        m = WHEN.search(r)
        if m:
            a, b, y, hh, mm = m.groups()
            # TikTok shows dd.mm in Russian UI and mm/dd in English UI
            day, mon = (int(a), int(b)) if '.' in m.group(0) else (int(b), int(a))
            y = int(y) + (2000 if y and len(y) == 2 else 0) if y else year
            times.add(f'{y:04d}-{mon:02d}-{day:02d} {int(hh):02d}:{mm}')
    out(scheduled=sorted(times), rows=len(rows))


# ---------------------------------------------------------------- upload + schedule
def click_text(page, *texts, timeout=10000):
    """Click the first visible element whose text is one of texts (Russian or English UI)."""
    loc = None
    for t in texts:
        cand = page.get_by_text(t, exact=True)
        if cand.count():
            loc = cand.first
            break
    if loc is None:
        loc = page.get_by_text(re.compile('|'.join(map(re.escape, texts)), re.I)).first
    loc.click(timeout=timeout)


def set_caption(page, caption):
    ed = page.locator('[contenteditable="true"]').first
    ed.click()
    page.keyboard.press('Meta+A')
    page.keyboard.press('Backspace')
    for i, line in enumerate(caption.split('\n')):
        if i:
            page.keyboard.press('Shift+Enter')
        for word in re.split(r'(\s+)', line):
            if not word:
                continue
            page.keyboard.type(word, delay=15)
            if word.startswith('#'):
                # a hashtag opens a suggestion list; wait for it, then Escape so it doesn't swallow the next key
                page.wait_for_timeout(700)
                page.keyboard.press('Escape')


def set_schedule(page, when):
    click_text(page, 'Schedule', 'Запланировать', 'Отложенная публикация')
    page.wait_for_timeout(1000)
    # a first-time dialog may ask to allow scheduling: accept it
    for t in ('Allow', 'Разрешить', 'Got it', 'Понятно'):
        b = page.get_by_role('button', name=t)
        if b.count() and b.first.is_visible():
            b.first.click()
    inputs = page.locator('input[type="text"]').filter(has_not=page.locator('[contenteditable]'))
    vals = [(i, inputs.nth(i).input_value()) for i in range(inputs.count())]
    tf = next(i for i, v in vals if re.fullmatch(r'\d{1,2}:\d{2}', v))
    df = next(i for i, v in vals if re.fullmatch(r'\d{4}-\d{2}-\d{2}|\d{1,2}[./]\d{1,2}[./]\d{2,4}', v))
    pick_date(page, inputs.nth(df), when)
    pick_time(page, inputs.nth(tf), when)


def pick_date(page, field, when):
    field.click()
    page.wait_for_timeout(500)
    # calendar: day cells are plain numbers; step months forward until the target month is shown
    for _ in range(3):
        cal = page.locator('[class*="calendar"], [class*="Calendar"]').last
        txt = cal.inner_text()
        if str(when.year) in txt and (when.strftime('%B') in txt or MONTHS_RU[when.month - 1] in txt.lower()
                                      or f'{when.month:02d}' in txt):
            break
        cal.locator('[class*="arrow"], [class*="next"]').last.click()
        page.wait_for_timeout(300)
    cal.get_by_text(str(when.day), exact=True).first.click()


MONTHS_RU = ['январ', 'феврал', 'март', 'апрел', 'ма', 'июн', 'июл', 'август', 'сентябр', 'октябр', 'ноябр', 'декабр']


def pick_time(page, field, when):
    field.click()
    page.wait_for_timeout(500)
    # time picker: two scroll lists (hours, minutes in 5-minute steps)
    lists = page.locator('[class*="time-picker"] [class*="list"], [class*="TimePicker"] [class*="list"]')
    for idx, val in ((0, f'{when.hour:02d}'), (1, f'{when.minute:02d}')):
        opt = lists.nth(idx).get_by_text(val, exact=True).first
        opt.scroll_into_view_if_needed()
        opt.click()
        page.wait_for_timeout(300)
    page.keyboard.press('Escape')


def cmd_upload(mp4, caption_file, when, submit=True):
    when = dt.datetime.strptime(when, '%Y-%m-%d %H:%M')
    caption = open(caption_file).read().strip()
    with sync_playwright() as p:
        ctx, page = browser(p)
        need_login(ctx)
        step = 'open'
        try:
            page.goto(UPLOAD)
            page.locator('input[type="file"]').first.wait_for(state='attached')
            step = 'file'
            page.locator('input[type="file"]').first.set_input_files(mp4)
            # the caption editor appears once the file is accepted; the post button enables once it's uploaded
            page.locator('[contenteditable="true"]').first.wait_for(timeout=120000)
            shot(page, '1_file')
            step = 'caption'
            set_caption(page, caption)
            shot(page, '2_caption')
            step = 'schedule'
            set_schedule(page, when)
            shot(page, '3_schedule')
            step = 'uploaded'
            post = page.locator('button[data-e2e="post_video_button"]')
            if not post.count():
                post = page.get_by_role('button', name=re.compile(r'^(Schedule|Запланировать)$'))
            for _ in range(300):
                if post.first.is_enabled() and post.first.get_attribute('aria-disabled') != 'true':
                    break
                page.wait_for_timeout(1000)
            if not submit:
                out(ok=True, submitted=False, shot=shot(page, '4_ready'))
                ctx.close()
                return
            step = 'submit'
            post.first.click()
            page.wait_for_url(re.compile(r'tiktokstudio/content|/manage'), timeout=120000)
            out(ok=True, submitted=True, at=when.strftime('%Y-%m-%d %H:%M'), shot=shot(page, '5_done'))
        except (PWTimeout, StopIteration, Exception) as e:
            out(ok=False, step=step, error=f'{type(e).__name__}: {str(e)[:300]}', shot=shot(page, f'fail_{step}'))
            ctx.close()
            sys.exit(1)
        ctx.close()


if __name__ == '__main__':
    a = sys.argv[1:]
    pos = [x for x in a if not x.startswith('--')]
    if pos[0] == 'login':
        cmd_login()
    elif pos[0] == 'scheduled':
        cmd_scheduled()
    elif pos[0] == 'dump':
        cmd_dump(pos[1])
    elif pos[0] == 'upload':
        cmd_upload(pos[1], pos[2], pos[3], submit='--no-submit' not in a)
    else:
        sys.exit(__doc__)
