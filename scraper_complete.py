"""Parallel scraper for TDF stage results 2020-2025.

Uses Playwright+Stealth with persistent browser profile for caching.
Processes stages in parallel batches to maximize throughput.
Stage type from profile icon on route/stages page.
"""
import asyncio
from playwright.async_api import async_playwright
from playwright_stealth import Stealth
from bs4 import BeautifulSoup
import pandas as pd
import random
import re
from tqdm import tqdm
import os
import json

BASE_URL = "https://www.procyclingstats.com"
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
CHECKPOINT = "checkpoint.json"
PARALLEL = 3


def icon_to_type(icon, is_itt):
    if is_itt:
        return 'ITT'
    if icon >= 4:
        return 'mountain'
    elif icon >= 2:
        return 'hilly'
    return 'flat'


def to_sec(s):
    if not s or s.strip() in ('', ',,', '..'):
        return 0
    s = s.strip().split(':')
    try:
        if len(s) == 3:
            return int(s[0]) * 3600 + int(s[1]) * 60 + int(s[2])
        elif len(s) == 2:
            return int(s[0]) * 60 + int(s[1])
        elif len(s) == 1:
            return int(s[0])
    except ValueError:
        return 0


def parse_results(html, year, stage, si):
    soup = BeautifulSoup(html, 'lxml')
    results = []
    table = soup.find('table', class_='results')
    if not table:
        for t in soup.find_all('table'):
            if 'Rnk' in t.get_text() and 'Rider' in t.get_text():
                table = t
                break
    if not table:
        return results
    tbodies = table.find_all('tbody')
    rows = []
    for tbody in tbodies:
        rows.extend(tbody.find_all('tr'))
    if not rows:
        rows = table.find_all('tr')[1:]

    st = si.get('stage_type', '')
    elev = si.get('elevation', 0)
    dist = si.get('distance_km', 0)
    ws = None

    for row in rows:
        cols = row.find_all('td')
        if len(cols) < 13:
            continue
        try:
            tc = cols[12]
            hid = tc.find('span', class_='hide')
            tr = (hid.get_text(strip=True) if hid else tc.get_text(strip=True))
            pos = int(cols[0].get_text(strip=True))
        except (ValueError, IndexError):
            continue
        rc = cols[7]
        rl = rc.find('a')
        rh = rl.get('href', '') if rl else ''
        rn = rl.get_text(strip=True) if rl else rc.get_text(strip=True)
        rt = cols[5].get_text(strip=True) if len(cols) > 5 else ''
        t = to_sec(tr)
        if pos == 1:
            ws = t
            cur_sec = 0.0
        elif t > 0:
            cur_sec = t
        gap = 0.0
        if ws and ws > 0:
            gap = round((cur_sec / ws) * 100, 4)
        results.append({
            'year': year, 'stage': stage, 'rider_name': rn,
            'position': pos, 'won': 1 if pos == 1 else 0,
            'stage_type': st, 'elevation': elev,
            'distance_km': dist, 'pcs_ranking': 0,
            'rider_type': rt, 'gap_pct': gap,
            'hist_stage_wins': 0, 'RCS_ranking': 0,
        })
    return results


async def fetch_result(page, year, stage):
    for _ in range(3):
        try:
            await page.goto(f"{BASE_URL}/race/tour-de-france/{year}/stage-{stage}/result/result",
                            timeout=60000, wait_until='domcontentloaded')
            await asyncio.sleep(random.uniform(1.5, 3))
            c = await page.content()
            if "Just a moment" not in c[:500] and len(c) > 500:
                return c
        except Exception:
            pass
        await asyncio.sleep(8)
    return None


async def worker(sem, page, year, stages_batch, pb):
    results = []
    for s in stages_batch:
        sn = s['stage']
        async with sem:
            html = await fetch_result(page, year, sn)
            if html:
                si = {
                    'stage_type': icon_to_type(s['profile_icon'], s['is_itt']),
                    'elevation': s['elevation'],
                    'distance_km': s['distance_km'],
                }
                r = parse_results(html, year, sn, si)
                results.extend(r)
            pb.update(1)
    return results


async def main():
    os.makedirs("cache", exist_ok=True)

    cp = {}
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            cp = json.load(f)

    all_results = []
    resume_year = cp.get('year', None)

    for year in YEARS:
        if resume_year and year < resume_year:
            try:
                df = pd.read_csv(f"cache/results_{year}.csv")
                all_results.extend(df.to_dict('records'))
                print(f"Cached {year} ({len(df)} rows)")
            except Exception:
                pass
            continue

        stealth = Stealth()
        async with stealth.use_async(async_playwright()) as pw:
            browser = await pw.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu',
                      '--disable-blink-features=AutomationControlled']
            )
            ctx = await browser.new_context(viewport={'width': 1920, 'height': 1080})
            pages = [await ctx.new_page() for _ in range(PARALLEL)]

            await pages[0].goto(BASE_URL, timeout=30000, wait_until='domcontentloaded')
            await asyncio.sleep(2)

            html = await fetch_result(pages[0], year, 1)
            if not html:
                html = None

            # Get route/stages page
            await pages[0].goto(f"{BASE_URL}/race/tour-de-france/{year}/route/stages",
                                timeout=30000, wait_until='domcontentloaded')
            await asyncio.sleep(2)
            content = await pages[0].content()

            soup = BeautifulSoup(content, 'lxml')
            table = soup.find('table', class_='basic')
            if not table:
                for t in soup.find_all('table'):
                    if 'Stage' in (t.get_text(strip=True) or ''):
                        table = t
                        break
            stages = []
            if table:
                for row in table.find_all('tr')[1:]:
                    cols = row.find_all('td')
                    if len(cols) < 5:
                        continue
                    sl = cols[2].find('a')
                    st = (sl.get_text(strip=True) if sl else cols[2].get_text(strip=True))
                    m = re.search(r'Stage\s*(\d+)', st)
                    if not m:
                        continue
                    dr = cols[5].get_text(strip=True) if len(cols) > 5 else ''
                    vr = cols[6].get_text(strip=True) if len(cols) > 6 else ''
                    pc = 0
                    if len(cols) > 1:
                        ps = cols[1].find('span')
                        if ps and ps.get('class'):
                            pm = re.search(r'p(\d)', ' '.join(ps.get('class')))
                            if pm:
                                pc = int(pm.group(1))
                    stages.append({
                        'stage': int(m.group(1)),
                        'distance_km': float(dr) if dr.replace('.', '', 1).lstrip('-').isdigit() else 0,
                        'elevation': int(vr) if vr.lstrip('-').isdigit() else 0,
                        'is_itt': '(ITT)' in st,
                        'profile_icon': pc,
                    })

            if not stages:
                print(f"  {year}: no stages")
                await browser.close()
                continue

            sem = asyncio.Semaphore(PARALLEL)
            pb = tqdm(total=len(stages), desc=f"  {year}")

            # Split stages into batches for parallel processing
            tasks = []
            batch_size = max(1, len(stages) // PARALLEL)
            for i in range(PARALLEL):
                batch = stages[i * batch_size:(i + 1) * batch_size]
                if batch:
                    tasks.append(worker(sem, pages[i], year, batch, pb))

            yr_results = []
            for task_result in await asyncio.gather(*tasks):
                yr_results.extend(task_result)

            pb.close()
            all_results.extend(yr_results)
            pd.DataFrame(yr_results).to_csv(f"cache/results_{year}.csv", index=False)
            print(f"  {year}: {len(yr_results)} results")
            await browser.close()

        resume_year = None

    df = pd.DataFrame(all_results)
    cols = ['year', 'stage', 'rider_name', 'position', 'won',
            'stage_type', 'elevation', 'distance_km', 'pcs_ranking',
            'rider_type', 'gap_pct', 'hist_stage_wins', 'RCS_ranking']
    df = df[cols]
    df = df.sort_values(['year', 'stage', 'position']).reset_index(drop=True)

    out = "tdf_stage_results_2020_2025.csv"
    df.to_csv(out, index=False)
    print(f"\n{'='*60}")
    print(f"Saved {len(df)} rows to {out}")
    print(f"Years: {sorted(df['year'].unique())}")
    print(f"\n{df.head(10).to_string()}")
    print(f"\nStage types: {df['stage_type'].value_counts().to_dict()}")
    print(f"Rows per year:\n{df.groupby('year').size().to_string()}")


if __name__ == "__main__":
    asyncio.run(main())
