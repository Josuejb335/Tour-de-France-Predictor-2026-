"""Scrape rider details from procyclingstats.com rider pages.

Reads unique rider names from the final CSV, builds rider URLs,
scrapes PCS ranking, career wins, and PCS points per rider.
Caches per-rider results so it can resume.
"""
import asyncio, os, json, time, unicodedata as ud
import regex as re
from playwright.async_api import async_playwright
from playwright_stealth import Stealth
from bs4 import BeautifulSoup
import pandas as pd

CSV = "tdf_stage_results_2020_2025.csv"
RIDER_CACHE = "rider_data.json"
PARALLEL = 2
DELAY = 1.5


def slugify(text):
    """Remove diacritics and lowercase, e.g. Pogačar → pogacar."""
    text = text.replace('ß', 'ss')
    nfkd = ud.normalize('NFKD', text)
    ascii_bytes = nfkd.encode('ascii', 'ignore')
    return ascii_bytes.decode().strip().lower()


def name_to_url(name):
    """Build rider URL from name. Handles both 'YatesAdam' and 'Adam Yates'."""
    if ' ' in name:
        parts_ws = name.split(None, 1)
        first, last = parts_ws[0], parts_ws[1]
    else:
        parts = re.findall(r'\p{Lu}\p{Ll}+', name)
        if len(parts) >= 2:
            first, last = parts[-1], ' '.join(parts[:-1])
        else:
            return None
    return f"{slugify(first)}-{slugify(last)}"


async def scrape_rider(page, rider_path):
    url = f"https://www.procyclingstats.com/rider/{rider_path}"
    await page.goto(url, timeout=30000, wait_until='domcontentloaded')
    await asyncio.sleep(2)
    content = await page.content()
    soup = BeautifulSoup(content, 'lxml')

    h1 = soup.find('h1')
    proper_name = h1.get_text(strip=True) if h1 else rider_path

    wins = 0
    for el in soup.find_all('div', class_='kpi'):
        txt = el.get_text(strip=True)
        if txt.isdigit():
            wins = int(txt)
            break

    rank = 0
    for mt5 in soup.find_all('div', class_='mt5'):
        txt = mt5.get_text()
        m = re.search(r'PCS Ranking\s*(\d+)', txt)
        if m:
            rank = int(m.group(1))
        if rank:
            break

    pts = 0
    rdr = soup.find('div', class_='rdrSeasonSum')
    if rdr:
        m = re.search(r'PCS points:\s*([\d,]+)', rdr.get_text())
        if m:
            pts = int(m.group(1).replace(',', ''))

    return proper_name, rank, wins, pts


async def main():
    if os.path.exists(RIDER_CACHE):
        with open(RIDER_CACHE) as f:
            rider_data = json.load(f)
    else:
        rider_data = {}

    df = pd.read_csv(CSV)
    unique_names = sorted(df['rider_name'].unique())
    print(f"Unique riders: {len(unique_names)}")

    # Handle manual name-to-URL overrides for edge cases
    slug_fix = {
        'KwiatkowskiMichał': 'michal-kwiatkowski',
        'MajkaRafał': 'rafal-majka',
        'MørkøvMichael': 'michael-morkov',
    }

    to_scrape = []
    for name in unique_names:
        if name in rider_data:
            continue
        url = slug_fix.get(name) or name_to_url(name)
        if not url:
            print(f"  SKIP {name} – cannot determine URL")
            continue
        to_scrape.append((name, url))

    if not to_scrape:
        print("All riders already cached.")
    else:
        print(f"Need to scrape: {len(to_scrape)} riders")

    stealth = Stealth()
    async with stealth.use_async(async_playwright()) as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu']
        )
        ctx = await browser.new_context(viewport={'width': 1920, 'height': 1080})

        idx = 0
        while idx < len(to_scrape):
            batch = to_scrape[idx:idx + PARALLEL]
            pages = [await ctx.new_page() for _ in range(len(batch))]

            tasks = []
            for (name, url), page in zip(batch, pages):
                async def do_one(pg=page, nm=name, ur=url):
                    try:
                        pn, rk, wn, pt = await scrape_rider(pg, ur)
                        return nm, {'name': pn, 'rank': rk, 'wins': wn, 'pts': pt}
                    except Exception as e:
                        print(f"  FAIL {nm} ({ur}): {e}")
                        return nm, None
                    finally:
                        await pg.close()

                tasks.append(do_one())

            for coro in tasks:
                name, result = await coro
                if result:
                    rider_data[name] = result
                    print(f"  {result['name']:30s} rank={result['rank']:3d} wins={result['wins']:3d} pts={result['pts']:5d}")

            idx += PARALLEL
            with open(RIDER_CACHE, 'w') as f:
                json.dump(rider_data, f, indent=1)
            if idx < len(to_scrape):
                await asyncio.sleep(DELAY * 2)

        await browser.close()

    # Merge back
    print(f"\nMerging rider data into CSV...")

    # Build reverse lookup: display name → data entry
    reverse = {}
    for orig_key, rd in rider_data.items():
        reverse[orig_key] = rd
        display = rd.get('name', '')
        if display:
            reverse[display] = rd

    rows = []
    for _, row in df.iterrows():
        r = row.to_dict()
        name = r['rider_name']
        rd = reverse.get(name, {})
        r['pcs_ranking'] = rd.get('rank', 0)
        r['hist_stage_wins'] = rd.get('wins', 0)
        r['RCS_ranking'] = rd.get('pts', 0)  # will rename next
        if rd.get('name'):
            r['rider_name'] = rd['name']
        rows.append(r)

    out = pd.DataFrame(rows)
    out = out.rename(columns={'RCS_ranking': 'PCS_points'})
    cols = ['year', 'stage', 'rider_name', 'position', 'won',
            'stage_type', 'elevation', 'distance_km', 'pcs_ranking',
            'rider_type', 'gap_pct', 'hist_stage_wins', 'PCS_points']
    out = out[cols]
    out = out.sort_values(['year', 'stage', 'position']).reset_index(drop=True)
    out.to_csv(CSV, index=False)
    print(f"Saved {len(out)} rows to {CSV}")
    print(f"Columns: {list(out.columns)}")


if __name__ == "__main__":
    asyncio.run(main())
