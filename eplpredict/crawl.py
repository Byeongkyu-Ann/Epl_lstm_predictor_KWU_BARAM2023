"""Crawl match results and statistics of one season from premierleague.com.

Example:
    python -m eplpredict.crawl --season-id 489 --out data/raw_2022_23.csv

The season id is the value of the `se` parameter in the results page URL.
The CSV is written from the oldest to the newest match.

The XPaths follow the site layout at the time of the original project (autumn 2023).
If the site changed, update the constants below.
"""
import argparse
import time

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By

from .config import DATA_DIR

RESULTS_URL = "https://www.premierleague.com/results?co=1&se={season}&cl=-1"

# --- XPaths -----------------------------------------------------------------
COOKIE_BUTTON = '//*[@id="onetrust-accept-btn-handler"]'
AD_CLOSE_BUTTON = '//*[@id="advertClose"]'

_LIST = '//*[@id="mainContent"]/div[3]/div[1]/div[2]/section/div[{i}]/div[2]/ul'
_ITEM = "/li/div/span/span[1]/span[2]"
RESULT_ITEM = _LIST + "/li[{j}]/div/span/span[1]/span[2]"   # j-th match of a date block
RESULT_ITEM_SINGLE = _LIST + _ITEM                           # date block with one match

_HEADER = '//*[@id="mainContent"]/div/section[2]/div[2]/section/div[1]/div[2]/div[1]'
HOME_TEAM = _HEADER + "/div[1]/div[1]/a[2]/span[1]"
AWAY_TEAM = _HEADER + "/div[3]/div[1]/a[2]/span[1]"
SCORE = _HEADER + "/div[2]/div[1]"

STATS_TAB = '//*[@id="mainContent"]/div/section[2]/div[2]/div/div[1]/div/div/ul/li[3]'
MATCH_WEEK = '//*[@id="mainContent"]/div/section[2]/div[1]/div[2]/div[1]/div'
STAT_CELL = ('//*[@id="mainContent"]/div/section[2]/div[2]/div/div[2]/section[3]'
             '/div[2]/div[2]/table/tbody/tr[{row}]/td[{col}]/p')

# Row of each statistic in the match stats table.
STAT_ROWS = {"possession": 1, "shots_on_target": 2, "shots": 3, "touches": 4,
             "passes": 5, "tackles": 6, "clearances": 7}


def click_if_present(driver, xpath: str) -> None:
    """Click an element such as a cookie banner or an advert if it exists."""
    for element in driver.find_elements(By.XPATH, xpath):
        try:
            element.click()
        except Exception:
            pass


def to_number(text: str) -> float:
    try:
        return float(text.replace("%", "").strip())
    except ValueError:
        return float("nan")


def goal_difference(score: str) -> int:
    """Convert a score such as '0-3' to home goals minus away goals."""
    home, away = score.split("-")
    return int(home) - int(away)


def collect_match(driver, item_xpath: str):
    """Open the match behind item_xpath and read its statistics.

    Returns None if the item does not exist or a field cannot be read.
    The driver is returned to the results page in both cases.
    """
    items = driver.find_elements(By.XPATH, item_xpath)
    if not items:
        return None
    ActionChains(driver).double_click(items[0]).perform()
    try:
        time.sleep(1)
        click_if_present(driver, AD_CLOSE_BUTTON)
        record = {
            "home_team": driver.find_element(By.XPATH, HOME_TEAM).text,
            "away_team": driver.find_element(By.XPATH, AWAY_TEAM).text,
            "goal_diff": goal_difference(driver.find_element(By.XPATH, SCORE).text),
        }
        driver.find_element(By.XPATH, STATS_TAB).click()
        time.sleep(1)
        record["match_week"] = driver.find_element(By.XPATH, MATCH_WEEK).text
        for name, row in STAT_ROWS.items():
            for side, col in (("home", 1), ("away", 3)):
                text = driver.find_element(By.XPATH, STAT_CELL.format(row=row, col=col)).text
                record[f"{side}_{name}"] = to_number(text)
        return record
    except Exception as err:
        print(f"  skipped a match: {type(err).__name__}")
        return None
    finally:
        driver.back()
        time.sleep(0.5)


def scroll_to_bottom(driver, interval: float) -> None:
    """Scroll until the page height stops growing so that every match is loaded."""
    previous = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(interval)
        current = driver.execute_script("return document.body.scrollHeight")
        if current == previous:
            return
        previous = current


def crawl_season(driver, max_blocks: int = 135, max_per_block: int = 100,
                 patience: int = 3, checkpoint=None) -> list:
    """Walk through the date blocks of the results page, newest first."""
    records, empty_blocks = [], 0
    for i in range(1, max_blocks + 1):
        found = 0
        for j in range(1, max_per_block + 1):
            record = collect_match(driver, RESULT_ITEM.format(i=i, j=j))
            if record is None and j == 1:
                # Date blocks with a single match use a different XPath.
                record = collect_match(driver, RESULT_ITEM_SINGLE.format(i=i))
                if record is not None:
                    records.append(record)
                    found += 1
                break
            if record is None:
                break
            records.append(record)
            found += 1
            if checkpoint and len(records) % 20 == 0:
                checkpoint(records)
        empty_blocks = 0 if found else empty_blocks + 1
        print(f"block {i}: {found} matches, total {len(records)}")
        if empty_blocks >= patience:
            break
    return records


def save(records: list, path) -> pd.DataFrame:
    df = pd.DataFrame(records).iloc[::-1].reset_index(drop=True)  # oldest first
    df.to_csv(path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season-id", type=int, required=True)
    parser.add_argument("--out", default=str(DATA_DIR / "raw_matches.csv"))
    parser.add_argument("--scroll-interval", type=float, default=10.0,
                        help="Seconds to wait after each scroll.")
    args = parser.parse_args()

    driver = webdriver.Chrome()
    driver.maximize_window()
    try:
        driver.get(RESULTS_URL.format(season=args.season_id))
        time.sleep(2)
        click_if_present(driver, COOKIE_BUTTON)
        click_if_present(driver, AD_CLOSE_BUTTON)
        scroll_to_bottom(driver, args.scroll_interval)
        time.sleep(3)
        records = crawl_season(driver, checkpoint=lambda r: save(r, args.out))
    finally:
        driver.quit()

    df = save(records, args.out)
    print(f"Saved {len(df)} matches to {args.out}")


if __name__ == "__main__":
    main()
