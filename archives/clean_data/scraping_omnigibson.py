#!/usr/bin/env python3
import logging
import sys
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://behavior.stanford.edu/knowledgebase/synsets/"


# ───────────────────────────── helper: scrape ONE page ──────────────────────────
def _scrape_single(url: str):
    """Return (list_of_box_tuples, list_of_properties) for one synset page."""
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Bounding boxes ────────────────────────────────────────────────────────────
    boxes = []
    bb_hdr = soup.find("h2", string=lambda s: s and "All Descendant Objects" in s)
    if bb_hdr:
        table = bb_hdr.find_next("table")
        cols = [th.get_text(strip=True) for th in table.thead.tr.find_all("th")]
        bb_idx = cols.index("Bounding Box Size")       # will raise if missing
        for row in table.tbody.find_all("tr"):
            raw = row.find_all("td")[bb_idx].get_text(strip=True)
            try:
                triple = tuple(float(v) for v in raw.split(","))
                if len(triple) == 3:
                    boxes.append(triple)
            except ValueError:
                logging.warning("Malformed bbox '%s' on %s – skipped", raw, url)

    # Properties ────────────────────────────────────────────────────────────────
    props = []
    prop_hdr = soup.find("h2", string="Properties")
    if prop_hdr:
        for tr in prop_hdr.find_next("table").tbody.find_all("tr"):
            props.append(tr.td.get_text(strip=True))

    return boxes, props


# ─────────────────────────── helper: make FULL URL ─────────────────────────────
def _canon(cell: str) -> str:
    """Accept a full URL or a bare 'truck.n.01' slug and return a canonical URL."""
    cell = cell.strip()
    return cell if cell.startswith("http") else urljoin(BASE, f"{cell.rstrip('/')}/")


# ──────────────────────── 1) main scraping entry point ─────────────────────────
def scrape_synsets(csv_path=None):
    # Look next to this .py file if no explicit path given
    if csv_path is None:
        csv_path = Path(__file__).with_name("mscoco_objects_omnigibson.csv")
    logging.info("Loading CSV %s", csv_path)
    df = pd.read_csv(csv_path)
    if "Category" not in df.columns:
        raise ValueError("CSV missing required 'Category' column")

    urls = sorted({_canon(x) for x in df["Category"].dropna()})
    logging.info("Found %d unique synset URL(s)", len(urls))

    name_lookup = {}         # url  -> (object_name, object_description)

    for _, r in df.dropna(subset=["Category"]).iterrows():
        url = _canon(r["Category"])
        name_lookup[url] = (r["object_name"], r["object_description"])

    urls = sorted(name_lookup.keys())                # keep your dedup + sort

    results = {}
    for i, url in enumerate(urls, 1):
        logging.info("[%d/%d] Scraping %s", i, len(urls), url)
        try:
            boxes, props = _scrape_single(url)
            obj_name, obj_desc = name_lookup[url]     # pull from the dict
            results[url] = {
                "object_name":        obj_name,
                "object_description": obj_desc,
                "boxes":              boxes,
                "properties":         props,
            }
            logging.info("      ✓ %d boxes, %d properties", len(boxes), len(props))
        except Exception as e:
            logging.error("      ✗ error on %s – %s", url, e)

    return results


# ────────────────────────── 2) statistics & tidy DF ────────────────────────────
def summarise(results: dict) -> pd.DataFrame:
    """
    Convert raw scrape results to a tidy DataFrame with descriptive stats.

    Columns returned:
        url
        n_boxes
        len_min, len_max, len_mean, len_std
        wid_min, wid_max, wid_mean, wid_std
        hei_min, hei_max, hei_mean, hei_std
        properties   (semicolon-separated list)
    """
    records = []
    for url, data in results.items():
        n = len(data["boxes"])
        arr = np.array(data["boxes"]) if n else np.empty((0, 3))

        if n:                     # avoid NaNs when no boxes
            L, W, H = arr[:, 0], arr[:, 1], arr[:, 2]
            stats = dict(
                len_min=L.min(), len_max=L.max(), len_mean=L.mean(), len_std=L.std(ddof=0),
                wid_min=W.min(), wid_max=W.max(), wid_mean=W.mean(), wid_std=W.std(ddof=0),
                hei_min=H.min(), hei_max=H.max(), hei_mean=H.mean(), hei_std=H.std(ddof=0),
            )
        else:
            stats = {k: np.nan for k in [
                "len_min","len_max","len_mean","len_std",
                "wid_min","wid_max","wid_mean","wid_std",
                "hei_min","hei_max","hei_mean","hei_std"]}

        records.append({
            "object_name": data["object_name"],
            "object_description": data["object_description"],
            "url": url,
            "n_boxes": n,                 # NEW COLUMN
            **stats,
            "properties": ";".join(data["properties"])
        })

    return pd.DataFrame(records)


# ────────────────────────────────── CLI demo ───────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    try:
        raw = scrape_synsets()                     # step 1
        df_stats = summarise(raw)                  # step 2

        out_path = Path(__file__).with_name("ground_truth_omnigibson.csv")  # SAME FOLDER
        df_stats.to_csv(out_path, index=False)
        logging.info("Wrote %s (%d rows)", out_path, len(df_stats))
    except Exception as exc:
        logging.error("Fatal: %s", exc)
        sys.exit(1)
