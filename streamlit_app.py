"""Streamlit front‑end for the UX‑SEO audit.

This app provides a simple user interface where anyone can enter a
website URL (and optionally a competitor URL) and instantly get
high‑level SEO and UX metrics.  It leverages the lightweight
analysis functions defined below to extract basic information about
a page: title, meta description, headings, links, images and
response characteristics.  The aim is to deliver a near‑instant
result without invoking the heavier Lighthouse audit, so that the
service can run on free hosting platforms.

To run locally:

    streamlit run streamlit_app.py

You can deploy this app for free on Streamlit Community Cloud
(https://streamlit.io/cloud) or similar services.  See the project
README for deployment instructions.
"""

from __future__ import annotations

import time
from typing import Dict, Tuple, Optional

import requests
from bs4 import BeautifulSoup  # type: ignore
# The Streamlit import is intentionally deferred to within the `main` function.
# This allows other modules to import `compute_metrics` and
# `compare_metrics` without requiring Streamlit to be installed.  When
# running the app via `streamlit run` or invoking `main()` directly,
# Streamlit will be available and imported on demand.
try:
    import streamlit as st  # type: ignore  # noqa: F401
except ImportError:
    st = None  # type: ignore
from urllib.parse import urlparse


def compute_metrics(url: str) -> Dict[str, object]:
    """Fetch a URL and compute basic SEO/UX metrics.

    The metrics collected are deliberately simple so they can be
    computed quickly on platforms with limited resources.  They
    include:

      * title and its length
      * meta description and its length
      * counts of heading tags (h1–h6)
      * number of images, and counts of images with/without alt
      * number of links, split into internal and external
      * response time and page size

    :param url: The absolute URL to analyse.
    :returns: A dictionary of metrics.
    """
    metrics: Dict[str, object] = {}
    # Measure response time
    start = time.perf_counter()
    resp = requests.get(url, timeout=30)
    elapsed = time.perf_counter() - start
    resp.raise_for_status()
    metrics["response_time_seconds"] = round(elapsed, 3)
    metrics["page_size_bytes"] = len(resp.content)

    soup = BeautifulSoup(resp.text, "html.parser")

    # Title and meta description
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    metrics["title"] = title
    metrics["title_length"] = len(title)

    meta_desc_tag = soup.find("meta", attrs={"name": "description"})
    meta_desc = meta_desc_tag["content"].strip() if meta_desc_tag and meta_desc_tag.get("content") else ""
    metrics["meta_description"] = meta_desc
    metrics["meta_description_length"] = len(meta_desc)

    # Headings counts h1-h6
    for i in range(1, 7):
        count = len(soup.find_all(f"h{i}"))
        metrics[f"h{i}_count"] = count

    # Images and alt attributes
    images = soup.find_all("img")
    metrics["image_count"] = len(images)
    with_alt = sum(1 for img in images if img.get("alt"))
    metrics["images_with_alt"] = with_alt
    metrics["images_without_alt"] = len(images) - with_alt

    # Links counts: internal vs external
    links = soup.find_all("a", href=True)
    metrics["link_count"] = len(links)
    # Determine the domain for internal/external classification
    parsed_base = urlparse(url)
    base_domain = parsed_base.netloc
    internal = 0
    external = 0
    for a in links:
        href = a["href"]
        # Ignore anchors and javascript:void(0)
        if not href or href.startswith("#") or href.startswith("javascript"):  # noqa
            continue
        # Convert relative links to absolute domain for counting
        if href.startswith("http://") or href.startswith("https://"):
            dest = urlparse(href).netloc
            if dest and dest != base_domain:
                external += 1
            else:
                internal += 1
        else:
            internal += 1
    metrics["internal_link_count"] = internal
    metrics["external_link_count"] = external
    return metrics


def compare_metrics(primary: Dict[str, object], competitor: Dict[str, object]) -> Dict[str, object]:
    """Compute simple comparisons between two metric dictionaries.

    Returns a dictionary where each key corresponds to a metric and the
    value is the difference primary - competitor when numeric, or
    tuple(primary, competitor) when not easily subtractable.

    :param primary: Metrics for the primary page.
    :param competitor: Metrics for the competitor page.
    :returns: A dictionary summarising differences.
    """
    diff: Dict[str, object] = {}
    keys = set(primary.keys()) & set(competitor.keys())
    for key in keys:
        p_val = primary[key]
        c_val = competitor[key]
        if isinstance(p_val, (int, float)) and isinstance(c_val, (int, float)):
            diff[key] = p_val - c_val
        else:
            diff[key] = (p_val, c_val)
    return diff


def main() -> None:
    """Run the Streamlit app."""
    if st is None:
        raise RuntimeError(
            "Streamlit is not installed. Run this app via `streamlit run` or "
            "install streamlit (pip install streamlit)."
        )

    # At this point Streamlit is importable
    st.set_page_config(page_title="UX‑SEO Audit", layout="wide")
    st.title("UX‑SEO Audit (Lite)")
    st.write(
        "Enter a website URL to get a quick UX/SEO overview. "
        "Optionally provide a competitor URL for comparison. This tool uses a simple "
        "Python analysis instead of Lighthouse so it runs quickly and can be deployed for free."
    )

    url = st.text_input("Page URL", "https://example.com")
    competitor_url = st.text_input("Competitor URL (optional)")
    run_audit = st.button("Run Audit")

    if run_audit:
        if not url:
            st.error("Please enter a valid URL.")
            return
        try:
            with st.spinner("Analyzing primary page..."):
                primary_metrics = compute_metrics(url)
            st.subheader("Primary Page Metrics")
            st.json(primary_metrics)
            if competitor_url:
                with st.spinner("Analyzing competitor page..."):
                    competitor_metrics = compute_metrics(competitor_url)
                st.subheader("Competitor Page Metrics")
                st.json(competitor_metrics)
                diff = compare_metrics(primary_metrics, competitor_metrics)
                st.subheader("Differences (Primary – Competitor)")
                st.json(diff)
        except Exception as exc:
            st.error(f"Error during analysis: {exc}")


if __name__ == "__main__":
    main()