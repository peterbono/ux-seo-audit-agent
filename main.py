"""Entry point for the UX‑SEO Audit Agent.

This script orchestrates the process of crawling a web page,
executing a Lighthouse audit and optionally invoking an AI model to
generate human‑readable recommendations.  It is deliberately
lightweight and composable.  You can import individual functions into
your own code or extend them with additional analyses.

Usage examples:

    python main.py --url https://example.com
    python main.py --url https://your‑site.com --competitor https://competitor.com
    python main.py --url https://example.com --use-openai

If you supply the ``--use-openai`` flag the script expects an
environment variable ``OPENAI_API_KEY`` to be set.  Without this flag
the OpenAI integration is skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Optional import; OpenAI is only used if the --use-openai flag is set.
try:
    import openai  # type: ignore
except ImportError:
    openai = None  # type: ignore


@dataclass
class PageData:
    """Simple container for extracted page information."""

    url: str
    title: Optional[str]
    meta_description: Optional[str]
    headings: Dict[str, list[str]]
    links: list[str]


@dataclass
class LighthouseReport:
    """Container for Lighthouse scores and raw JSON."""

    performance: Optional[float]
    accessibility: Optional[float]
    best_practices: Optional[float]
    seo: Optional[float]
    pwa: Optional[float]
    raw_json: Optional[Dict[str, Any]]


def crawl_page(url: str) -> PageData:
    """Download a web page and extract high‑level information.

    This function uses ``requests`` to fetch the page and
    ``BeautifulSoup`` to parse it.  It extracts the ``<title>`` tag,
    meta description, all headings (h1–h6), and all absolute links.

    :param url: URL of the page to crawl.
    :returns: :class:`PageData` with extracted information.
    """
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    title_tag = soup.find("title")
    meta_desc_tag = soup.find("meta", attrs={"name": "description"})
    headings: Dict[str, list[str]] = {f"h{i}": [] for i in range(1, 7)}
    for i in range(1, 7):
        for tag in soup.find_all(f"h{i}"):
            headings[f"h{i}"].append(tag.get_text(strip=True))
    links: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # convert relative URLs to absolute
        if href.startswith("http://") or href.startswith("https://"):
            links.append(href)
        elif href.startswith("/"):
            links.append(url.rstrip("/") + href)
    return PageData(
        url=url,
        title=title_tag.get_text(strip=True) if title_tag else None,
        meta_description=meta_desc_tag.get("content") if meta_desc_tag else None,
        headings=headings,
        links=links,
    )


def run_lighthouse(url: str, output_path: str = "lighthouse_report.json") -> LighthouseReport:
    """Run the Lighthouse CLI on a URL and return key scores.

    This function calls the ``lighthouse`` executable via ``subprocess``.
    It writes the JSON report to ``output_path``.  If Lighthouse is not
    installed on your system or fails to run, the function returns a
    report with ``None`` fields.

    :param url: URL to audit.
    :param output_path: Path to write the JSON output.
    :returns: :class:`LighthouseReport` with scores and raw JSON (or
              ``None`` values if the audit fails).
    """
    try:
        cmd = [
            "lighthouse",
            url,
            "--quiet",
            "--chrome-flags=--headless",
            f"--output-path={output_path}",
            "--output=json",
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with open(output_path, "r", encoding="utf-8") as f:
            report_json = json.load(f)
        categories = report_json.get("categories", {})
        return LighthouseReport(
            performance=categories.get("performance", {}).get("score"),
            accessibility=categories.get("accessibility", {}).get("score"),
            best_practices=categories.get("best-practices", {}).get("score"),
            seo=categories.get("seo", {}).get("score"),
            pwa=categories.get("pwa", {}).get("score"),
            raw_json=report_json,
        )
    except Exception as exc:
        # In case of any failure, return a report with None fields
        sys.stderr.write(f"Warning: Lighthouse audit failed for {url}: {exc}\n")
        return LighthouseReport(None, None, None, None, None, None)


def call_openai(prompt: str) -> str:
    """Send a prompt to OpenAI and return the completion.

    Requires that ``openai`` is installed and ``OPENAI_API_KEY`` is set.
    Raises a RuntimeError if those conditions are not met.
    """
    if openai is None:
        raise RuntimeError(
            "openai package is not installed. Install it with 'pip install openai' and try again."
        )
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return response["choices"][0]["message"]["content"]


def generate_ai_prompt(page: PageData, competitor: Optional[PageData], lighthouse: LighthouseReport) -> str:
    """Construct a prompt for the AI model based on collected data."""
    prompt_parts = [
        f"You are a UX and SEO expert. Analyse the website at {page.url}.",
    ]
    if page.title:
        prompt_parts.append(f"The page title is: '{page.title}'.")
    if page.meta_description:
        prompt_parts.append(f"Its meta description is: '{page.meta_description}'.")
    if lighthouse.performance is not None:
        prompt_parts.append(f"Lighthouse performance score: {lighthouse.performance * 100:.0f}.")
    if lighthouse.accessibility is not None:
        prompt_parts.append(f"Accessibility score: {lighthouse.accessibility * 100:.0f}.")
    if lighthouse.seo is not None:
        prompt_parts.append(f"SEO score: {lighthouse.seo * 100:.0f}.")
    if competitor:
        prompt_parts.append(
            f"The competitor site is {competitor.url}. Compare the two and suggest ways to outperform the competitor in UX and SEO."
        )
    prompt_parts.append("Provide a concise list of recommendations.")
    return "\n".join(prompt_parts)


def generate_report(url: str, competitor_url: Optional[str], use_openai: bool) -> Dict[str, Any]:
    """Generate the audit report for a given URL and optional competitor.

    :param url: The primary URL to audit.
    :param competitor_url: Optional competitor URL for benchmarking.
    :param use_openai: Whether to call the OpenAI API for a narrative report.
    :returns: Dictionary containing collected data and, if requested, AI analysis.
    """
    report: Dict[str, Any] = {}
    # Crawl primary page
    print(f"Crawling {url}…", file=sys.stderr)
    primary_page = crawl_page(url)
    report["page"] = asdict(primary_page)
    # Lighthouse audit
    print(f"Running Lighthouse audit for {url}…", file=sys.stderr)
    lighthouse_report = run_lighthouse(url)
    report["lighthouse"] = asdict(lighthouse_report)
    # If competitor specified, crawl and audit
    competitor_page: Optional[PageData] = None
    if competitor_url:
        print(f"Crawling competitor {competitor_url}…", file=sys.stderr)
        competitor_page = crawl_page(competitor_url)
        report["competitor_page"] = asdict(competitor_page)
        print(f"Running Lighthouse audit for competitor {competitor_url}…", file=sys.stderr)
        competitor_lighthouse = run_lighthouse(competitor_url)
        report["competitor_lighthouse"] = asdict(competitor_lighthouse)
    # Optional AI analysis
    if use_openai:
        try:
            prompt = generate_ai_prompt(primary_page, competitor_page, lighthouse_report)
            print("Calling OpenAI API…", file=sys.stderr)
            ai_response = call_openai(prompt)
            report["ai_recommendations"] = ai_response
        except Exception as exc:
            report["ai_error"] = str(exc)
    return report


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="UX‑SEO Audit Agent")
    parser.add_argument("--url", required=True, help="URL to audit")
    parser.add_argument("--competitor", help="Competitor URL to benchmark against")
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Enable OpenAI analysis (requires OPENAI_API_KEY)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    report = generate_report(args.url, args.competitor, args.use_openai)
    json.dump(report, sys.stdout, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()