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
from typing import Dict, Tuple, Optional, List

import requests
import re
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

    # ---------------------------------------------------------------------------
    # Additional on‑page signals
    # Many subtle elements influence UX and SEO beyond the basic counts above.  We
    # capture a few of them here to differentiate the tool from a simple GPT
    # prompt.  These include canonical/robots tags, social sharing metadata,
    # structured data, responsive design hints and accessibility attributes.

    # Canonical tag tells search engines which version of a URL to index.  The
    # canonical tag helps consolidate signals when there are multiple URLs for
    # the same content【600788209180035†L501-L540】.  We record whether one is
    # present.
    canonical_tag = soup.find("link", rel="canonical")
    metrics["canonical_present"] = bool(canonical_tag and canonical_tag.get("href"))

    # Robots meta tag can instruct search engines how to crawl a page.  We
    # simply detect its presence.
    robots_tag = soup.find("meta", attrs={"name": "robots"})
    metrics["robots_meta_present"] = bool(robots_tag and robots_tag.get("content"))

    # Open Graph tags control how a page appears when shared on social media.  A
    # complete set of og:title/og:description/og:image tags improves click‑through
    # rates【600788209180035†L553-L599】.  We record whether each tag exists.
    metrics["og_title_present"] = bool(soup.find("meta", property="og:title"))
    metrics["og_description_present"] = bool(
        soup.find("meta", property="og:description")
    )
    metrics["og_image_present"] = bool(soup.find("meta", property="og:image"))

    # Structured data (JSON‑LD) helps search engines understand content and is
    # associated with improved SERP features and UX【600788209180035†L664-L676】.  We
    # detect whether any JSON‑LD script is present.
    metrics["structured_data_present"] = bool(
        soup.find("script", attrs={"type": "application/ld+json"})
    )

    # Viewport meta tag indicates responsive design and accessibility.  Its
    # absence can trigger Google Search Console warnings【600788209180035†L609-L627】.
    metrics["viewport_present"] = bool(
        soup.find("meta", attrs={"name": "viewport"})
    )

    # Accessibility attributes (ARIA).  We compute the ratio of elements with
    # ARIA attributes to total elements.  This serves as a proxy for how well
    # interactive components are labelled for assistive technologies.
    total_elements = 0
    aria_count = 0
    for tag in soup.find_all(True):
        total_elements += 1
        for attr_name in tag.attrs.keys():
            if isinstance(attr_name, str) and attr_name.startswith("aria-"):
                aria_count += 1
                break
    metrics["aria_ratio"] = (
        aria_count / total_elements if total_elements > 0 else 0.0
    )

    # Lazy loading images improve page performance by deferring off‑screen
    # downloads.  We compute the fraction of images using loading="lazy".
    lazy_images = sum(
        1 for img in images if img.get("loading", "").lower() == "lazy"
    )
    metrics["lazy_loading_ratio"] = (
        lazy_images / len(images) if images else 0.0
    )

    # ---------------------------------------------------------------------------
    # Readability and text statistics
    text = soup.get_text(separator=" ", strip=True)
    # Basic word, sentence and syllable counts for readability
    metrics["word_count"] = len(text.split())
    # Count sentences using ., ! and ? as delimiters
    metrics["sentence_count"] = text.count(".") + text.count("!") + text.count("?")
    metrics["syllable_count"] = _count_syllables_in_text(text)
    metrics["flesch_reading_ease"] = _flesch_reading_ease(
        metrics["word_count"], metrics["sentence_count"], metrics["syllable_count"]
    )
    # Language attribute
    html_tag = soup.find("html")
    metrics["lang"] = html_tag.get("lang") if html_tag else None
    return metrics


def _count_syllables_in_word(word: str) -> int:
    """Approximate the number of syllables in an English word.

    Uses a simple heuristic based on vowel groups. Words without vowels are
    counted as one syllable. Consecutive vowels are treated as a single
    syllable. A final silent 'e' is not counted.
    """
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False
    for char in word:
        if char in vowels:
            if not prev_is_vowel:
                count += 1
            prev_is_vowel = True
        else:
            prev_is_vowel = False
    # Subtract one for silent trailing 'e'
    if word.endswith("e") and count > 1:
        count -= 1
    return count or 1


def _count_syllables_in_text(text: str) -> int:
    """Count approximate syllables in a block of text."""
    syllables = 0
    for token in text.split():
        # Strip non‑alphabetic characters
        clean = re.sub(r"[^a-zA-Z]", "", token)
        if clean:
            syllables += _count_syllables_in_word(clean)
    return syllables


def _flesch_reading_ease(word_count: int, sentence_count: int, syllable_count: int) -> float:
    """Compute the Flesch reading ease score for English text.

    The formula is: 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words).
    Returns 0.0 if there are no sentences or words.
    """
    if sentence_count == 0 or word_count == 0:
        return 0.0
    return round(
        206.835
        - 1.015 * (word_count / sentence_count)
        - 84.6 * (syllable_count / word_count),
        2,
    )


def evaluate_metrics(metrics: Dict[str, object]) -> Tuple[int, List[str]]:
    """Evaluate metrics and return a score (0‑100) and improvement suggestions.

    This function applies simple heuristics for SEO and UX best practices. Each
    deviation deducts points from a starting score of 100. Suggestions are
    returned explaining how to improve. The score is clipped between 0 and 100.

    :param metrics: The metrics dictionary returned from ``compute_metrics``.
    :returns: A tuple ``(score, suggestions)``.
    """
    score = 100
    suggestions: List[str] = []

    # Title length
    title_len = metrics.get("title_length", 0)
    if title_len == 0:
        suggestions.append("Add a <title> tag to the page.")
        score -= 10
    elif title_len < 30 or title_len > 60:
        suggestions.append(
            f"Adjust title length (current {title_len} characters; recommended 30–60)."
        )
        score -= 5

    # Meta description
    meta_len = metrics.get("meta_description_length", 0)
    if meta_len == 0:
        suggestions.append("Add a meta description to summarise the page.")
        score -= 10
    elif meta_len < 70 or meta_len > 160:
        suggestions.append(
            f"Adjust meta description length (current {meta_len} characters; recommended 70–160)."
        )
        score -= 5

    # H1 count
    h1_count = metrics.get("h1_count", 0)
    if h1_count != 1:
        suggestions.append(f"Use exactly one H1 heading (current {h1_count}).")
        score -= 5

    # Images alt ratio
    img_count = metrics.get("image_count", 0)
    imgs_with_alt = metrics.get("images_with_alt", 0)
    if img_count > 0:
        alt_ratio = imgs_with_alt / img_count
        if alt_ratio < 0.9:
            missing = img_count - imgs_with_alt
            suggestions.append(
                f"Add alt attributes to images ({missing} of {img_count} missing)."
            )
            score -= 5

    # Response time
    response_time = metrics.get("response_time_seconds", 0.0)
    if isinstance(response_time, (int, float)) and response_time > 1.0:
        suggestions.append(
            f"Improve server response time (currently {response_time:.2f} s > 1 s)."
        )
        score -= 5

    # Page size
    page_size_kb = metrics.get("page_size_bytes", 0) / 1024
    if isinstance(page_size_kb, (int, float)) and page_size_kb > 1024:
        suggestions.append(
            f"Reduce page size (currently {int(page_size_kb)} KB; aim for <1 MB)."
        )
        score -= 5

    # Internal vs external links
    int_links = metrics.get("internal_link_count", 0)
    ext_links = metrics.get("external_link_count", 0)
    total_links = metrics.get("link_count", 0)
    if total_links and ext_links > int_links:
        suggestions.append("Increase internal linking to strengthen site structure.")
        score -= 3

    # Readability
    flesch = metrics.get("flesch_reading_ease", 0)
    if isinstance(flesch, (int, float)) and flesch < 60:
        suggestions.append(
            f"Simplify your copy (Flesch reading ease {flesch}; aim for >60)."
        )
        score -= 4

    # Language attribute
    if not metrics.get("lang"):
        suggestions.append(
            "Specify the language on the <html> tag (e.g., lang=\"en\" or lang=\"fr\")."
        )
        score -= 2

    # Additional heuristics for extended analysis
    # -----------------------------------------------------------------------
    # Canonical tag presence
    if not metrics.get("canonical_present"):
        suggestions.append(
            "Add a rel=\"canonical\" link tag to consolidate duplicate URLs and improve SEO."
        )
        score -= 3

    # Robots meta tag
    if not metrics.get("robots_meta_present"):
        suggestions.append(
            "Add a meta robots tag to control how search engines crawl and index the page."
        )
        score -= 1

    # Open Graph tags
    if not metrics.get("og_title_present"):
        suggestions.append(
            "Add an Open Graph title (og:title) to improve social sharing previews."
        )
        score -= 1
    if not metrics.get("og_description_present"):
        suggestions.append(
            "Add an Open Graph description (og:description) for better link previews."
        )
        score -= 1
    if not metrics.get("og_image_present"):
        suggestions.append(
            "Add an Open Graph image (og:image) to control the preview image on social media."
        )
        score -= 1

    # Structured data
    if not metrics.get("structured_data_present"):
        suggestions.append(
            "Implement JSON‑LD structured data to help search engines understand your content and win rich snippets."
        )
        score -= 3

    # Viewport meta
    if not metrics.get("viewport_present"):
        suggestions.append(
            "Add a responsive viewport meta tag (e.g., <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">)."
        )
        score -= 2

    # ARIA ratio (accessibility)
    aria_ratio = metrics.get("aria_ratio", 0)
    if isinstance(aria_ratio, (int, float)) and aria_ratio < 0.05:
        suggestions.append(
            "Increase the use of ARIA attributes on interactive elements to improve accessibility."
        )
        score -= 2

    # Lazy loading images
    lazy_ratio = metrics.get("lazy_loading_ratio", 0)
    if (
        isinstance(lazy_ratio, (int, float))
        and metrics.get("image_count", 0) > 0
        and lazy_ratio < 0.5
    ):
        suggestions.append(
            "Enable lazy loading (loading=\"lazy\") on images to improve perceived performance."
        )
        score -= 2

    # Heading structure: encourage at least one H2 for content hierarchy
    h2_count = metrics.get("h2_count", 0)
    if isinstance(h2_count, int) and h2_count == 0:
        suggestions.append(
            "Add H2 headings to structure your content and aid navigation."
        )
        score -= 2

    # Final clipping
    score = max(0, min(100, score))
    return score, suggestions


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
            # Evaluate the primary page
            primary_score, primary_suggestions = evaluate_metrics(primary_metrics)
            st.subheader("Primary Page Metrics")
            st.json(primary_metrics)
            st.markdown(f"**UX/SEO Score:** {primary_score}/100")
            if primary_suggestions:
                st.markdown("**Suggestions to improve:**")
                for s in primary_suggestions:
                    st.write("- " + s)
            if competitor_url:
                with st.spinner("Analyzing competitor page..."):
                    competitor_metrics = compute_metrics(competitor_url)
                competitor_score, competitor_suggestions = evaluate_metrics(competitor_metrics)
                st.subheader("Competitor Page Metrics")
                st.json(competitor_metrics)
                st.markdown(f"**Competitor UX/SEO Score:** {competitor_score}/100")
                # Show competitor suggestions
                if competitor_suggestions:
                    st.markdown("**Competitor Suggestions:**")
                    for s in competitor_suggestions:
                        st.write("- " + s)
                # Show score difference
                score_diff = primary_score - competitor_score
                st.markdown(f"**Score difference (Primary – Competitor):** {score_diff}")
                # Show metric differences
                diff = compare_metrics(primary_metrics, competitor_metrics)
                st.subheader("Metric Differences (Primary – Competitor)")
                st.json(diff)
        except Exception as exc:
            st.error(f"Error during analysis: {exc}")


if __name__ == "__main__":
    main()