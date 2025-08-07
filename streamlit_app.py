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
from io import BytesIO  # for reading uploaded images


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

    # Capture text of H2 headings for potential content gap analysis
    h2_tags = soup.find_all("h2")
    metrics["h2_texts"] = [tag.get_text(strip=True) for tag in h2_tags]

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
    # Semantic HTML landmarks
    # Semantic landmarks like <nav>, <header>, <main> and <footer> help
    # assistive technologies and improve overall structure.  We record
    # whether these elements are present.  Missing landmarks may indicate
    # accessibility or structural issues.
    metrics["nav_present"] = bool(soup.find("nav"))
    metrics["header_present"] = bool(soup.find("header"))
    metrics["main_present"] = bool(soup.find("main"))
    metrics["footer_present"] = bool(soup.find("footer"))

    # ---------------------------------------------------------------------------
    # Readability and text statistics
    # Extract all visible text from the page
    text = soup.get_text(separator=" ", strip=True)
    # Expose plain text so other functions (e.g. content gap, tone) can reuse it
    metrics["plain_text"] = text

    # Basic word, sentence and syllable counts for readability
    words = text.split()
    metrics["word_count"] = len(words)
    # Count sentences using ., ! and ? as delimiters
    metrics["sentence_count"] = text.count(".") + text.count("!") + text.count("?")
    metrics["syllable_count"] = _count_syllables_in_text(text)
    metrics["flesch_reading_ease"] = _flesch_reading_ease(
        metrics["word_count"], metrics["sentence_count"], metrics["syllable_count"]
    )
    # Language attribute
    html_tag = soup.find("html")
    metrics["lang"] = html_tag.get("lang") if html_tag else None

    # ----------------------------------------------------------------------
    # Tone analysis and keyword extraction
    # Normalise words: lowercase, strip punctuation
    cleaned_words: List[str] = []
    for token in words:
        # Remove any leading/trailing punctuation
        cleaned = re.sub(r"[^a-zA-Z0-9]", "", token.lower())
        if cleaned:
            cleaned_words.append(cleaned)
    # Define simple stopwords list (English and a few common French words)
    stopwords = {
        "a", "the", "and", "or", "of", "to", "in", "on", "for", "with", "is",
        "it", "this", "that", "an", "as", "at", "be", "by", "from", "are",
        "was", "were", "but", "not", "can", "we", "you", "your", "our", "nous",
        "vous", "pour", "que", "qui", "par", "dans", "les", "des", "une", "un",
        "le", "la", "et", "du", "de", "en", "se", "au", "aux", "ce", "ces", "sa",
        "ses", "sur", "plus", "pas", "ne"
    }
    filtered_words = [w for w in cleaned_words if w not in stopwords]
    # Compute top keywords (up to 10) by frequency
    from collections import Counter
    counter = Counter(filtered_words)
    metrics["top_keywords"] = counter.most_common(10)
    # Pronoun ratio as proxy for informal tone
    personal_pronouns = {
        "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "yourselves"
    }
    pronoun_count = sum(1 for w in cleaned_words if w in personal_pronouns)
    metrics["pronoun_ratio"] = (
        pronoun_count / len(cleaned_words) if cleaned_words else 0.0
    )
    # Count exclamation marks as proxy for tone excitement
    metrics["exclamation_count"] = text.count("!")
    # Determine a simplistic tone classification
    metrics["tone"] = "informal" if (
        metrics["pronoun_ratio"] > 0.05 or metrics["exclamation_count"] > 2
    ) else "formal"

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

    # Tone checks: high pronoun ratio or many exclamation marks may indicate an overly informal tone
    pronoun_ratio = metrics.get("pronoun_ratio", 0)
    exclamations = metrics.get("exclamation_count", 0)
    if isinstance(pronoun_ratio, (int, float)) and pronoun_ratio > 0.2:
        suggestions.append(
            "The copy appears highly informal (many personal pronouns); ensure tone suits your audience."
        )
        score -= 1
    if isinstance(exclamations, int) and exclamations > 3:
        suggestions.append(
            "Reduce the number of exclamation marks to maintain a professional tone."
        )
        score -= 1

    # Heading structure: encourage at least one H2 for content hierarchy
    h2_count = metrics.get("h2_count", 0)
    if isinstance(h2_count, int) and h2_count == 0:
        suggestions.append(
            "Add H2 headings to structure your content and aid navigation."
        )
        score -= 2

    # Semantic landmarks: nav, header, main, footer
    if not metrics.get("nav_present"):
        suggestions.append(
            "Add a <nav> landmark to define the main navigation region."
        )
        score -= 1
    if not metrics.get("header_present"):
        suggestions.append(
            "Include a <header> element to group introductory content and site identity."
        )
        score -= 1
    if not metrics.get("main_present"):
        suggestions.append(
            "Wrap the primary content in a <main> element to improve accessibility."
        )
        score -= 1
    if not metrics.get("footer_present"):
        suggestions.append(
            "Add a <footer> element to contain footer information and navigation."
        )
        score -= 1

    # End tone and content heuristics

    # Final clipping
    score = max(0, min(100, score))
    return score, suggestions


def content_gap(primary: Dict[str, object], competitor: Dict[str, object]) -> Dict[str, List[str]]:
    """Identify content gaps between the primary and competitor pages.

    The analysis considers both the top keywords extracted from each page
    and the H2 headings used.  Keywords or headings that appear on the
    competitor's page but not on the primary page are treated as
    opportunities for improvement.  This function returns a dictionary
    with two keys: ``keywords`` and ``headings``, each mapping to a list
    of missing terms.

    :param primary: Metrics for the primary page.
    :param competitor: Metrics for the competitor page.
    :returns: A dict ``{"keywords": [...], "headings": [...]}`` where each
              list contains items present on the competitor page but not on the
              primary.
    """
    # Top keywords difference
    primary_keywords = {kw for kw, _ in primary.get("top_keywords", [])}
    competitor_keywords = {kw for kw, _ in competitor.get("top_keywords", [])}
    missing_keywords = list(competitor_keywords - primary_keywords)
    # Heading differences (case‑insensitive comparison)
    primary_headings = {h.lower() for h in primary.get("h2_texts", [])}
    competitor_headings = {h.lower() for h in competitor.get("h2_texts", [])}
    missing_headings = [h for h in competitor_headings if h not in primary_headings]
    return {
        "keywords": sorted(missing_keywords),
        "headings": sorted(missing_headings),
    }


def generate_heatmap(metrics: Dict[str, object]):
    """Create a heatmap representing the relative distribution of key page elements.

    This helper visualises how many headings (H1–H3), images, links and words
    appear on a page by displaying a one‑row heatmap.  It is intended as a
    lightweight alternative to true user‑behaviour heatmaps and requires only
    ``matplotlib`` and ``numpy``.  If these libraries are missing, the
    function returns ``None`` so the caller can degrade gracefully.

    :param metrics: Metrics dictionary for a page.
    :returns: A matplotlib figure ready for rendering, or ``None`` if the
              plotting libraries are unavailable.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        return None
    labels = ["H1", "H2", "H3", "Images", "Links", "Words"]
    values = [
        metrics.get("h1_count", 0),
        metrics.get("h2_count", 0),
        metrics.get("h3_count", 0),
        metrics.get("image_count", 0),
        metrics.get("link_count", 0),
        metrics.get("word_count", 0),
    ]
    data = np.array([values])
    fig, ax = plt.subplots(figsize=(len(labels) * 0.8, 2))
    im = ax.imshow(data, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks([])
    ax.set_title("Distribution of page elements")
    for i, val in enumerate(values):
        ax.text(i, 0, f"{val}", va="center", ha="center", fontsize=9)
    return fig


def _spectral_residual_saliency(img) -> "np.ndarray":
    """Compute a saliency map using the spectral residual algorithm.

    This algorithm approximates visual attention by analysing the log spectrum of
    the image's Fourier transform.  It is a lightweight technique that does not
    require training data and can run on serverless platforms.  The returned
    saliency map is normalised to the range [0, 1].

    :param img: A BGR image as a NumPy array.
    :returns: A 2‑D NumPy array representing the saliency map.
    """
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    # Compute the 2D FFT and separate magnitude and phase
    dft = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT)
    real, imag = dft[:, :, 0], dft[:, :, 1]
    mag, phase = cv2.cartToPolar(real, imag)
    # Log amplitude spectrum
    log_mag = np.log(mag + 1e-8)
    # Average filter (spectral residual)
    avg_log_mag = cv2.boxFilter(log_mag, -1, (3, 3))
    spectral_residual = log_mag - avg_log_mag
    # Convert back to complex form
    exp_residual = np.exp(spectral_residual)
    real2, imag2 = cv2.polarToCart(exp_residual, phase)
    complex_spectrum = cv2.merge([real2, imag2])
    # Inverse FFT
    inverse = cv2.idft(complex_spectrum, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    saliency_map = inverse ** 2
    saliency_map = cv2.GaussianBlur(saliency_map, (9, 9), sigmaX=0)
    cv2.normalize(saliency_map, saliency_map, 0, 1, cv2.NORM_MINMAX)
    return saliency_map


def generate_saliency_heatmap(image_bytes: bytes):
    """Generate a saliency heatmap from an uploaded image.

    The function accepts raw image bytes (PNG or JPEG) and computes a
    visual saliency map using the spectral residual algorithm.  It returns
    a matplotlib figure displaying the heatmap.  If the required
    libraries (Pillow, OpenCV, NumPy, Matplotlib) are unavailable, ``None``
    is returned.

    :param image_bytes: Raw bytes of the uploaded image file.
    :returns: A matplotlib figure or ``None`` if dependencies are missing.
    """
    try:
        from PIL import Image  # type: ignore
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        return None
    # Read image from bytes
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    saliency_map = _spectral_residual_saliency(img_bgr)
    # Display original and saliency overlay side by side.  Normalise saliency for overlay.
    saliency_norm = saliency_map
    # Create two subplots: original screenshot and saliency overlay on top of original
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    # Original image
    axes[0].imshow(img)
    axes[0].axis("off")
    axes[0].set_title("Capture originale")
    # Overlay: show the original image with saliency map overlay as a heatmap
    axes[1].imshow(img)
    axes[1].imshow(saliency_norm, cmap="hot", alpha=0.6)
    axes[1].axis("off")
    axes[1].set_title("Carte de saillance")
    fig.suptitle("Analyse prédite de l'attention visuelle", fontsize=10)
    return fig


def check_gambling_compliance(url: str, metrics: Dict[str, object]) -> List[str]:
    """Check gambling‑site compliance elements such as responsible gambling notices.

    If the domain suggests a gambling or betting site (keywords like 'casino',
    'poker', 'bet', 'game'), ensure that the page contains typical warnings
    about responsible gambling and age restrictions.  This function returns
    suggestions if required notices are missing.

    :param url: The URL of the page being analysed.
    :param metrics: Metrics dictionary containing the plain text.
    :returns: A list of additional suggestions related to responsible gambling.
    """
    domain = urlparse(url).netloc.lower()
    gambling_keywords = ["casino", "poker", "bet", "gambling", "jeu"]
    if not any(kw in domain for kw in gambling_keywords):
        return []
    text = metrics.get("plain_text", "").lower()
    suggestions: List[str] = []
    # Warnings expected: responsible gambling notice and age restrictions
    if "responsible" not in text and "jouez" not in text:
        suggestions.append(
            "Include a notice about playing responsibly and support for problem gamblers."
        )
    if not any(age in text for age in ["18+", "21+", "18 ans", "21 ans"]):
        suggestions.append(
            "Display a clear age restriction (e.g., 18+ or 21+) to comply with regulations."
        )
    return suggestions


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
    """Run the Streamlit app with an enhanced UI and additional analyses."""
    if st is None:
        raise RuntimeError(
            "Streamlit is not installed. Run this app via `streamlit run` or "
            "install streamlit (pip install streamlit)."
        )

    st.set_page_config(page_title="UX‑SEO Audit", layout="wide")

    # ---------------------------------------------------------------------
    # Inject a bit of CSS to improve the visual presentation.  The design
    # draws inspiration from modern tools like Framer: light background,
    # rounded panels with gentle shadows and consistent typography.  This
    # styling is intentionally scoped to elements we add via HTML below.
    st.markdown(
        """
        <style>
        /* Base styles: use a clean sans‑serif font and light background */
        html, body, .main, .stApp {background-color: #f9fafb; color: #111827; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;}
        /* Card container with rounded corners and subtle shadow */
        .card {
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 1.5rem;
            margin-top: 1rem;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }
        /* Section header styling */
        .section-title {
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        /* Suggestions list styling */
        .suggestions ul {list-style-type: disc; margin-left: 1rem;}
        .suggestions li {margin-bottom: 0.35rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("UX‑SEO Audit (Lite)")
    st.write(
        "Obtenez un aperçu rapide de l'expérience utilisateur (UX) et du référencement (SEO) "
        "d'une page en entrant son URL. Facultativement, comparez‑la à un concurrent. "
        "Cette version légère fonctionne sans Lighthouse pour garantir un temps d'attente court et un hébergement gratuit."
    )

    url = st.text_input("Page URL", "https://example.com")
    competitor_url = st.text_input("Competitor URL (optional)")
    run_audit = st.button("Run Audit")

    if run_audit:
        if not url:
            st.error("Please enter a valid URL.")
            return
        try:
            # Analyse primaire
            with st.spinner("Analyzing primary page..."):
                primary_metrics = compute_metrics(url)
            primary_score, primary_suggestions = evaluate_metrics(primary_metrics)
            # Domain‑specific compliance
            primary_gamble_suggestions = check_gambling_compliance(url, primary_metrics)
            primary_suggestions.extend(primary_gamble_suggestions)

            # UI layout for primary page
            st.header("Primary Page Results")
            tabs = st.tabs(["Summary", "Heatmap", "Keywords", "Suggestions"])
            # Summary tab: show key metrics and score, wrapped in a card
            with tabs[0]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                # Use columns to display metrics in rows of three
                col1, col2, col3 = st.columns(3)
                col1.metric("UX/SEO Score", f"{primary_score}/100")
                col2.metric("Word Count", int(primary_metrics.get("word_count", 0)))
                col3.metric("Flesch Reading Ease", float(primary_metrics.get("flesch_reading_ease", 0)))
                col1.metric("Response Time (s)", float(primary_metrics.get("response_time_seconds", 0)))
                col2.metric("Page Size (KB)", int(primary_metrics.get("page_size_bytes", 0) / 1024))
                col3.metric("Tone", str(primary_metrics.get("tone", "unknown")).capitalize())
                st.markdown("<div class='section-title'>Detailed Metrics</div>", unsafe_allow_html=True)
                st.json(primary_metrics)
                st.markdown('</div>', unsafe_allow_html=True)
            # Heatmap tab
            with tabs[1]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                # Element distribution heatmap
                st.markdown("<div class='section-title'>Element Distribution</div>", unsafe_allow_html=True)
                fig_dist = generate_heatmap(primary_metrics)
                if fig_dist is not None:
                    st.pyplot(fig_dist)
                else:
                    st.info(
                        "Matplotlib (or NumPy) is not installed in this environment. "
                        "Distribution heatmap generation is disabled."
                    )
                # Visual saliency heatmap (requires uploaded screenshot)
                st.markdown("<div class='section-title'>Visual Saliency (optional)</div>", unsafe_allow_html=True)
                uploaded_primary = st.file_uploader(
                    "Téléchargez une capture d'écran de la page (PNG/JPEG) pour estimer l'attention visuelle",
                    type=["png", "jpg", "jpeg"],
                    key="saliency_primary"
                )
                if uploaded_primary is not None:
                    fig_sal = generate_saliency_heatmap(uploaded_primary.getvalue())
                    if fig_sal is not None:
                        st.pyplot(fig_sal)
                    else:
                        st.info(
                            "La génération de carte de saillance n'est pas disponible (bibliothèques manquantes)."
                        )
                st.markdown('</div>', unsafe_allow_html=True)
            # Keywords tab
            with tabs[2]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("<div class='section-title'>Top Keywords</div>", unsafe_allow_html=True)
                keywords = primary_metrics.get("top_keywords", [])
                if keywords:
                    st.table(keywords)
                else:
                    st.write("No significant keywords extracted.")
                st.markdown('</div>', unsafe_allow_html=True)
            # Suggestions tab
            with tabs[3]:
                st.markdown('<div class="card suggestions">', unsafe_allow_html=True)
                st.markdown("<div class='section-title'>Suggestions</div>", unsafe_allow_html=True)
                if primary_suggestions:
                    # Use HTML list for better spacing
                    html_list = "<ul>" + "".join(f"<li>{s}</li>" for s in primary_suggestions) + "</ul>"
                    st.markdown(html_list, unsafe_allow_html=True)
                else:
                    st.write("No suggestions – well done!")
                st.markdown('</div>', unsafe_allow_html=True)

            # If competitor provided
            if competitor_url:
                with st.spinner("Analyzing competitor page..."):
                    competitor_metrics = compute_metrics(competitor_url)
                competitor_score, competitor_suggestions = evaluate_metrics(competitor_metrics)
                competitor_gamble_suggestions = check_gambling_compliance(competitor_url, competitor_metrics)
                competitor_suggestions.extend(competitor_gamble_suggestions)
                # Content gap analysis
                gap = content_gap(primary_metrics, competitor_metrics)
                # Score difference
                score_diff = primary_score - competitor_score
                st.header("Competitor Comparison")
                cmp_tabs = st.tabs(["Competitor Summary", "Content Gap", "Metric Differences", "Competitor Suggestions"])
                with cmp_tabs[0]:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Competitor Score", f"{competitor_score}/100", delta=f"{score_diff:+}")
                    c2.metric("Words", int(competitor_metrics.get("word_count", 0)), delta=int(primary_metrics.get("word_count", 0)) - int(competitor_metrics.get("word_count", 0)))
                    c3.metric("Flesch", float(competitor_metrics.get("flesch_reading_ease", 0)), delta=float(primary_metrics.get("flesch_reading_ease", 0)) - float(competitor_metrics.get("flesch_reading_ease", 0)))
                    st.markdown("<div class='section-title'>Competitor Metrics</div>", unsafe_allow_html=True)
                    st.json(competitor_metrics)
                    st.markdown('</div>', unsafe_allow_html=True)
                with cmp_tabs[1]:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("<div class='section-title'>Content Gap Analysis</div>", unsafe_allow_html=True)
                    # Display missing keywords and headings separately.  Use Streamlit columns for clarity.
                    missing_kw = gap.get("keywords", [])
                    missing_h2 = gap.get("headings", [])
                    if not missing_kw and not missing_h2:
                        st.write("No significant content gaps detected – your page covers similar topics.")
                    else:
                        kw_col, h_col = st.columns(2)
                        kw_col.subheader("Mots‑clés manquants")
                        if missing_kw:
                            kw_col.write(", ".join(missing_kw))
                        else:
                            kw_col.write("Aucun mot‑clé manquant.")
                        h_col.subheader("Rubriques H2 manquantes")
                        if missing_h2:
                            h_col.write("; ".join(missing_h2))
                        else:
                            h_col.write("Aucune rubrique manquante.")
                    st.markdown('</div>', unsafe_allow_html=True)
                with cmp_tabs[2]:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("<div class='section-title'>Metric Differences (Primary – Competitor)</div>", unsafe_allow_html=True)
                    diff = compare_metrics(primary_metrics, competitor_metrics)
                    st.json(diff)
                    st.markdown('</div>', unsafe_allow_html=True)
                with cmp_tabs[3]:
                    st.markdown('<div class="card suggestions">', unsafe_allow_html=True)
                    st.markdown("<div class='section-title'>Competitor Suggestions</div>", unsafe_allow_html=True)
                    if competitor_suggestions:
                        html_list = "<ul>" + "".join(f"<li>{s}</li>" for s in competitor_suggestions) + "</ul>"
                        st.markdown(html_list, unsafe_allow_html=True)
                    else:
                        st.write("No suggestions – competitor page looks strong.")
                    st.markdown('</div>', unsafe_allow_html=True)
        except Exception as exc:
            st.error(f"Error during analysis: {exc}")


if __name__ == "__main__":
    main()