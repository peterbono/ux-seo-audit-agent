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
# We no longer accept uploaded screenshots for saliency analysis, so
# BytesIO is no longer needed.  Removing this import avoids confusion.


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
    # Business‑oriented signals: calls‑to‑action, trust elements and forms
    # Many affiliate and marketing sites rely on clear calls‑to‑action (CTA)
    # buttons or links to drive conversions.  We scan all <a> and <button>
    # elements and count how many contain common CTA verbs.  Examples include
    # "buy", "sign up", "join", "subscribe", "download" and "learn more".
    cta_keywords = {
        "buy", "order", "download", "get", "join", "sign up",
        "sign-up", "register", "subscribe", "start", "learn more",
        "discover", "claim", "book", "try", "go"
    }
    cta_count = 0
    cta_texts: List[str] = []
    for elem in soup.find_all(["a", "button"]):
        text_content = elem.get_text(strip=True).lower()
        if not text_content:
            continue
        for kw in cta_keywords:
            # match keyword at start or anywhere for multi‑word phrases
            if kw in text_content:
                cta_count += 1
                cta_texts.append(text_content)
                break
    metrics["cta_count"] = cta_count
    metrics["cta_texts"] = cta_texts

    # Trust signals: look for words that imply security, licensing or social proof.
    # We search the full page text for occurrences of these keywords.  Each
    # unique keyword found counts once, so the count reflects the variety of
    # trust signals rather than raw frequency.  This simple heuristic helps
    # identify whether the page includes badges like "SSL", "licensed", or
    # "trusted reviews".
    trust_keywords = {
        "ssl", "secure", "safe", "trusted", "licence", "license",
        "licensed", "regulated", "mga", "malta gaming", "trust", "fair",
        "état", "témoignages", "avis", "review", "testimonials"
    }
    # Extract the full page text once for trust signal detection.  We defer storing
    # this as metrics["plain_text"] until later, but reusing it here avoids
    # referencing an undefined variable.  Without this, referencing ``text``
    # before it is assigned causes a runtime error (see issue reported by users).
    full_page_text = soup.get_text(separator=" ", strip=True)
    lower_text = full_page_text.lower()
    found_trust = set()
    for kw in trust_keywords:
        if kw in lower_text:
            found_trust.add(kw)
    metrics["trust_keyword_count"] = len(found_trust)
    metrics["trust_keywords_found"] = sorted(found_trust)

    # Forms detection: count the number of <form> elements.  Forms are
    # indicative of lead capture (newsletter signup, contact, registration) and
    # are important for affiliate marketing pages aiming to capture user
    # information.
    metrics["form_count"] = len(soup.find_all("form"))

    # ---------------------------------------------------------------------------
    # Business growth and content signals
    # In addition to basic conversion signals, affiliate and marketing pages
    # often leverage social proof, promotions, comparisons, cross‑sells and
    # help sections to encourage engagement and conversion.  We scan the full
    # text for keywords that indicate these elements and record simple counts.
    # These counts feed into suggestions and business‑oriented KPIs later on.

    # Social proof: testimonials, reviews, ratings
    social_keywords = {
        "testimonial", "testimonials", "avis", "reviews", "review",
        "note", "notes", "évaluation", "ratings", "rating", "retour",
        "feedback", "témoignage"
    }
    social_count = sum(1 for kw in social_keywords if kw in lower_text)
    metrics["social_proof_count"] = social_count

    # Promotions and bonuses
    promo_keywords = {
        "bonus", "promo", "promotion", "réduction", "discount",
        "code", "coupon", "gratuit", "offre", "offres"
    }
    promo_count = sum(1 for kw in promo_keywords if kw in lower_text)
    metrics["promo_count"] = promo_count

    # Benefits and advantages
    benefit_keywords = {
        "avantage", "avantages", "benefit", "benefits", "pourquoi",
        "pourquoi nous", "avantageux", "bénéfice"
    }
    benefit_count = sum(1 for kw in benefit_keywords if kw in lower_text)
    metrics["benefit_count"] = benefit_count

    # Comparison or versus content (e.g., "X vs Y", "compare")
    comparison_keywords = {" vs ", " versus ", "compare", "comparaison"}
    comparison_count = sum(1 for kw in comparison_keywords if kw in lower_text)
    metrics["comparison_count"] = comparison_count

    # Cross‑sell and recommendation cues
    recommendation_keywords = {
        "recommend", "recommended", "similar", "related", "vous pourriez aussi aimer",
        "people also", "articles similaires", "produits similaires"
    }
    recommendation_count = sum(1 for kw in recommendation_keywords if kw in lower_text)
    metrics["recommendation_count"] = recommendation_count

    # Frequently asked questions or help section
    faq_keywords = {"faq", "questions fréquentes", "foire aux questions",
                    "faqs", "aide", "help"}
    metrics["faq_present"] = any(kw in lower_text for kw in faq_keywords)

    # Search bar presence: look for <input type="search"> elements or input fields
    # with a 'search' role.  We also check for text placeholders commonly used
    # in search fields (e.g., "search", "rechercher").
    search_inputs = soup.find_all("input", attrs={"type": "search"})
    # Check for text inputs with placeholder containing search terms
    for input_elem in soup.find_all("input", attrs={"type": "text"}):
        placeholder = input_elem.get("placeholder", "").lower()
        if "search" in placeholder or "rechercher" in placeholder:
            search_inputs.append(input_elem)
    metrics["search_present"] = len(search_inputs) > 0

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


def business_analysis(metrics: Dict[str, object]) -> Tuple[int, Dict[str, float], List[str]]:
    """Evaluate business‑oriented KPIs and return a score, ratios and suggestions.

    This function interprets counts of conversion and growth signals—CTAs,
    trust keywords, forms, social proof, promotions, benefits, comparisons,
    recommendations and FAQs—to produce a business score between 0 and 100.
    Ratios per 1000 words help normalise counts across pages of different
    lengths. Suggestions are generated when ratios fall outside typical
    ranges. The aim is to provide concrete actions that maximise
    conversion potential and growth for marketing or affiliate sites.

    :param metrics: Metrics dictionary from ``compute_metrics``.
    :returns: ``(score, ratios, suggestions)`` where ``score`` is an
              integer between 0 and 100, ``ratios`` maps descriptive
              keys (e.g., "cta_per_1000_words") to floats, and
              ``suggestions`` is a list of improvement recommendations.
    """
    score = 100
    suggestions: List[str] = []
    word_count = max(1, int(metrics.get("word_count", 0)))
    # Calculate ratios per 1000 words for normalization
    cta_count = int(metrics.get("cta_count", 0))
    trust_count = int(metrics.get("trust_keyword_count", 0))
    form_count = int(metrics.get("form_count", 0))
    social_count = int(metrics.get("social_proof_count", 0))
    promo_count = int(metrics.get("promo_count", 0))
    benefit_count = int(metrics.get("benefit_count", 0))
    comparison_count = int(metrics.get("comparison_count", 0))
    recommendation_count = int(metrics.get("recommendation_count", 0))
    faq_present = bool(metrics.get("faq_present"))
    search_present = bool(metrics.get("search_present"))
    # Ratios per 1000 words
    cta_ratio = (cta_count / word_count) * 1000
    trust_ratio = (trust_count / word_count) * 1000
    social_ratio = (social_count / word_count) * 1000
    promo_ratio = (promo_count / word_count) * 1000
    benefit_ratio = (benefit_count / word_count) * 1000
    comparison_ratio = (comparison_count / word_count) * 1000
    recommendation_ratio = (recommendation_count / word_count) * 1000
    ratios: Dict[str, float] = {
        "cta_per_1000_words": cta_ratio,
        "trust_per_1000_words": trust_ratio,
        "social_proof_per_1000_words": social_ratio,
        "promo_per_1000_words": promo_ratio,
        "benefit_per_1000_words": benefit_ratio,
        "comparison_per_1000_words": comparison_ratio,
        "recommendation_per_1000_words": recommendation_ratio,
        "forms": float(form_count),
        "faq_present": 1.0 if faq_present else 0.0,
        "search_present": 1.0 if search_present else 0.0,
    }
    # Define heuristics: target ranges per 1000 words
    # CTA: 1–3 per 1000 words is optimal; below or above suggests action
    if cta_ratio < 1:
        suggestions.append(
            "Ajoutez un ou plusieurs appels à l’action clairs (1–3 pour 1000 mots) pour guider les conversions."
        )
        score -= 5
    elif cta_ratio > 3:
        suggestions.append(
            "Réduisez le nombre d’appels à l’action (1–3 pour 1000 mots) afin d’éviter une surcharge et de ne pas disperser l’attention."
        )
        score -= 3
    # Trust signals: au moins 2 mots de confiance par 1000 mots
    if trust_ratio < 2:
        suggestions.append(
            "Ajoutez des éléments de confiance (mots comme ‘sécurisé’, ‘licencié’, avis clients) pour rassurer les visiteurs."
        )
        score -= 3
    # Forms: au moins un formulaire pour les captures de leads
    if form_count == 0:
        suggestions.append(
            "Ajoutez un formulaire d’inscription ou de contact pour capter des prospects."
        )
        score -= 4
    # Social proof: 1–2 mentions par 1000 mots
    if social_ratio < 1:
        suggestions.append(
            "Ajoutez des preuves sociales (avis, témoignages, notes) pour augmenter la crédibilité."
        )
        score -= 2
    # Promotions: 0.5–2 par 1000 mots
    if promo_ratio < 0.5:
        suggestions.append(
            "Mettez en avant des promotions ou bonus pour inciter à l’action."
        )
        score -= 2
    # Benefits: 1–3 par 1000 mots
    if benefit_ratio < 1:
        suggestions.append(
            "Soulignez clairement les avantages et bénéfices de votre offre pour convaincre les visiteurs."
        )
        score -= 2
    # Comparisons: suggérer d’inclure comparatifs si absent
    if comparison_ratio < 0.1:
        suggestions.append(
            "Ajoutez des tableaux comparatifs ou des articles ‘X vs Y’ pour aider les visiteurs à choisir."
        )
        score -= 1
    # Recommendations/cross‑selling: ratio < 0.5 -> suggestion
    if recommendation_ratio < 0.5:
        suggestions.append(
            "Proposez des recommandations de produits ou d’articles similaires pour augmenter la valeur moyenne."
        )
        score -= 1
    # FAQ and search presence
    if not faq_present:
        suggestions.append(
            "Intégrez une FAQ ou section d’aide pour répondre aux questions courantes et améliorer la confiance."
        )
        score -= 1
    if not search_present:
        suggestions.append(
            "Ajoutez une barre de recherche ou des filtres pour faciliter la navigation et la découverte de contenu."
        )
        score -= 1
    # Clip score between 0 and 100
    score = max(0, min(100, score))
    return score, ratios, suggestions


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

    # -------------------------------------------------------------------
    # Business conversion heuristics
    # Calls‑to‑action (CTA), trust signals and forms are critical for
    # conversion‑focused affiliate or marketing pages.  A page with no CTA
    # buttons or links leaves visitors unsure how to proceed, a lack of trust
    # indicators can raise doubts, and the absence of a form limits lead
    # capture.  Conversely, an excess of CTAs may appear overly aggressive.
    cta_count = metrics.get("cta_count", 0)
    if isinstance(cta_count, int):
        if cta_count == 0:
            suggestions.append(
                "Add a clear call‑to‑action button or link (e.g., ‘S’inscrire’, ‘Obtenir le bonus’) to guide users towards conversion."
            )
            score -= 4
        elif cta_count > 5:
            suggestions.append(
                "Trop de boutons ou liens d’appel à l’action peuvent sembler agressifs ; privilégiez quelques CTA bien visibles."
            )
            score -= 2
    trust_count = metrics.get("trust_keyword_count", 0)
    if isinstance(trust_count, int) and trust_count == 0:
        suggestions.append(
            "Incluez des signes de confiance (badges SSL, licences, avis clients) pour rassurer les visiteurs."
        )
        score -= 3
    form_count = metrics.get("form_count", 0)
    if isinstance(form_count, int) and form_count == 0:
        suggestions.append(
            "Ajoutez un formulaire simple d’inscription ou de contact pour capter des leads et faciliter les conversions."
        )
        score -= 3

    # -------------------------------------------------------------------
    # High‑level UX and design heuristics (Gestalt and Nielsen)
    # These recommendations are independent of specific metric thresholds
    # and serve as general guidance for business‑oriented optimisation.
    # Gestalt principles emphasise grouping, contrast and alignment, while
    # Nielsen’s heuristics encourage simplicity, feedback and consistency.
    suggestions.append(
        "Utilisez des lois de Gestalt pour améliorer la hiérarchie visuelle : regroupez les éléments liés par proximité et similaires par couleur ou forme, et veillez à des espacements cohérents."
    )
    suggestions.append(
        "Mettez en avant vos appels à l’action avec des contrastes de couleurs et un alignement clair afin qu’ils ressortent immédiatement."
    )
    suggestions.append(
        "Simplifiez l’interface et adoptez des modèles de navigation familiers pour réduire la charge cognitive (heuristique de Nielsen : cohérence et normes)."
    )
    suggestions.append(
        "Donnez un feedback explicite aux actions des utilisateurs (messages de confirmation, changements visuels) et offrez la possibilité d’annuler facilement (heuristiques de visibilité de l’état du système et de contrôle par l’utilisateur)."
    )

    # -------------------------------------------------------------------
    # Business and growth heuristics
    # Le manque de certaines sections clés (ex. témoignages, promotions, FAQ) peut
    # limiter la conversion ou la confiance.  Nous émettons des recommandations
    # spécifiques basées sur les nouveaux signaux de croissance calculés dans
    # compute_metrics().  Chaque absence déduit un petit nombre de points afin
    # de mettre l’accent sur leur importance sans alourdir excessivement
    # l’impact sur le score global.
    social_count = metrics.get("social_proof_count", 0)
    if isinstance(social_count, int) and social_count == 0:
        suggestions.append(
            "Ajoutez des preuves sociales (témoignages, avis ou notes) pour renforcer la confiance et la crédibilité."
        )
        score -= 2
    promo_count = metrics.get("promo_count", 0)
    if isinstance(promo_count, int) and promo_count == 0:
        suggestions.append(
            "Envisagez d’inclure des promotions, codes promo ou bonus pour stimuler les conversions et l’engagement."
        )
        score -= 1
    benefit_count = metrics.get("benefit_count", 0)
    if isinstance(benefit_count, int) and benefit_count == 0:
        suggestions.append(
            "Mettez clairement en avant les bénéfices et avantages de votre offre pour persuader les visiteurs."
        )
        score -= 1
    comparison_count = metrics.get("comparison_count", 0)
    if isinstance(comparison_count, int) and comparison_count == 0:
        suggestions.append(
            "Intégrez des comparatifs ou des tableaux ‘X vs Y’ pour montrer votre valeur face aux concurrents."
        )
        score -= 1
    recommendation_count = metrics.get("recommendation_count", 0)
    if isinstance(recommendation_count, int) and recommendation_count == 0:
        suggestions.append(
            "Ajoutez des recommandations ou des sections ‘vous pourriez aussi aimer’ pour augmenter la valeur moyenne d’achat."
        )
        score -= 1
    if not metrics.get("faq_present"):
        suggestions.append(
            "Proposez une FAQ ou une rubrique d’aide pour répondre aux questions courantes et réduire le taux de rebond."
        )
        score -= 1
    if not metrics.get("search_present"):
        suggestions.append(
            "Intégrez un champ de recherche ou des filtres pour aider les visiteurs à trouver rapidement ce qu’ils cherchent."
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


def layout_analysis(metrics: Dict[str, object]) -> Tuple[int, Dict[str, float], List[str]]:
    """Assess the structural layout and visual hierarchy of a page.

    Unlike a saliency heatmap that requires a screenshot, this function
    computes a synthetic "Layout Score" based on ratios of headings,
    images and links relative to the overall word count and the presence
    of semantic landmarks (nav, header, main, footer).  The aim is to
    approximate whether the content is well structured and visually
    balanced.  A higher score (closer to 100) indicates a healthy mix
    of structural elements, while very low or very high ratios suggest
    cluttered or sparse layouts.

    The returned ratios dictionary contains the actual ratios used for
    scoring, which can be displayed in the UI.  Suggestions provide
    guidance on improving the page layout.

    :param metrics: A metrics dictionary from ``compute_metrics``.
    :returns: ``(score, ratios, suggestions)`` where ``score`` is an
              integer between 0 and 100, ``ratios`` maps descriptive keys
              ("heading_ratio", "image_ratio", "link_ratio", "landmark_count")
              to numeric values, and ``suggestions`` is a list of
              human‑readable recommendations.
    """
    # Extract counts
    word_count = max(1, int(metrics.get("word_count", 0)))
    total_headings = int(metrics.get("h1_count", 0)) + int(metrics.get("h2_count", 0)) + int(metrics.get("h3_count", 0))
    image_count = int(metrics.get("image_count", 0))
    link_count = int(metrics.get("link_count", 0))
    # Compute ratios relative to words
    heading_ratio = total_headings / word_count  # headings per word
    image_ratio = image_count / word_count       # images per word
    link_ratio = link_count / word_count         # links per word
    # Count semantic landmarks present
    landmark_count = sum(
        1 for key in ["nav_present", "header_present", "main_present", "footer_present"]
        if metrics.get(key)
    )
    # Ratios dictionary for display
    ratios = {
        "heading_ratio": heading_ratio,
        "image_ratio": image_ratio,
        "link_ratio": link_ratio,
        "landmark_count": landmark_count,
    }
    # Scoring heuristics: assign up to 25 points for each category
    def subscore(actual: float, target: float, weight: float) -> float:
        """Compute a subscore based on proximity of actual ratio to target.

        If the ratio is within a factor of 0.5–2 of the target, full weight
        is awarded.  If it is moderately off (0.2–5×), 70 % of the weight
        is awarded.  Otherwise only 40 % of the weight is given.  This
        simple scheme penalises extremely high or low ratios without
        over‑penalising reasonable variation.
        """
        if actual == 0 or target == 0:
            return weight * 0.4
        ratio = actual / target
        if 0.5 <= ratio <= 2.0:
            return weight
        elif 0.2 <= ratio <= 5.0:
            return weight * 0.7
        else:
            return weight * 0.4
    # Define target ratios: these are empirical values representing a
    # healthy distribution for typical web pages.
    target_heading_ratio = 1 / 300  # roughly one heading for every 300 words
    target_image_ratio = 1 / 500   # roughly one image for every 500 words
    target_link_ratio = 1 / 100    # roughly one link for every 100 words
    score = 0
    score += subscore(heading_ratio, target_heading_ratio, 25)
    score += subscore(image_ratio, target_image_ratio, 25)
    score += subscore(link_ratio, target_link_ratio, 25)
    # Landmarks contribute linearly: each present landmark gives 6.25 points
    score += (landmark_count / 4) * 25
    # Clip score and convert to int
    layout_score = int(min(max(score, 0), 100))
    # Suggestions
    suggestions: List[str] = []
    # Headings suggestions
    if heading_ratio < target_heading_ratio * 0.5:
        suggestions.append(
            "Add more headings (H1–H3) to break up long stretches of text and improve hierarchy."
        )
    elif heading_ratio > target_heading_ratio * 2:
        suggestions.append(
            "Reduce the number of headings or group sections more logically to avoid clutter."
        )
    # Images suggestions
    if image_ratio < target_image_ratio * 0.5 and image_count > 0:
        suggestions.append(
            "Consider adding more images or illustrations to enrich the content visually."
        )
    elif image_ratio > target_image_ratio * 2:
        suggestions.append(
            "Too many images relative to text can distract readers; remove or combine images where appropriate."
        )
    # Links suggestions
    if link_ratio < target_link_ratio * 0.5 and link_count > 0:
        suggestions.append(
            "Increase internal linking to guide users and search engines through your content."
        )
    elif link_ratio > target_link_ratio * 2:
        suggestions.append(
            "Reduce the number of links or ensure they serve clear navigation and are not overwhelming."
        )
    # Semantic landmarks suggestions
    if landmark_count < 4:
        missing = []
        for key, label in [
            ("nav_present", "<nav>"),
            ("header_present", "<header>"),
            ("main_present", "<main>"),
            ("footer_present", "<footer>")
        ]:
            if not metrics.get(key):
                missing.append(label)
        suggestions.append(
            f"Add semantic landmarks {', '.join(missing)} to improve structure and accessibility."
        )
    return layout_score, ratios, suggestions


def generate_heatmap(metrics: Dict[str, object]):
    """Deprecated: previously created a heatmap of element counts.

    This function is retained for backwards compatibility but now simply
    returns ``None``.  Earlier versions of the app displayed a heatmap of
    headings, images, links and word counts.  Feedback indicated that this
    chart did not provide meaningful insights, so it has been removed from
    the interface.  The import and plotting code remain commented out to
    avoid requiring Matplotlib and NumPy on lightweight deployments.

    :param metrics: Unused.
    :returns: ``None`` always.
    """
    return None




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
            # Layout and visual hierarchy analysis
            layout_score, layout_ratios, layout_suggestions = layout_analysis(primary_metrics)
            primary_suggestions.extend(layout_suggestions)
            # Business analysis suggestions
            biz_score, biz_ratios, biz_suggestions = business_analysis(primary_metrics)
            primary_suggestions.extend(biz_suggestions)

            # UI layout for primary page
            st.header("Primary Page Results")
            # Tabs: Summary, Structure and Visual Hierarchy, Keywords, Conversion, Growth & Business, Suggestions
            tabs = st.tabs([
                "Summary",
                "Structure",
                "Keywords",
                "Conversion",
                "Growth",
                "Suggestions",
            ])
            # Summary tab: show key metrics and scores, wrapped in a card
            with tabs[0]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                # Use columns to display metrics in rows of three
                col1, col2, col3 = st.columns(3)
                col1.metric("UX/SEO Score", f"{primary_score}/100")
                col2.metric("Layout Score", f"{layout_score}/100")
                col3.metric("Tone", str(primary_metrics.get("tone", "unknown")).capitalize())
                col1.metric("Words", int(primary_metrics.get("word_count", 0)))
                col2.metric("Flesch Reading Ease", float(primary_metrics.get("flesch_reading_ease", 0)))
                col3.metric("Response Time (s)", float(primary_metrics.get("response_time_seconds", 0)))
                col1.metric("Page Size (KB)", int(primary_metrics.get("page_size_bytes", 0) / 1024))
                col2.metric("Internal Links", int(primary_metrics.get("internal_link_count", 0)))
                col3.metric("External Links", int(primary_metrics.get("external_link_count", 0)))
                st.markdown("<div class='section-title'>Detailed Metrics</div>", unsafe_allow_html=True)
                st.json(primary_metrics)
                st.markdown('</div>', unsafe_allow_html=True)
            # Structure tab
            with tabs[1]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("<div class='section-title'>Layout & Visual Hierarchy</div>", unsafe_allow_html=True)
                # Show ratios and landmark count in columns
                lc1, lc2, lc3, lc4 = st.columns(4)
                heading_ratio = layout_ratios.get("heading_ratio", 0)
                image_ratio = layout_ratios.get("image_ratio", 0)
                link_ratio = layout_ratios.get("link_ratio", 0)
                landmark_count = layout_ratios.get("landmark_count", 0)
                # Format ratios as elements per thousand words for readability
                hr_perk = heading_ratio * 1000
                ir_perk = image_ratio * 1000
                lr_perk = link_ratio * 1000
                lc1.metric("Headings per 1000 words", f"{hr_perk:.2f}", "Target ~3")
                lc2.metric("Images per 1000 words", f"{ir_perk:.2f}", "Target ~2")
                lc3.metric("Links per 1000 words", f"{lr_perk:.2f}", "Target ~10")
                lc4.metric("Semantic Landmarks", f"{int(landmark_count)}/4")
                # Provide explanation and tips
                st.markdown(
                    "Plus le ratio d'éléments est proche des valeurs cibles, mieux la hiérarchie visuelle est équilibrée. "
                    "La présence des éléments <nav>, <header>, <main> et <footer> améliore l'accessibilité et la structure.",
                    unsafe_allow_html=True,
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
            # Conversion tab
            with tabs[3]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("<div class='section-title'>Conversion & Trust</div>", unsafe_allow_html=True)
                cc, tc, fc = st.columns(3)
                cta_count = int(primary_metrics.get("cta_count", 0))
                trust_count = int(primary_metrics.get("trust_keyword_count", 0))
                form_count = int(primary_metrics.get("form_count", 0))
                cc.metric("CTA count", cta_count, "+" if cta_count > 0 else "")
                tc.metric("Trust signals", trust_count, "+" if trust_count > 0 else "")
                fc.metric("Forms", form_count, "+" if form_count > 0 else "")
                # Show details: CTA texts and trust keywords
                if primary_metrics.get("cta_texts"):
                    st.subheader("Calls‑to‑action trouvées")
                    st.write(", ".join(primary_metrics.get("cta_texts", [])))
                if primary_metrics.get("trust_keywords_found"):
                    st.subheader("Mots‑clés de confiance trouvés")
                    st.write(", ".join(primary_metrics.get("trust_keywords_found", [])))
                st.markdown('</div>', unsafe_allow_html=True)

            # Growth & Business tab
            with tabs[4]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("<div class='section-title'>Growth & Business</div>", unsafe_allow_html=True)
                # Compute business score, ratios and suggestions
                biz_score, biz_ratios, biz_suggestions = business_analysis(primary_metrics)
                # Display key KPIs in two rows of columns
                r1c1, r1c2, r1c3, r1c4 = st.columns(4)
                r1c1.metric("Business Score", f"{biz_score}/100")
                r1c2.metric("CTA per 1000 words", f"{biz_ratios['cta_per_1000_words']:.2f}", "Target 1–3")
                r1c3.metric("Trust per 1000 words", f"{biz_ratios['trust_per_1000_words']:.2f}", "Target ≥2")
                r1c4.metric("Forms", int(biz_ratios['forms']))
                r2c1, r2c2, r2c3, r2c4 = st.columns(4)
                r2c1.metric("Social proof per 1000", f"{biz_ratios['social_proof_per_1000_words']:.2f}", "Target ≥1")
                r2c2.metric("Promos per 1000", f"{biz_ratios['promo_per_1000_words']:.2f}", "Target ≥0.5")
                r2c3.metric("Benefits per 1000", f"{biz_ratios['benefit_per_1000_words']:.2f}", "Target ≥1")
                r2c4.metric("Comparisons per 1000", f"{biz_ratios['comparison_per_1000_words']:.2f}", "Target ≥0.1")
                r3c1, r3c2, r3c3, r3c4 = st.columns(4)
                r3c1.metric("Recomm. per 1000", f"{biz_ratios['recommendation_per_1000_words']:.2f}", "Target ≥0.5")
                r3c2.metric("FAQ present", "Yes" if biz_ratios['faq_present'] else "No")
                r3c3.metric("Search bar", "Yes" if biz_ratios['search_present'] else "No")
                # Explanation and suggestions
                st.markdown(
                    "Ces indicateurs mesurent l’orientation conversion et croissance de la page. "
                    "Des CTA et éléments de confiance bien dosés renforcent la conversion, tandis que les preuves sociales, promotions, bénéfices, comparatifs et recommandations favorisent l’engagement. "
                    "La présence d’une FAQ et d’une barre de recherche améliore la découverte de contenu.",
                    unsafe_allow_html=True,
                )
                if biz_suggestions:
                    st.markdown("<div class='section-title'>Recommandations Business</div>", unsafe_allow_html=True)
                    html_list = "<ul>" + "".join(f"<li>{s}</li>" for s in biz_suggestions) + "</ul>"
                    st.markdown(html_list, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            # Suggestions tab
            with tabs[4]:
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
                # Layout analysis for competitor
                competitor_layout_score, competitor_layout_ratios, competitor_layout_suggestions = layout_analysis(competitor_metrics)
                competitor_suggestions.extend(competitor_layout_suggestions)
                # Business analysis for competitor
                cmp_biz_score, cmp_biz_ratios, cmp_biz_suggestions = business_analysis(competitor_metrics)
                competitor_suggestions.extend(cmp_biz_suggestions)
                # Content gap analysis
                gap = content_gap(primary_metrics, competitor_metrics)
                # Score differences
                score_diff = primary_score - competitor_score
                layout_diff = layout_score - competitor_layout_score
                st.header("Competitor Comparison")
                cmp_tabs = st.tabs([
                    "Competitor Summary",
                    "Content Gap",
                    "Metric Differences",
                    "Conversion",
                    "Growth",
                    "Competitor Suggestions",
                ])
                # Competitor summary
                with cmp_tabs[0]:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Competitor Score", f"{competitor_score}/100", delta=f"{score_diff:+}")
                    c2.metric("Layout Score", f"{competitor_layout_score}/100", delta=f"{layout_diff:+}")
                    c3.metric("Words", int(competitor_metrics.get("word_count", 0)), delta=int(primary_metrics.get("word_count", 0)) - int(competitor_metrics.get("word_count", 0)))
                    # Additional row for readability and response time
                    c4, c5, c6 = st.columns(3)
                    c4.metric("Flesch", float(competitor_metrics.get("flesch_reading_ease", 0)), delta=float(primary_metrics.get("flesch_reading_ease", 0)) - float(competitor_metrics.get("flesch_reading_ease", 0)))
                    c5.metric("Response Time (s)", float(competitor_metrics.get("response_time_seconds", 0)), delta=float(primary_metrics.get("response_time_seconds", 0)) - float(competitor_metrics.get("response_time_seconds", 0)))
                    c6.metric("Links", int(competitor_metrics.get("link_count", 0)), delta=int(primary_metrics.get("link_count", 0)) - int(competitor_metrics.get("link_count", 0)))
                    st.markdown("<div class='section-title'>Competitor Metrics</div>", unsafe_allow_html=True)
                    st.json(competitor_metrics)
                    st.markdown('</div>', unsafe_allow_html=True)
                # Content gap analysis tab
                with cmp_tabs[1]:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("<div class='section-title'>Content Gap Analysis</div>", unsafe_allow_html=True)
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
                # Metric differences tab
                with cmp_tabs[2]:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("<div class='section-title'>Metric Differences (Primary – Competitor)</div>", unsafe_allow_html=True)
                    diff = compare_metrics(primary_metrics, competitor_metrics)
                    # Include layout ratios differences for completeness
                    diff["layout_score_difference"] = layout_diff
                    diff["heading_ratio_difference"] = layout_ratios.get("heading_ratio", 0) - competitor_layout_ratios.get("heading_ratio", 0)
                    diff["image_ratio_difference"] = layout_ratios.get("image_ratio", 0) - competitor_layout_ratios.get("image_ratio", 0)
                    diff["link_ratio_difference"] = layout_ratios.get("link_ratio", 0) - competitor_layout_ratios.get("link_ratio", 0)
                    diff["semantic_landmark_difference"] = layout_ratios.get("landmark_count", 0) - competitor_layout_ratios.get("landmark_count", 0)
                    st.json(diff)
                    st.markdown('</div>', unsafe_allow_html=True)
                # Conversion tab
                with cmp_tabs[3]:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("<div class='section-title'>Conversion & Trust</div>", unsafe_allow_html=True)
                    # Display competitor metrics and differences relative to primary
                    cc1, cc2, cc3 = st.columns(3)
                    c_cta = int(competitor_metrics.get("cta_count", 0))
                    p_cta = int(primary_metrics.get("cta_count", 0))
                    c_trust = int(competitor_metrics.get("trust_keyword_count", 0))
                    p_trust = int(primary_metrics.get("trust_keyword_count", 0))
                    c_form = int(competitor_metrics.get("form_count", 0))
                    p_form = int(primary_metrics.get("form_count", 0))
                    cc1.metric("CTA", c_cta, delta=f"{p_cta - c_cta:+}")
                    cc2.metric("Trust signals", c_trust, delta=f"{p_trust - c_trust:+}")
                    cc3.metric("Forms", c_form, delta=f"{p_form - c_form:+}")
                    # Details
                    if competitor_metrics.get("cta_texts"):
                        st.subheader("CTAs trouvées")
                        st.write(", ".join(competitor_metrics.get("cta_texts", [])))
                    if competitor_metrics.get("trust_keywords_found"):
                        st.subheader("Mots‑clés de confiance trouvés")
                        st.write(", ".join(competitor_metrics.get("trust_keywords_found", [])))
                    st.markdown('</div>', unsafe_allow_html=True)

                # Growth & Business tab
                with cmp_tabs[4]:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("<div class='section-title'>Growth & Business</div>", unsafe_allow_html=True)
                    # Compute competitor business analysis
                    cmp_biz_score, cmp_biz_ratios, cmp_biz_suggestions = business_analysis(competitor_metrics)
                    prim_biz_score, prim_biz_ratios, _ = business_analysis(primary_metrics)
                    # Display metrics with deltas relative to primary
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Business Score", f"{cmp_biz_score}/100", delta=f"{prim_biz_score - cmp_biz_score:+}")
                    c2.metric("CTA/1000", f"{cmp_biz_ratios['cta_per_1000_words']:.2f}", delta=f"{prim_biz_ratios['cta_per_1000_words'] - cmp_biz_ratios['cta_per_1000_words']:+.2f}")
                    c3.metric("Trust/1000", f"{cmp_biz_ratios['trust_per_1000_words']:.2f}", delta=f"{prim_biz_ratios['trust_per_1000_words'] - cmp_biz_ratios['trust_per_1000_words']:+.2f}")
                    c4.metric("Forms", int(cmp_biz_ratios['forms']), delta=f"{int(prim_biz_ratios['forms']) - int(cmp_biz_ratios['forms']):+}")
                    c5, c6, c7, c8 = st.columns(4)
                    c5.metric("Social/1000", f"{cmp_biz_ratios['social_proof_per_1000_words']:.2f}", delta=f"{prim_biz_ratios['social_proof_per_1000_words'] - cmp_biz_ratios['social_proof_per_1000_words']:+.2f}")
                    c6.metric("Promo/1000", f"{cmp_biz_ratios['promo_per_1000_words']:.2f}", delta=f"{prim_biz_ratios['promo_per_1000_words'] - cmp_biz_ratios['promo_per_1000_words']:+.2f}")
                    c7.metric("Benefits/1000", f"{cmp_biz_ratios['benefit_per_1000_words']:.2f}", delta=f"{prim_biz_ratios['benefit_per_1000_words'] - cmp_biz_ratios['benefit_per_1000_words']:+.2f}")
                    c8.metric("Comparisons/1000", f"{cmp_biz_ratios['comparison_per_1000_words']:.2f}", delta=f"{prim_biz_ratios['comparison_per_1000_words'] - cmp_biz_ratios['comparison_per_1000_words']:+.2f}")
                    c9, c10, c11, c12 = st.columns(4)
                    c9.metric("Recomm./1000", f"{cmp_biz_ratios['recommendation_per_1000_words']:.2f}", delta=f"{prim_biz_ratios['recommendation_per_1000_words'] - cmp_biz_ratios['recommendation_per_1000_words']:+.2f}")
                    c10.metric("FAQ", "Yes" if cmp_biz_ratios['faq_present'] else "No", delta="+" if (prim_biz_ratios['faq_present'] and not cmp_biz_ratios['faq_present']) else ("-" if (not prim_biz_ratios['faq_present'] and cmp_biz_ratios['faq_present']) else "0"))
                    c11.metric("Search", "Yes" if cmp_biz_ratios['search_present'] else "No", delta="+" if (prim_biz_ratios['search_present'] and not cmp_biz_ratios['search_present']) else ("-" if (not prim_biz_ratios['search_present'] and cmp_biz_ratios['search_present']) else "0"))
                    # Explanation
                    st.markdown(
                        "Ces indicateurs comparent la performance business et croissance entre la page primaire et la concurrente. "
                        "Un score positif indique que la page primaire est plus optimisée, tandis qu’un score négatif signale des opportunités d’amélioration.",
                        unsafe_allow_html=True,
                    )
                    # Append competitor business suggestions to competitor suggestions list
                    competitor_suggestions.extend(cmp_biz_suggestions)
                    st.markdown('</div>', unsafe_allow_html=True)
                # Competitor suggestions tab
                with cmp_tabs[5]:
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