# UX‑SEO Audit Agent

**UX‑SEO Audit Agent** is an open‑source project that automates the
assessment of a website’s user experience (UX) and search engine
optimisation (SEO), and compares those results against a competing site.
The goal is to make it easy for designers, developers and product teams
to obtain actionable insights without needing a suite of expensive tools.

## 🔍 Why build another audit tool?

Existing products often specialise in only one discipline.  UX audit
services focus on usability, while SEO tools concentrate on keywords and
technical optimisation.  There are few solutions that combine both
perspectives and provide a clear benchmark against competitors.  In
addition, many professional tools are expensive or offer limited free
tier access.  **UX‑SEO Audit Agent** aims to close that gap by
combining open‑source components into an agent‑driven workflow that
produces a coherent report.

Google’s own [Lighthouse](https://developers.google.com/web/tools/lighthouse)
provides a great starting point.  It is a free and open‑source site
monitoring tool that helps businesses track page performance,
accessibility and SEO【762196631858810†L138-L140】.  When you run an audit,
Lighthouse loads a website multiple times to gather information about
its structure, tags and performance.  The tool emphasises mobile
experiences by emulating a smartphone and throttling the network to
simulate slower connections【762196631858810†L154-L159】.  According to
Google, Lighthouse exists to help you identify and fix common problems
that affect your site’s performance, accessibility and user experience【762196631858810†L162-L169】.

While Lighthouse is extremely useful, it doesn’t provide detailed
feedback on design patterns, content hierarchy or competitor
positioning.  This project therefore supplements Lighthouse with
additional analyses:

- **HTML and content extraction** using Python (via
  `requests` and `BeautifulSoup`) to capture titles, headings, links,
  images, alt text and other on‑page elements.
- **Accessibility and performance metrics** via the Lighthouse CLI
  (`lighthouse`), executed as a subprocess.  The tool can produce a JSON
  output which is parsed and merged with other data.
- **Competitor comparison** by running the same steps on a competitor’s
  URL and calculating relative scores (e.g. differences in heading
  structure, keyword frequency or Core Web Vitals).
- **AI‑powered recommendations (optional)**.  If you supply an OpenAI API
  key, the agent can send the collected data to GPT to generate a
  narrative report with design recommendations, CRO suggestions and
  keyword opportunities.  This step is optional and only executed when
  the `--use-openai` flag and the `OPENAI_API_KEY` environment variable
  are provided.

## ✨ Features

- **Combined UX & SEO audit** – collects on‑page content, accessibility and
  performance metrics in one report.  The lightweight analyser extracts
  titles, meta descriptions, headings, links, images and alt text and
  measures response time and page weight.  It also detects canonical
  tags, robots meta, Open Graph metadata, structured data, responsive
  viewport settings and ARIA attributes for accessibility.  The analyser
  looks for semantic landmarks like `<nav>`, `<header>`, `<main>` and
  `<footer>` and flags their absence, since these elements help
  search engines and assistive technologies understand page structure.
  In addition, it computes readability (Flesch reading ease), measures
  the prevalence of personal pronouns and exclamation points to infer
  tone, extracts the top keywords on the page and performs a
  **layout & visual hierarchy analysis**.  This analysis derives a
  "Layout Score" based on ratios of headings, images and links
  relative to the amount of text, and checks for semantic landmarks
  (`<nav>`, `<header>`, `<main>`, `<footer>`).  A balanced ratio of
  structural elements indicates that the page is neither overly
  cluttered nor too sparse.  The layout score and accompanying
  suggestions help you understand how to organise content for better
  legibility and user flow.  In the Streamlit interface, these structural ratios are presented clearly so you can see how close your page is to recommended targets (e.g. three headings and two images per 1 000 words, and all four semantic landmarks) without adding distracting progress bars.
  Beyond structure and tone, the analyser also inspects business‑oriented growth signals: it looks for
  **social proof** keywords (testimonials, reviews, ratings),
  **promotions and bonuses**, clearly articulated **benefits and
  advantages**, **comparison or versus language** (e.g. “X vs Y”),
  **cross‑sell and recommendation cues**, the presence of a **FAQ or
  help section**, and whether a **search bar** is available to aid
  navigation.  Each absence triggers targeted recommendations to
  strengthen the page’s conversion potential and user journey.

  Beyond structural and technical metrics, the analyser also
  **captures business conversion and growth signals**.  It counts how many
  calls‑to‑action (CTA) buttons or links appear on the page,
  detects the presence of trust‑building keywords (e.g. “SSL”,
  “licence”, “avis clients”) and records the number of forms used for
  lead capture.  It also scans for social proof (testimonials,
  reviews), promotions and bonuses, explicit statements of benefits,
  comparison phrases like “vs” or “comparatif”, cross‑sell cues,
  FAQ/help sections and a search bar.  To make these metrics
  actionable, the app normalises them per 1 000 words and calculates
  a **Business Score** out of 100 along with key performance indicators
  (CTA/1 000 words, trust words/1 000 words, social proof/1 000 words,
  etc.).  Ratios outside recommended ranges trigger targeted
  suggestions.  Each KPI and the resulting Business Score are
  displayed in the **Growth & Business** tab, along with concise indicators showing how well your page meets typical benchmarks (for example around two CTAs and at least two trust words per 1 000 words, and at least one social proof and a form).  The tool then
  recommends adding or balancing these elements – such as
  including testimonials, highlighting key benefits, offering a
  promotion, or adding a FAQ – to enhance the page’s credibility,
  usability and conversion rate.  Collectively, these extra signals
  go beyond a simple GPT prompt to provide concrete data for growth
  and conversion optimisation.

  A **zoning engine** groups all recommendations into familiar page
  zones – Meta (head), Navigation & Header, Hero & CTA, Main Content,
  Trust & Social Proof, Growth & Promotions, Footer & Compliance,
  Accessibility & Semantic, and Design & UX.  This helps you see
  which part of your page needs attention.  When you analyse a
  competitor, the tool adds example notes to the relevant zones
  highlighting where the competitor performs better (e.g., more
  prominent CTAs or stronger trust signals).

  - **Improved result presentation** – numeric scores are now paired with
    qualitative labels (e.g. *Excellent*, *Good*, *Fair*, *Poor*) to help
    readers instantly interpret their meaning.  Rather than cluttering the
    view with progress bars, the scores stand alone and are coloured via
    labels.  When you analyse a competitor, a small bar chart still
    compares the three high‑level scores (UX/SEO, layout and business) side
    by side so you can immediately spot relative strengths and weaknesses.
    Detailed raw metrics (including all extracted HTML counts and ratios)
    are no longer displayed by default; they remain accessible in
    collapsible sections or can be omitted entirely, keeping the main view
    focused on actionable insights.  The **Keywords** tab now shows both
    the most frequent single‑word keywords and the most common
    two‑word phrases to reveal deeper patterns in your content.  It also
    reports **lexical richness** (the proportion of unique words to the total
    vocabulary) and the total count of unique words, giving you a quick
    indication of how varied or repetitive your language is.
- **Competitor benchmarking** – specify one or more competitor URLs to
  compare scores and identify differentiators.
 - **Content gap analysis** – highlights important keywords and H2
   headings present on a competitor site but missing on your own,
   giving you a list of topics and structural sections to address when
   expanding content.
 - **Domain‑specific compliance** – detects if a site appears to belong to the
   gambling or betting sector (e.g. poker, casino, jeux d’argent) and
   reminds you to include responsible gambling notices and age restrictions
   where required.
 - **Semantic and accessibility checks** – notes missing `<nav>`, `<header>`,
   `<main>` or `<footer>` elements and low ratios of ARIA labels,
   encouraging better structure and inclusivity.
- **Automated workflow** – runs headlessly via command line and can be
  integrated into CI pipelines.
- **Extensible prompts** – prompts for OpenAI can be customised to
  generate reports in different styles (e.g. professional, casual,
  summarised).
- **Export formats** – prints structured JSON to stdout by default;
  additional exporters (e.g. Markdown or Notion API) can be implemented
  easily.

### Quick audit via Streamlit

The repository includes a `streamlit_app.py` that powers a free, hosted
web interface.  This app offers an instant audit by analysing the
page’s HTML directly (no Lighthouse needed).  It computes a UX/SEO
score based on heuristics like title length, meta description length,
presence of a single H1, alt text ratio, internal/external link
  balance, readability, and the extra signals mentioned above (canonical
  tag, robots meta, Open Graph tags, JSON‑LD structured data, viewport
  meta, ARIA usage, lazy loading).  It also measures the prevalence of
  personal pronouns and exclamation points to infer tone.  The tool
  extracts the top keywords and the most common two‑word phrases on the page.  It also
  measures lexical richness (unique words divided by total words) so you
  can gauge how varied your copy is.  These keywords and phrases are compared
  against a competitor to highlight content gaps.  The app calculates a **Layout Score**
  along with a summary of how many headings, images and links appear per
  thousand words and counts how many semantic landmarks (nav, header,
  main, footer) are present to assess structural completeness.
  The Structure tab clearly shows how close these ratios are to recommended values for a balanced hierarchy, without using progress bars.

  To serve marketers and affiliate sites, the Streamlit app also
  reports the number of calls‑to‑action, trust keywords (e.g. “secure”,
  “licence”, “avis clients”) and forms found on the page.  It goes
  further by scanning for social proof (testimonials, reviews),
  promotions and bonuses, explicit benefits, comparison phrases,
  cross‑sell cues, FAQ/help sections and the presence of a search bar.
  These statistics are displayed in dedicated **Conversion** and
  **Growth & Business** tabs, where you can see CTA texts, trust
  keywords and growth signals detected and compare them with a
  competitor.  For gambling or betting sites, the app checks for
  responsible gaming notices and age restrictions.  The tool then
  generates a set of actionable suggestions to improve the page.  These
  heuristics draw on industry guidance: canonical tags tell search
  engines which version of a page to index【600788209180035†L501-L540】,
  Open Graph tags control how links appear on social media【600788209180035†L553-L599】,
  and alt attributes help both users and search engines understand
  images【467421527218735†L89-L96】【600788209180035†L864-L869】.

  To help you interpret the results at a glance, the app assigns
  qualitative labels—such as *Excellent*, *Good*, *Fair* or *Poor*—to
  each score.  When you provide a competitor URL, a small bar chart
  appears in the competitor summary comparing your page’s UX/SEO,
  layout and business scores against those of the rival site.
  Detailed raw metrics and lists of CTA or trust keywords are hidden
  behind expanders, keeping the
  interface clean while allowing you to drill down when needed.

  Finally, all recommendations are organised into a **Zoning & Suggestions**
  tab.  This view groups suggestions by common page regions (meta
  head, navigation & header, hero & CTA, main content, trust & social
  proof, growth & promotions, footer & compliance, accessibility &
  semantic, and design & UX).  Seeing feedback in context helps you
  prioritise improvements where they matter most.  When you provide a
  competitor URL, the zoning tab also notes where the competitor site
  excels – for example, highlighting that a rival page includes more
  calls‑to‑action ou de plus forts signaux de confiance – ce qui vous permet de visualiser concrètement des exemples à suivre.

## 📦 Installation

This project requires **Python 3.10+** and **Node.js 18+** because
Lighthouse runs as a Node package.  The easiest way to get started is
to install the Python dependencies and the Lighthouse CLI globally:

```bash
# clone the repository
git clone https://github.com/your‑username/ux‑seo‑audit-agent.git
cd ux‑seo‑audit-agent

# install Python dependencies
pip install -r requirements.txt

# install Lighthouse CLI (requires Node)
npm install -g lighthouse
```

Alternatively you can run Lighthouse via Docker or another container,
but this repository assumes it is available on your system `PATH`.

## 🚀 Usage

Run the script from the command line, specifying at least one URL:

```bash
python main.py --url https://example.com
```

To benchmark against a competitor, pass the `--competitor` option:

```bash
python main.py --url https://your‑site.com --competitor https://competitor.com
```

If you want the agent to generate written recommendations, export
`OPENAI_API_KEY` and pass the `--use-openai` flag.  Without the flag,
the tool will skip the AI step and simply return collected data.

```bash
export OPENAI_API_KEY=sk‑...
python main.py --url https://example.com --use-openai
```

By default the report is printed to `stdout` in JSON.  You can redirect
this to a file or write custom exporters.

## 🗂 Project structure

```
ux_seo_audit_agent/
├── docs/
│   └── index.html        # Simple static site explaining the project
├── main.py              # Entry point script
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── LICENSE              # MIT License
```

### docs/index.html

The `docs` folder contains a basic static website.  When this repository
is hosted on GitHub Pages the content of this folder will be served
automatically.  The site introduces the project, explains how it works
and links back to the GitHub repository.  You can customise the design
using your own CSS or frameworks.

### main.py

`main.py` is a thin orchestration layer that ties together the
components.  It accepts command‑line arguments, invokes the page
crawler and Lighthouse, calculates comparison metrics, and optionally
calls OpenAI.  You are encouraged to extend the code with additional
analyses such as keyword density, more nuanced readability or
visual‑hierarchy predictions.

### requirements.txt

This file lists the Python packages needed to run the script.  At a
minimum it includes:

* `requests` for downloading HTML
* `beautifulsoup4` for parsing HTML
* `tqdm` for progress bars

# The following dependencies are optional and only required if you enable
# advanced visualisations (e.g. saliency heatmaps) in future:
# * `matplotlib` and `numpy` for generating charts and working with
#   numerical arrays
# * `opencv-python-headless` and `Pillow` for image processing

If you enable the OpenAI step you will also need `openai`.  The
dependency list is intentionally small; additional libraries can be
added as the project evolves.

## 📄 License

This project is licensed under the MIT License.  See the `LICENSE`
file for details.

## 🙏 Acknowledgements

This project builds upon the excellent work of the Chrome
Lighthouse team.  Lighthouse is a free, open source and automated site
monitoring tool that helps businesses track performance,
accessibility and SEO【762196631858810†L138-L140】.  During an audit it
loads a target website multiple times, emulates mobile devices and
gathers page structure and performance metrics【762196631858810†L154-L159】.
Google notes that Lighthouse is designed to help you identify and fix
common problems affecting performance, accessibility and user
experience【762196631858810†L162-L169】.
