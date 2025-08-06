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
  performance metrics in one report.
- **Competitor benchmarking** – specify one or more competitor URLs to
  compare scores and identify differentiators.
- **Automated workflow** – runs headlessly via command line and can be
  integrated into CI pipelines.
- **Extensible prompts** – prompts for OpenAI can be customised to
  generate reports in different styles (e.g. professional, casual,
  summarised).
- **Export formats** – prints structured JSON to stdout by default;
  additional exporters (e.g. Markdown or Notion API) can be implemented
  easily.

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
analyses such as keyword density, readability or heatmap predictions.

### requirements.txt

This file lists the Python packages needed to run the script.  At a
minimum it includes:

- `requests` for downloading HTML
- `beautifulsoup4` for parsing HTML
- `tqdm` for progress bars

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
