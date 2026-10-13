# ianchen28.github.io

Personal AI research portfolio and bilingual blog: agent evaluation, rubric
quality, reward modeling and research engineering.

**Live**: [ianchen28.github.io](https://ianchen28.github.io)

## Stack

- [Hugo](https://gohugo.io/) (static site generator)
- [PaperMod](https://github.com/adityatelange/hugo-PaperMod) theme
- GitHub Pages (via GitHub Actions)
- Bilingual (English + Chinese)
- Hugo Extended **0.147.0**, matching the deployment workflow

## Local Development

```bash
# Install Hugo extended
# https://gohugo.io/installation/

# Clone with submodules
git clone --recurse-submodules git@github.com:ianchen28/ianchen28.github.io.git

# Restore the pinned theme in an existing clone
git submodule update --init --recursive

# Local review, excluding drafts and future posts
hugo server --bind 127.0.0.1 --port 1313 --disableFastRender --noBuildLock

# Production build, including both languages
hugo --minify --printPathWarnings --panicOnWarning --noBuildLock
```

Open `http://localhost:1313/` for English and `http://localhost:1313/zh/` for
Chinese. Project pages are `/projects/agent-improvement-lab/` and
`/zh/projects/agent-improvement-lab/`. Serve this Hugo site, not the separate
research repository. Use `-D` only when intentionally reviewing drafts.

## Structure

```
content/
├── _index.md        # Portfolio homepage (with _index.zh.md)
├── about/           # About page (bilingual)
├── projects/        # Independent bilingual project bundles
├── posts/
│   ├── research/    # LLM/Agent/Rubric research
│   ├── tech/        # Technical articles
│   └── archive/     # Earlier game AI posts
├── search.md        # Search page
└── archives.md      # Archives page
data/portfolio.yaml  # Featured project and selected-post IDs
layouts/index.html   # Portfolio homepage
layouts/projects/    # Project presentation
assets/css/extended/portfolio.css # Scoped styles; PaperMod theme variables
```

## Content and preservation

Author new project/article pages as Markdown bundles with real English and
Chinese content. Homepage prose and labels live in native multilingual
`content/_index.md` and `content/_index.zh.md`. Keep shared featured/selected
page IDs in `data/portfolio.yaml`; do not duplicate article bodies there.
Localized writing-card titles and descriptions live in the homepage front
matter; historical article Markdown stays unchanged. The homepage and project
list share `layouts/partials/portfolio-project-card.html`. Project JSON-LD is
`WebPage` and omits dates until real dates are supplied.

Keep original post Markdown, dates, tags, attribution, draft flags, bundle
images and URLs. In particular, `torchcraftai-code-learning.md/` is an existing
public URL segment. Homepage branding does not rename the blog. Projects are
searchable but excluded from legacy blog archives and RSS item selection.
Publications are omitted until real references are available.

The Chinese article image-link repair is a separate maintenance commit:
resource-aware image/figure templates resolve shared page-bundle resources,
reducing the baseline's 21 missing local references to zero. Article and media
sources stay in place. Regression checks permit only image `src` repairs in
article, feed and search-summary output. Existing math uses KaTeX CDN resources.

About pages use a neutral research biography. Professional title, affiliation,
education and any additional contact details remain omitted pending owner
confirmation. The existing Ian Chen name, avatar and GitHub contact were
confirmed for this release; record future changes in review materials, not
public-page placeholders. Describe Agent
Improvement Lab as a public research project until its license is confirmed;
this website does not add or choose a research license. The old title is not
reused. This website is separate from Agent Improvement Lab's Python
environment. Publish only manually reviewed summaries, never private runs,
logs, trajectories, credentials or automatically synchronized artifacts.

## Validation

`tests/portfolio_browser.cjs` checks bilingual home, project, project-list and
About pages, plus blog lists and a historical article, at
320/390/768/1050/1440px in light/dark themes. Supply an existing
Playwright module with `PLAYWRIGHT_MODULE` if it is not installed locally:

```bash
node tests/portfolio_browser.cjs http://localhost:1313 /tmp/portfolio-browser
```

`tests/preserve_legacy.py` compares a production build with the legacy baseline
and inventories from the workspace audit. It checks source hashes, routes,
article DOM/anchors, feeds, search and missing local references. Use
`--image-fix` for the current release including the maintenance commit; that
mode allows image `src` repairs while preserving the remaining article/feed/
search data. Omit it only when checking the pre-maintenance portfolio commit.

```bash
python3 tests/preserve_legacy.py SOURCE BASELINE_PUBLIC PUBLIC INVENTORY OUTPUT_JSON --image-fix
```

Keep generated output outside the source repository. No new browser or Python
dependency is required by the preservation checker (Python standard library).

## Deployment

The existing Hugo/Pages workflow now has **manual `workflow_dispatch` only**.
It builds this repository and uploads `public/`; pushes do not trigger deployment.
Implementation approval does not authorize a push, merge or dispatch.
Publication/deployment requires separate explicit owner approval.
