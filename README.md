# ianchen28.github.io

Personal blog — LLM Agent evaluation, rubric quality, and reward modeling research.

**Live**: [ianchen28.github.io](https://ianchen28.github.io)

## Stack

- [Hugo](https://gohugo.io/) (static site generator)
- [PaperMod](https://github.com/adityatelange/hugo-PaperMod) theme
- GitHub Pages (via GitHub Actions)
- Bilingual (English + Chinese)

## Local Development

```bash
# Install Hugo extended
# https://gohugo.io/installation/

# Clone with submodules
git clone --recurse-submodules git@github.com:ianchen28/ianchen28.github.io.git

# Local preview (including drafts)
hugo server -D

# Build
hugo --gc --minify
```

## Structure

```
content/
├── about/           # About page (bilingual)
├── posts/
│   ├── research/    # LLM/Agent/Rubric research
│   ├── tech/        # Technical articles
│   └── archive/     # Earlier game AI posts
├── search.md        # Search page
└── archives.md      # Archives page
```

## Deployment

Push to `main` → GitHub Actions builds and deploys automatically.
