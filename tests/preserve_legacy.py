import csv
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit


# Usage: python3 tests/preserve_legacy.py SOURCE BASELINE PUBLIC INVENTORY OUTPUT [--image-fix]
IMAGE_FIX = "--image-fix" in sys.argv
SOURCE, BASELINE, PUBLIC, INVENTORY, OUTPUT = map(Path, [a for a in sys.argv[1:] if a != "--image-fix"])
HOST = "https://ianchen28.github.io/"
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__(convert_charrefs=True)
        self.ids = []
        self.references = []
        self.canonical = ""
        self.translations = []
        self.language = ""
        self.main_count = 0
        self.heading_count = 0
        self.content = []
        self.capture_depth = 0
        self.feed(path.read_text())
        self.close()

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "html":
            self.language = attributes.get("lang")
        if tag == "main":
            self.main_count += 1
        if tag == "h1":
            self.heading_count += 1
        if "id" in attributes:
            self.ids.append(attributes["id"])
        if tag == "link":
            if attributes.get("rel") == "canonical":
                self.canonical = attributes["href"]
            if "hreflang" in attributes:
                self.translations.append(attributes["href"])
        if tag in {"a", "img", "script", "link"}:
            key = "href" if tag in {"a", "link"} else "src"
            if key in attributes:
                self.references.append((tag, attributes[key]))
        if tag == "div" and "post-content" in attributes.get("class", "").split():
            assert not self.capture_depth
            self.capture_depth = 1
        elif self.capture_depth and tag not in VOID:
            self.capture_depth += 1
        if self.capture_depth:
            self.content.append(("start", tag, tuple(sorted(attrs))))

    def handle_endtag(self, tag):
        if self.capture_depth:
            self.content.append(("end", tag))
            self.capture_depth -= 1

    def handle_data(self, data):
        if self.capture_depth:
            self.content.append(("text", data))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def html_pages(directory):
    return {str(path.relative_to(directory)): Page(path) for path in directory.rglob("*.html")}


def local_target(public, page_path, reference):
    base = urljoin(HOST, page_path.removesuffix("index.html"))
    target = urlsplit(urljoin(base, reference))
    if target.netloc != urlsplit(HOST).netloc:
        return None
    path = public / unquote(target.path).lstrip("/")
    return path if path.is_file() else path / "index.html"


def missing_references(directory, pages):
    missing = []
    for path, page in pages.items():
        for tag, reference in page.references:
            if not reference or reference.startswith(("#", "mailto:", "tel:", "data:")):
                continue
            target = local_target(directory, path, reference)
            if target is not None and not target.is_file():
                missing.append((path, tag, reference))
    return Counter(missing)


def normalized_content(events):
    # Only image src may change in the independently reviewed maintenance patch.
    return [(event[0], event[1], tuple(a for a in event[2] if a[0] != "src"))
            if event[0] == "start" and event[1] == "img" else event for event in events]


def without_image_sources(html):
    return re.sub(r'(<img\b[^>]*?\s)src=(?:"[^"]*"|\'[^\']*\'|[^\s>]+)', r'\1', html)


def feed_item(item):
    if IMAGE_FIX:
        description = item.find("description")
        if description is not None and description.text:
            description.text = without_image_sources(description.text)
    return ET.tostring(item)


baseline_pages = html_pages(BASELINE)
pages = html_pages(PUBLIC)
assert baseline_pages.keys() <= pages.keys(), "A legacy HTML route disappeared"
baseline_missing = missing_references(BASELINE, baseline_pages)
missing = missing_references(PUBLIC, pages)
if IMAGE_FIX:
    assert not missing, f"Image repair must resolve every local reference: {missing}"
else:
    assert missing == baseline_missing, (missing - baseline_missing, baseline_missing - missing)
    assert sum(missing.values()) == 21, missing

source_rows = list(csv.DictReader((INVENTORY / "content-inventory.csv").open()))
legacy_posts = [row for row in source_rows if row["source"].startswith("content/posts/")]
for row in legacy_posts:
    assert sha(SOURCE / row["source"]) == row["sha256"], row["source"]

assets = list(csv.DictReader((INVENTORY / "asset-inventory.csv").open()))
for row in assets:
    assert sha(SOURCE / row["source"]) == row["sha256"], row["source"]
for path in BASELINE.rglob("*"):
    if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}:
        current = PUBLIC / path.relative_to(BASELINE)
        assert current.is_file() and sha(path) == sha(current), path

published_articles = [
    row for row in legacy_posts if row["kind"] == "page" and row["draft"] == "false"
]
article_urls = {row["url"] for row in published_articles}
for row in published_articles:
    path = unquote(urlsplit(row["url"]).path).lstrip("/") + "index.html"
    original, current = baseline_pages[path], pages[path]
    if IMAGE_FIX:
        assert normalized_content(original.content) == normalized_content(current.content), f"Article content changed beyond image src: {path}"
        original_images = [r for r in original.references if r[0] == "img"]
        current_images = [r for r in current.references if r[0] == "img"]
        assert len(original_images) == len(current_images), f"Image source removed: {path}"
        assert all(reference for _, reference in current_images), f"Empty image source: {path}"
        for (_, reference), (_, after_reference) in zip(original_images, current_images):
            if reference != after_reference:
                assert urlsplit(reference).netloc == "", f"External image changed: {reference}"
                before_target = local_target(BASELINE, path, reference)
                if before_target.is_file():
                    after_target = local_target(PUBLIC, path, after_reference)
                    assert sha(before_target) == sha(after_target), f"Working image content changed: {reference}"
                else:
                    resource = SOURCE / "content" / path.removeprefix("zh/").removesuffix("index.html") / unquote(urlsplit(reference).path)
                    assert resource.is_file(), f"Original bundle resource absent: {resource}"
                    assert sha(resource) == sha(local_target(PUBLIC, path, after_reference)), f"Repair points to wrong image: {reference}"
    else:
        assert original.content == current.content, f"Article body DOM changed: {path}"
    assert original.ids == current.ids, f"Article heading/anchor IDs changed: {path}"
    assert current.canonical == row["url"]
    assert current.translations == original.translations
for row in legacy_posts:
    if row["draft"] == "true":
        assert not (PUBLIC / unquote(urlsplit(row["url"]).path).lstrip("/") / "index.html").exists()

feed_count = 0
for path in BASELINE.rglob("*.xml"):
    original_tree = ET.parse(path)
    if original_tree.getroot().tag != "rss":
        continue
    current_path = PUBLIC / path.relative_to(BASELINE)
    assert current_path.is_file(), path
    current_tree = ET.parse(current_path)
    original_items = {
        item.findtext("link"): feed_item(item)
        for item in original_tree.findall("./channel/item")
        if item.findtext("link") in article_urls
    }
    current_items = {
        item.findtext("link"): feed_item(item)
        for item in current_tree.findall("./channel/item")
        if item.findtext("link") in article_urls
    }
    assert original_items == current_items, f"Legacy feed items changed: {path}"
    assert all(
        "/projects/" not in item.findtext("link", "")
        for item in current_tree.findall("./channel/item")
    )
    feed_count += 1

for prefix in ("", "zh/"):
    original = json.loads((BASELINE / prefix / "index.json").read_text())
    current = json.loads((PUBLIC / prefix / "index.json").read_text())
    original_posts = {item["permalink"]: item for item in original if item["permalink"] in article_urls}
    current_posts = {item["permalink"]: item for item in current if item["permalink"] in article_urls}
    if IMAGE_FIX:
        # One historical summary contains figure HTML. Preserve every other
        # search field and all summary markup except the repaired image URL.
        for items in (original_posts, current_posts):
            for item in items.values():
                item["summary"] = without_image_sources(item["summary"])
    assert original_posts == current_posts, f"Legacy search entries changed: {prefix}"
    assert any("/projects/agent-improvement-lab/" in item["permalink"] for item in current)
    assert "Agent Improvement Lab" not in (PUBLIC / prefix / "archives/index.html").read_text()

new_paths = ["index.html", "zh/index.html", "projects/index.html", "zh/projects/index.html",
             "projects/agent-improvement-lab/index.html", "zh/projects/agent-improvement-lab/index.html"]
for path in new_paths:
    page = pages[path]
    assert page.language == ("zh" if path.startswith("zh/") else "en"), (path, page.language)
    assert page.main_count == page.heading_count == 1
    assert len(page.ids) == len(set(page.ids)), f"Duplicate IDs: {path}"
    text = (PUBLIC / path).read_text()
    assert "Principal Applied Scientist" not in text and "首席应用科学家" not in text
    counterpart = path.removeprefix("zh/") if path.startswith("zh/") else "zh/" + path
    assert urljoin(HOST, counterpart.removesuffix("index.html")) in page.translations
for path in ("projects/agent-improvement-lab/index.html", "zh/projects/agent-improvement-lab/index.html"):
    text = (PUBLIC / path).read_text()
    assert 'marker-end="url(#feedback-arrow-h)"' in text or "marker-end=url(#feedback-arrow-h)" in text
    assert "heldout-evaluation" in text
    assert "/projects/agent-improvement-lab/" in text and "/zh/projects/agent-improvement-lab/" in text

workflow = (SOURCE / ".github/workflows/hugo.yaml").read_text()
assert "workflow_dispatch:" in workflow and "\n  push:" not in workflow
assert "path: ./public" in workflow
assert not any(path.name.startswith(".env") or path.name == ".git" for path in PUBLIC.rglob("*"))
assert not (PUBLIC / "runs").exists() and not (PUBLIC / "site-audit").exists()

report = {
    "legacy_post_markdown_unchanged": len(legacy_posts),
    "source_media_unchanged": len(assets),
    "published_article_bodies_and_anchors_unchanged": len(published_articles),
    "legacy_html_routes_preserved": len(baseline_pages),
    "current_html_routes": len(pages),
    "rss_feeds_preserved": feed_count,
    "rss_comparison": "image src changes only" if IMAGE_FIX else "identical legacy items",
    "article_dom_comparison": "image src changes only" if IMAGE_FIX else "identical",
    "search_languages_with_preserved_article_entries": 2,
    "search_comparison": "image src changes only" if IMAGE_FIX else "identical legacy records",
    "native_multilingual_home_list_project_pages": len(new_paths),
    "missing_local_references": sum(missing.values()),
    "image_fix": IMAGE_FIX,
    "new_missing_local_references": 0,
    "manual_only_workflow": True,
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
