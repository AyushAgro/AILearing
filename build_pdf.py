#!/usr/bin/env python3
"""Build PDFs from the volume HTML files.

    python3 build_pdf.py                              # every volumes/*.html
    python3 build_pdf.py volumes/00-design-system.html

What happens in Chromium, before printing:
  - chapters (h2), sections (h3, h4), figures, listings and tables are numbered
  - <nav class="toc"></nav> is filled with the contents
  - Mermaid diagrams and KaTeX maths are rendered to vector graphics
  - contents page numbers are filled from a first print pass
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor"
OUT = ROOT / "output"

MERMAID_CONFIG = {
    "startOnLoad": False,
    "theme": "base",
    "htmlLabels": False,
    "flowchart": {"htmlLabels": False, "curve": "basis", "useMaxWidth": True},
    "themeVariables": {
        "fontFamily": "Inter, Arial, sans-serif",
        "fontSize": "13px",
        "primaryColor": "#eef4fb",
        "primaryBorderColor": "#1f5fae",
        "primaryTextColor": "#1b2130",
        "lineColor": "#5d6679",
        "secondaryColor": "#edf7f1",
        "tertiaryColor": "#f5f6f8",
        "clusterBkg": "#f8f9fb",
        "clusterBorder": "#b9c0cd",
        "actorBkg": "#eef4fb",
        "actorBorder": "#1f5fae",
    },
}

PREPARE_JS = """
() => {
  const slug = t => t.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 60);
  const used = new Set([...document.querySelectorAll('[id]')].map(e => e.id));
  const ensureId = (el, base) => {
    if (el.id) return el.id;
    let id = slug(base) || 'section', n = 2, c = id;
    while (used.has(c)) c = `${id}-${n++}`;
    used.add(c); el.id = c; return c;
  };
  const toc = [];
  let ch = 0, sec = 0, sub = 0, fig = 0, lst = 0, tab = 0;
  const heads = document.querySelectorAll('h1:not(.no-toc), h2, h3, h4, figure, table');
  for (const el of heads) {
    const tag = el.tagName;
    const inReview = !!el.closest('.review');
    if (tag === 'H1') { ensureId(el, el.textContent); toc.push([1, el.id, '', el.textContent]); }
    else if (tag === 'H2') {
      ensureId(el, el.textContent);
      if (inReview || el.classList.contains('unnumbered')) { el.classList.add('unnumbered'); toc.push([2, el.id, '', el.textContent, true]); continue; }
      ch++; sec = sub = fig = lst = tab = 0;
      el.dataset.num = ch;
      toc.push([1, el.id, String(ch), el.textContent]);
    } else if (tag === 'H3' && ch && !inReview) {
      sec++; sub = 0;
      const n = `${ch}.${sec}`; const title = el.textContent;
      el.insertAdjacentHTML('afterbegin', `<span class="secno">${n}</span>`);
      ensureId(el, title); toc.push([2, el.id, n, title]);
    } else if (tag === 'H4' && ch && !inReview) {
      sub++; el.insertAdjacentHTML('afterbegin', `<span class="secno">${ch}.${sec}.${sub}</span>`);
    } else if (tag === 'FIGURE' || tag === 'TABLE') {
      const cap = el.querySelector(':scope > figcaption, :scope > caption');
      if (!cap) continue;
      let word, n;
      if (tag === 'TABLE') { word = 'Table'; n = ++tab; }
      else if (el.classList.contains('listing')) { word = 'Listing'; n = ++lst; }
      else { word = 'Figure'; n = ++fig; }
      cap.insertAdjacentHTML('afterbegin', `<span class="caption-label">${word} ${ch || 'I'}.${n}</span>`);
    }
  }
  const nav = document.querySelector('nav.toc');
  if (nav) {
    const row = ([lvl, id, num, title, part]) =>
      `<li class="toc-l${lvl}${part ? ' toc-sub' : ''}"><a href="#${id}"><span class="toc-num">${num}</span>` +
      `<span class="toc-title">${title}</span><span class="toc-leader"></span><span class="toc-page"></span></a></li>`;
    nav.innerHTML = '<h1 class="no-toc">Contents</h1><ol>' + toc.map(row).join('') + '</ol>';
  }
  return toc.length;
}
"""

KATEX_JS = """
() => {
  const errors = [];
  document.querySelectorAll('span.math, div.math').forEach(el => {
    const tex = el.textContent.trim();
    try { katex.render(tex, el, {displayMode: el.tagName === 'DIV', throwOnError: true}); }
    catch (e) { errors.push(tex.slice(0, 60) + ': ' + e.message); }
  });
  return errors;
}
"""

MERMAID_JS = """
async (config) => {
  mermaid.initialize(config);
  const errors = [];
  const blocks = [...document.querySelectorAll('pre.mermaid')];
  for (let i = 0; i < blocks.length; i++) {
    const code = blocks[i].textContent.trim();
    try {
      const { svg } = await mermaid.render('mmd-' + i, code);
      const div = document.createElement('div');
      div.className = 'mermaid-rendered'; div.innerHTML = svg;
      blocks[i].replaceWith(div);
    } catch (e) { errors.push(code.split('\\n')[0] + ': ' + String(e.message || e).split('\\n')[0]); }
    document.querySelectorAll('[id^="dmmd-"]').forEach(x => x.remove());
  }
  return errors;
}
"""

FILL_TOC_JS = """
(pages) => document.querySelectorAll('nav.toc a').forEach(a => {
  const id = a.getAttribute('href').slice(1);
  a.querySelector('.toc-page').textContent = pages[id] || '';
})
"""

HEADER = """<div style="width:100%; margin:0 21mm; padding-bottom:1.5mm; border-bottom:0.5pt solid #d8dce4;
 display:flex; justify-content:space-between; font-family:Inter,Arial,sans-serif; font-size:7pt; color:#5d6679;">
 <span style="color:#1f5fae; font-weight:600; letter-spacing:0.08em; text-transform:uppercase;">{left}</span>
 <span>AI Engineering: From First Principles to Production</span></div>"""
FOOTER = """<div style="width:100%; text-align:center; font-family:Inter,Arial,sans-serif; font-size:8pt;
 color:#3b4458;"><span class="pageNumber"></span></div>"""


def dest_pages(pdf: bytes) -> tuple[dict[str, int], int]:
    reader = PdfReader(io.BytesIO(pdf))
    pages = {}
    for name, dest in reader.named_destinations.items():
        try:
            pages[str(name).lstrip("/")] = reader.get_destination_page_number(dest) + 1
        except Exception:  # noqa: BLE001
            pass
    return pages, len(reader.pages)


def build(browser, html_path: Path) -> Path:
    page = browser.new_page()
    page.goto(html_path.resolve().as_uri(), wait_until="load")
    page.evaluate("document.fonts.ready.then(() => true)")
    page.evaluate(PREPARE_JS)
    problems: list[str] = []
    if page.locator("span.math, div.math").count():
        page.add_script_tag(path=str(VENDOR / "katex" / "katex.min.js"))
        problems += page.evaluate(KATEX_JS)
    if page.locator("pre.mermaid").count():
        page.add_script_tag(path=str(VENDOR / "mermaid.min.js"))
        problems += page.evaluate(MERMAID_JS, MERMAID_CONFIG)
    page.evaluate("document.fonts.ready.then(() => true)")

    left = page.evaluate("document.querySelector('meta[name=volume]')?.content || document.title")
    options = {
        "prefer_css_page_size": True,
        "print_background": True,
        "outline": True,
        "tagged": True,
        "display_header_footer": True,
        "header_template": HEADER.format(left=left),
        "footer_template": FOOTER,
    }
    pdf = page.pdf(**options)
    for _ in range(3):  # fill contents page numbers until stable
        pages, _n = dest_pages(pdf)
        page.evaluate(FILL_TOC_JS, pages)
        pdf = page.pdf(**options)
        if dest_pages(pdf)[0] == pages:
            break
    page.close()

    OUT.mkdir(exist_ok=True)
    out = OUT / (html_path.stem + ".pdf")
    out.write_bytes(pdf)
    for p in problems:
        print(f"  ! {p}")
    print(f"{html_path.name} -> {out.relative_to(ROOT)} ({dest_pages(pdf)[1]} pages)")
    return out


def main(args: list[str]) -> int:
    files = [Path(a) for a in args] or sorted((ROOT / "volumes").glob("*.html"))
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for f in files:
            build(browser, f)
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
