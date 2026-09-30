# Style

The single style reference for the book. Every volume follows it.

## Files

```text
ai-engineering-book/
├── style.md          this file
├── MASTER_PLAN.md    curriculum (25 volumes)
├── book.css          the only stylesheet
├── build_pdf.py      HTML -> PDF (python3 build_pdf.py volumes/NN-name.html)
├── vendor/           fonts, Mermaid, KaTeX (offline rendering)
├── volumes/          one HTML file per volume: NN-slug.html
└── output/           generated PDFs
```

## Writing

- British English: optimise, tokeniser, behaviour, modelling, artefact.
- Reader is a senior engineer. No beginner Python, ML, API or database material.
- Bullet first. Paragraphs are at most 4–5 sentences.
- Problem first: Problem → Possible Solutions (what each does *not* solve) → how it works →
  code → trade-offs → failure modes → evaluation → when to use or avoid.
- Complex systems: Simple Version → Production Version → Why It Is Complicated.
- Real products: label every internal claim as Documented (with source) or Inferred.

## Headings

| Tag | Use | Numbering (automatic) |
|-----|-----|-----------------------|
| H1 | Title page, Volume Review | none |
| H2 | Chapter, one per `<section class="chapter">` | Chapter N (new page) |
| H3 | Major concept | N.M |
| H4 | Technical subsection | N.M.K |
| H5 | Detailed topic | none |

Never use H6. Never type numbers into headings.

## Colours and components

| Markup | Colour | Printed label |
|--------|--------|---------------|
| `<aside class="callout core">` | Blue | CORE |
| `<aside class="callout impl">` | Green | IMPLEMENTATION |
| `<aside class="callout warning">` | Orange | WARNING |
| `<aside class="callout security">` | Red | SECURITY |
| `<aside class="callout advanced">` | Purple | ADVANCED |

An optional title goes first: `<aside class="callout core"><span class="title">Key Idea</span>…`

Frames (neutral boxes): `<div class="frame problem|simple|production|why-complex|exercise|checklist">`.

Other components:

- Facet heading: `<p class="facet">Advantages</p>` followed by a list.
- Pros and cons: `<div class="tradeoffs"><div class="pros">…</div><div class="cons">…</div></div>`.
- Failure modes: `<ul class="failure-modes">`.
- Evidence labels: `<span class="evidence documented">Documented</span>` and `<span class="evidence inferred">Inferred</span>`.
- Paper cards: `<section class="paper-card">` with a head and a `<dl>` covering Problem, Previous limitation, Approach, Contribution, Impact and Limitations.

## Code

```html
<figure class="listing">
<figcaption>What the code does.</figcaption>
<pre><code>…python…</code></pre>
</figure>
<pre class="output">…printed result…</pre>
```

- Escape `<` as `&lt;` and `>` in `->` as `&gt;` inside code.
- Keep lines to 88 characters or fewer, and use shape comments like `# (B, T, C)`.
- Code shown should be code that was actually run.

## Diagrams (no images)

- ASCII: `<figure class="diagram"><pre class="ascii">…</pre><figcaption>…</figcaption></figure>`
- Mermaid: `<pre class="mermaid">flowchart TD …</pre>`. Prefer `TD` (top-down), keep to 15 nodes or fewer.
- Architecture (HLD): `<div class="arch">` with `<div class="arch-layer" data-label="…">` rows and
  `<div class="node service|model|gpu|store|queue|guard|eval|obs|client|gateway|external">`.

## Maths and tables

- Maths: `<span class="math">\sqrt{d_k}</span>` (inline), `<div class="math">…</div>` (display).
- Tables: `<table><caption>…</caption><thead>…</thead><tbody>…</tbody></table>`, with `class="num"`
  on numeric columns.

## Volume skeleton

```html
<head> … <meta name="volume" content="Volume N · Title">
  <link rel="stylesheet" href="../vendor/katex/katex.min.css">
  <link rel="stylesheet" href="../book.css"> </head>
<section class="title-page">…</section>
<nav class="toc"></nav>                         <!-- filled automatically -->
<section class="chapter"><h2>…</h2>…</section>  <!-- repeat -->
<section class="review"><h1>Volume Review</h1>
  <h2>What You Learned</h2> … <h2>Next Volume</h2></section>
```

Volume Review sections: What You Learned · What You Built · Important Code · Important
Architecture · Failure Modes · Evaluation · Practical Project · Exercises · Research Papers ·
Interview Questions · Next Volume.
