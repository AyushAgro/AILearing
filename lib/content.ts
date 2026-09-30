import fs from "fs";
import path from "path";
import { HTMLElement, parse } from "node-html-parser";
import type { Catalog, Heading, ModuleRecord, PageKind, PageRecord } from "./types";

const WORDS_PER_MINUTE = 230;

let cache: { stamp: string; modules: ModuleRecord[] } | null = null;

function volumesDir(): string {
  const candidates = [
    process.env.VOLUMES_DIR,
    path.join(process.cwd(), "volumes"),
  ].filter((dir): dir is string => Boolean(dir));

  for (const dir of candidates) {
    if (fs.existsSync(dir)) return dir;
  }
  throw new Error("Could not find volumes/*.html. Set VOLUMES_DIR or run from the book directory.");
}

function stampOf(dir: string): string {
  return fs
    .readdirSync(dir)
    .filter((name) => name.endsWith(".html"))
    .map((name) => {
      const stat = fs.statSync(path.join(dir, name));
      return `${name}:${stat.mtimeMs}`;
    })
    .join("|");
}

function collapse(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

function slugify(value: string): string {
  const slug = value
    .toLowerCase()
    .replace(/['’]/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return slug.slice(0, 80) || "section";
}

function uniqueSlug(base: string, used: Set<string>): string {
  let slug = base;
  let n = 2;
  while (used.has(slug)) {
    slug = `${base}-${n}`;
    n += 1;
  }
  used.add(slug);
  return slug;
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

const TEX_SYMBOLS: Record<string, string> = {
  approx: "≈", times: "×", cdot: "·", le: "≤", leq: "≤", ge: "≥", geq: "≥", neq: "≠", ne: "≠",
  to: "→", rightarrow: "→", leftarrow: "←", sim: "~", propto: "∝", infty: "∞", pm: "±",
  ldots: "…", cdots: "…", in: "∈", sum: "Σ", prod: "Π", partial: "∂", nabla: "∇",
  alpha: "α", beta: "β", gamma: "γ", delta: "δ", epsilon: "ε", varepsilon: "ε", eta: "η",
  theta: "θ", lambda: "λ", mu: "μ", pi: "π", rho: "ρ", sigma: "σ", tau: "τ", phi: "φ",
  chi: "χ", psi: "ψ", omega: "ω", Delta: "Δ", Sigma: "Σ", Theta: "Θ", Omega: "Ω",
};

/** Rough plain-text form of a TeX snippet, for ledes, outlines and search (not the article). */
function texToText(tex: string): string {
  let out = tex;
  for (let i = 0; i < 3; i += 1) {
    out = out
      .replace(/\\(?:text|mathrm|mathbf|mathit|mathsf|mathtt|operatorname|boldsymbol|textbf)\s*\{([^{}]*)\}/g, "$1")
      .replace(/\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}/g, "$1/$2")
      .replace(/\\sqrt\s*\{([^{}]*)\}/g, "√$1");
  }
  return out
    .replace(/\\(?:left|right|big|Big|bigl|bigr)\b/g, "")
    .replace(/\\[,;:! ]|\\q?quad\b/g, " ")
    .replace(/\\([A-Za-z]+)/g, (_, name: string) => TEX_SYMBOLS[name] ?? name)
    .replace(/[{}\\]/g, "");
}

/** Text of an element with any `.math` spans flattened from TeX to readable text. */
function plainText(el: HTMLElement): string {
  if (!el.querySelector(".math")) return el.text;
  const copy = parse(el.outerHTML);
  for (const math of copy.querySelectorAll(".math")) math.set_content(escapeHtml(texToText(math.text)));
  return copy.text;
}

function childSections(body: HTMLElement): HTMLElement[] {
  return body.childNodes.filter(
    (node): node is HTMLElement =>
      node instanceof HTMLElement && node.tagName.toUpperCase() === "SECTION",
  );
}

function ensureHeadingIds(section: HTMLElement, selector: string): Heading[] {
  const targets = section.querySelectorAll(selector);
  // Reserve ids already used elsewhere in the page (e.g. the chapter h2) so a generated
  // slug such as "special-tokens" cannot collide with them.
  const used = new Set<string>(
    section
      .querySelectorAll("[id]")
      .filter((node) => !targets.includes(node))
      .map((node) => node.getAttribute("id")?.trim() || "")
      .filter(Boolean),
  );
  if (section.getAttribute("id")) used.add(section.getAttribute("id")!.trim());
  const headings: Heading[] = [];
  for (const heading of targets) {
    const level = heading.tagName.toUpperCase() === "H2" ? 2 : 3;
    const text = collapse(plainText(heading));
    if (!text) continue;
    let id = heading.getAttribute("id")?.trim() || "";
    if (!id || used.has(id)) id = uniqueSlug(slugify(text), used);
    else used.add(id);
    heading.setAttribute("id", id);
    headings.push({ id, text, level });
  }
  return headings;
}

function readingMinutes(section: HTMLElement): number {
  const words = collapse(section.text).split(" ").filter(Boolean).length;
  return Math.max(1, Math.round(words / WORDS_PER_MINUTE));
}

function ledeOf(section: HTMLElement, kind: PageKind): string {
  // Skip facet labels ("Objective", "Package layout"): they are headings, not prose.
  const source =
    section.querySelector(".chapter-lede") ||
    section.querySelectorAll("p, li").find((el) => !el.classList.contains("facet") && collapse(el.text));
  const lede = source ? plainText(source) : "";
  const text = collapse(lede);
  if (!text || kind === "review") return text.slice(0, 220);
  return text.length > 220 ? `${text.slice(0, 217).trimEnd()}…` : text;
}

function decorate(section: HTMLElement): void {
  for (const frame of section.querySelectorAll(".frame[data-title]")) {
    const title = frame.getAttribute("data-title");
    if (!title || frame.querySelector(":scope > .frame-name, .frame-name")) continue;
    frame.innerHTML = `<span class="frame-name">${escapeHtml(title)}</span>${frame.innerHTML}`;
  }
  for (const node of section.querySelectorAll("script, iframe, object, embed")) node.remove();
  for (const node of section.querySelectorAll("*")) {
    for (const name of Object.keys(node.attributes)) {
      if (name.toLowerCase().startsWith("on")) node.removeAttribute(name);
    }
  }
  wrapTables(section);
}

function wrapTables(section: HTMLElement) {
  for (const table of [...section.querySelectorAll("table")]) {
    const parent = table.parentNode;
    if (parent instanceof HTMLElement && parent.classList.contains("table-scroll")) continue;
    const wrapper = parse(`<div class="table-scroll">${table.outerHTML}</div>`).querySelector(".table-scroll");
    if (wrapper) table.replaceWith(wrapper);
  }
}

function pageFromSection(
  section: HTMLElement,
  kind: PageKind,
  chapterNumber: number | null,
  usedSlugs: Set<string>,
): PageRecord {
  const heading = section.querySelector(kind === "chapter" ? "h2" : "h1, h2");
  const title = collapse((heading && plainText(heading)) || (kind === "intro" ? "About this volume" : "Volume review"));
  const explicit = kind === "chapter" ? heading?.getAttribute("id")?.trim() : undefined;
  const fallback = kind === "intro" ? "about" : kind === "review" ? "review" : slugify(title);
  const slug = uniqueSlug(explicit || fallback, usedSlugs);

  if (kind === "chapter" && heading && chapterNumber) {
    heading.setAttribute("data-num", String(chapterNumber));
    if (!heading.getAttribute("id")) heading.setAttribute("id", slug);
  }

  const outlineSelector = kind === "chapter" ? "h3" : kind === "review" ? "h2" : "h3";
  const headings = ensureHeadingIds(section, outlineSelector);
  decorate(section);

  return {
    slug,
    title,
    kind,
    number: chapterNumber,
    minutes: readingMinutes(section),
    lede: ledeOf(section, kind),
    headings,
    html: section.outerHTML,
  };
}

function readVolume(file: string, filename: string): ModuleRecord | null {
  const match = filename.match(/^(\d+)-(.+)\.html$/);
  if (!match) return null;
  const number = Number(match[1]);
  if (!Number.isFinite(number) || number <= 0) return null;

  const doc = parse(fs.readFileSync(file, "utf8"));
  const body = doc.querySelector("body");
  if (!body) return null;

  const title = collapse(doc.querySelector(".title-page h1")?.text || doc.querySelector("title")?.text || match[2]);
  const subtitle = collapse(doc.querySelector(".tp-subtitle")?.text || "");
  const pages: PageRecord[] = [];
  const usedSlugs = new Set<string>();
  let chapter = 0;

  for (const section of childSections(body)) {
    const classes = section.classList;
    if (classes.contains("title-page")) continue;
    if (classes.contains("intro")) {
      pages.push(pageFromSection(section, "intro", null, usedSlugs));
      continue;
    }
    if (classes.contains("chapter")) {
      chapter += 1;
      pages.push(pageFromSection(section, "chapter", chapter, usedSlugs));
      continue;
    }
    if (classes.contains("review")) {
      pages.push(pageFromSection(section, "review", null, usedSlugs));
    }
  }

  if (pages.length === 0) return null;
  return {
    slug: filename.replace(/\.html$/, ""),
    number,
    title,
    subtitle,
    pages,
  };
}

export function getModules(): ModuleRecord[] {
  const dir = volumesDir();
  const stamp = stampOf(dir);
  if (cache && cache.stamp === stamp) return cache.modules;

  const modules = fs
    .readdirSync(dir)
    .filter((name) => name.endsWith(".html"))
    .sort()
    .map((name) => readVolume(path.join(dir, name), name))
    .filter((mod): mod is ModuleRecord => mod !== null)
    .sort((a, b) => a.number - b.number);

  cache = { stamp, modules };
  return modules;
}

export function getCatalog(): Catalog {
  return {
    modules: getModules().map((mod) => ({
      slug: mod.slug,
      number: mod.number,
      title: mod.title,
      subtitle: mod.subtitle,
      pages: mod.pages.map(({ html: _html, ...meta }) => meta),
    })),
  };
}

export function getModule(slug: string): ModuleRecord | undefined {
  return getModules().find((mod) => mod.slug === slug);
}

export function flatPages(modules = getModules()) {
  return modules.flatMap((mod) =>
    mod.pages.map((page) => ({
      moduleSlug: mod.slug,
      moduleTitle: mod.title,
      moduleNumber: mod.number,
      page,
    })),
  );
}

export function getPageContext(moduleSlug: string, pageSlug: string) {
  const modules = getModules();
  const mod = modules.find((item) => item.slug === moduleSlug);
  const page = mod?.pages.find((item) => item.slug === pageSlug);
  if (!mod || !page) return null;

  const flat = flatPages(modules);
  const index = flat.findIndex((item) => item.moduleSlug === moduleSlug && item.page.slug === pageSlug);
  const prev = index > 0 ? flat[index - 1] : null;
  const next = index >= 0 && index < flat.length - 1 ? flat[index + 1] : null;
  return { module: mod, page, prev, next, index, total: flat.length };
}
