"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { Article } from "@/components/Article";
import { DoneToggle } from "@/components/DoneToggle";
import { ThemeButton } from "@/components/ThemeButton";
import { useProgress } from "@/components/ProgressProvider";
import type { ModuleMeta, PageMeta } from "@/lib/types";
import { countDone } from "@/lib/progress";

export type Neighbor = {
  moduleSlug: string;
  moduleTitle: string;
  moduleNumber: number;
  page: PageMeta;
};

type Size = "s" | "m" | "l";

function hrefOf(item: Neighbor) {
  return `/m/${item.moduleSlug}/${item.page.slug}`;
}

function kindLabel(page: PageMeta) {
  if (page.kind === "intro") return "Opening";
  if (page.kind === "review") return "Review";
  return `Chapter ${page.number}`;
}

export function Reader({
  module,
  page,
  html,
  prev,
  next,
}: {
  module: ModuleMeta;
  page: PageMeta;
  html: string;
  prev: Neighbor | null;
  next: Neighbor | null;
}) {
  const router = useRouter();
  const progress = useProgress();
  const progressRef = useRef(progress);
  progressRef.current = progress;

  const [navOpen, setNavOpen] = useState(false);
  const [notesOpen, setNotesOpen] = useState(false);
  const [size, setSize] = useState<Size>("m");
  const [activeId, setActiveId] = useState("");
  const [bar, setBar] = useState(0);
  const userMoved = useRef(false);
  const ignoreUntil = useRef(0);

  const entry = progress.entry(module.slug, page.slug);
  const doneCount = countDone(progress.store, module);

  useEffect(() => {
    try {
      const saved = localStorage.getItem("aie-learn-size");
      if (saved === "s" || saved === "m" || saved === "l") setSize(saved);
    } catch {
      /* Storage blocked: keep the default size. */
    }
  }, []);

  useEffect(() => {
    setNavOpen(false);
    userMoved.current = false;
    setActiveId("");
  }, [module.slug, page.slug]);

  function ratioNow() {
    const max = document.documentElement.scrollHeight - window.innerHeight;
    if (max <= 0) return 0;
    return Math.min(1, Math.max(0, window.scrollY / max));
  }

  function restore() {
    ignoreUntil.current = performance.now() + 700;
    const hash = window.location.hash.slice(1);
    if (hash) {
      const target = document.getElementById(decodeURIComponent(hash));
      if (target) {
        target.scrollIntoView({ block: "start" });
        setBar(ratioNow());
        return;
      }
    }
    const saved = progressRef.current.entry(module.slug, page.slug).scroll;
    const max = document.documentElement.scrollHeight - window.innerHeight;
    window.scrollTo(0, saved * Math.max(0, max));
    setBar(saved);
  }

  useEffect(() => {
    if (!progress.ready) return;
    let cancelled = false;
    const run = () => {
      if (!cancelled) restore();
    };
    run();
    const frame = requestAnimationFrame(run);
    const timer = window.setTimeout(run, 60);
    return () => {
      cancelled = true;
      cancelAnimationFrame(frame);
      window.clearTimeout(timer);
    };
    // Restore only when the page changes. `restore` reads the latest progress ref.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [progress.ready, module.slug, page.slug]);

  useEffect(() => {
    if (!progress.ready) return;
    progressRef.current.touch(module.slug, page.slug);
  }, [progress.ready, module.slug, page.slug]);

  useEffect(() => {
    let timer = 0;
    const onScroll = () => {
      const ratio = ratioNow();
      setBar(ratio);
      if (performance.now() < ignoreUntil.current) return;
      userMoved.current = true;
      window.clearTimeout(timer);
      timer = window.setTimeout(() => {
        progressRef.current.setScroll(module.slug, page.slug, ratio);
      }, 280);
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.clearTimeout(timer);
    };
  }, [module.slug, page.slug]);

  useEffect(() => {
    const nodes = page.headings
      .map((heading) => document.getElementById(heading.id))
      .filter((node): node is HTMLElement => Boolean(node));
    if (!nodes.length) return;
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible[0]?.target.id) setActiveId(visible[0].target.id);
      },
      { rootMargin: "-15% 0px -70% 0px", threshold: [0, 1] },
    );
    for (const node of nodes) observer.observe(node);
    return () => observer.disconnect();
  }, [page.headings, html]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)) return;
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (event.key === "Escape") {
        setNavOpen(false);
        setNotesOpen(false);
      } else if (event.key === "d") {
        event.preventDefault();
        progressRef.current.toggleDone(module.slug, page.slug);
      } else if (event.key === "n") {
        setNotesOpen((open) => !open);
      } else if (event.key === "]" && next) {
        router.push(hrefOf(next), { scroll: false });
      } else if (event.key === "[" && prev) {
        router.push(hrefOf(prev), { scroll: false });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [module.slug, page.slug, next, prev, router]);

  useEffect(() => {
    document.getElementById(`side-${page.slug}`)?.scrollIntoView({ block: "nearest" });
  }, [page.slug]);

  function cycleSize() {
    const order: Size[] = ["s", "m", "l"];
    const nextSize = order[(order.indexOf(size) + 1) % order.length];
    setSize(nextSize);
    try {
      localStorage.setItem("aie-learn-size", nextSize);
    } catch {
      /* Size still applies for this visit. */
    }
  }

  const outline = page.headings.length ? (
    <ol>
      {page.headings.map((heading) => (
        <li key={heading.id}>
          <a href={`#${heading.id}`} className={activeId === heading.id ? "is-active" : undefined}>
            {heading.text}
          </a>
        </li>
      ))}
    </ol>
  ) : null;

  return (
    <div className={page.headings.length ? "reader-shell has-outline" : "reader-shell"}>
      <div className="reading-progress" style={{ ["--p" as string]: bar }} />
      {navOpen ? <button type="button" className="scrim" aria-label="Close contents" onClick={() => setNavOpen(false)} /> : null}

      <aside className={navOpen ? "sidebar is-open" : "sidebar"} aria-label="Volume contents">
        <Link href="/" className="side-library">
          Library
        </Link>
        <Link href={`/m/${module.slug}`} className="side-volume">
          <span>Volume {module.number}</span>
          <strong>{module.title}</strong>
        </Link>
        <nav>
          {module.pages.map((item) => {
            const itemEntry = progress.entry(module.slug, item.slug);
            const current = item.slug === page.slug;
            return (
              <div key={item.slug} id={`side-${item.slug}`} className={current ? "side-row is-current" : "side-row"}>
                <DoneToggle
                  done={itemEntry.done}
                  label={itemEntry.done ? `Mark “${item.title}” as not done` : `Mark “${item.title}” as done`}
                  onToggle={() => progress.toggleDone(module.slug, item.slug)}
                />
                <Link href={`/m/${module.slug}/${item.slug}`} scroll={false} aria-current={current ? "page" : undefined}>
                  <span>{kindLabel(item)}</span>
                  <strong>{item.title}</strong>
                </Link>
              </div>
            );
          })}
        </nav>
        <div className="side-foot">
          <div className="meter" aria-hidden="true">
            <span style={{ width: `${module.pages.length ? (doneCount / module.pages.length) * 100 : 0}%` }} />
          </div>
          <p>
            {doneCount} of {module.pages.length} done
          </p>
          <p className="keys">
            <kbd>[</kbd> <kbd>]</kbd> pages · <kbd>d</kbd> done · <kbd>n</kbd> notes
          </p>
        </div>
      </aside>

      <div className="reader-main">
        <header className="topbar">
          <button type="button" className="icon-btn nav-toggle" aria-label="Open contents" onClick={() => setNavOpen(true)}>
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M4 7h16M4 12h16M4 17h12" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
            </svg>
          </button>
          <div className="topbar-title">
            <Link href={`/m/${module.slug}`}>
              Volume {module.number}
            </Link>
            <strong>{page.title}</strong>
          </div>
          <div className="topbar-actions">
            <button type="button" className="icon-btn size-btn" onClick={cycleSize} aria-label={`Text size ${size}. Activate to change.`}>
              A
            </button>
            <ThemeButton />
            <button
              type="button"
              className={entry.note ? "icon-btn has-note" : "icon-btn"}
              aria-pressed={notesOpen}
              aria-label="Notes"
              onClick={() => setNotesOpen((open) => !open)}
            >
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M7 4.5h8.2L19 8.2V19.5H7z" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
                <path d="M15 4.8V8.4H18.6M9.2 12h6.2M9.2 15.4h4.4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
              </svg>
            </button>
            <button
              type="button"
              className={entry.done ? "mark is-done" : "mark"}
              aria-pressed={entry.done}
              aria-label={entry.done ? "Mark not complete" : "Mark complete"}
              onClick={() => progress.toggleDone(module.slug, page.slug)}
            >
              <span className="mark-long">{entry.done ? "Mark not complete" : "Mark complete"}</span>
              <span className="mark-short" aria-hidden="true">
                {entry.done ? "Completed" : "Complete"}
              </span>
            </button>
          </div>
        </header>

        <div className="column">
          {outline ? (
            <details className="article-outline">
              <summary>On this page</summary>
              {outline}
            </details>
          ) : null}
          <article className="article-card" id="content" data-size={size}>
            <Article
              html={html}
              onReady={() => {
                if (!userMoved.current) restore();
              }}
            />
          </article>
          <nav className="pager" aria-label="Pages">
            {prev ? (
              <Link href={hrefOf(prev)} scroll={false} className="pager-prev">
                <span>{prev.moduleSlug === module.slug ? "Previous" : `Volume ${prev.moduleNumber}`}</span>
                <strong>{prev.page.title}</strong>
                {prev.moduleSlug !== module.slug ? <em>{prev.moduleTitle}</em> : null}
              </Link>
            ) : (
              <span />
            )}
            {next ? (
              <Link href={hrefOf(next)} scroll={false} className="pager-next">
                <span>{next.moduleSlug === module.slug ? "Next" : `Volume ${next.moduleNumber}`}</span>
                <strong>{next.page.title}</strong>
                {next.moduleSlug !== module.slug ? <em>{next.moduleTitle}</em> : null}
              </Link>
            ) : (
              <span />
            )}
          </nav>
        </div>
      </div>

      {page.headings.length ? (
        <aside className="outline" aria-label="On this page">
          <p>On this page</p>
          {outline}
        </aside>
      ) : null}

      {notesOpen ? (
        <aside className="notes" aria-label="Notes">
          <header>
            <strong>Notes</strong>
            <button type="button" onClick={() => setNotesOpen(false)}>
              Close
            </button>
          </header>
          <textarea
            value={entry.note}
            onChange={(event) => progress.setNote(module.slug, page.slug, event.target.value)}
            placeholder="What do you want to remember from this page?"
          />
          <p>Saved with your progress on this browser.</p>
        </aside>
      ) : null}
    </div>
  );
}
