"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { ThemeButton } from "@/components/ThemeButton";
import { useProgress } from "@/components/ProgressProvider";
import { SERIES, type Catalog } from "@/lib/types";
import {
  continueTarget,
  countDone,
  formatDuration,
  getEntry,
  minutesLeft,
  moduleStatus,
  normalizeProgress,
} from "@/lib/progress";

function pad(n: number) {
  return String(n).padStart(2, "0");
}

export function Library({ catalog }: { catalog: Catalog }) {
  const progress = useProgress();
  const [query, setQuery] = useState("");
  const [notice, setNotice] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  const totals = useMemo(() => {
    const pages = catalog.modules.flatMap((mod) =>
      mod.pages.map((page) => ({
        minutes: page.minutes,
        done: progress.ready && getEntry(progress.store, mod.slug, page.slug).done,
      })),
    );
    const done = pages.filter((page) => page.done).length;
    return {
      pages: pages.length,
      done,
      left: minutesLeft(pages),
      minutes: pages.reduce((sum, page) => sum + page.minutes, 0),
    };
  }, [catalog, progress.ready, progress.store]);

  const target = progress.ready ? continueTarget(catalog, progress.store) : null;
  const targetEntry = target ? getEntry(progress.store, target.mod.slug, target.page.slug) : null;
  const started = progress.ready && Object.values(progress.store.pages).some((page) => page.seenAt || page.done);

  const results = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return [];
    const found: {
      href: string;
      moduleTitle: string;
      moduleNumber: number;
      title: string;
      detail: string;
    }[] = [];
    for (const mod of catalog.modules) {
      for (const page of mod.pages) {
        const heading = page.headings.find((item) => item.text.toLowerCase().includes(q));
        const hay = [page.title, page.lede, mod.title, ...page.headings.map((item) => item.text)].join(" ").toLowerCase();
        if (!hay.includes(q)) continue;
        found.push({
          href: `/m/${mod.slug}/${page.slug}${heading ? `#${heading.id}` : ""}`,
          moduleTitle: mod.title,
          moduleNumber: mod.number,
          title: page.title,
          detail: heading ? heading.text : page.lede,
        });
        if (found.length >= 24) return found;
      }
    }
    return found;
  }, [catalog, query]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const targetEl = event.target as HTMLElement | null;
      const typing = targetEl && (targetEl.tagName === "INPUT" || targetEl.tagName === "TEXTAREA");
      if (event.key === "/" && !typing) {
        event.preventDefault();
        searchRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  function exportProgress() {
    const blob = new Blob([JSON.stringify(progress.store, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "ai-engineering-progress.json";
    link.click();
    URL.revokeObjectURL(url);
    setNotice("Progress downloaded.");
  }

  async function importProgress(file: File) {
    try {
      const parsed = normalizeProgress(JSON.parse(await file.text()));
      if (!parsed.pages || parsed.version !== 1) throw new Error("invalid");
      progress.replace(parsed);
      setNotice("Progress imported on this browser.");
    } catch {
      setNotice("That file is not a progress export.");
    }
  }

  const percent = totals.pages ? Math.round((totals.done / totals.pages) * 100) : 0;
  const placeLabel = !started ? "Start" : target?.resuming ? "Continue" : "Up next";

  return (
    <main className="study" id="content">
      <header className="study-top">
        <p className="kicker">{SERIES.subtitle}</p>
        <div className="study-title-row">
          <h1>{SERIES.title}</h1>
          <ThemeButton />
        </div>
        <p className="study-stat">
          {catalog.modules.length} volumes · {totals.pages} pages · {formatDuration(totals.minutes)} of reading
          {progress.ready && totals.done > 0
            ? totals.done === totals.pages
              ? " · every page marked done"
              : ` · ${totals.done} done · ${formatDuration(totals.left)} left`
            : ""}
        </p>
      </header>

      {!progress.ready || !target ? (
        <div className="continue skeleton" aria-hidden="true" />
      ) : (
        <Link className="continue" href={`/m/${target.mod.slug}/${target.page.slug}`}>
          <div>
            <p className="kicker">{placeLabel}</p>
            <p className="continue-volume">
              Volume {target.mod.number} · {target.mod.title}
            </p>
            <h2>{target.page.title}</h2>
            <p className="continue-meta">
              {target.page.kind === "chapter" ? `Chapter ${target.page.number}` : target.page.kind === "review" ? "Volume review" : "Opening"}
              {" · "}
              {target.page.minutes} min
              {target.resuming && targetEntry && targetEntry.scroll > 0.08 && targetEntry.scroll < 0.92
                ? " · partway through"
                : ""}
              {target.resuming && targetEntry && targetEntry.scroll >= 0.92 ? " · near the end" : ""}
            </p>
          </div>
          <div className="continue-side">
            <div className="ring" style={{ ["--p" as string]: percent }}>
              <span>{percent}%</span>
            </div>
            <span className="btn primary">Resume</span>
          </div>
        </Link>
      )}

      <div className="search-wrap">
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <circle cx="11" cy="11" r="6.5" fill="none" stroke="currentColor" strokeWidth="1.8" />
          <path d="M16 16.5 20 20.5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
        </svg>
        <input
          ref={searchRef}
          className="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search chapters and headings"
          aria-label="Search the book"
        />
        {query ? (
          <button type="button" className="search-clear" onClick={() => setQuery("")}>
            Clear
          </button>
        ) : (
          <kbd className="search-kbd">/</kbd>
        )}
      </div>

      {query.trim() ? (
        <section className="results" aria-label="Search results">
          {results.length === 0 ? (
            <p className="empty">No pages match “{query.trim()}”.</p>
          ) : (
            <ul>
              {results.map((result) => (
                <li key={result.href}>
                  <Link href={result.href}>
                    <span>
                      Volume {result.moduleNumber} · {result.moduleTitle}
                    </span>
                    <strong>{result.title}</strong>
                    {result.detail ? <em>{result.detail}</em> : null}
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </section>
      ) : (
        <section className="rail" aria-label="Volumes">
          <h2 className="kicker rail-label">Volumes</h2>
          {catalog.modules.map((mod) => {
            const done = progress.ready ? countDone(progress.store, mod) : 0;
            const status = progress.ready ? moduleStatus(progress.store, mod) : "new";
            const pct = mod.pages.length ? (done / mod.pages.length) * 100 : 0;
            const active = target?.mod.slug === mod.slug && status !== "done";
            const left = minutesLeft(
              mod.pages.map((page) => ({
                minutes: page.minutes,
                done: progress.ready && getEntry(progress.store, mod.slug, page.slug).done,
              })),
            );
            return (
              <Link
                key={mod.slug}
                href={`/m/${mod.slug}`}
                className={active ? "volume is-active" : status === "done" ? "volume is-done" : "volume"}
              >
                <span className="vol-num">{pad(mod.number)}</span>
                <span className="vol-body">
                  <span className="vol-top">
                    <strong>{mod.title}</strong>
                    {status === "reading" ? <em>Reading</em> : null}
                    {status === "done" ? <em className="done-label">Done</em> : null}
                  </span>
                  {mod.subtitle ? <span className="vol-sub">{mod.subtitle}</span> : null}
                  <span className="meter" aria-hidden="true">
                    <span style={{ width: `${pct}%` }} />
                  </span>
                  <span className="vol-meta">
                    {progress.ready ? `${done} of ${mod.pages.length} pages` : `${mod.pages.length} pages`}
                    {progress.ready && status !== "done" && done > 0 ? ` · ${formatDuration(left)} left` : ""}
                    {status === "done" ? "" : progress.ready && done === 0 ? ` · ${formatDuration(mod.pages.reduce((sum, page) => sum + page.minutes, 0))}` : ""}
                  </span>
                </span>
              </Link>
            );
          })}
        </section>
      )}

      <footer className="study-foot">
        <p>Progress is saved in this browser. Export it when you want the same place on another device.</p>
        <div className="foot-actions">
          <button type="button" onClick={exportProgress}>
            Export
          </button>
          <button type="button" onClick={() => fileRef.current?.click()}>
            Import
          </button>
          <button
            type="button"
            onClick={() => {
              if (window.confirm("Clear reading progress on this browser?")) {
                progress.clear();
                setNotice("Progress cleared.");
              }
            }}
          >
            Clear
          </button>
          <input
            ref={fileRef}
            type="file"
            accept="application/json"
            hidden
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) void importProgress(file);
              event.target.value = "";
            }}
          />
        </div>
        {notice ? <p className="notice">{notice}</p> : null}
      </footer>
    </main>
  );
}
