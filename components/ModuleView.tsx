"use client";

import Link from "next/link";
import { DoneToggle } from "@/components/DoneToggle";
import { ThemeButton } from "@/components/ThemeButton";
import { useProgress } from "@/components/ProgressProvider";
import type { ModuleMeta } from "@/lib/types";
import { countDone, formatDuration, getEntry, minutesLeft, moduleResume } from "@/lib/progress";

function pageLabel(page: ModuleMeta["pages"][number]) {
  if (page.kind === "intro") return "Opening";
  if (page.kind === "review") return "Review";
  return `Chapter ${page.number}`;
}

export function ModuleView({
  module,
  prev,
  next,
}: {
  module: ModuleMeta;
  prev: { slug: string; number: number; title: string } | null;
  next: { slug: string; number: number; title: string } | null;
}) {
  const progress = useProgress();
  const done = progress.ready ? countDone(progress.store, module) : 0;
  const resume = progress.ready ? moduleResume(module, progress.store) : module.pages[0];
  const left = minutesLeft(
    module.pages.map((page) => ({
      minutes: page.minutes,
      done: progress.ready && getEntry(progress.store, module.slug, page.slug).done,
    })),
  );
  const complete = progress.ready && done === module.pages.length;

  return (
    <main className="study" id="content">
      <header className="study-top">
        <div className="study-title-row">
          <Link href="/" className="back">
            Library
          </Link>
          <ThemeButton />
        </div>
        <p className="kicker">Volume {module.number}</p>
        <h1>{module.title}</h1>
        {module.subtitle ? <p className="lede">{module.subtitle}</p> : null}
        <p className="study-stat">
          {progress.ready ? `${done} of ${module.pages.length} pages` : `${module.pages.length} pages`}
          {complete ? " · volume complete" : progress.ready ? ` · ${formatDuration(left)} left` : ""}
        </p>
        <div className="module-actions">
          {resume ? (
            <Link className="btn primary resume-btn" href={`/m/${module.slug}/${resume.slug}`}>
              {complete ? "Read again" : getEntry(progress.store, module.slug, resume.slug).seenAt ? "Resume" : "Start"}
              <span>{resume.title}</span>
            </Link>
          ) : null}
          <button
            type="button"
            className="btn"
            disabled={!progress.ready || complete}
            onClick={() => progress.setModuleDone(module.slug, module.pages.map((page) => page.slug), true)}
          >
            Mark volume complete
          </button>
          <button
            type="button"
            className="btn"
            disabled={!progress.ready || done === 0}
            onClick={() => progress.setModuleDone(module.slug, module.pages.map((page) => page.slug), false)}
          >
            Mark volume not complete
          </button>
        </div>
      </header>

      <ol className="chapters">
        {module.pages.map((page) => {
          const entry = progress.entry(module.slug, page.slug);
          const isResume = resume?.slug === page.slug && !entry.done && !complete;
          return (
            <li key={page.slug} className={isResume ? "chapter is-resume" : entry.done ? "chapter is-done" : "chapter"}>
              <DoneToggle
                done={entry.done}
                label={entry.done ? `Mark “${page.title}” as not done` : `Mark “${page.title}” as done`}
                onToggle={() => progress.toggleDone(module.slug, page.slug)}
              />
              <Link href={`/m/${module.slug}/${page.slug}`} className="chapter-link">
                <span className="chapter-label">
                  {pageLabel(page)}
                  {isResume ? <em>{entry.seenAt ? "Resume" : "Next"}</em> : null}
                  {entry.note ? <em className="note-flag">Note</em> : null}
                </span>
                <span className="chapter-title">{page.title}</span>
                {page.lede ? <span className="chapter-lede">{page.lede}</span> : null}
              </Link>
              <span className="chapter-min">{page.minutes} min</span>
            </li>
          );
        })}
      </ol>

      <nav className="volume-jump" aria-label="Nearby volumes">
        {prev ? (
          <Link href={`/m/${prev.slug}`}>
            <span>Previous volume</span>
            <strong>
              {prev.number}. {prev.title}
            </strong>
          </Link>
        ) : (
          <span />
        )}
        {next ? (
          <Link href={`/m/${next.slug}`}>
            <span>Next volume</span>
            <strong>
              {next.number}. {next.title}
            </strong>
          </Link>
        ) : (
          <span />
        )}
      </nav>
    </main>
  );
}
