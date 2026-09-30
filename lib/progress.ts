import type { Catalog, ModuleMeta } from "./types";

export const PROGRESS_KEY = "aie-learn-progress-v1";

export type PageProgress = {
  done: boolean;
  doneAt?: string;
  scroll: number;
  note: string;
  seenAt?: string;
};

export type ProgressStore = {
  version: 1;
  last: { module: string; page: string } | null;
  pages: Record<string, PageProgress>;
};

export type Place = {
  module: string;
  page: string;
};

export const EMPTY_PROGRESS: ProgressStore = {
  version: 1,
  last: null,
  pages: {},
};

export function pageKey(moduleSlug: string, pageSlug: string): string {
  return `${moduleSlug}/${pageSlug}`;
}

function blankPage(): PageProgress {
  return { done: false, scroll: 0, note: "" };
}

export function normalizeProgress(raw: unknown): ProgressStore {
  if (!raw || typeof raw !== "object") return { ...EMPTY_PROGRESS, pages: {} };
  const value = raw as Partial<ProgressStore>;
  if (value.version !== 1 || !value.pages || typeof value.pages !== "object") {
    return { ...EMPTY_PROGRESS, pages: {} };
  }

  const pages: Record<string, PageProgress> = {};
  for (const [key, entry] of Object.entries(value.pages)) {
    if (!entry || typeof entry !== "object") continue;
    const item = entry as Partial<PageProgress>;
    pages[key] = {
      done: Boolean(item.done),
      doneAt: typeof item.doneAt === "string" ? item.doneAt : undefined,
      scroll: typeof item.scroll === "number" && Number.isFinite(item.scroll) ? Math.min(1, Math.max(0, item.scroll)) : 0,
      note: typeof item.note === "string" ? item.note : "",
      seenAt: typeof item.seenAt === "string" ? item.seenAt : undefined,
    };
  }

  const last = value.last;
  const safeLast =
    last && typeof last.module === "string" && typeof last.page === "string"
      ? { module: last.module, page: last.page }
      : null;

  return { version: 1, last: safeLast, pages };
}

export function readProgress(): ProgressStore {
  if (typeof window === "undefined") return { ...EMPTY_PROGRESS, pages: {} };
  try {
    return normalizeProgress(JSON.parse(localStorage.getItem(PROGRESS_KEY) || "null"));
  } catch {
    return { ...EMPTY_PROGRESS, pages: {} };
  }
}

export function writeProgress(store: ProgressStore) {
  try {
    localStorage.setItem(PROGRESS_KEY, JSON.stringify(store));
  } catch {
    /* Storage can be full or blocked; keep the in-memory store working. */
  }
}

export function getEntry(store: ProgressStore, moduleSlug: string, pageSlug: string): PageProgress {
  return store.pages[pageKey(moduleSlug, pageSlug)] ?? blankPage();
}

export function patchEntry(
  store: ProgressStore,
  moduleSlug: string,
  pageSlug: string,
  patch: Partial<PageProgress>,
): ProgressStore {
  const key = pageKey(moduleSlug, pageSlug);
  const prev = store.pages[key] ?? blankPage();
  return {
    ...store,
    pages: {
      ...store.pages,
      [key]: { ...prev, ...patch },
    },
  };
}

export function countDone(store: ProgressStore, module: Pick<ModuleMeta, "slug" | "pages">): number {
  return module.pages.filter((page) => getEntry(store, module.slug, page.slug).done).length;
}

export function moduleStatus(store: ProgressStore, module: ModuleMeta): "new" | "reading" | "done" {
  const done = countDone(store, module);
  if (done === module.pages.length) return "done";
  const seen = module.pages.some((page) => {
    const entry = getEntry(store, module.slug, page.slug);
    return entry.done || entry.seenAt;
  });
  return seen ? "reading" : "new";
}

function locate(catalog: Catalog, place: Place | null) {
  if (!place) return null;
  const mod = catalog.modules.find((item) => item.slug === place.module);
  const page = mod?.pages.find((item) => item.slug === place.page);
  if (!mod || !page) return null;
  return { mod, page };
}

export function continueTarget(catalog: Catalog, store: ProgressStore): { mod: ModuleMeta; page: ModuleMeta["pages"][number]; resuming: boolean } | null {
  if (catalog.modules.length === 0) return null;
  const flat = catalog.modules.flatMap((mod) => mod.pages.map((page) => ({ mod, page })));
  const last = locate(catalog, store.last);
  if (last) {
    const lastEntry = getEntry(store, last.mod.slug, last.page.slug);
    if (!lastEntry.done) {
      return { mod: last.mod, page: last.page, resuming: true };
    }
    const index = flat.findIndex((item) => item.mod.slug === last.mod.slug && item.page.slug === last.page.slug);
    const upcoming = flat.slice(index + 1).find((item) => !getEntry(store, item.mod.slug, item.page.slug).done);
    if (upcoming) return { ...upcoming, resuming: false };
  }
  const firstOpen = flat.find((item) => !getEntry(store, item.mod.slug, item.page.slug).done);
  if (firstOpen) return { ...firstOpen, resuming: false };
  return { ...flat[0], resuming: false };
}

export function moduleResume(module: ModuleMeta, store: ProgressStore) {
  let latest: { page: ModuleMeta["pages"][number]; at: string } | null = null;
  for (const page of module.pages) {
    const seenAt = getEntry(store, module.slug, page.slug).seenAt;
    if (seenAt && (!latest || seenAt > latest.at)) latest = { page, at: seenAt };
  }
  if (latest && !getEntry(store, module.slug, latest.page.slug).done) return latest.page;
  return module.pages.find((page) => !getEntry(store, module.slug, page.slug).done) ?? module.pages[module.pages.length - 1];
}

export function minutesLeft(pages: { minutes: number; done: boolean }[]): number {
  return pages.reduce((sum, page) => sum + (page.done ? 0 : page.minutes), 0);
}

export function formatDuration(minutes: number): string {
  if (minutes < 90) return `${minutes} min`;
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  if (rest < 15) return `${hours} hr`;
  return `${hours} hr ${rest} min`;
}
