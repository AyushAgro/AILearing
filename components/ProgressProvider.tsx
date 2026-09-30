"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import {
  EMPTY_PROGRESS,
  getEntry,
  normalizeProgress,
  patchEntry,
  PROGRESS_KEY,
  readProgress,
  writeProgress,
  type PageProgress,
  type ProgressStore,
} from "@/lib/progress";

type Api = {
  ready: boolean;
  store: ProgressStore;
  entry: (moduleSlug: string, pageSlug: string) => PageProgress;
  toggleDone: (moduleSlug: string, pageSlug: string) => void;
  setModuleDone: (moduleSlug: string, pageSlugs: string[], done: boolean) => void;
  setScroll: (moduleSlug: string, pageSlug: string, scroll: number) => void;
  setNote: (moduleSlug: string, pageSlug: string, note: string) => void;
  touch: (moduleSlug: string, pageSlug: string) => void;
  replace: (store: ProgressStore) => void;
  clear: () => void;
};

function markPage(store: ProgressStore, moduleSlug: string, pageSlug: string, done: boolean): ProgressStore {
  const prev = getEntry(store, moduleSlug, pageSlug);
  if (prev.done === done) return store;
  return patchEntry(store, moduleSlug, pageSlug, {
    done,
    doneAt: done ? new Date().toISOString() : undefined,
  });
}

const ProgressContext = createContext<Api | null>(null);

export function ProgressProvider({ children }: { children: React.ReactNode }) {
  const [store, setStore] = useState<ProgressStore>({ ...EMPTY_PROGRESS, pages: {} });
  const [ready, setReady] = useState(false);

  useEffect(() => {
    setStore(readProgress());
    setReady(true);
    const onStorage = (event: StorageEvent) => {
      if (event.key === PROGRESS_KEY) setStore(readProgress());
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  const commit = useCallback((update: (current: ProgressStore) => ProgressStore) => {
    setStore((current) => {
      const next = update(current);
      if (next !== current) writeProgress(next);
      return next;
    });
  }, []);

  const api = useMemo<Api>(
    () => ({
      ready,
      store,
      entry: (moduleSlug, pageSlug) => getEntry(store, moduleSlug, pageSlug),
      toggleDone: (moduleSlug, pageSlug) => {
        commit((current) => {
          const prev = getEntry(current, moduleSlug, pageSlug);
          return markPage(current, moduleSlug, pageSlug, !prev.done);
        });
      },
      setModuleDone: (moduleSlug, pageSlugs, done) => {
        commit((current) => pageSlugs.reduce((store, pageSlug) => markPage(store, moduleSlug, pageSlug, done), current));
      },
      setScroll: (moduleSlug, pageSlug, scroll) => {
        commit((current) => {
          const prev = getEntry(current, moduleSlug, pageSlug);
          if (Math.abs(prev.scroll - scroll) < 0.004) return current;
          return patchEntry(current, moduleSlug, pageSlug, { scroll });
        });
      },
      setNote: (moduleSlug, pageSlug, note) => {
        commit((current) => {
          const prev = getEntry(current, moduleSlug, pageSlug);
          if (prev.note === note) return current;
          return patchEntry(current, moduleSlug, pageSlug, { note });
        });
      },
      touch: (moduleSlug, pageSlug) => {
        commit((current) => ({
          ...patchEntry(current, moduleSlug, pageSlug, { seenAt: new Date().toISOString() }),
          last: { module: moduleSlug, page: pageSlug },
        }));
      },
      replace: (next) => {
        const safe = normalizeProgress(next);
        writeProgress(safe);
        setStore(safe);
      },
      clear: () => {
        const empty: ProgressStore = { version: 1, last: null, pages: {} };
        writeProgress(empty);
        setStore(empty);
      },
    }),
    [commit, ready, store],
  );

  return <ProgressContext.Provider value={api}>{children}</ProgressContext.Provider>;
}

export function useProgress() {
  const value = useContext(ProgressContext);
  if (!value) throw new Error("useProgress must be used within ProgressProvider");
  return value;
}
