"use client";

import { useEffect, useState } from "react";

const THEME_KEY = "aie-learn-theme";

/**
 * Re-applies the saved theme after hydration. The inline head script sets it before paint,
 * but when React re-renders <html> on the client (e.g. a notFound() page) the attribute is lost.
 */
export function ThemeSync() {
  useEffect(() => {
    try {
      if (localStorage.getItem(THEME_KEY) === "dark") document.documentElement.dataset.theme = "dark";
    } catch {
      /* Storage blocked: stay on the light theme. */
    }
  }, []);
  return null;
}

export function ThemeButton() {
  const [dark, setDark] = useState(false);

  useEffect(() => {
    setDark(document.documentElement.dataset.theme === "dark");
  }, []);

  function toggle() {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    if (next === "dark") document.documentElement.dataset.theme = "dark";
    else delete document.documentElement.dataset.theme;
    try {
      localStorage.setItem(THEME_KEY, next);
    } catch {
      /* Theme still applies for this visit. */
    }
    setDark(next === "dark");
  }

  return (
    <button type="button" className="icon-btn" onClick={toggle} aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}>
      {dark ? (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <circle cx="12" cy="12" r="4" fill="none" stroke="currentColor" strokeWidth="1.8" />
          <path d="M12 3v2.2M12 18.8V21M3 12h2.2M18.8 12H21M5.6 5.6l1.6 1.6M16.8 16.8l1.6 1.6M18.4 5.6l-1.6 1.6M7.2 16.8l-1.6 1.6" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M16.5 14.5A6.5 6.5 0 0 1 9.2 6 6.8 6.8 0 1 0 16.5 14.5Z" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
        </svg>
      )}
    </button>
  );
}
