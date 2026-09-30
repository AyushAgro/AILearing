"use client";

import { useEffect, useMemo, useRef } from "react";
import "katex/dist/katex.min.css";

export function Article({ html, onReady }: { html: string; onReady?: () => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const readyRef = useRef(onReady);
  readyRef.current = onReady;
  // React 19 re-applies innerHTML whenever this object changes identity, which would wipe
  // the KaTeX and Mermaid output on every parent re-render. Keep it stable per `html`.
  const markup = useMemo(() => ({ __html: html }), [html]);

  useEffect(() => {
    const root = ref.current;
    if (!root) return;
    let cancelled = false;

    async function enhance() {
      if (!root) return;
      const katex = (await import("katex")).default;
      if (cancelled) return;
      for (const el of root.querySelectorAll<HTMLElement>(".math")) {
        if (el.querySelector(".katex")) continue;
        const tex = el.textContent ?? "";
        if (!tex.trim()) continue;
        katex.render(tex, el, {
          displayMode: el.tagName === "DIV",
          throwOnError: false,
          trust: false,
        });
      }

      const diagrams = [...root.querySelectorAll<HTMLElement>("pre.mermaid")].filter(
        (node) => node.dataset.rendered !== "1",
      );
      if (diagrams.length) {
        const mermaid = (await import("mermaid")).default;
        if (cancelled) return;
        mermaid.initialize({
          startOnLoad: false,
          securityLevel: "strict",
          theme: "neutral",
          fontFamily: "inherit",
        });
        try {
          await mermaid.run({ nodes: diagrams });
          for (const node of diagrams) node.dataset.rendered = "1";
        } catch {
          /* Leave the source visible if a diagram cannot render. */
        }
      }
      if (!cancelled) readyRef.current?.();
    }

    void enhance();
    return () => {
      cancelled = true;
    };
  }, [html]);

  return <div ref={ref} className="reader" dangerouslySetInnerHTML={markup} />;
}
