"use client";

export function DoneToggle({
  done,
  label,
  onToggle,
}: {
  done: boolean;
  label: string;
  onToggle: () => void;
}) {
  return (
    <button
      type="button"
      className={done ? "check is-done" : "check"}
      aria-pressed={done}
      aria-label={label}
      onClick={(event) => {
        event.preventDefault();
        event.stopPropagation();
        onToggle();
      }}
    >
      <svg viewBox="0 0 16 16" aria-hidden="true">
        <path
          d="M3.2 8.2 6.3 11.2 12.8 4.6"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </button>
  );
}
