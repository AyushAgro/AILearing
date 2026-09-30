import Link from "next/link";

export default function NotFound() {
  return (
    <main className="study" id="content">
      <p className="kicker">Missing page</p>
      <h1>That page is not in the book.</h1>
      <p className="study-stat">
        <Link href="/" className="back">
          Library
        </Link>
      </p>
    </main>
  );
}
