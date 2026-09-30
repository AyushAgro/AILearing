import type { Metadata } from "next";
import { Inter, JetBrains_Mono, Source_Serif_4 } from "next/font/google";
import { ProgressProvider } from "@/components/ProgressProvider";
import { ThemeSync } from "@/components/ThemeButton";
import "./globals.css";
import "./reader.css";

const sans = Inter({
  subsets: ["latin"],
  style: ["normal", "italic"],
  variable: "--font-sans",
  display: "swap",
});

const serif = Source_Serif_4({
  subsets: ["latin"],
  style: ["normal", "italic"],
  variable: "--font-serif",
  display: "swap",
});

const mono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    default: "AI Engineering",
    template: "%s · AI Engineering",
  },
  description: "Read the AI Engineering book, resume the last page, and mark chapters done.",
};

const themeScript = `
try {
  if (localStorage.getItem("aie-learn-theme") === "dark") {
    document.documentElement.dataset.theme = "dark";
  }
} catch (e) {}
`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en-GB" className={`${sans.variable} ${serif.variable} ${mono.variable}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body>
        <a className="skip" href="#content">
          Skip to content
        </a>
        <ThemeSync />
        <ProgressProvider>{children}</ProgressProvider>
      </body>
    </html>
  );
}
