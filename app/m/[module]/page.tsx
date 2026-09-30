import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { ModuleView } from "@/components/ModuleView";
import { getCatalog } from "@/lib/content";

type Params = { module: string };

export function generateStaticParams() {
  return getCatalog().modules.map((mod) => ({ module: mod.slug }));
}

export async function generateMetadata({ params }: { params: Promise<Params> }): Promise<Metadata> {
  const { module: slug } = await params;
  const mod = getCatalog().modules.find((item) => item.slug === slug);
  return { title: mod ? mod.title : "Not found" };
}

export default async function ModulePage({ params }: { params: Promise<Params> }) {
  const { module: slug } = await params;
  const modules = getCatalog().modules;
  const index = modules.findIndex((item) => item.slug === slug);
  const mod = modules[index];
  if (!mod) notFound();
  const prev = index > 0 ? modules[index - 1] : null;
  const next = index < modules.length - 1 ? modules[index + 1] : null;
  return (
    <ModuleView
      module={mod}
      prev={prev ? { slug: prev.slug, number: prev.number, title: prev.title } : null}
      next={next ? { slug: next.slug, number: next.number, title: next.title } : null}
    />
  );
}
