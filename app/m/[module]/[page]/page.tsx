import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { Reader, type Neighbor } from "@/components/Reader";
import { getModules, getPageContext } from "@/lib/content";
import type { ModuleMeta } from "@/lib/types";

type Params = { module: string; page: string };
type Flat = NonNullable<ReturnType<typeof getPageContext>>["prev"];

export function generateStaticParams() {
  return getModules().flatMap((mod) => mod.pages.map((page) => ({ module: mod.slug, page: page.slug })));
}

export async function generateMetadata({ params }: { params: Promise<Params> }): Promise<Metadata> {
  const { module: moduleSlug, page: pageSlug } = await params;
  const ctx = getPageContext(moduleSlug, pageSlug);
  return { title: ctx ? ctx.page.title : "Not found" };
}

function neighbor(item: Flat): Neighbor | null {
  if (!item) return null;
  const { html: _html, ...page } = item.page;
  return {
    moduleSlug: item.moduleSlug,
    moduleTitle: item.moduleTitle,
    moduleNumber: item.moduleNumber,
    page,
  };
}

export default async function ReaderPage({ params }: { params: Promise<Params> }) {
  const { module: moduleSlug, page: pageSlug } = await params;
  const ctx = getPageContext(moduleSlug, pageSlug);
  if (!ctx) notFound();

  const moduleMeta: ModuleMeta = {
    slug: ctx.module.slug,
    number: ctx.module.number,
    title: ctx.module.title,
    subtitle: ctx.module.subtitle,
    pages: ctx.module.pages.map(({ html: _html, ...meta }) => meta),
  };

  return (
    <Reader
      module={moduleMeta}
      page={moduleMeta.pages.find((item) => item.slug === pageSlug) as ModuleMeta["pages"][number]}
      html={ctx.page.html}
      prev={neighbor(ctx.prev)}
      next={neighbor(ctx.next)}
    />
  );
}
