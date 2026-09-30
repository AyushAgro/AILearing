export type Heading = {
  id: string;
  text: string;
  level: 2 | 3;
};

export type PageKind = "intro" | "chapter" | "review";

export type PageMeta = {
  slug: string;
  title: string;
  kind: PageKind;
  number: number | null;
  minutes: number;
  lede: string;
  headings: Heading[];
};

export type ModuleMeta = {
  slug: string;
  number: number;
  title: string;
  subtitle: string;
  pages: PageMeta[];
};

export type Catalog = {
  modules: ModuleMeta[];
};

export type PageRecord = PageMeta & { html: string };

export type ModuleRecord = Omit<ModuleMeta, "pages"> & { pages: PageRecord[] };

export const SERIES = {
  title: "AI Engineering",
  subtitle: "From First Principles to Production",
};
