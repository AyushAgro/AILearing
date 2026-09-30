import { Library } from "@/components/Library";
import { getCatalog } from "@/lib/content";

export default function HomePage() {
  return <Library catalog={getCatalog()} />;
}
