import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ConsoleTopbar } from "@/components/console/topbar";
import { currentUsername } from "@/lib/current-user";
import { CompareView } from "./compare-view";

export default async function BenchmarkComparePage({
  searchParams,
}: {
  searchParams: Promise<{ ids?: string }>;
}) {
  const [sp, username] = await Promise.all([searchParams, currentUsername()]);
  // De-dupe while preserving order; cap so a hand-typed URL can't fan out
  // hundreds of S3 fetches.
  const ids = Array.from(
    new Set((sp.ids ?? "").split(",").map((s) => s.trim()).filter(Boolean)),
  ).slice(0, 12);

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[{ label: "Benchmark", href: "/benchmark" }, { label: "Compare" }]}
        username={username}
      />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        {ids.length < 2 ? (
          <div className="mx-auto max-w-md rounded-md border border-dashed border-border px-6 py-12 text-center">
            <p className="text-sm text-muted-foreground">
              Select two or more benchmarks to compare. Go to the benchmark list, click{" "}
              <span className="font-medium text-foreground">Select</span>, tick the runs, then{" "}
              <span className="font-medium text-foreground">Compare</span>.
            </p>
            <Button asChild variant="outline" size="sm" className="mt-4">
              <Link href="/benchmark">
                <ArrowLeft className="h-4 w-4" /> Back to benchmarks
              </Link>
            </Button>
          </div>
        ) : (
          <CompareView ids={ids} />
        )}
      </div>
    </div>
  );
}
