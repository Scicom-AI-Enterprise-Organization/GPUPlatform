import Link from "next/link";
import Image from "next/image";

export default function AuthLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="grid min-h-screen w-full lg:grid-cols-2">
      {/* Left — Brand hero. Hidden on mobile to free up space for the form. */}
      <div className="relative hidden overflow-hidden bg-gradient-to-br from-primary/15 via-background to-primary/5 lg:block">
        {/* Faint dot grid for visual texture */}
        <div
          aria-hidden
          className="absolute inset-0 opacity-[0.4] [background-image:radial-gradient(circle,_var(--muted-foreground)_1px,_transparent_1px)] [background-size:18px_18px]"
        />
        <div className="relative z-10 flex h-full flex-col justify-between p-10">
          <Link href="/" className="inline-flex items-center gap-3 self-start">
            <Image
              src="/logos/scicom-logo.png"
              alt="Scicom"
              width={140}
              height={36}
              priority
              className="h-9 w-auto select-none"
            />
            <span className="text-sm font-medium uppercase tracking-wide text-muted-foreground">
              GPU Platform
            </span>
          </Link>
          <div className="max-w-md space-y-3">
            <h2 className="text-3xl font-semibold tracking-tight text-foreground">
              Run inference without owning the rack.
            </h2>
            <p className="text-sm text-muted-foreground">
              Deploy vLLM endpoints in one click, autoscale GPU workers across
              clouds, and pay per second of compute.
            </p>
          </div>
        </div>
      </div>

      {/* Right — Form. Carries its own logo on small screens since the hero is hidden. */}
      <div className="flex flex-col items-center justify-center bg-background px-6 py-10 sm:px-10">
        <div className="w-full max-w-sm">
          <Link
            href="/"
            className="mb-6 inline-flex items-center gap-2 lg:hidden"
          >
            <Image
              src="/logos/scicom-logo.png"
              alt="Scicom"
              width={120}
              height={32}
              priority
              className="h-8 w-auto select-none"
            />
            <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              GPU Platform
            </span>
          </Link>
          {children}
        </div>
      </div>
    </div>
  );
}
