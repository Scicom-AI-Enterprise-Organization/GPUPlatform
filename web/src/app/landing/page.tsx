import Link from "next/link";
import { ArrowRight, KeyRound, Zap, Cpu, Gauge, Code2, Rocket, Coffee, Mountain, Sparkles } from "lucide-react";
import { cookies } from "next/headers";
import { Button } from "@/components/ui/button";
import { TOKEN_COOKIE } from "@/lib/auth-cookie";
import { LandingDots } from "@/components/landing-dots";
import { TypewriterCycle } from "@/components/typewriter";
import { HeroCycler } from "@/components/hero-cycler";

const SECTION_IDS = ["hero", "how", "features", "api", "cta"];

export default async function Landing() {
  const jar = await cookies();
  const signedIn = !!jar.get(TOKEN_COOKIE)?.value;

  return (
    <div className="relative isolate h-screen overflow-hidden bg-background text-foreground">
      <div aria-hidden className="fixed inset-0 -z-10 bg-gradient-to-b from-background/50 via-background/80 to-background" />

      <header className="fixed top-0 inset-x-0 z-30 border-b border-border/30 bg-background/60 backdrop-blur-xl">
        <nav className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4 sm:px-6">
          <Link href="/landing" className="flex items-center gap-2 font-semibold tracking-tight">
            <span className="inline-flex h-7 w-7 items-center justify-center rounded-md bg-gradient-to-br from-violet-500 to-fuchsia-500 text-white shadow-[0_0_30px_-5px] shadow-fuchsia-500/60">
              <Rocket className="h-4 w-4" />
            </span>
            <span>GPU Platform</span>
          </Link>
          <Button asChild size="sm">
            <Link href={signedIn ? "/serverless" : "/login"}>
              {signedIn ? "Console" : "Sign in"} <ArrowRight className="h-4 w-4" />
            </Link>
          </Button>
        </nav>
      </header>

      <LandingDots ids={SECTION_IDS} />

      <main
        id="landing-scroll"
        className="h-screen overflow-y-auto scroll-smooth pt-14 scrollbar-thin md:snap-y md:snap-mandatory"
      >
        {/* HERO */}
        <section id="hero" className="relative flex min-h-[calc(100vh-3.5rem)] snap-start snap-always items-center justify-center overflow-hidden px-4 py-12 sm:px-6">
          <div aria-hidden className="absolute -top-24 left-1/4 -z-10 h-[480px] w-[480px] rounded-full bg-fuchsia-500/20 blur-[120px]" />
          <div aria-hidden className="absolute top-40 right-1/4 -z-10 h-[420px] w-[420px] rounded-full bg-violet-500/20 blur-[120px]" />
          <div className="mx-auto max-w-3xl text-center">
            <HeroCycler />
            <div className="mt-10">
              <Button asChild size="lg">
                <Link href={signedIn ? "/serverless" : "/login"}>
                  {signedIn ? "Open console" : "Get me in"}
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
            </div>
            <div className="mt-12 grid grid-cols-1 gap-3 sm:grid-cols-3">
              <Tile icon={<Zap className="h-4 w-4" />} title="Wakes up in" value="~60s" />
              <Tile icon={<Coffee className="h-4 w-4" />} title="Idle bill" value="$0.00" />
              <Tile icon={<KeyRound className="h-4 w-4" />} title="Auth" value="Per-user, no shared keys" />
            </div>
          </div>
        </section>

        {/* HOW */}
        <section id="how" className="relative flex min-h-screen snap-start snap-always items-center justify-center px-4 py-12 sm:px-6">
          <div className="mx-auto max-w-6xl">
            <div className="mb-10 text-center">
              <h2 className="text-3xl font-semibold tracking-tight sm:text-4xl">How it works</h2>
              <p className="mt-3 text-muted-foreground">From request to response, with autoscaling in between.</p>
            </div>
            <ol className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5">
              <Step n={1} title="Request received" body="The gateway accepts an inbound request and validates the caller." />
              <Step n={2} title="Job enqueued" body="The job is placed on a per-app Redis queue. A request ID is returned to the caller." />
              <Step n={3} title="Autoscaler reacts" body="If no worker is available for the app, the autoscaler provisions a GPU pod from the configured provider." />
              <Step n={4} title="Worker processes" body="The worker pulls the model, registers with the gateway, and drains the queue." />
              <Step n={5} title="Idle scale-down" body="After the configured idle period with no traffic, the worker is terminated and billing stops." />
            </ol>
            <p className="mt-10 text-center text-sm text-muted-foreground">
              GPUs run only while requests are being served. You pay only for the seconds of compute you actually use.
            </p>
          </div>
        </section>

        {/* FEATURES */}
        <section id="features" className="relative flex min-h-screen snap-start snap-always items-center justify-center px-4 py-12 sm:px-6">
          <div aria-hidden className="absolute inset-0 -z-10 bg-gradient-to-br from-violet-500/5 via-transparent to-fuchsia-500/5" />
          <div aria-hidden className="absolute -top-32 right-0 -z-10 h-[400px] w-[400px] rounded-full bg-violet-500/15 blur-[140px]" />
          <div aria-hidden className="absolute -bottom-32 left-0 -z-10 h-[400px] w-[400px] rounded-full bg-fuchsia-500/15 blur-[140px]" />
          <div className="mx-auto max-w-6xl">
            <div className="mb-10 text-center">
              <h2 className="bg-gradient-to-b from-foreground to-foreground/70 bg-clip-text text-3xl font-semibold tracking-tight text-transparent sm:text-4xl">
                Stuff that actually matters
              </h2>
              <p className="mt-3 text-muted-foreground">A control plane. Not a wrapper. Not a Discord bot.</p>
            </div>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
              <Feature icon={<Cpu className="h-5 w-5" />} title="Multi-cloud GPUs" body="RunPod today. Tomorrow whoever's cheapest. We can ghost any of them." />
              <Feature icon={<Gauge className="h-5 w-5" />} title="Per-app autoscaler" body="Per-endpoint queue depth + idle timeout. Caps prevent your boss's panic attack." />
              <Feature icon={<KeyRound className="h-5 w-5" />} title="Multi-user, owner-scoped" body="Each user owns their endpoints. Admin sees everything. No shared keys." />
              <Feature icon={<Code2 className="h-5 w-5" />} title="OpenAI-compatible" body="Drop-in for any OpenAI client. Switch base_url, watch your bill drop." />
              <Feature icon={<Rocket className="h-5 w-5" />} title="Streaming first" body="SSE end-to-end. Tokens flow as soon as the GPU emits them." />
              <Feature icon={<Sparkles className="h-5 w-5" />} title="Persistent history" body="Every payload + result lands in Postgres. We remember it after Redis forgets." />
            </div>
          </div>
        </section>

        {/* API */}
        <section id="api" className="relative flex min-h-screen snap-start snap-always items-center justify-center px-4 py-12 sm:px-6">
          <div className="mx-auto max-w-4xl">
            <div className="mb-8 text-center">
              <h2 className="text-3xl font-semibold tracking-tight sm:text-4xl">One curl, real GPU</h2>
              <p className="mt-3 text-muted-foreground">No SDK. No Terraform. No tears.</p>
            </div>
            <pre className="overflow-x-auto rounded-xl border border-border/60 bg-zinc-950/80 p-6 text-xs leading-relaxed text-zinc-100 shadow-2xl shadow-violet-500/10 sm:text-sm">
{`# 1. Login (returns a token)
curl -X POST https://serverlessgpu.aies.scicom.dev/auth/login \\
  -H 'Content-Type: application/json' \\
  -d '{"username":"you","password":"..."}'

# 2. Create an endpoint
curl -X POST https://serverlessgpu.aies.scicom.dev/apps \\
  -H 'Authorization: Bearer <token>' \\
  -H 'Content-Type: application/json' \\
  -d '{"name":"qwen","model":"Qwen/Qwen1.5-0.5B","gpu":"rtx3090"}'

# 3. Run inference (OpenAI-compatible)
curl -X POST https://serverlessgpu.aies.scicom.dev/v1/chat/completions \\
  -H 'Authorization: Bearer <token>' \\
  -H 'Content-Type: application/json' \\
  -d '{"model":"qwen","messages":[{"role":"user","content":"yo"}]}'`}
            </pre>
          </div>
        </section>

        {/* CTA */}
        <section id="cta" className="relative flex min-h-screen snap-start snap-always items-center justify-center px-4 py-12 sm:px-6">
          <div aria-hidden className="absolute inset-0 -z-10 bg-gradient-to-b from-transparent via-violet-500/5 to-transparent" />
          <div aria-hidden className="absolute left-1/2 top-1/2 -z-10 h-[420px] w-[760px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-fuchsia-500/15 blur-[140px]" />
          <div className="mx-auto max-w-3xl text-center">
            <Mountain className="mx-auto mb-6 h-10 w-10 text-foreground/70" />
            <h2 className="bg-gradient-to-b from-foreground to-foreground/60 bg-clip-text text-3xl font-semibold tracking-tight text-transparent sm:text-5xl">
              Stop paying for idle pods. Go{" "}
              <TypewriterCycle
                words={["climb", "race", "run", "swipe", "fly"]}
                className="bg-gradient-to-r from-violet-500 to-fuchsia-500 bg-clip-text text-transparent"
              />
              .
            </h2>
            <p className="mt-3 text-muted-foreground">First endpoint takes a minute. Then it&apos;s your problem only when somebody actually uses it.</p>
            <div className="mt-8 flex justify-center">
              <Button asChild size="lg">
                <Link href={signedIn ? "/serverless" : "/login"}>
                  {signedIn ? "Open console" : "Sign in"}
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
            </div>
            <p className="mt-12 text-xs text-muted-foreground">
              Built at SCICOM AIES · Powered by RunPod, vLLM, and a healthy disrespect for idle pods.
            </p>
          </div>
        </section>
      </main>
    </div>
  );
}

function Tile({ icon, title, value }: { icon: React.ReactNode; title: string; value: string }) {
  return (
    <div className="rounded-xl border border-border/60 bg-background/60 px-4 py-3 text-left backdrop-blur">
      <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground">
        {icon}
        {title}
      </div>
      <div className="mt-1 text-lg font-semibold text-foreground">{value}</div>
    </div>
  );
}

function Step({ n, title, body }: { n: number; title: string; body: string }) {
  return (
    <li className="rounded-xl border border-border/40 bg-gradient-to-b from-background/60 to-background/20 p-5 backdrop-blur transition hover:border-border/80 hover:from-background/80">
      <div className="font-mono text-xs text-muted-foreground">Step {n}</div>
      <div className="mt-1 text-base font-semibold text-foreground">{title}</div>
      <div className="mt-2 text-sm text-muted-foreground">{body}</div>
    </li>
  );
}

function Feature({ icon, title, body }: { icon: React.ReactNode; title: string; body: string }) {
  return (
    <div className="group rounded-xl border border-border/40 bg-background/40 p-6 backdrop-blur-sm transition hover:border-border/80 hover:bg-background/70">
      <div className="inline-flex h-10 w-10 items-center justify-center rounded-md bg-gradient-to-br from-violet-500/20 to-fuchsia-500/20 text-foreground ring-1 ring-border/60">
        {icon}
      </div>
      <div className="mt-4 text-base font-semibold text-foreground">{title}</div>
      <p className="mt-2 text-sm text-muted-foreground">{body}</p>
    </div>
  );
}
