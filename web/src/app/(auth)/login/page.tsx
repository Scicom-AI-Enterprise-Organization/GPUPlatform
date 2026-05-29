import { LoginForm } from "./login-form";
import Link from "next/link";
import { GithubSsoButton } from "./github-sso-button";

export default async function LoginPage({
  searchParams,
}: {
  searchParams: Promise<{ next?: string; sso_error?: string }>;
}) {
  const { next, sso_error: ssoError } = await searchParams;
  const githubEnabled = !!process.env.GITHUB_CLIENT_ID;
  const registrationDisabled = process.env.REGISTRATION_DISABLED === "1";
  const target = next ?? "/serverless";
  return (
    <div className="space-y-6">
      <div className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">Sign in</h1>
        <p className="text-sm text-muted-foreground">
          Welcome back to GPU Platform.
        </p>
      </div>

      {ssoError && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          GitHub sign-in failed: {ssoError}
        </div>
      )}

      {githubEnabled && (
        <>
          <GithubSsoButton next={target} />
          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <span className="w-full border-t border-border" />
            </div>
            <div className="relative flex justify-center text-[11px] uppercase tracking-wide">
              <span className="bg-background px-2 text-muted-foreground">or with email</span>
            </div>
          </div>
        </>
      )}

      <LoginForm next={target} />
      {!registrationDisabled && (
        <p className="text-xs text-muted-foreground">
          No account?{" "}
          <Link href="/register" className="text-primary hover:underline">
            Create one
          </Link>
        </p>
      )}
    </div>
  );
}
