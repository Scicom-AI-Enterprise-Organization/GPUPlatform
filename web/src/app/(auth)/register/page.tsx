import { RegisterForm } from "./register-form";
import Link from "next/link";
import { redirect } from "next/navigation";

export default function RegisterPage() {
  if (process.env.REGISTRATION_DISABLED === "1") {
    redirect("/login");
  }
  return (
    <div className="space-y-6">
      <div className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">Create account</h1>
        <p className="text-sm text-muted-foreground">
          Get an isolated workspace for your endpoints.
        </p>
      </div>
      <RegisterForm />
      <p className="text-xs text-muted-foreground">
        Have an account already?{" "}
        <Link href="/login" className="text-primary hover:underline">
          Sign in
        </Link>
      </p>
    </div>
  );
}
