import { redirect } from "next/navigation";
import { cookies } from "next/headers";
import { TOKEN_COOKIE } from "@/lib/auth-cookie";

export default async function Index() {
  const jar = await cookies();
  const signedIn = !!jar.get(TOKEN_COOKIE)?.value;
  redirect(signedIn ? "/inference" : "/login");
}
