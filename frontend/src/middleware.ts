import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

/**
 * NovaCare Route Protection Middleware
 *
 * Since we store tokens in localStorage (client-side), this middleware
 * performs lightweight checks only.  The real auth guard lives in the
 * AuthContext on the client side.
 *
 * Public routes: /, /auth/*
 * Protected routes: /rover/*, /guardian/*, /medical/*, /admin/*
 */

const PUBLIC_PATHS = ["/", "/auth"];

function isPublicPath(pathname: string): boolean {
  return PUBLIC_PATHS.some(
    (pub) => pathname === pub || pathname.startsWith(`${pub}/`),
  );
}

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // Allow public routes, static assets, and API routes
  if (
    isPublicPath(pathname) ||
    pathname.startsWith("/_next") ||
    pathname.startsWith("/api") ||
    pathname.includes(".")
  ) {
    return NextResponse.next();
  }

  // For protected routes, we rely on client-side AuthContext to handle
  // redirect logic since tokens are in localStorage (not cookies).
  // This middleware is a placeholder for future cookie-based auth.
  return NextResponse.next();
}

export const config = {
  matcher: [
    /*
     * Match all request paths except:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     */
    "/((?!_next/static|_next/image|favicon.ico).*)",
  ],
};
