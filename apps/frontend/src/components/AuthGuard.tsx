"use client";

/**
 * AuthGuard — Client-side route protection component.
 * Wraps protected pages and redirects unauthenticated users to login.
 * Optionally checks for specific roles.
 */

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth, type UserRole } from "@/context/AuthContext";

interface AuthGuardProps {
  children: React.ReactNode;
  requiredRole?: UserRole;
  fallbackPath?: string;
}

export function AuthGuard({
  children,
  requiredRole,
  fallbackPath = "/auth/login",
}: AuthGuardProps) {
  const { isAuthenticated, isLoading, roles } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (isLoading) return;

    if (!isAuthenticated) {
      router.replace(fallbackPath);
      return;
    }

    if (requiredRole && !roles.includes(requiredRole)) {
      // User is authenticated but lacks the required role — redirect to their dashboard
      if (roles.includes("rover")) {
        router.replace("/rover");
      } else if (roles.includes("caregiver")) {
        router.replace("/guardian");
      } else if (roles.includes("doctor")) {
        router.replace("/medical");
      } else {
        router.replace("/");
      }
    }
  }, [isAuthenticated, isLoading, roles, requiredRole, router, fallbackPath]);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center gradient-bg">
        <div className="text-center animate-fade-in">
          <div className="w-16 h-16 rounded-2xl gradient-primary flex items-center justify-center mx-auto mb-4 animate-pulse">
            <span className="text-white text-2xl font-bold">N</span>
          </div>
          <p className="text-text-muted dark:text-gray-400 text-lg">Loading NovaCare...</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return null;
  }

  if (requiredRole && !roles.includes(requiredRole)) {
    return null;
  }

  return <>{children}</>;
}
