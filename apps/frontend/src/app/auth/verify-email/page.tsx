"use client";

import { useEffect, useState, Suspense } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Heart, CheckCircle, XCircle, Loader2 } from "lucide-react";
import { Button, Card } from "@/components/ui";
import { verifyEmail } from "@/lib/auth-api";

function VerifyEmailContent() {
  const searchParams = useSearchParams();
  const token = searchParams.get("token");

  const [status, setStatus] = useState<"loading" | "success" | "error">("loading");
  const [message, setMessage] = useState("");

  useEffect(() => {
    if (!token) {
      setStatus("error");
      setMessage("No verification token provided.");
      return;
    }

    const verify = async () => {
      const res = await verifyEmail(token);
      if (res.status === "success") {
        setStatus("success");
        setMessage("Your email has been verified successfully!");
      } else {
        setStatus("error");
        setMessage(res.error || "Verification failed. The token may be invalid or expired.");
      }
    };

    verify();
  }, [token]);

  return (
    <div className="min-h-screen gradient-bg flex items-center justify-center p-4">
      <div className="w-full max-w-md animate-fade-in">
        {/* Logo */}
        <div className="flex items-center justify-center gap-2 mb-8">
          <div className="w-10 h-10 rounded-xl gradient-primary flex items-center justify-center">
            <Heart className="w-5 h-5 text-white" />
          </div>
          <span className="text-xl font-display font-bold text-text-primary dark:text-white">NovaCare</span>
        </div>

        <Card variant="elevated" padding="lg">
          <div className="text-center py-4">
            {status === "loading" && (
              <>
                <Loader2 className="w-16 h-16 text-primary animate-spin mx-auto mb-4" />
                <h2 className="text-2xl font-display font-bold text-text-primary dark:text-white mb-2">Verifying Email...</h2>
                <p className="text-text-muted dark:text-gray-400">Please wait while we verify your email address.</p>
              </>
            )}

            {status === "success" && (
              <>
                <div className="w-16 h-16 rounded-full bg-success-100 dark:bg-success-900/30 flex items-center justify-center mx-auto mb-4">
                  <CheckCircle className="w-8 h-8 text-success" />
                </div>
                <h2 className="text-2xl font-display font-bold text-text-primary dark:text-white mb-2">Email Verified!</h2>
                <p className="text-text-muted dark:text-gray-400 mb-6">{message}</p>
                <Link href="/auth/login">
                  <Button className="w-full" size="lg">Continue to Sign In</Button>
                </Link>
              </>
            )}

            {status === "error" && (
              <>
                <div className="w-16 h-16 rounded-full bg-accent-100 dark:bg-accent-900/30 flex items-center justify-center mx-auto mb-4">
                  <XCircle className="w-8 h-8 text-accent" />
                </div>
                <h2 className="text-2xl font-display font-bold text-text-primary dark:text-white mb-2">Verification Failed</h2>
                <p className="text-text-muted dark:text-gray-400 mb-6">{message}</p>
                <Link href="/auth/login">
                  <Button variant="outline" className="w-full" size="lg">Back to Sign In</Button>
                </Link>
              </>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}

export default function VerifyEmailPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen gradient-bg flex items-center justify-center p-4">
        <div className="w-full max-w-md text-center py-8">
          <Loader2 className="w-16 h-16 text-primary animate-spin mx-auto mb-4" />
          <h2 className="text-xl font-bold text-text-primary dark:text-white">Loading Verification Page...</h2>
        </div>
      </div>
    }>
      <VerifyEmailContent />
    </Suspense>
  );
}
