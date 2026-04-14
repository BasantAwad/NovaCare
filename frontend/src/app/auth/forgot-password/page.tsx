"use client";

import { useState } from "react";
import Link from "next/link";
import { Heart, Mail, ArrowLeft, Shield, Lock, CheckCircle } from "lucide-react";
import { Button, Input, Card } from "@/components/ui";
import { forgotPassword } from "@/lib/auth-api";

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    await forgotPassword(email);
    setSubmitted(true);
    setIsLoading(false);
  };

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
          {!submitted ? (
            <>
              <h2 className="text-2xl font-display font-bold text-text-primary dark:text-white mb-2">
                Forgot Password?
              </h2>
              <p className="text-text-muted dark:text-gray-400 mb-8">
                Enter your email and we&apos;ll send you a link to reset your password.
              </p>

              <form onSubmit={handleSubmit} className="space-y-5">
                <Input
                  label="Email Address"
                  type="email"
                  placeholder="you@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  leftIcon={<Mail className="w-5 h-5" />}
                  required
                />

                <Button type="submit" className="w-full" size="lg" isLoading={isLoading}>
                  Send Reset Link
                </Button>
              </form>
            </>
          ) : (
            <div className="text-center py-4">
              <div className="w-16 h-16 rounded-full bg-success-100 dark:bg-success-900/30 flex items-center justify-center mx-auto mb-4">
                <CheckCircle className="w-8 h-8 text-success" />
              </div>
              <h2 className="text-2xl font-display font-bold text-text-primary dark:text-white mb-2">
                Check Your Email
              </h2>
              <p className="text-text-muted dark:text-gray-400 mb-6">
                If an account with <strong className="text-text-primary dark:text-white">{email}</strong> exists, we&apos;ve sent a password reset link.
              </p>
              <p className="text-sm text-text-muted dark:text-gray-400">
                Didn&apos;t receive it? Check your spam folder or{" "}
                <button onClick={() => setSubmitted(false)} className="text-primary font-medium hover:underline">
                  try again
                </button>
              </p>
            </div>
          )}

          <div className="mt-6 text-center">
            <Link href="/auth/login" className="text-sm text-primary font-medium hover:text-primary-800 inline-flex items-center gap-1">
              <ArrowLeft className="w-4 h-4" />
              Back to Sign In
            </Link>
          </div>
        </Card>

        {/* Trust Indicators */}
        <div className="mt-8 flex items-center justify-center gap-6 text-text-muted dark:text-gray-400">
          <div className="flex items-center gap-2">
            <Shield className="w-4 h-4" />
            <span className="text-xs">HIPAA Compliant</span>
          </div>
          <div className="flex items-center gap-2">
            <Lock className="w-4 h-4" />
            <span className="text-xs">256-bit Encryption</span>
          </div>
        </div>
      </div>
    </div>
  );
}
