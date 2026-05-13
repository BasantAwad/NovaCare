"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Heart, Mail, Lock, Shield, Stethoscope, Users, Bot, Eye, EyeOff, AlertCircle } from "lucide-react";
import { FcGoogle } from "react-icons/fc";
import { Button, Input, Card } from "@/components/ui";
import { useAuth, type UserRole } from "@/context/AuthContext";
import { loginWithGoogle } from "@/lib/auth-api";

type LoginRole = "rover" | "caregiver" | "doctor";

const ROLE_CONFIG: Record<LoginRole, { label: string; icon: React.ElementType; dashboard: string; description: string }> = {
  rover:     { label: "Rover (Patient)",       icon: Bot,         dashboard: "/rover",    description: "Access your rover assistant" },
  caregiver: { label: "Caregiver",             icon: Users,       dashboard: "/guardian", description: "Monitor your loved ones" },
  doctor:    { label: "Medical Professional",  icon: Stethoscope, dashboard: "/medical",  description: "Manage patient care" },
};

export default function LoginPage() {
  const router = useRouter();
  const { login } = useAuth();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(false);
  const [selectedRole, setSelectedRole] = useState<LoginRole>("rover");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    const result = await login(email, password);

    if (result.success && result.roles) {
      // Map role to dashboard path
      const roleMap: Record<string, string> = {
        rover: "/rover",
        caregiver: "/guardian",
        doctor: "/medical",
      };

      // Use the selected role if available, otherwise first role
      const targetRole = result.roles.includes(selectedRole) ? selectedRole : result.roles[0];
      const dashboardPath = roleMap[targetRole] || "/rover";
      router.push(dashboardPath);
    } else {
      setError(result.error || "Invalid email or password");
      setIsLoading(false);
    }
  };

  const handleGoogleLogin = async () => {
    setIsLoading(true);
    setError(null);
    
    // Simulate getting a mock Google payload
    const mockGoogleId = "g-12345";
    const mockEmail = "sarah@novacare.demo"; // Simulating existing user for demo

    const result = await loginWithGoogle(mockGoogleId, mockEmail);
    if (result.status === "success" && result.data?.roles) {
      const roleMap: Record<string, string> = {
        rover: "/rover",
        caregiver: "/guardian",
        doctor: "/medical",
      };
      const targetRole = result.data.roles.includes(selectedRole) ? selectedRole : result.data.roles[0];
      router.push(roleMap[targetRole] || "/rover");
    } else {
      setError(result.error || "Failed to log in with Google. You may need to Sign Up first.");
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex">
      {/* Left Side - Form */}
      <div className="w-full lg:w-1/2 flex items-center justify-center p-8 gradient-bg">
        <div className="w-full max-w-md animate-fade-in">
          {/* Logo */}
          <div className="flex items-center gap-3 mb-8">
            <div className="w-12 h-12 rounded-xl gradient-primary flex items-center justify-center">
              <Heart className="w-7 h-7 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-display font-bold text-text-primary dark:text-white">NovaCare</h1>
              <p className="text-sm text-text-muted dark:text-gray-400">Empowering Independence</p>
            </div>
          </div>

          {/* Welcome Text */}
          <div className="mb-8">
            <h2 className="text-3xl font-display font-bold text-text-primary dark:text-white mb-2">
              Welcome Back
            </h2>
            <p className="text-text-secondary dark:text-gray-300">
              Sign in to continue caring for your loved ones
            </p>
          </div>

          {/* Error Alert */}
          {error && (
            <div className="mb-6 flex items-center gap-3 p-4 bg-accent-50 dark:bg-accent-900/30 border border-accent-200 dark:border-accent-800 rounded-xl animate-slide-down">
              <AlertCircle className="w-5 h-5 text-accent shrink-0" />
              <p className="text-sm text-accent-700 dark:text-accent-300">{error}</p>
            </div>
          )}

          {/* Role Selector - 3 roles */}
          <div className="grid grid-cols-3 gap-2 p-1.5 bg-gray-100 dark:bg-gray-800 rounded-xl mb-6">
            {(Object.keys(ROLE_CONFIG) as LoginRole[]).map((role) => {
              const config = ROLE_CONFIG[role];
              const Icon = config.icon;
              const isActive = selectedRole === role;

              return (
                <button
                  key={role}
                  type="button"
                  onClick={() => setSelectedRole(role)}
                  className={`flex flex-col items-center gap-1 py-3 px-2 rounded-lg font-medium transition-all text-center ${
                    isActive
                      ? "bg-white dark:bg-gray-700 text-primary shadow-sm"
                      : "text-text-muted dark:text-gray-400 hover:text-text-secondary dark:hover:text-gray-300"
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  <span className="text-xs leading-tight">{config.label}</span>
                </button>
              );
            })}
          </div>

          {/* Login Form */}
          <form onSubmit={handleLogin} className="space-y-5">
            <Input
              label="Email Address"
              type="email"
              placeholder="you@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              leftIcon={<Mail className="w-5 h-5" />}
              required
            />

            <div className="relative">
              <Input
                label="Password"
                type={showPassword ? "text" : "password"}
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                leftIcon={<Lock className="w-5 h-5" />}
                required
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-4 top-[42px] text-text-muted dark:text-gray-400 hover:text-text-secondary dark:hover:text-gray-300 transition-colors"
              >
                {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
              </button>
            </div>

            <div className="flex items-center justify-between">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={rememberMe}
                  onChange={(e) => setRememberMe(e.target.checked)}
                  className="w-4 h-4 rounded border-gray-300 dark:border-gray-600 text-primary focus:ring-primary"
                />
                <span className="text-sm text-text-secondary dark:text-gray-300">Remember me</span>
              </label>
              <Link
                href="/auth/forgot-password"
                className="text-sm text-primary hover:text-primary-800 font-medium"
              >
                Forgot password?
              </Link>
            </div>

            <Button type="submit" className="w-full" size="lg" isLoading={isLoading}>
              Sign In
            </Button>

            <div className="relative my-4">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-gray-200 dark:border-gray-700"></div>
              </div>
              <div className="relative flex justify-center text-sm">
                <span className="px-2 bg-white dark:bg-gray-900 text-text-muted dark:text-gray-400">Or</span>
              </div>
            </div>

            <Button 
              type="button" 
              variant="outline" 
              className="w-full flex items-center justify-center gap-2" 
              size="lg"
              onClick={handleGoogleLogin}
              disabled={isLoading}
            >
              <FcGoogle className="w-5 h-5 text-xl" />
              Continue with Google
            </Button>
          </form>

          {/* Demo Accounts */}
          <div className="mt-6 p-4 bg-primary-50 dark:bg-primary-900/20 rounded-xl border border-primary-100 dark:border-primary-800">
            <p className="text-xs font-medium text-primary-700 dark:text-primary-300 mb-2">🔑 Demo Accounts (Password: NovaCare2026!)</p>
            <div className="space-y-1">
              <button
                type="button"
                onClick={() => { setEmail("sarah@novacare.demo"); setSelectedRole("rover"); }}
                className="text-xs text-primary-600 dark:text-primary-400 hover:underline block"
              >
                Rover: sarah@novacare.demo
              </button>
              <button
                type="button"
                onClick={() => { setEmail("john.guardian@novacare.demo"); setSelectedRole("caregiver"); }}
                className="text-xs text-primary-600 dark:text-primary-400 hover:underline block"
              >
                Caregiver: john.guardian@novacare.demo
              </button>
              <button
                type="button"
                onClick={() => { setEmail("dr.smith@novacare.demo"); setSelectedRole("doctor"); }}
                className="text-xs text-primary-600 dark:text-primary-400 hover:underline block"
              >
                Doctor: dr.smith@novacare.demo
              </button>
            </div>
          </div>

          {/* Sign Up Link */}
          <p className="mt-6 text-center text-text-secondary dark:text-gray-300">
            Don&apos;t have an account?{" "}
            <Link href="/auth/signup" className="text-primary font-semibold hover:text-primary-800">
              Sign Up
            </Link>
          </p>

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

      {/* Right Side - Hero Image */}
      <div className="hidden lg:flex w-1/2 bg-primary-700 items-center justify-center p-12">
        <div className="text-center text-white max-w-lg animate-slide-up">
          <div className="flex justify-center mb-10">
            <div className="w-32 h-32 bg-white/10 rounded-full flex items-center justify-center backdrop-blur-xl shadow-2xl border border-white/20 relative group">
              <div className="absolute inset-0 rounded-full bg-white/20 opacity-0 group-hover:opacity-100 transition-opacity duration-500 blur-xl"></div>
              <Shield className="w-16 h-16 text-white" strokeWidth={1.5} />
            </div>
          </div>
          <h2 className="text-4xl font-display font-bold mb-4">
            Care With Confidence
          </h2>
          <p className="text-primary-100 text-lg">
            NovaCare provides 24/7 intelligent assistance, ensuring your loved ones
            are safe, healthy, and connected.
          </p>
          <div className="mt-10 grid grid-cols-3 gap-6">
            <div className="p-4 bg-white/10 rounded-2xl backdrop-blur">
              <div className="text-3xl font-bold">24/7</div>
              <div className="text-sm text-primary-200">Monitoring</div>
            </div>
            <div className="p-4 bg-white/10 rounded-2xl backdrop-blur">
              <div className="text-3xl font-bold">99.9%</div>
              <div className="text-sm text-primary-200">Uptime</div>
            </div>
            <div className="p-4 bg-white/10 rounded-2xl backdrop-blur">
              <div className="text-3xl font-bold">1M+</div>
              <div className="text-sm text-primary-200">Alerts Sent</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
