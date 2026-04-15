"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  Heart, Mail, Lock, User, Phone, Check, ArrowLeft, ArrowRight,
  Shield, Stethoscope, Users, Bot, Calendar, Droplets, AlertTriangle,
  FileText, Building, Award, Eye, EyeOff, AlertCircle,
} from "lucide-react";
import { FcGoogle } from "react-icons/fc";
import { Button, Input, Card, ProgressBar } from "@/components/ui";
import { cn } from "@/lib/utils";
import {
  signupRover,
  signupCaregiver,
  signupDoctor,
  getReferenceData,
  type ReferenceData,
} from "@/lib/auth-api";

type AccountType = "rover" | "caregiver" | "doctor" | null;

export default function SignUpPage() {
  const router = useRouter();
  const [step, setStep] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showPassword, setShowPassword] = useState(false);
  const [referenceData, setReferenceData] = useState<ReferenceData | null>(null);

  // Shared fields
  const [accountType, setAccountType] = useState<AccountType>(null);
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [email, setEmail] = useState("");
  const [phone, setPhone] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [agreeTerms, setAgreeTerms] = useState(false);
  const [agreePrivacy, setAgreePrivacy] = useState(false);
  const [googleId, setGoogleId] = useState("");

  // Rover-specific
  const [dateOfBirth, setDateOfBirth] = useState("");
  const [gender, setGender] = useState("");
  const [bloodType, setBloodType] = useState("");
  const [selectedConditions, setSelectedConditions] = useState<string[]>([]);
  const [selectedAllergies, setSelectedAllergies] = useState<string[]>([]);
  const [primaryConditionName, setPrimaryConditionName] = useState("");
  const [customCondition, setCustomCondition] = useState("");
  const [customAllergies, setCustomAllergies] = useState(""); // Fallback

  const [needsCaregiver, setNeedsCaregiver] = useState(false);
  const [emergencyName, setEmergencyName] = useState("");
  const [emergencyPhone, setEmergencyPhone] = useState("");
  const [emergencyRelationship, setEmergencyRelationship] = useState("");

  // Caregiver-specific
  const [govIdType, setGovIdType] = useState("");
  const [govIdNumber, setGovIdNumber] = useState("");
  const [govIdExpiry, setGovIdExpiry] = useState("");
  const [roverEmail, setRoverEmail] = useState("");
  const [hasExistingRover, setHasExistingRover] = useState(false);

  // Doctor-specific
  const [specializationId, setSpecializationId] = useState("");
  const [licenseNumber, setLicenseNumber] = useState("");
  const [licenseCountry, setLicenseCountry] = useState("");
  const [licenseExpiry, setLicenseExpiry] = useState("");
  const [boardRegNumber, setBoardRegNumber] = useState("");
  const [clinicId, setClinicId] = useState("");

  // Load reference data on mount
  useEffect(() => {
    const load = async () => {
      const res = await getReferenceData();
      if (res.status === "success" && res.data) {
        setReferenceData(res.data);
      }
    };
    load();
  }, []);

  // Step counts per role
  const getMaxSteps = (): number => {
    switch (accountType) {
      case "rover": return 5;
      case "caregiver": return 5;
      case "doctor": return 5;
      default: return 1;
    }
  };

  const getPasswordStrength = (pw: string) => {
    let s = 0;
    if (pw.length >= 8) s++;
    if (/[a-z]/.test(pw)) s++;
    if (/[A-Z]/.test(pw)) s++;
    if (/[0-9]/.test(pw)) s++;
    if (/[^a-zA-Z0-9]/.test(pw)) s++;
    return s;
  };

  const passwordStrength = getPasswordStrength(password);
  const strengthLabels = ["Very Weak", "Weak", "Fair", "Strong", "Very Strong"];
  const strengthColors = ["danger", "danger", "warning", "success", "success"] as const;

  const canProceed = (): boolean => {
    switch (step) {
      case 1: return accountType !== null;
      case 2:
        if (googleId) return !!(firstName && lastName && email);
        return !!(firstName && lastName && email && password && password === confirmPassword && passwordStrength >= 3);
      case 3:
        if (accountType === "rover") return !!(dateOfBirth && gender);
        if (accountType === "caregiver") return !!(dateOfBirth && phone);
        if (accountType === "doctor") return !!(specializationId && licenseNumber);
        return false;
      case 4:
        if (accountType === "rover") return !!(emergencyName && emergencyPhone);
        if (accountType === "caregiver") return true; // Gov ID is optional for now
        if (accountType === "doctor") return true;
        return false;
      case 5:
        return agreeTerms && agreePrivacy;
      default: return false;
    }
  };

  const handleSubmit = async () => {
    setIsLoading(true);
    setError(null);

    try {
      let res;

      if (accountType === "rover") {
        res = await signupRover({
          email, password: password || undefined, google_id: googleId || undefined, first_name: firstName, last_name: lastName,
          date_of_birth: dateOfBirth, gender, phone_number: phone,
          blood_type: bloodType, needs_caregiver: needsCaregiver,
          health_conditions: selectedConditions.map((id) => ({ condition_id: id, severity: "mild" })),
          allergies: selectedAllergies.length > 0
            ? selectedAllergies.map((id) => ({ allergy_id: id, severity: "mild" }))
            : customAllergies ? customAllergies.split(",").map(a => ({ allergy_name: a.trim() })) : undefined,
          primary_condition_name: primaryConditionName === "other" ? customCondition : primaryConditionName,
          emergency_contact: emergencyName ? {
            name: emergencyName, phone_number: emergencyPhone,
            relationship: emergencyRelationship,
          } : undefined,
        });
      } else if (accountType === "caregiver") {
        res = await signupCaregiver({
          email, password: password || undefined, google_id: googleId || undefined, first_name: firstName, last_name: lastName,
          date_of_birth: dateOfBirth, phone_number: phone,
          government_id_type_id: govIdType || undefined,
          government_id_number: govIdNumber || undefined,
          id_expiry_date: govIdExpiry || undefined,
          rover_email: hasExistingRover ? roverEmail : undefined,
        });
      } else if (accountType === "doctor") {
        res = await signupDoctor({
          email, password: password || undefined, google_id: googleId || undefined, first_name: firstName, last_name: lastName,
          specialization_id: specializationId,
          medical_license_num: licenseNumber,
          license_country_id: licenseCountry || undefined,
          license_expiry_date: licenseExpiry || undefined,
          board_reg_number: boardRegNumber || undefined,
          clinic_organization_id: clinicId || undefined,
        });
      }

      if (res?.status === "success") {
        const dashboardMap: Record<string, string> = {
          rover: "/rover", caregiver: "/guardian", doctor: "/medical",
        };
        router.push(dashboardMap[accountType!] || "/");
      } else {
        setError(res?.error || "Signup failed. Please try again.");
        setIsLoading(false);
      }
    } catch {
      setError("We couldn't connect to the authentication service. Please ensure the backend server is actively running.");
      setIsLoading(false);
    }
  };

  const getStepLabel = (): string => {
    if (step === 1) return "Choose Account Type";
    if (step === 2) return "Personal Information";

    if (accountType === "rover") {
      if (step === 3) return "Health Information";
      if (step === 4) return "Emergency Contact";
      if (step === 5) return "Review & Confirm";
    }
    if (accountType === "caregiver") {
      if (step === 3) return "Basic Details";
      if (step === 4) return "Government ID";
      if (step === 5) return "Review & Confirm";
    }
    if (accountType === "doctor") {
      if (step === 3) return "Professional Info";
      if (step === 4) return "Clinic & License";
      if (step === 5) return "Review & Confirm";
    }
    return "";
  };

  return (
    <div className="min-h-screen gradient-bg py-8 px-4">
      <div className="max-w-2xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <Link href="/" className="flex items-center gap-2">
            <div className="w-10 h-10 rounded-xl gradient-primary flex items-center justify-center">
              <Heart className="w-5 h-5 text-white" />
            </div>
            <span className="text-xl font-display font-bold text-text-primary dark:text-white">NovaCare</span>
          </Link>
          <Link href="/auth/login" className="text-sm text-text-muted dark:text-gray-400 hover:text-primary transition-colors">
            Already have an account? <span className="font-semibold text-primary">Sign In</span>
          </Link>
        </div>

        {/* Progress Bar */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-text-secondary dark:text-gray-300">Step {step} of {getMaxSteps()}</span>
            <span className="text-sm text-text-muted dark:text-gray-400">{getStepLabel()}</span>
          </div>
          <ProgressBar value={step} max={getMaxSteps()} variant="primary" size="md" />
        </div>

        {/* Error Alert */}
        {error && (
          <div className="mb-6 flex items-center gap-3 p-4 bg-accent-50 dark:bg-accent-900/30 border border-accent-200 dark:border-accent-800 rounded-xl animate-slide-down">
            <AlertCircle className="w-5 h-5 text-accent shrink-0" />
            <p className="text-sm text-accent-700 dark:text-accent-300">{error}</p>
          </div>
        )}

        {/* Form Card */}
        <Card variant="elevated" padding="lg" className="animate-fade-in">
          {/* ============= STEP 1: Account Type ============= */}
          {step === 1 && (
            <div>
              <h2 className="text-2xl font-display font-bold text-text-primary dark:text-white mb-2">
                Choose Your Account Type
              </h2>
              <p className="text-text-muted dark:text-gray-400 mb-8">
                Select the role that best describes how you&apos;ll be using NovaCare
              </p>

              <div className="grid gap-4">
                {/* Rover */}
                <button
                  onClick={() => setAccountType("rover")}
                  className={cn(
                    "p-6 rounded-2xl border-2 text-left transition-all",
                    accountType === "rover"
                      ? "border-primary bg-primary-50 dark:bg-primary-900/30"
                      : "border-gray-200 dark:border-gray-700 hover:border-primary-300 hover:bg-gray-50 dark:hover:bg-gray-700"
                  )}
                >
                  <div className="flex items-start gap-4">
                    <div className="w-14 h-14 rounded-xl bg-success-100 dark:bg-success-900/50 flex items-center justify-center shrink-0">
                      <Bot className="w-7 h-7 text-success" />
                    </div>
                    <div className="flex-1">
                      <h3 className="text-lg font-semibold text-text-primary dark:text-white mb-1">Rover (Patient)</h3>
                      <p className="text-sm text-text-muted dark:text-gray-400">
                        Individual using the NovaCare rover for daily assistance, health monitoring, and AI companion interaction
                      </p>
                    </div>
                    {accountType === "rover" && <Check className="w-6 h-6 text-primary shrink-0" />}
                  </div>
                </button>

                {/* Caregiver */}
                <button
                  onClick={() => setAccountType("caregiver")}
                  className={cn(
                    "p-6 rounded-2xl border-2 text-left transition-all",
                    accountType === "caregiver"
                      ? "border-primary bg-primary-50 dark:bg-primary-900/30"
                      : "border-gray-200 dark:border-gray-700 hover:border-primary-300 hover:bg-gray-50 dark:hover:bg-gray-700"
                  )}
                >
                  <div className="flex items-start gap-4">
                    <div className="w-14 h-14 rounded-xl bg-secondary-100 dark:bg-secondary-900/50 flex items-center justify-center shrink-0">
                      <Users className="w-7 h-7 text-secondary" />
                    </div>
                    <div className="flex-1">
                      <h3 className="text-lg font-semibold text-text-primary dark:text-white mb-1">Caregiver</h3>
                      <p className="text-sm text-text-muted dark:text-gray-400">
                        Family member or professional caregiver monitoring and supporting a patient remotely
                      </p>
                    </div>
                    {accountType === "caregiver" && <Check className="w-6 h-6 text-primary shrink-0" />}
                  </div>
                </button>

                {/* Doctor */}
                <button
                  onClick={() => setAccountType("doctor")}
                  className={cn(
                    "p-6 rounded-2xl border-2 text-left transition-all",
                    accountType === "doctor"
                      ? "border-primary bg-primary-50 dark:bg-primary-900/30"
                      : "border-gray-200 dark:border-gray-700 hover:border-primary-300 hover:bg-gray-50 dark:hover:bg-gray-700"
                  )}
                >
                  <div className="flex items-start gap-4">
                    <div className="w-14 h-14 rounded-xl bg-primary-100 dark:bg-primary-900/50 flex items-center justify-center shrink-0">
                      <Stethoscope className="w-7 h-7 text-primary" />
                    </div>
                    <div className="flex-1">
                      <h3 className="text-lg font-semibold text-text-primary dark:text-white mb-1">Medical Professional</h3>
                      <p className="text-sm text-text-muted dark:text-gray-400">
                        Doctor or healthcare provider managing patient care, prescriptions, and medical records
                      </p>
                    </div>
                    {accountType === "doctor" && <Check className="w-6 h-6 text-primary shrink-0" />}
                  </div>
                </button>
              </div>
            </div>
          )}

          {/* ============= STEP 2: Personal Info (all roles) ============= */}
          {step === 2 && (
            <div>
              <h2 className="text-2xl font-display font-bold text-text-primary dark:text-white mb-2">
                Personal Information
              </h2>
              <p className="text-text-muted dark:text-gray-400 mb-8">
                Create your secure NovaCare account
              </p>

              <div className="space-y-5">
                <Button 
                  type="button" 
                  variant="outline" 
                  className="w-full flex items-center justify-center gap-2" 
                  size="lg"
                  onClick={() => {
                    setGoogleId("g-12345");
                    setFirstName("Google");
                    setLastName("User");
                    setEmail("google.user@gmail.com");
                    setPassword("");
                    setConfirmPassword("");
                  }}
                >
                  <FcGoogle className="w-5 h-5 text-xl" />
                  Continue with Google
                </Button>
                
                <div className="relative my-6">
                  <div className="absolute inset-0 flex items-center">
                    <div className="w-full border-t border-gray-200 dark:border-gray-700"></div>
                  </div>
                  <div className="relative flex justify-center text-sm">
                    <span className="px-2 bg-white dark:bg-gray-900 text-text-muted dark:text-gray-400">Or enter details</span>
                  </div>
                </div>

                <div className="grid md:grid-cols-2 gap-4">
                  <Input label="First Name" placeholder="John" value={firstName} onChange={(e) => setFirstName(e.target.value)} leftIcon={<User className="w-5 h-5" />} />
                  <Input label="Last Name" placeholder="Doe" value={lastName} onChange={(e) => setLastName(e.target.value)} />
                </div>

                <Input label="Email Address" type="email" placeholder="you@example.com" value={email} onChange={(e) => setEmail(e.target.value)} leftIcon={<Mail className="w-5 h-5" />} />

                <Input label="Phone Number (Optional)" type="tel" placeholder="+20-100-000-0000" value={phone} onChange={(e) => setPhone(e.target.value)} leftIcon={<Phone className="w-5 h-5" />} helperText="For emergency alerts and verification" />

                {!googleId && (
                  <>
                    <div className="relative">
                      <Input label="Password" type={showPassword ? "text" : "password"} placeholder="Create a strong password" value={password} onChange={(e) => setPassword(e.target.value)} leftIcon={<Lock className="w-5 h-5" />} />
                      <button type="button" onClick={() => setShowPassword(!showPassword)} className="absolute right-4 top-[42px] text-text-muted dark:text-gray-400 hover:text-text-secondary transition-colors">
                        {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                      </button>
                    </div>

                    {password && (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-text-muted dark:text-gray-400">Password strength</span>
                          <span className={cn("text-sm font-medium", passwordStrength <= 1 ? "text-accent" : passwordStrength <= 2 ? "text-secondary" : "text-success")}>
                            {strengthLabels[passwordStrength - 1] || "Too Weak"}
                          </span>
                        </div>
                        <ProgressBar value={passwordStrength} max={5} variant={strengthColors[passwordStrength - 1] || "danger"} size="sm" />
                      </div>
                    )}

                    <Input label="Confirm Password" type="password" placeholder="Confirm your password" value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} leftIcon={<Lock className="w-5 h-5" />}
                      error={confirmPassword && password !== confirmPassword ? "Passwords do not match" : undefined}
                      success={!!(confirmPassword && password === confirmPassword)}
                    />
                  </>
                )}
              </div>
            </div>
          )}

          {/* ============= STEP 3: Role-specific info ============= */}
          {step === 3 && accountType === "rover" && (
            <div>
              <h2 className="text-2xl font-display font-bold text-text-primary dark:text-white mb-2">Health Information</h2>
              <p className="text-text-muted dark:text-gray-400 mb-8">Help us personalize your care experience</p>

              <div className="space-y-5">
                <div className="grid md:grid-cols-2 gap-4">
                  <Input label="Date of Birth" type="date" value={dateOfBirth} onChange={(e) => setDateOfBirth(e.target.value)} leftIcon={<Calendar className="w-5 h-5" />} />

                  <div>
                    <label className="block text-sm font-medium text-text-secondary dark:text-gray-300 mb-2">Gender</label>
                    <select value={gender} onChange={(e) => setGender(e.target.value)} className="input-field">
                      <option value="">Select gender</option>
                      <option value="male">Male</option>
                      <option value="female">Female</option>
                      <option value="other">Other</option>
                      <option value="prefer_not_to_say">Prefer not to say</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-text-secondary dark:text-gray-300 mb-2">Blood Type</label>
                  <select value={bloodType} onChange={(e) => setBloodType(e.target.value)} className="input-field">
                    <option value="">Select (optional)</option>
                    {["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"].map((bt) => (
                      <option key={bt} value={bt}>{bt}</option>
                    ))}
                  </select>
                </div>

                {/* Primary Disability / Condition */}
                <div>
                  <label className="block text-sm font-medium text-text-secondary dark:text-gray-300 mb-2">Primary Condition / Disability</label>
                  {referenceData?.health_conditions && referenceData.health_conditions.length > 0 ? (
                    <div className="space-y-3">
                      <select value={primaryConditionName} onChange={(e) => {
                          setPrimaryConditionName(e.target.value);
                          if (e.target.value !== "other") setCustomCondition("");
                        }} className="input-field border-primary focus:ring-primary/20">
                        <option value="">Select primary condition</option>
                        {referenceData.health_conditions.map((hc) => (
                          <option key={hc.id} value={hc.name}>{hc.name}</option>
                        ))}
                        <option value="other">Other (Please specify)</option>
                      </select>
                      {primaryConditionName === "other" && (
                        <Input placeholder="Enter your condition" value={customCondition} onChange={(e) => setCustomCondition(e.target.value)} />
                      )}
                    </div>
                  ) : (
                    <Input placeholder="E.g., Visually Impaired (Text fallback)" value={customCondition} onChange={(e) => {
                      setPrimaryConditionName("other");
                      setCustomCondition(e.target.value);
                    }} helperText="Server fetch failed: Please enter manually." />
                  )}
                </div>

                {/* Other Health Conditions */}
                <div>
                  <label className="block text-sm font-medium text-text-secondary dark:text-gray-300 mb-2">Additional Health Conditions (optional)</label>
                  {referenceData?.health_conditions && referenceData.health_conditions.length > 0 ? (
                    <div className="grid grid-cols-2 gap-2 max-h-[160px] overflow-y-auto p-2 border border-gray-200 dark:border-gray-700 rounded-xl">
                      {referenceData.health_conditions.map((hc) => (
                        <label key={hc.id} className={cn(
                          "flex items-center gap-2 p-2 rounded-xl border cursor-pointer transition-all",
                          selectedConditions.includes(hc.id)
                            ? "border-primary bg-primary-50 dark:bg-primary-900/30"
                            : "border-transparent hover:bg-gray-50 dark:hover:bg-gray-800"
                        )}>
                          <input type="checkbox" checked={selectedConditions.includes(hc.id)} onChange={(e) => {
                            if (e.target.checked) setSelectedConditions([...selectedConditions, hc.id]);
                            else setSelectedConditions(selectedConditions.filter((id) => id !== hc.id));
                          }} className="w-4 h-4 rounded border-gray-300 text-primary focus:ring-primary" />
                          <span className="text-sm truncate text-text-primary dark:text-white" title={hc.name}>{hc.name}</span>
                        </label>
                      ))}
                    </div>
                  ) : (
                    <p className="text-xs text-text-muted">Options unavailable right now.</p>
                  )}
                </div>

                {/* Allergies */}
                <div>
                  <label className="block text-sm font-medium text-text-secondary dark:text-gray-300 mb-2">Allergies (optional)</label>
                  {referenceData?.allergies && referenceData.allergies.length > 0 ? (
                    <div className="grid grid-cols-2 gap-2 max-h-[160px] overflow-y-auto p-2 border border-gray-200 dark:border-gray-700 rounded-xl">
                      {referenceData.allergies.map((alg) => (
                        <label key={alg.id} className={cn(
                          "flex items-center gap-2 p-2 rounded-xl border cursor-pointer transition-all",
                          selectedAllergies.includes(alg.id)
                            ? "border-accent bg-accent-50 dark:bg-accent-900/30"
                            : "border-transparent hover:bg-gray-50 dark:hover:bg-gray-800"
                        )}>
                          <input type="checkbox" checked={selectedAllergies.includes(alg.id)} onChange={(e) => {
                            if (e.target.checked) setSelectedAllergies([...selectedAllergies, alg.id]);
                            else setSelectedAllergies(selectedAllergies.filter((id) => id !== alg.id));
                          }} className="w-4 h-4 rounded border-gray-300 text-accent focus:ring-accent" />
                          <span className="text-sm truncate text-text-primary dark:text-white" title={alg.name}>{alg.name}</span>
                        </label>
                      ))}
                    </div>
                  ) : (
                     <Input placeholder="E.g., Peanuts, Latex" value={customAllergies} onChange={(e) => setCustomAllergies(e.target.value)} helperText="Server fetch failed: Enter comma separated." />
                  )}
                </div>

                {/* Caregiver question */}
                <div className="p-4 bg-primary-50 dark:bg-primary-900/20 rounded-xl border border-primary-100 dark:border-primary-800">
                  <label className="flex items-center gap-3 cursor-pointer">
                    <input type="checkbox" checked={needsCaregiver} onChange={(e) => setNeedsCaregiver(e.target.checked)} className="w-5 h-5 rounded border-gray-300 text-primary focus:ring-primary" />
                    <div>
                      <span className="text-sm font-medium text-text-primary dark:text-white">Do you need a caregiver?</span>
                      <p className="text-xs text-text-muted dark:text-gray-400 mt-1">A caregiver will be able to monitor your health data and receive alerts</p>
                    </div>
                  </label>
                </div>
              </div>
            </div>
          )}

          {step === 3 && accountType === "caregiver" && (
            <div>
              <h2 className="text-2xl font-display font-bold text-text-primary dark:text-white mb-2">Basic Details</h2>
              <p className="text-text-muted dark:text-gray-400 mb-8">Tell us a bit more about yourself</p>
              <div className="space-y-5">
                <Input label="Date of Birth" type="date" value={dateOfBirth} onChange={(e) => setDateOfBirth(e.target.value)} leftIcon={<Calendar className="w-5 h-5" />} />
                <Input label="Phone Number" type="tel" placeholder="+20-100-000-0000" value={phone} onChange={(e) => setPhone(e.target.value)} leftIcon={<Phone className="w-5 h-5" />} />
              </div>
            </div>
          )}

          {step === 3 && accountType === "doctor" && (
            <div>
              <h2 className="text-2xl font-display font-bold text-text-primary dark:text-white mb-2">Professional Information</h2>
              <p className="text-text-muted dark:text-gray-400 mb-8">Your medical credentials</p>
              <div className="space-y-5">
                <div>
                  <label className="block text-sm font-medium text-text-secondary dark:text-gray-300 mb-2">Specialization</label>
                  <select value={specializationId} onChange={(e) => setSpecializationId(e.target.value)} className="input-field">
                    <option value="">Select specialization</option>
                    {(referenceData?.specializations || []).map((s) => (
                      <option key={s.id} value={s.id}>{s.name}</option>
                    ))}
                  </select>
                </div>
                <Input label="Medical License Number" placeholder="MED-XX-XXXX-XXXXX" value={licenseNumber} onChange={(e) => setLicenseNumber(e.target.value)} leftIcon={<FileText className="w-5 h-5" />} />
                <div>
                  <label className="block text-sm font-medium text-text-secondary dark:text-gray-300 mb-2">License Country</label>
                  <select value={licenseCountry} onChange={(e) => setLicenseCountry(e.target.value)} className="input-field">
                    <option value="">Select country</option>
                    {(referenceData?.countries || []).map((c) => (
                      <option key={c.id} value={c.id}>{c.name}</option>
                    ))}
                  </select>
                </div>
                <Input label="Board Registration Number (Optional)" placeholder="BRD-XXXXX" value={boardRegNumber} onChange={(e) => setBoardRegNumber(e.target.value)} leftIcon={<Award className="w-5 h-5" />} />
              </div>
            </div>
          )}

          {/* ============= STEP 4: Role-specific extended info ============= */}
          {step === 4 && accountType === "rover" && (
            <div>
              <h2 className="text-2xl font-display font-bold text-text-primary dark:text-white mb-2">Emergency Contact</h2>
              <p className="text-text-muted dark:text-gray-400 mb-8">Someone we can reach in case of emergency</p>
              <div className="space-y-5">
                <Input label="Contact Name" placeholder="Full name" value={emergencyName} onChange={(e) => setEmergencyName(e.target.value)} leftIcon={<User className="w-5 h-5" />} />
                <Input label="Contact Phone" type="tel" placeholder="+20-100-000-0000" value={emergencyPhone} onChange={(e) => setEmergencyPhone(e.target.value)} leftIcon={<Phone className="w-5 h-5" />} />
                <Input label="Relationship" placeholder="e.g. Parent, Spouse, Sibling" value={emergencyRelationship} onChange={(e) => setEmergencyRelationship(e.target.value)} leftIcon={<Users className="w-5 h-5" />} />
              </div>
            </div>
          )}

          {step === 4 && accountType === "caregiver" && (
            <div>
              <h2 className="text-2xl font-display font-bold text-text-primary dark:text-white mb-2">Government ID (Optional)</h2>
              <p className="text-text-muted dark:text-gray-400 mb-8">For identity verification — you can complete this later</p>
              <div className="space-y-5">
                <div>
                  <label className="block text-sm font-medium text-text-secondary dark:text-gray-300 mb-2">ID Type</label>
                  <select value={govIdType} onChange={(e) => setGovIdType(e.target.value)} className="input-field">
                    <option value="">Select ID type (optional)</option>
                    {(referenceData?.id_types || []).map((t) => (
                      <option key={t.id} value={t.id}>{t.name}</option>
                    ))}
                  </select>
                </div>
                {govIdType && (
                  <>
                    <Input label="ID Number" placeholder="Enter ID number" value={govIdNumber} onChange={(e) => setGovIdNumber(e.target.value)} leftIcon={<FileText className="w-5 h-5" />} />
                    <Input label="ID Expiry Date" type="date" value={govIdExpiry} onChange={(e) => setGovIdExpiry(e.target.value)} leftIcon={<Calendar className="w-5 h-5" />} />
                  </>
                )}

                <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
                  <label className="flex items-center gap-3 cursor-pointer mb-4">
                    <input type="checkbox" checked={hasExistingRover} onChange={(e) => setHasExistingRover(e.target.checked)} className="w-5 h-5 rounded border-gray-300 text-primary focus:ring-primary" />
                    <div>
                      <span className="text-sm font-medium text-text-primary dark:text-white">I&apos;m already caring for someone on NovaCare</span>
                      <p className="text-xs text-text-muted dark:text-gray-400">Enter their email to request access</p>
                    </div>
                  </label>
                  {hasExistingRover && (
                    <Input label="Rover's Email" type="email" placeholder="patient@example.com" value={roverEmail} onChange={(e) => setRoverEmail(e.target.value)} leftIcon={<Mail className="w-5 h-5" />} helperText="An access request will be sent to them for approval" />
                  )}
                </div>
              </div>
            </div>
          )}

          {step === 4 && accountType === "doctor" && (
            <div>
              <h2 className="text-2xl font-display font-bold text-text-primary dark:text-white mb-2">Clinic & License Details</h2>
              <p className="text-text-muted dark:text-gray-400 mb-8">Additional professional details</p>
              <div className="space-y-5">
                <Input label="License Expiry Date" type="date" value={licenseExpiry} onChange={(e) => setLicenseExpiry(e.target.value)} leftIcon={<Calendar className="w-5 h-5" />} />
                <div>
                  <label className="block text-sm font-medium text-text-secondary dark:text-gray-300 mb-2">Clinic / Organization (Optional)</label>
                  <select value={clinicId} onChange={(e) => setClinicId(e.target.value)} className="input-field">
                    <option value="">Select or skip</option>
                    {(referenceData?.clinic_organizations || []).map((c) => (
                      <option key={c.id} value={c.id}>{c.name}</option>
                    ))}
                  </select>
                </div>
              </div>
            </div>
          )}

          {/* ============= STEP 5: Review & Confirm (all roles) ============= */}
          {step === 5 && (
            <div>
              <h2 className="text-2xl font-display font-bold text-text-primary dark:text-white mb-2">Review & Confirm</h2>
              <p className="text-text-muted dark:text-gray-400 mb-6">Please review your information before creating your account</p>

              {/* Review summary */}
              <div className="space-y-4 mb-6">
                <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-xl space-y-2">
                  <h4 className="text-sm font-semibold text-text-secondary dark:text-gray-300">Account</h4>
                  <p className="text-sm text-text-primary dark:text-white">{firstName} {lastName} · {email}</p>
                  <p className="text-xs text-text-muted dark:text-gray-400 capitalize">Role: {accountType}</p>
                </div>

                {accountType === "rover" && (
                  <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-xl space-y-2">
                    <h4 className="text-sm font-semibold text-text-secondary dark:text-gray-300">Health</h4>
                    <p className="text-sm text-text-primary dark:text-white">DOB: {dateOfBirth} · Gender: {gender} · Blood: {bloodType || "N/A"}</p>
                    <p className="text-sm text-text-primary dark:text-white">Conditions: {selectedConditions.length || "None"} · Allergies: {selectedAllergies.length || "None"}</p>
                    <p className="text-sm text-text-primary dark:text-white">Needs caregiver: {needsCaregiver ? "Yes" : "No"}</p>
                  </div>
                )}

                {accountType === "caregiver" && (
                  <div className="p-4 bg-primary-50 dark:bg-primary-900/20 rounded-xl space-y-2">
                    <h4 className="text-sm font-semibold text-primary-700 dark:text-primary-300">⏳ Verification Status</h4>
                    <p className="text-sm text-text-primary dark:text-white">Your account will start as <strong>Pending</strong> and will be verified by an admin within 24-48 hours.</p>
                  </div>
                )}

                {accountType === "doctor" && (
                  <div className="p-4 bg-primary-50 dark:bg-primary-900/20 rounded-xl space-y-2">
                    <h4 className="text-sm font-semibold text-primary-700 dark:text-primary-300">⏳ License Verification</h4>
                    <p className="text-sm text-text-primary dark:text-white">Your medical license will be verified. License: {licenseNumber}</p>
                  </div>
                )}
              </div>

              {/* Terms */}
              <div className="space-y-3">
                <label className="flex items-start gap-3 cursor-pointer">
                  <input type="checkbox" checked={agreeTerms} onChange={(e) => setAgreeTerms(e.target.checked)} className="mt-1 w-4 h-4 rounded border-gray-300 dark:border-gray-600 text-primary focus:ring-primary" />
                  <span className="text-sm text-text-secondary dark:text-gray-300">
                    I agree to NovaCare&apos;s <a href="#" className="text-primary hover:underline">Terms of Service</a>
                  </span>
                </label>
                <label className="flex items-start gap-3 cursor-pointer">
                  <input type="checkbox" checked={agreePrivacy} onChange={(e) => setAgreePrivacy(e.target.checked)} className="mt-1 w-4 h-4 rounded border-gray-300 dark:border-gray-600 text-primary focus:ring-primary" />
                  <span className="text-sm text-text-secondary dark:text-gray-300">
                    I have read and accept the <a href="#" className="text-primary hover:underline">Privacy Policy</a> and consent to data processing
                  </span>
                </label>
              </div>
            </div>
          )}

          {/* Navigation Buttons */}
          <div className="flex items-center justify-between mt-8 pt-6 border-t border-gray-100 dark:border-gray-700">
            {step > 1 ? (
              <Button variant="ghost" onClick={() => { setStep(step - 1); setError(null); }} leftIcon={<ArrowLeft className="w-5 h-5" />}>
                Back
              </Button>
            ) : (
              <div />
            )}
            {step < getMaxSteps() ? (
              <Button onClick={() => { setStep(step + 1); setError(null); }} disabled={!canProceed()} rightIcon={<ArrowRight className="w-5 h-5" />}>
                Continue
              </Button>
            ) : (
              <Button onClick={handleSubmit} disabled={!canProceed()} isLoading={isLoading}>
                Create Account
              </Button>
            )}
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
