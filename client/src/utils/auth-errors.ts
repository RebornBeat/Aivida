export function handleAuthError(error: unknown): string {
  if (typeof error === "string") return error;

  const errorMap: Record<string, string> = {
    InvalidCredentials: "Invalid email or password",
    AccountLocked: "Account has been locked. Please contact support",
    EmailNotVerified: "Please verify your email address",
    // Add more error mappings
  };

  return errorMap[error.code] || "An unexpected error occurred";
}
