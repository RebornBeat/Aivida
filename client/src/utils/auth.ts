export function handleAuthError(error: unknown): string {
  if (typeof error === "string") return error;

  const errorMap: Record<string, string> = {
    InvalidCredentials: "Invalid email or password",
    AccountLocked: "Account has been locked",
    EmailNotVerified: "Please verify your email",
    DuplicateEmail: "Email already in use",
    ServerError: "Server error occurred",
    NetworkError: "Network connection error",
  };

  if (error instanceof Error) {
    return errorMap[error.message] || error.message;
  }

  return "An unexpected error occurred";
}

// Client-side password hashing before sending to server
export async function hashPassword(password: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(password);
  const hashBuffer = await crypto.subtle.digest("SHA-256", data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
}
