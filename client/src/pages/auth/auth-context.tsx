import { createContext, useContext, useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { setupApiInterceptor } from "@/utils/api";
import { handleAuthError, hashPassword } from "@/utils/auth";

interface User {
  id: string;
  username: string;
  role: "admin" | "manager" | "worker" | "client";
  status: "active" | "suspended" | "inactive";
}

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
}

interface AuthContextType extends AuthState {
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [authState, setAuthState] = useState<AuthState>({
    user: null,
    token: null,
    isAuthenticated: false,
  });

  useEffect(() => {
    const checkAuth = async () => {
      try {
        const token = localStorage.getItem("auth_token");
        if (token) {
          const user = await invoke<User>("verify_token", { token });
          setAuthState({
            user,
            token,
            isAuthenticated: true,
          });
        }
      } catch (err) {
        localStorage.removeItem("auth_token");
        localStorage.removeItem("user");
        setAuthState({
          user: null,
          token: null,
          isAuthenticated: false,
        });
      }
    };

    checkAuth();
  }, []);

  useEffect(() => {
    if (authState.isAuthenticated) {
      const interval = setInterval(refreshToken, 14 * 60 * 1000); // Refresh every 14 minutes
      return () => clearInterval(interval);
    }
  }, [authState.isAuthenticated]);

  const login = async (email: string, password: string) => {
    try {
      const hashedPassword = await hashPassword(password);

      const response = await invoke<{ user: User; token: string }>(
        "login_user",
        {
          email,
          password: hashedPassword,
        },
      );

      const { user, token } = response;

      localStorage.setItem("auth_token", token);
      localStorage.setItem("user", JSON.stringify(user));

      setAuthState({
        user,
        token,
        isAuthenticated: true,
      });
    } catch (err) {
      throw new Error(handleAuthError(err));
    }
  };

  const register = async (email: string, password: string) => {
    try {
      const hashedPassword = await hashPassword(password);

      await invoke("register_user", {
        email,
        password: hashedPassword,
      });

      await login(email, password);
    } catch (err) {
      throw new Error(handleAuthError(err));
    }
  };

  const logout = async () => {
    try {
      await invoke("logout_user");
      localStorage.removeItem("auth_token");
      localStorage.removeItem("user");
      setAuthState({
        user: null,
        token: null,
        isAuthenticated: false,
      });
    } catch (err) {
      throw new Error(handleAuthError(err));
    }
  };

  const refreshToken = async () => {
    try {
      const newToken = await invoke<string>("refresh_token", {
        token: authState.token,
      });
      setAuthState((prev) => ({
        ...prev,
        token: newToken,
      }));
    } catch (err) {
      logout();
    }
  };

  return (
    <AuthContext.Provider
      value={{
        ...authState,
        login,
        register,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
