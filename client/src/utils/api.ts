import { AuthContextType } from "@/contexts/auth-context";

export function setupApiInterceptor(auth: AuthContextType) {
  window.__TAURI__.invoke = new Proxy(window.__TAURI__.invoke, {
    apply: async function (target, thisArg, argumentsList) {
      const [command, payload = {}] = argumentsList;

      // Add token to all requests
      const authenticatedPayload = {
        ...payload,
        token: auth.token,
      };

      try {
        return await target.apply(thisArg, [command, authenticatedPayload]);
      } catch (error) {
        // Handle auth errors
        if (error.message === "Token expired") {
          await auth.refreshToken();
          authenticatedPayload.token = auth.token;
          return await target.apply(thisArg, [command, authenticatedPayload]);
        }
        throw error;
      }
    },
  });
}
