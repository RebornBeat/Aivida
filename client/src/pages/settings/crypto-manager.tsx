import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  Card,
  Button,
  Alert,
  AlertTitle,
  AlertDescription,
} from "@/components/ui";
import { Shield, RefreshCw, Key } from "lucide-react";

interface CryptoKeys {
  mlkem_public_key: string;
  dilithium_public_key: string;
  last_rotation: string;
  key_health: "good" | "needs_rotation" | "critical";
}

interface CryptoManagerProps {
  onError: (error: string) => void;
  onSuccess: (message: string) => void;
}

export function CryptoManager({ onError, onSuccess }: CryptoManagerProps) {
  const [keys, setKeys] = useState<CryptoKeys | null>(null);
  const [isRotating, setIsRotating] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchKeys();
  }, []);

  const fetchKeys = async () => {
    try {
      setIsLoading(true);
      const cryptoKeys = (await invoke("get_crypto_keys")) as CryptoKeys;
      setKeys(cryptoKeys);
    } catch (err) {
      onError("Failed to fetch crypto keys");
      console.error("Crypto keys fetch error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const rotateKeys = async () => {
    try {
      setIsRotating(true);
      await invoke("rotate_crypto_keys");
      await fetchKeys(); // Refresh keys after rotation
      onSuccess("Keys rotated successfully");
    } catch (err) {
      onError("Failed to rotate keys");
      console.error("Key rotation error:", err);
    } finally {
      setIsRotating(false);
    }
  };

  const getKeyHealthColor = (health: CryptoKeys["key_health"]) => {
    switch (health) {
      case "good":
        return "text-green-500";
      case "needs_rotation":
        return "text-yellow-500";
      case "critical":
        return "text-red-500";
      default:
        return "text-gray-500";
    }
  };

  if (isLoading) {
    return (
      <div className="flex justify-center py-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Crypto Key Management</h2>
        <Button onClick={rotateKeys} disabled={isRotating}>
          <RefreshCw
            className={cn("h-4 w-4 mr-2", isRotating && "animate-spin")}
          />
          Rotate Keys
        </Button>
      </div>

      {keys?.key_health === "needs_rotation" && (
        <Alert variant="warning">
          <AlertTitle>Key Rotation Recommended</AlertTitle>
          <AlertDescription>
            It's recommended to rotate your keys for enhanced security.
          </AlertDescription>
        </Alert>
      )}

      {keys?.key_health === "critical" && (
        <Alert variant="destructive">
          <AlertTitle>Key Rotation Required</AlertTitle>
          <AlertDescription>
            Your keys require immediate rotation for security purposes.
          </AlertDescription>
        </Alert>
      )}

      <div className="grid gap-6 md:grid-cols-2">
        <Card className="p-4">
          <div className="flex items-start space-x-4">
            <div className="p-2 bg-primary/10 rounded-lg">
              <Key className="h-5 w-5 text-primary" />
            </div>
            <div>
              <h3 className="font-medium">ML-KEM Public Key</h3>
              <p className="text-sm text-gray-500 mt-1 font-mono break-all">
                {keys?.mlkem_public_key}
              </p>
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-start space-x-4">
            <div className="p-2 bg-primary/10 rounded-lg">
              <Shield className="h-5 w-5 text-primary" />
            </div>
            <div>
              <h3 className="font-medium">Dilithium Public Key</h3>
              <p className="text-sm text-gray-500 mt-1 font-mono break-all">
                {keys?.dilithium_public_key}
              </p>
            </div>
          </div>
        </Card>
      </div>

      <Card className="p-4">
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="font-medium">Key Health Status</h3>
            <span
              className={cn(
                "font-medium",
                getKeyHealthColor(keys?.key_health || "good"),
              )}
            >
              {keys?.key_health?.replace("_", " ").toUpperCase()}
            </span>
          </div>

          <div className="text-sm text-gray-500">
            <p>
              Last Key Rotation:{" "}
              {keys?.last_rotation
                ? new Date(keys.last_rotation).toLocaleString()
                : "Never"}
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
}
