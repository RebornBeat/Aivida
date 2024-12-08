import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  Card,
  Tabs,
  TabsList,
  TabsTrigger,
  TabsContent,
  Button,
  Alert,
  AlertTitle,
  AlertDescription,
  Switch,
} from "@/components/ui";
import { AccountSettings } from "./components/account-settings";
import { WorkerSettings } from "./components/worker-settings";
import { PerformanceSettings } from "./components/performance-settings";
import { SecuritySettings } from "./components/security-settings";
import { NotificationSettings } from "./components/notification-settings";
import { NetworkSettings } from "./components/network-settings";
import { CryptoManager } from "./components/crypto-manager";
import { SystemPreferences } from "./components/system-preferences";

export function Settings() {
  const [activeTab, setActiveTab] = useState("account");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const handleError = (error: unknown) => {
    setError(
      typeof error === "string" ? error : "An unexpected error occurred",
    );
    setTimeout(() => setError(null), 5000);
  };

  const handleSuccess = (message: string) => {
    setSuccess(message);
    setTimeout(() => setSuccess(null), 5000);
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">Settings</h1>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {success && (
        <Alert>
          <AlertTitle>Success</AlertTitle>
          <AlertDescription>{success}</AlertDescription>
        </Alert>
      )}

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid grid-cols-4 lg:grid-cols-8 w-full">
          <TabsTrigger value="account">Account</TabsTrigger>
          <TabsTrigger value="worker">Worker</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="security">Security</TabsTrigger>
          <TabsTrigger value="notifications">Notifications</TabsTrigger>
          <TabsTrigger value="network">Network</TabsTrigger>
          <TabsTrigger value="crypto">Crypto Keys</TabsTrigger>
          <TabsTrigger value="system">System</TabsTrigger>
        </TabsList>

        <TabsContent value="account">
          <AccountSettings onError={handleError} onSuccess={handleSuccess} />
        </TabsContent>

        <TabsContent value="worker">
          <WorkerSettings onError={handleError} onSuccess={handleSuccess} />
        </TabsContent>

        <TabsContent value="performance">
          <PerformanceSettings
            onError={handleError}
            onSuccess={handleSuccess}
          />
        </TabsContent>

        <TabsContent value="security">
          <SecuritySettings onError={handleError} onSuccess={handleSuccess} />
        </TabsContent>

        <TabsContent value="notifications">
          <NotificationSettings
            onError={handleError}
            onSuccess={handleSuccess}
          />
        </TabsContent>

        <TabsContent value="network">
          <NetworkSettings onError={handleError} onSuccess={handleSuccess} />
        </TabsContent>

        <TabsContent value="crypto">
          <CryptoManager onError={handleError} onSuccess={handleSuccess} />
        </TabsContent>

        <TabsContent value="system">
          <SystemPreferences onError={handleError} onSuccess={handleSuccess} />
        </TabsContent>
      </Tabs>
    </div>
  );
}
