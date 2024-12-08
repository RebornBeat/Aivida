import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  Card,
  Button,
  Switch,
  Alert,
  AlertTitle,
  AlertDescription,
  Form,
  FormField,
  FormItem,
  FormLabel,
  FormControl,
  FormDescription,
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
  Progress,
  Separator,
} from "@/components/ui";
import {
  Shield,
  Lock,
  Globe,
  AlertTriangle,
  FileKey,
  Activity,
  UserCheck,
  Radio,
} from "lucide-react";
import { z } from "zod";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";

interface SecuritySettingsProps {
  onError: (error: string) => void;
  onSuccess: (message: string) => void;
}

interface SecurityStatus {
  level: "basic" | "standard" | "maximum";
  lastAssessment: string;
  locationStatus: {
    country: string;
    region: string;
    isRestricted: boolean;
    requiredLevel: "basic" | "standard" | "maximum";
  };
  threatDetection: {
    isEnabled: boolean;
    lastScan: string;
    threats: number;
  };
  connectionSecurity: {
    encryptionEnabled: boolean;
    p2pEnabled: boolean;
    activeConnections: number;
  };
  keyStatus: {
    lastRotation: string;
    health: "good" | "needs_rotation" | "critical";
  };
}

const securitySchema = z.object({
  threatDetection: z.boolean(),
  automaticKeyRotation: z.boolean(),
  connectionMode: z.enum(["standard", "p2p"]),
  dataRetention: z.enum(["24h", "7d", "30d", "90d"]),
  loggingLevel: z.enum(["basic", "detailed", "debug"]),
});

export function SecuritySettings({
  onError,
  onSuccess,
}: SecuritySettingsProps) {
  const [securityStatus, setSecurityStatus] = useState<SecurityStatus | null>(
    null,
  );
  const [isScanning, setIsScanning] = useState(false);
  const [scanProgress, setScanProgress] = useState(0);
  const [isLoading, setIsLoading] = useState(true);

  const form = useForm<z.infer<typeof securitySchema>>({
    resolver: zodResolver(securitySchema),
    defaultValues: {
      threatDetection: true,
      automaticKeyRotation: true,
      connectionMode: "standard",
      dataRetention: "7d",
      loggingLevel: "detailed",
    },
  });

  useEffect(() => {
    fetchSecurityStatus();
  }, []);

  const fetchSecurityStatus = async () => {
    try {
      setIsLoading(true);
      const [status, settings] = await Promise.all([
        invoke("get_security_status"),
        invoke("get_security_settings"),
      ]);

      setSecurityStatus(status as SecurityStatus);
      form.reset(settings as z.infer<typeof securitySchema>);
    } catch (err) {
      onError("Failed to fetch security status");
      console.error("Security status error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const runSecurityScan = async () => {
    try {
      setIsScanning(true);
      setScanProgress(0);

      // Simulate scan progress
      for (let i = 0; i <= 100; i += 10) {
        setScanProgress(i);
        await new Promise((resolve) => setTimeout(resolve, 500));
      }

      await invoke("run_security_scan");
      await fetchSecurityStatus();
      onSuccess("Security scan completed successfully");
    } catch (err) {
      onError("Failed to complete security scan");
    } finally {
      setIsScanning(false);
      setScanProgress(0);
    }
  };

  const onSubmit = async (data: z.infer<typeof securitySchema>) => {
    try {
      await invoke("update_security_settings", { settings: data });
      onSuccess("Security settings updated successfully");
      await fetchSecurityStatus();
    } catch (err) {
      onError("Failed to update security settings");
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
      {/* Security Status Overview */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-lg font-semibold">Security Status</h3>
            <p className="text-sm text-gray-500">
              Last assessment: {securityStatus?.lastAssessment}
            </p>
          </div>
          <Button onClick={runSecurityScan} disabled={isScanning}>
            {isScanning ? (
              <>
                <Activity className="mr-2 h-4 w-4 animate-spin" />
                Scanning...
              </>
            ) : (
              <>
                <Shield className="mr-2 h-4 w-4" />
                Run Security Scan
              </>
            )}
          </Button>
        </div>

        {isScanning && (
          <div className="mb-6">
            <p className="text-sm text-gray-500 mb-2">
              Scanning system security...
            </p>
            <Progress value={scanProgress} />
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Location Status */}
          <Card className="p-4">
            <div className="flex items-center space-x-3">
              <Globe
                className={`h-5 w-5 ${
                  securityStatus?.locationStatus.isRestricted
                    ? "text-red-500"
                    : "text-green-500"
                }`}
              />
              <div>
                <p className="font-medium">Location Status</p>
                <p className="text-sm text-gray-500">
                  {securityStatus?.locationStatus.country} -{" "}
                  {securityStatus?.locationStatus.isRestricted
                    ? "Restricted"
                    : "Allowed"}
                </p>
              </div>
            </div>
          </Card>

          {/* Connection Security */}
          <Card className="p-4">
            <div className="flex items-center space-x-3">
              <Lock
                className={`h-5 w-5 ${
                  securityStatus?.connectionSecurity.encryptionEnabled
                    ? "text-green-500"
                    : "text-red-500"
                }`}
              />
              <div>
                <p className="font-medium">Connection Security</p>
                <p className="text-sm text-gray-500">
                  {securityStatus?.connectionSecurity.activeConnections} active
                  connections
                </p>
              </div>
            </div>
          </Card>

          {/* Threat Status */}
          <Card className="p-4">
            <div className="flex items-center space-x-3">
              <AlertTriangle
                className={`h-5 w-5 ${
                  securityStatus?.threatDetection.threats === 0
                    ? "text-green-500"
                    : "text-red-500"
                }`}
              />
              <div>
                <p className="font-medium">Threat Detection</p>
                <p className="text-sm text-gray-500">
                  {securityStatus?.threatDetection.threats || "No"} threats
                  detected
                </p>
              </div>
            </div>
          </Card>

          {/* Key Health */}
          <Card className="p-4">
            <div className="flex items-center space-x-3">
              <FileKey
                className={`h-5 w-5 ${
                  securityStatus?.keyStatus.health === "good"
                    ? "text-green-500"
                    : securityStatus?.keyStatus.health === "needs_rotation"
                      ? "text-yellow-500"
                      : "text-red-500"
                }`}
              />
              <div>
                <p className="font-medium">Key Health</p>
                <p className="text-sm text-gray-500">
                  {securityStatus?.keyStatus.health.replace("_", " ")}
                </p>
              </div>
            </div>
          </Card>
        </div>
      </Card>

      {/* Location-based Security Alert */}
      {securityStatus?.locationStatus.isRestricted && (
        <Alert variant="warning">
          <AlertTitle>Location-based Security Requirements</AlertTitle>
          <AlertDescription>
            Your current location requires a minimum security level of{" "}
            <span className="font-medium">
              {securityStatus.locationStatus.requiredLevel}
            </span>
          </AlertDescription>
        </Alert>
      )}

      {/* Security Configuration */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-6">Security Configuration</h3>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <FormField
                control={form.control}
                name="threatDetection"
                render={({ field }) => (
                  <FormItem className="flex flex-row items-center justify-between rounded-lg border p-4">
                    <div className="space-y-0.5">
                      <FormLabel className="text-base">
                        Threat Detection
                      </FormLabel>
                      <FormDescription>
                        Continuously monitor for security threats
                      </FormDescription>
                    </div>
                    <FormControl>
                      <Switch
                        checked={field.value}
                        onCheckedChange={field.onChange}
                      />
                    </FormControl>
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="automaticKeyRotation"
                render={({ field }) => (
                  <FormItem className="flex flex-row items-center justify-between rounded-lg border p-4">
                    <div className="space-y-0.5">
                      <FormLabel className="text-base">
                        Automatic Key Rotation
                      </FormLabel>
                      <FormDescription>
                        Automatically rotate encryption keys
                      </FormDescription>
                    </div>
                    <FormControl>
                      <Switch
                        checked={field.value}
                        onCheckedChange={field.onChange}
                      />
                    </FormControl>
                  </FormItem>
                )}
              />
            </div>

            <Separator />

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <FormField
                control={form.control}
                name="connectionMode"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Connection Mode</FormLabel>
                    <Select value={field.value} onValueChange={field.onChange}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="standard">Standard</SelectItem>
                        <SelectItem value="p2p">
                          P2P (Enhanced Security)
                        </SelectItem>
                      </SelectContent>
                    </Select>
                    <FormDescription>
                      Choose how your worker connects to the network
                    </FormDescription>
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="dataRetention"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Data Retention Period</FormLabel>
                    <Select value={field.value} onValueChange={field.onChange}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="24h">24 Hours</SelectItem>
                        <SelectItem value="7d">7 Days</SelectItem>
                        <SelectItem value="30d">30 Days</SelectItem>
                        <SelectItem value="90d">90 Days</SelectItem>
                      </SelectContent>
                    </Select>
                    <FormDescription>
                      How long to keep activity logs and data
                    </FormDescription>
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="loggingLevel"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Logging Level</FormLabel>
                    <Select value={field.value} onValueChange={field.onChange}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="basic">Basic</SelectItem>
                        <SelectItem value="detailed">Detailed</SelectItem>
                        <SelectItem value="debug">Debug</SelectItem>
                      </SelectContent>
                    </Select>
                    <FormDescription>
                      Level of detail in security logs
                    </FormDescription>
                  </FormItem>
                )}
              />
            </div>

            <Button type="submit" className="w-full">
              Save Security Settings
            </Button>
          </form>
        </Form>
      </Card>
    </div>
  );
}
