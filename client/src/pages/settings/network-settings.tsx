import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  Card,
  Button,
  Switch,
  Progress,
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
  Alert,
  AlertTitle,
  AlertDescription,
  Separator,
} from "@/components/ui";
import {
  Network,
  Wifi,
  Globe,
  Activity,
  Zap,
  Share2,
  Signal,
  Users,
  Radio,
} from "lucide-react";
import { z } from "zod";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { formatBytes } from "@/utils/format";

interface NetworkSettingsProps {
  onError: (error: string) => void;
  onSuccess: (message: string) => void;
}

interface NetworkStats {
  upload: {
    current: number;
    peak: number;
    average: number;
  };
  download: {
    current: number;
    peak: number;
    average: number;
  };
  latency: {
    current: number;
    average: number;
  };
  connections: {
    active: number;
    total: number;
    p2p: number;
  };
  bandwidth: {
    available: number;
    used: number;
  };
}

interface P2PNode {
  id: string;
  status: "connected" | "disconnected";
  latency: number;
  bandwidth: number;
  location: string;
  lastSeen: string;
}

const networkSchema = z.object({
  connectionMode: z.enum(["auto", "p2p", "server"]),
  bandwidthLimit: z.number().min(0),
  p2pSettings: z.object({
    maxConnections: z.number().min(1).max(100),
    preferredRegions: z.array(z.string()),
    discoveryMode: z.enum(["active", "passive", "hybrid"]),
  }),
  optimizations: z.object({
    autoOptimize: z.boolean(),
    prioritizeLatency: z.boolean(),
    enableQos: z.boolean(),
  }),
});

export function NetworkSettings({ onError, onSuccess }: NetworkSettingsProps) {
  const [networkStats, setNetworkStats] = useState<NetworkStats | null>(null);
  const [p2pNodes, setP2PNodes] = useState<P2PNode[]>([]);
  const [isTestingConnection, setIsTestingConnection] = useState(false);
  const [testProgress, setTestProgress] = useState(0);
  const [isLoading, setIsLoading] = useState(true);

  const form = useForm<z.infer<typeof networkSchema>>({
    resolver: zodResolver(networkSchema),
    defaultValues: {
      connectionMode: "auto",
      bandwidthLimit: 0,
      p2pSettings: {
        maxConnections: 50,
        preferredRegions: [],
        discoveryMode: "hybrid",
      },
      optimizations: {
        autoOptimize: true,
        prioritizeLatency: true,
        enableQos: true,
      },
    },
  });

  useEffect(() => {
    fetchNetworkData();
    const interval = setInterval(fetchNetworkData, 5000); // Update every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchNetworkData = async () => {
    try {
      setIsLoading(true);
      const [stats, nodes, settings] = await Promise.all([
        invoke("get_network_stats"),
        invoke("get_p2p_nodes"),
        invoke("get_network_settings"),
      ]);

      setNetworkStats(stats as NetworkStats);
      setP2PNodes(nodes as P2PNode[]);
      form.reset(settings as z.infer<typeof networkSchema>);
    } catch (err) {
      console.error("Failed to fetch network data:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const testConnection = async () => {
    try {
      setIsTestingConnection(true);
      setTestProgress(0);

      // Start connection test
      await invoke("start_network_test");

      // Simulate progress updates
      for (let i = 0; i <= 100; i += 10) {
        setTestProgress(i);
        await new Promise((resolve) => setTimeout(resolve, 500));
      }

      const results = await invoke("get_network_test_results");
      onSuccess("Network test completed successfully");
      await fetchNetworkData(); // Refresh network data
    } catch (err) {
      onError("Failed to complete network test");
    } finally {
      setIsTestingConnection(false);
      setTestProgress(0);
    }
  };

  const onSubmit = async (data: z.infer<typeof networkSchema>) => {
    try {
      await invoke("update_network_settings", { settings: data });
      onSuccess("Network settings updated successfully");
      await fetchNetworkData();
    } catch (err) {
      onError("Failed to update network settings");
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
      {/* Network Status Overview */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-lg font-semibold">Network Status</h3>
            <p className="text-sm text-gray-500">
              Current network performance and statistics
            </p>
          </div>
          <Button onClick={testConnection} disabled={isTestingConnection}>
            {isTestingConnection ? (
              <>
                <Activity className="mr-2 h-4 w-4 animate-spin" />
                Testing...
              </>
            ) : (
              <>
                <Network className="mr-2 h-4 w-4" />
                Test Connection
              </>
            )}
          </Button>
        </div>

        {isTestingConnection && (
          <div className="mb-6">
            <p className="text-sm text-gray-500 mb-2">
              Testing network performance...
            </p>
            <Progress value={testProgress} />
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Upload Speed */}
          <Card className="p-4">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-green-100 rounded-lg">
                <Share2 className="h-5 w-5 text-green-600" />
              </div>
              <div>
                <p className="text-sm text-gray-500">Upload</p>
                <p className="font-medium">
                  {formatBytes(networkStats?.upload.current || 0)}/s
                </p>
                <p className="text-xs text-gray-500">
                  Peak: {formatBytes(networkStats?.upload.peak || 0)}/s
                </p>
              </div>
            </div>
          </Card>

          {/* Download Speed */}
          <Card className="p-4">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-blue-100 rounded-lg">
                <Share2 className="h-5 w-5 text-blue-600 transform rotate-180" />
              </div>
              <div>
                <p className="text-sm text-gray-500">Download</p>
                <p className="font-medium">
                  {formatBytes(networkStats?.download.current || 0)}/s
                </p>
                <p className="text-xs text-gray-500">
                  Peak: {formatBytes(networkStats?.download.peak || 0)}/s
                </p>
              </div>
            </div>
          </Card>

          {/* Latency */}
          <Card className="p-4">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-yellow-100 rounded-lg">
                <Activity className="h-5 w-5 text-yellow-600" />
              </div>
              <div>
                <p className="text-sm text-gray-500">Latency</p>
                <p className="font-medium">
                  {networkStats?.latency.current || 0}ms
                </p>
                <p className="text-xs text-gray-500">
                  Avg: {networkStats?.latency.average || 0}ms
                </p>
              </div>
            </div>
          </Card>

          {/* Connections */}
          <Card className="p-4">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-purple-100 rounded-lg">
                <Users className="h-5 w-5 text-purple-600" />
              </div>
              <div>
                <p className="text-sm text-gray-500">Connections</p>
                <p className="font-medium">
                  {networkStats?.connections.active || 0} active
                </p>
                <p className="text-xs text-gray-500">
                  P2P: {networkStats?.connections.p2p || 0}
                </p>
              </div>
            </div>
          </Card>
        </div>
      </Card>

      {/* P2P Network Status */}
      {p2pNodes.length > 0 && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">P2P Network</h3>
          <div className="space-y-4">
            {p2pNodes.map((node) => (
              <div
                key={node.id}
                className="flex items-center justify-between p-4 rounded-lg border"
              >
                <div className="flex items-center space-x-4">
                  <Radio
                    className={`h-5 w-5 ${
                      node.status === "connected"
                        ? "text-green-500"
                        : "text-gray-400"
                    }`}
                  />
                  <div>
                    <p className="font-medium">{node.id.slice(0, 8)}</p>
                    <p className="text-sm text-gray-500">{node.location}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-sm font-medium">{node.latency}ms</p>
                  <p className="text-xs text-gray-500">
                    {formatBytes(node.bandwidth)}/s
                  </p>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Network Configuration */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-6">Network Configuration</h3>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
            {/* Connection Mode */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
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
                        <SelectItem value="auto">Automatic</SelectItem>
                        <SelectItem value="p2p">P2P Preferred</SelectItem>
                        <SelectItem value="server">Server Only</SelectItem>
                      </SelectContent>
                    </Select>
                    <FormDescription>
                      Choose how to connect to the network
                    </FormDescription>
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="bandwidthLimit"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Bandwidth Limit (MB/s)</FormLabel>
                    <Select
                      value={field.value.toString()}
                      onValueChange={(value) => field.onChange(parseInt(value))}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="0">Unlimited</SelectItem>
                        <SelectItem value="10">10 MB/s</SelectItem>
                        <SelectItem value="50">50 MB/s</SelectItem>
                        <SelectItem value="100">100 MB/s</SelectItem>
                        <SelectItem value="500">500 MB/s</SelectItem>
                      </SelectContent>
                    </Select>
                    <FormDescription>
                      Limit network bandwidth usage
                    </FormDescription>
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="p2pSettings.discoveryMode"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Discovery Mode</FormLabel>
                    <Select value={field.value} onValueChange={field.onChange}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="active">Active</SelectItem>
                        <SelectItem value="passive">Passive</SelectItem>
                        <SelectItem value="hybrid">Hybrid</SelectItem>
                      </SelectContent>
                    </Select>
                    <FormDescription>How to discover P2P nodes</FormDescription>
                  </FormItem>
                )}
              />
            </div>

            <Separator />

            {/* Optimizations */}
            <div className="space-y-4">
              <h4 className="text-sm font-medium">Network Optimizations</h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <FormField
                  control={form.control}
                  name="optimizations.autoOptimize"
                  render={({ field }) => (
                    <FormItem className="flex flex-row items-center justify-between rounded-lg border p-4">
                      <div className="space-y-0.5">
                        <FormLabel className="text-base">
                          Auto-Optimize
                        </FormLabel>
                        <FormDescription>
                          Automatically optimize network settings
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
                  name="optimizations.prioritizeLatency"
                  render={({ field }) => (
                    <FormItem className="flex flex-row items-center justify-between rounded-lg border p-4">
                      <div className="space-y-0.5">
                        <FormLabel className="text-base">
                          Prioritize Latency
                        </FormLabel>
                        <FormDescription>
                          Optimize for lower latency
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
                  name="optimizations.enableQos"
                  render={({ field }) => (
                    <FormItem className="flex flex-row items-center justify-between rounded-lg border p-4">
                      <div className="space-y-0.5">
                        <FormLabel className="text-base">Enable QoS</FormLabel>
                        <FormDescription>
                          Quality of Service prioritization
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
            </div>

            <Button type="submit" className="w-full">
              Save Network Settings
            </Button>
          </form>
        </Form>
      </Card>
    </div>
  );
}
