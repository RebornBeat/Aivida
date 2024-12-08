import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  Card,
  Button,
  Switch,
  Slider,
  Form,
  FormField,
  FormItem,
  FormLabel,
  FormControl,
  FormDescription,
  Alert,
  AlertTitle,
  AlertDescription,
  Progress,
  Separator,
} from "@/components/ui";
import {
  Zap,
  Gauge,
  BarChart3,
  CPU,
  Memory,
  Network,
  Maximize2,
  MinusCircle,
} from "lucide-react";
import { z } from "zod";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { formatBytes } from "@/utils/format";

interface PerformanceSettingsProps {
  onError: (error: string) => void;
  onSuccess: (message: string) => void;
}

interface PerformanceMetrics {
  cpu: {
    currentUsage: number;
    averageUsage: number;
    temperature: number;
    powerDraw: number;
  };
  gpu?: {
    currentUsage: number;
    averageUsage: number;
    temperature: number;
    powerDraw: number;
    memoryUsage: number;
  };
  memory: {
    used: number;
    total: number;
    swap: number;
  };
  network: {
    bandwidth: number;
    latency: number;
  };
}

const performanceSchema = z.object({
  optimizationMode: z.enum(["auto", "performance", "efficiency", "balanced"]),
  autoScaling: z.boolean(),
  batchSize: z.number().min(1).max(1000),
  memoryBuffer: z.number().min(0).max(50),
  networkPriority: z.enum(["high", "normal", "low"]),
  processingThrottling: z.number().min(0).max(100),
  cacheSize: z.number().min(100).max(10000),
});

export function PerformanceSettings({
  onError,
  onSuccess,
}: PerformanceSettingsProps) {
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [optimizationStatus, setOptimizationStatus] = useState<string | null>(
    null,
  );
  const [isLoading, setIsLoading] = useState(true);

  const form = useForm<z.infer<typeof performanceSchema>>({
    resolver: zodResolver(performanceSchema),
    defaultValues: {
      optimizationMode: "auto",
      autoScaling: true,
      batchSize: 32,
      memoryBuffer: 20,
      networkPriority: "normal",
      processingThrottling: 0,
      cacheSize: 1000,
    },
  });

  useEffect(() => {
    fetchPerformanceData();
    const interval = setInterval(fetchPerformanceData, 5000); // Update every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchPerformanceData = async () => {
    try {
      const [metricsData, settings] = await Promise.all([
        invoke("get_performance_metrics"),
        invoke("get_performance_settings"),
      ]);

      setMetrics(metricsData as PerformanceMetrics);
      form.reset(settings as z.infer<typeof performanceSchema>);
    } catch (err) {
      console.error("Failed to fetch performance data:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const onSubmit = async (data: z.infer<typeof performanceSchema>) => {
    try {
      await invoke("update_performance_settings", { settings: data });
      onSuccess("Performance settings updated successfully");
    } catch (err) {
      onError("Failed to update performance settings");
    }
  };

  const runOptimization = async () => {
    try {
      setIsOptimizing(true);
      setOptimizationStatus("Analyzing system performance...");

      // Start optimization process
      await invoke("start_performance_optimization");

      setOptimizationStatus("Optimizing resource allocation...");
      await new Promise((resolve) => setTimeout(resolve, 2000));

      setOptimizationStatus("Adjusting batch sizes...");
      await new Promise((resolve) => setTimeout(resolve, 1500));

      setOptimizationStatus("Finalizing optimizations...");
      await new Promise((resolve) => setTimeout(resolve, 1000));

      onSuccess("Performance optimization completed");
      await fetchPerformanceData(); // Refresh data
    } catch (err) {
      onError("Failed to optimize performance");
    } finally {
      setIsOptimizing(false);
      setOptimizationStatus(null);
    }
  };

  const renderMetricCard = (
    icon: React.ReactNode,
    title: string,
    value: number,
    unit: string,
    trend?: { value: number; label: string },
  ) => (
    <Card className="p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-primary/10 rounded-lg">{icon}</div>
          <div>
            <p className="text-sm text-gray-500">{title}</p>
            <p className="text-lg font-semibold">
              {value}
              <span className="text-sm font-normal text-gray-500 ml-1">
                {unit}
              </span>
            </p>
          </div>
        </div>
        {trend && (
          <div
            className={`text-sm ${
              trend.value >= 0 ? "text-green-500" : "text-red-500"
            }`}
          >
            {trend.value >= 0 ? "↑" : "↓"} {Math.abs(trend.value)}%
            <p className="text-xs text-gray-500">{trend.label}</p>
          </div>
        )}
      </div>
    </Card>
  );

  if (isLoading) {
    return (
      <div className="flex justify-center py-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Current Performance Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {metrics && (
          <>
            {renderMetricCard(
              <CPU className="h-5 w-5 text-primary" />,
              "CPU Usage",
              metrics.cpu.currentUsage,
              "%",
              {
                value: metrics.cpu.currentUsage - metrics.cpu.averageUsage,
                label: "vs average",
              },
            )}

            {metrics.gpu &&
              renderMetricCard(
                <Gauge className="h-5 w-5 text-primary" />,
                "GPU Usage",
                metrics.gpu.currentUsage,
                "%",
                {
                  value: metrics.gpu.currentUsage - metrics.gpu.averageUsage,
                  label: "vs average",
                },
              )}

            {renderMetricCard(
              <Memory className="h-5 w-5 text-primary" />,
              "Memory Usage",
              (metrics.memory.used / metrics.memory.total) * 100,
              "%",
            )}

            {renderMetricCard(
              <Network className="h-5 w-5 text-primary" />,
              "Network Latency",
              metrics.network.latency,
              "ms",
            )}
          </>
        )}
      </div>

      {/* Quick Optimization */}
      <Card className="p-6">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold">Performance Optimization</h3>
            <p className="text-sm text-gray-500">
              Automatically optimize system performance based on current
              workload
            </p>
          </div>
          <Button onClick={runOptimization} disabled={isOptimizing}>
            {isOptimizing ? (
              <>
                <Gauge className="mr-2 h-4 w-4 animate-spin" />
                Optimizing...
              </>
            ) : (
              <>
                <Zap className="mr-2 h-4 w-4" />
                Optimize Now
              </>
            )}
          </Button>
        </div>

        {optimizationStatus && (
          <div className="mt-4">
            <p className="text-sm text-gray-500 mb-2">{optimizationStatus}</p>
            <Progress value={isOptimizing ? 66 : 100} />
          </div>
        )}
      </Card>

      {/* Performance Settings Form */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-6">
          Performance Configuration
        </h3>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <FormField
                control={form.control}
                name="optimizationMode"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Optimization Mode</FormLabel>
                    <div className="grid grid-cols-2 gap-4">
                      {["auto", "performance", "efficiency", "balanced"].map(
                        (mode) => (
                          <Card
                            key={mode}
                            className={`p-4 cursor-pointer hover:border-primary transition-colors ${
                              field.value === mode
                                ? "border-primary bg-primary/5"
                                : ""
                            }`}
                            onClick={() => field.onChange(mode)}
                          >
                            <div className="flex flex-col items-center space-y-2">
                              {mode === "auto" && (
                                <BarChart3 className="h-5 w-5 text-primary" />
                              )}
                              {mode === "performance" && (
                                <Maximize2 className="h-5 w-5 text-red-500" />
                              )}
                              {mode === "efficiency" && (
                                <MinusCircle className="h-5 w-5 text-green-500" />
                              )}
                              {mode === "balanced" && (
                                <Gauge className="h-5 w-5 text-blue-500" />
                              )}
                              <span className="capitalize">{mode}</span>
                            </div>
                          </Card>
                        ),
                      )}
                    </div>
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="autoScaling"
                render={({ field }) => (
                  <FormItem className="flex flex-row items-center justify-between rounded-lg border p-4">
                    <div className="space-y-0.5">
                      <FormLabel className="text-base">Auto Scaling</FormLabel>
                      <FormDescription>
                        Automatically adjust resources based on workload
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
                name="batchSize"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Batch Size</FormLabel>
                    <FormControl>
                      <div className="flex items-center space-x-3">
                        <Slider
                          value={[field.value]}
                          onValueChange={([value]) => field.onChange(value)}
                          min={1}
                          max={1000}
                          step={1}
                          className="flex-1"
                        />
                        <span className="w-12 text-right">{field.value}</span>
                      </div>
                    </FormControl>
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="memoryBuffer"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Memory Buffer (%)</FormLabel>
                    <FormControl>
                      <div className="flex items-center space-x-3">
                        <Slider
                          value={[field.value]}
                          onValueChange={([value]) => field.onChange(value)}
                          min={0}
                          max={50}
                          step={5}
                          className="flex-1"
                        />
                        <span className="w-12 text-right">{field.value}%</span>
                      </div>
                    </FormControl>
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="processingThrottling"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Processing Throttling (%)</FormLabel>
                    <FormControl>
                      <div className="flex items-center space-x-3">
                        <Slider
                          value={[field.value]}
                          onValueChange={([value]) => field.onChange(value)}
                          min={0}
                          max={100}
                          step={5}
                          className="flex-1"
                        />
                        <span className="w-12 text-right">{field.value}%</span>
                      </div>
                    </FormControl>
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="cacheSize"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Cache Size (MB)</FormLabel>
                    <FormControl>
                      <div className="flex items-center space-x-3">
                        <Slider
                          value={[field.value]}
                          onValueChange={([value]) => field.onChange(value)}
                          min={100}
                          max={10000}
                          step={100}
                          className="flex-1"
                        />
                        <span className="w-16 text-right">{field.value}MB</span>
                      </div>
                    </FormControl>
                  </FormItem>
                )}
              />
            </div>

            <Button type="submit" className="w-full">
              Save Configuration
            </Button>
          </form>
        </Form>
      </Card>
    </div>
  );
}
