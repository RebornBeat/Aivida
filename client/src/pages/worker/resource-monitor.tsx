import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Card } from "@/components/ui";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { formatBytes, formatDuration } from "@/utils/format";

interface ResourceMetrics {
  timestamp: number;
  cpu: {
    utilization: number;
    temperature: number;
    frequency: number;
  };
  gpu: Array<{
    id: string;
    utilization: number;
    memoryUsed: number;
    temperature: number;
    powerUsage: number;
  }>;
  memory: {
    used: number;
    total: number;
    swap: number;
  };
  network: {
    upload: number;
    download: number;
    activeConnections: number;
  };
  jobs: {
    training: number;
    inference: number;
    annotation: number;
  };
}

interface ResourceMonitorProps {
  refreshInterval?: number; // in milliseconds
}

export function ResourceMonitor({
  refreshInterval = 1000,
}: ResourceMonitorProps) {
  const [metrics, setMetrics] = useState<ResourceMetrics[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const currentMetrics = (await invoke(
          "get_resource_metrics",
        )) as ResourceMetrics;
        setMetrics((prev) => [...prev, currentMetrics].slice(-60)); // Keep last 60 data points (1 minute)
        setError(null);
      } catch (err) {
        setError("Failed to fetch resource metrics");
        console.error("Resource metrics error:", err);
      } finally {
        setIsLoading(false);
      }
    };

    // Initial fetch
    fetchMetrics();

    // Set up interval for real-time updates
    const interval = setInterval(fetchMetrics, refreshInterval);
    return () => clearInterval(interval);
  }, [refreshInterval]);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload) return null;

    return (
      <div className="bg-white p-3 border rounded-lg shadow-lg">
        <p className="text-sm font-medium">
          {new Date(label).toLocaleTimeString()}
        </p>
        {payload.map((entry: any) => (
          <p key={entry.name} className="text-sm">
            <span
              className="inline-block w-3 h-3 rounded-full mr-2"
              style={{ backgroundColor: entry.color }}
            />
            {entry.name}: {entry.value.toFixed(2)}
            {entry.unit}
          </p>
        ))}
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* CPU Usage Chart */}
      <Card className="p-4">
        <h3 className="text-lg font-semibold mb-4">CPU Utilization</h3>
        <div className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={metrics}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="timestamp"
                tickFormatter={(timestamp) =>
                  new Date(timestamp).toLocaleTimeString()
                }
              />
              <YAxis />
              <Tooltip content={<CustomTooltip />} />
              <Line
                type="monotone"
                dataKey="cpu.utilization"
                stroke="#3b82f6"
                name="Utilization"
                unit="%"
              />
              <Line
                type="monotone"
                dataKey="cpu.temperature"
                stroke="#ef4444"
                name="Temperature"
                unit="°C"
              />
              <Line
                type="monotone"
                dataKey="cpu.frequency"
                stroke="#10b981"
                name="Frequency"
                unit="GHz"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Card>

      {/* GPU Usage Chart (if available) */}
      {metrics[0]?.gpu.length > 0 && (
        <Card className="p-4">
          <h3 className="text-lg font-semibold mb-4">GPU Utilization</h3>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={metrics}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="timestamp"
                  tickFormatter={(timestamp) =>
                    new Date(timestamp).toLocaleTimeString()
                  }
                />
                <YAxis />
                <Tooltip content={<CustomTooltip />} />
                {metrics[0].gpu.map((gpu, index) => (
                  <Line
                    key={gpu.id}
                    type="monotone"
                    dataKey={`gpu[${index}].utilization`}
                    stroke={`hsl(${index * 60}, 70%, 50%)`}
                    name={`GPU ${index + 1}`}
                    unit="%"
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mt-4">
            {metrics[0].gpu.map((gpu, index) => (
              <div key={gpu.id} className="p-3 bg-gray-50 rounded-lg">
                <h4 className="font-medium mb-2">GPU {index + 1}</h4>
                <div className="space-y-1 text-sm">
                  <p>Memory: {formatBytes(gpu.memoryUsed)}</p>
                  <p>Temperature: {gpu.temperature}°C</p>
                  <p>Power: {gpu.powerUsage}W</p>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Memory Usage Chart */}
      <Card className="p-4">
        <h3 className="text-lg font-semibold mb-4">Memory Usage</h3>
        <div className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={metrics}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="timestamp"
                tickFormatter={(timestamp) =>
                  new Date(timestamp).toLocaleTimeString()
                }
              />
              <YAxis tickFormatter={(value) => formatBytes(value)} />
              <Tooltip content={<CustomTooltip />} />
              <Area
                type="monotone"
                dataKey="memory.used"
                stackId="1"
                stroke="#3b82f6"
                fill="#3b82f6"
                name="Used"
                unit="GB"
              />
              <Area
                type="monotone"
                dataKey="memory.swap"
                stackId="1"
                stroke="#ef4444"
                fill="#ef4444"
                name="Swap"
                unit="GB"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </Card>

      {/* Network Usage Chart */}
      <Card className="p-4">
        <h3 className="text-lg font-semibold mb-4">Network Activity</h3>
        <div className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={metrics}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="timestamp"
                tickFormatter={(timestamp) =>
                  new Date(timestamp).toLocaleTimeString()
                }
              />
              <YAxis tickFormatter={(value) => `${formatBytes(value)}/s`} />
              <Tooltip content={<CustomTooltip />} />
              <Line
                type="monotone"
                dataKey="network.upload"
                stroke="#3b82f6"
                name="Upload"
                unit="MB/s"
              />
              <Line
                type="monotone"
                dataKey="network.download"
                stroke="#10b981"
                name="Download"
                unit="MB/s"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-4 text-sm text-gray-500">
          Active Connections:{" "}
          {metrics[metrics.length - 1]?.network.activeConnections || 0}
        </div>
      </Card>

      {/* Active Jobs Overview */}
      <Card className="p-4">
        <h3 className="text-lg font-semibold mb-4">Active Jobs</h3>
        <div className="h-[200px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={metrics}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="timestamp"
                tickFormatter={(timestamp) =>
                  new Date(timestamp).toLocaleTimeString()
                }
              />
              <YAxis />
              <Tooltip content={<CustomTooltip />} />
              <Area
                type="monotone"
                dataKey="jobs.training"
                stackId="1"
                stroke="#3b82f6"
                fill="#3b82f6"
                name="Training"
              />
              <Area
                type="monotone"
                dataKey="jobs.inference"
                stackId="1"
                stroke="#10b981"
                fill="#10b981"
                name="Inference"
              />
              <Area
                type="monotone"
                dataKey="jobs.annotation"
                stackId="1"
                stroke="#8b5cf6"
                fill="#8b5cf6"
                name="Annotation"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </Card>
    </div>
  );
}
