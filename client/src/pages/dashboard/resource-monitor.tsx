import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from "recharts";
import type { ResourceMetrics } from "../types";
import { formatBytes } from "@/utils/format";

interface ResourceMonitorProps {
  metrics: ResourceMetrics | null;
}

export function ResourceMonitor({ metrics }: ResourceMonitorProps) {
  if (!metrics) return null;

  // For demo purposes, generate some historical data
  const historicalData = Array.from({ length: 20 }, (_, i) => ({
    time: new Date(Date.now() - (19 - i) * 5000).toLocaleTimeString(),
    cpu: metrics.cpuUtilization * (0.8 + Math.random() * 0.4),
    gpu: metrics.gpuUtilization * (0.8 + Math.random() * 0.4),
    memory: metrics.memoryUsed * (0.8 + Math.random() * 0.4),
    network: {
      upload: metrics.networkBandwidth.upload * (0.8 + Math.random() * 0.4),
      download: metrics.networkBandwidth.download * (0.8 + Math.random() * 0.4),
    },
  }));

  return (
    <div className="space-y-6">
      {/* CPU & GPU Usage */}
      <div className="rounded-lg border border-border p-4">
        <h3 className="text-lg font-medium mb-4">CPU & GPU Usage</h3>
        <div className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={historicalData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} domain={[0, 100]} unit="%" />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="cpu"
                stroke="#3B82F6"
                name="CPU"
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="gpu"
                stroke="#10B981"
                name="GPU"
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Memory Usage */}
      <div className="rounded-lg border border-border p-4">
        <h3 className="text-lg font-medium mb-4">Memory Usage</h3>
        <div className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={historicalData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" tick={{ fontSize: 12 }} />
              <YAxis
                tick={{ fontSize: 12 }}
                tickFormatter={(value) => formatBytes(value)}
              />
              <Tooltip formatter={(value) => formatBytes(value as number)} />
              <Area
                type="monotone"
                dataKey="memory"
                stroke="#8B5CF6"
                fill="#8B5CF6"
                fillOpacity={0.2}
                name="Memory"
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Network Activity */}
      <div className="rounded-lg border border-border p-4">
        <h3 className="text-lg font-medium mb-4">Network Activity</h3>
        <div className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={historicalData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" tick={{ fontSize: 12 }} />
              <YAxis
                tick={{ fontSize: 12 }}
                tickFormatter={(value) => `${formatBytes(value)}/s`}
              />
              <Tooltip
                formatter={(value) => `${formatBytes(value as number)}/s`}
              />
              <Line
                type="monotone"
                dataKey="network.upload"
                stroke="#EC4899"
                name="Upload"
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="network.download"
                stroke="#6366F1"
                name="Download"
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
