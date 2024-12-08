import { Progress } from "recharts";
import type { ResourceMetrics } from "./index";
import { formatBytes } from "../../utils/format";

interface SystemHealthProps {
  metrics: ResourceMetrics | null;
}

export function SystemHealth({ metrics }: SystemHealthProps) {
  if (!metrics) return null;

  const healthMetrics = [
    {
      label: "CPU",
      value: metrics.cpuUtilization,
      format: (value: number) => `${value.toFixed(1)}%`,
      color: "bg-primary",
    },
    {
      label: "GPU",
      value: metrics.gpuUtilization,
      format: (value: number) => `${value.toFixed(1)}%`,
      color: "bg-info",
    },
    {
      label: "Memory",
      value: metrics.memoryUsed / 1024 / 1024 / 1024,
      format: (value: number) => formatBytes(value * 1024 * 1024 * 1024),
      color: "bg-warning",
    },
    {
      label: "Network",
      value:
        (metrics.networkBandwidth.upload + metrics.networkBandwidth.download) /
        1024 /
        1024,
      format: (value: number) => `${value.toFixed(1)} MB/s`,
      color: "bg-success",
    },
  ];

  return (
    <div className="space-y-4">
      {healthMetrics.map((metric) => (
        <div key={metric.label} className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-secondary">{metric.label}</span>
            <span>{metric.format(metric.value)}</span>
          </div>
          <div className="h-2 bg-secondary-light rounded-full overflow-hidden">
            <div
              className={`h-full ${metric.color} rounded-full transition-all duration-500`}
              style={{ width: `${Math.min(metric.value, 100)}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
