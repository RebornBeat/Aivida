import { Cpu, ActivitySquare, ChartLine } from "lucide-react";
import { formatBytes } from "@/utils/format";
import type { ResourceMetrics } from "./index";

interface MetricsOverviewProps {
  metrics: ResourceMetrics | null;
}

export function MetricsOverview({ metrics }: MetricsOverviewProps) {
  if (!metrics) return null;

  const metricCards = [
    {
      title: "CPU Usage",
      value: `${metrics.cpuUtilization.toFixed(1)}%`,
      icon: <Cpu className="h-5 w-5 text-primary" />,
      trend: metrics.cpuTrend,
    },
    {
      title: "GPU Usage",
      value: `${metrics.gpuUtilization.toFixed(1)}%`,
      icon: <ActivitySquare className="h-5 w-5 text-primary" />,
      trend: metrics.gpuTrend,
    },
    {
      title: "Memory Used",
      value: formatBytes(metrics.memoryUsed),
      icon: <ChartLine className="h-5 w-5 text-primary" />,
      trend: metrics.memoryTrend,
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {metricCards.map((card) => (
        <div
          key={card.title}
          className="rounded-lg border border-border bg-background p-4"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-primary/10 rounded-lg">{card.icon}</div>
              <div>
                <p className="text-sm text-secondary">{card.title}</p>
                <p className="text-2xl font-bold">{card.value}</p>
              </div>
            </div>
            {card.trend !== undefined && (
              <div
                className={`text-sm ${
                  card.trend >= 0 ? "text-success" : "text-danger"
                }`}
              >
                {card.trend >= 0 ? "↑" : "↓"} {Math.abs(card.trend)}%
                <p className="text-xs text-secondary">vs last hour</p>
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
