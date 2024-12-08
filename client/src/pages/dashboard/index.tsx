import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
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
import { Cpu, ActivitySquare, ChartLine, Loader2 } from "lucide-react";
import { formatBytes } from "../../utils/format";
import { MetricsOverview } from "./metrics-overview";
import { JobList } from "./job-list";
import { ResourceMonitor } from "./resource-monitor";
import { SystemHealth } from "./system-health";

interface ResourceMetrics {
  cpuUtilization: number;
  gpuUtilization: number;
  memoryUsed: number;
  networkBandwidth: {
    upload: number;
    download: number;
  };
  cpuTrend?: number;
  gpuTrend?: number;
  memoryTrend?: number;
}

interface Job {
  id: string;
  title: string;
  type: "training" | "inference" | "annotation";
  status: "pending" | "processing" | "completed" | "failed";
  progress: {
    current: number;
    total: number;
  };
  resourceUsage: {
    cpu: number;
    memory: number;
    gpu?: number;
  };
  startTime: Date;
  estimatedCompletion?: Date;
}

export function Dashboard() {
  const [metrics, setMetrics] = useState<ResourceMetrics | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [metricsData, jobsData] = await Promise.all([
          invoke<ResourceMetrics>("get_resources"),
          invoke<Job[]>("get_active_jobs"),
        ]);

        setMetrics(metricsData);
        setJobs(jobsData);
        setError(null);
      } catch (err) {
        setError("Failed to fetch dashboard data");
        console.error("Dashboard data fetch error:", err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000); // Update every 5 seconds
    return () => clearInterval(interval);
  }, []);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-lg border border-danger bg-danger-light p-4 text-danger">
        <h3 className="font-medium">Error</h3>
        <p>{error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">Dashboard</h1>
      </div>

      {/* Metrics Overview */}
      <MetricsOverview metrics={metrics} />

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Active Jobs */}
        <div className="rounded-lg border border-border bg-background p-6">
          <h2 className="text-lg font-semibold mb-4">Active Jobs</h2>
          <JobList jobs={jobs} />
        </div>

        {/* System Health */}
        <div className="rounded-lg border border-border bg-background p-6">
          <h2 className="text-lg font-semibold mb-4">System Health</h2>
          <SystemHealth metrics={metrics} />
        </div>
      </div>

      {/* Resource Monitor */}
      <div className="rounded-lg border border-border bg-background p-6">
        <h2 className="text-lg font-semibold mb-4">Resource Monitor</h2>
        <ResourceMonitor metrics={metrics} />
      </div>
    </div>
  );
}
