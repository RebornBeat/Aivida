import { useState, useEffect } from "react";
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
} from "@/components/ui";
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
import { WorkerRegistration } from "./components/worker-registration";
import { PerformanceHistory } from "./components/performance-history";
import { JobHistory } from "./components/job-history";
import { ResourceMonitor } from "./components/resource-monitor";
import { EarningsChart } from "./components/earnings-chart";
import { formatBytes, formatDuration } from "@/utils/format";
import { WorkerStatus, JobType, WorkerMetrics } from "@/types";

export function WorkerDashboard() {
  const [isRegistered, setIsRegistered] = useState(false);
  const [workerStatus, setWorkerStatus] = useState<WorkerStatus | null>(null);
  const [metrics, setMetrics] = useState<WorkerMetrics | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const checkRegistration = async () => {
      try {
        setIsLoading(true);
        const info = await invoke("get_worker_info");
        if (info) {
          setIsRegistered(true);
          setWorkerStatus(info as WorkerStatus);
          // Fetch initial metrics
          const workerMetrics = await invoke("get_worker_metrics");
          setMetrics(workerMetrics as WorkerMetrics);
        }
      } catch (err) {
        console.error("Failed to check worker registration:", err);
        setIsRegistered(false);
      } finally {
        setIsLoading(false);
      }
    };

    checkRegistration();
  }, []);

  // Regular metrics update
  useEffect(() => {
    if (!isRegistered) return;

    const updateMetrics = async () => {
      try {
        const workerMetrics = await invoke("get_worker_metrics");
        setMetrics(workerMetrics as WorkerMetrics);
      } catch (err) {
        console.error("Failed to update metrics:", err);
      }
    };

    const interval = setInterval(updateMetrics, 5000);
    return () => clearInterval(interval);
  }, [isRegistered]);

  const toggleAvailability = async () => {
    try {
      await invoke("toggle_worker_availability");
      const updatedStatus = await invoke("get_worker_info");
      setWorkerStatus(updatedStatus as WorkerStatus);
    } catch (err) {
      setError("Failed to update worker status");
    }
  };

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-[500px]">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (!isRegistered) {
    return <WorkerRegistration onRegistered={() => setIsRegistered(true)} />;
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">Worker Dashboard</h1>
        <div className="flex items-center space-x-4">
          <div className="text-right">
            <p className="text-sm text-gray-500">Total Earnings</p>
            <p className="text-lg font-bold text-green-600">
              {metrics?.totalEarnings || 0} credits
            </p>
          </div>
          <Button
            onClick={toggleAvailability}
            variant={workerStatus?.available ? "outline" : "default"}
          >
            {workerStatus?.available ? "Go Offline" : "Go Online"}
          </Button>
        </div>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Current Status Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="p-4">
          <h3 className="font-semibold mb-2">Active Jobs</h3>
          <div className="grid grid-cols-3 gap-2 text-center">
            <div>
              <p className="text-sm text-gray-500">Training</p>
              <p className="font-bold">{metrics?.activeJobs.training || 0}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Inference</p>
              <p className="font-bold">{metrics?.activeJobs.inference || 0}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Annotation</p>
              <p className="font-bold">{metrics?.activeJobs.annotation || 0}</p>
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <h3 className="font-semibold mb-2">Resource Usage</h3>
          <div className="grid grid-cols-3 gap-2 text-center">
            <div>
              <p className="text-sm text-gray-500">CPU</p>
              <p className="font-bold">{metrics?.resourceUsage.cpu}%</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">GPU</p>
              <p className="font-bold">{metrics?.resourceUsage.gpu}%</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Memory</p>
              <p className="font-bold">
                {formatBytes(metrics?.resourceUsage.memory || 0)}
              </p>
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <h3 className="font-semibold mb-2">Performance</h3>
          <div className="grid grid-cols-2 gap-2 text-center">
            <div>
              <p className="text-sm text-gray-500">Success Rate</p>
              <p className="font-bold">{metrics?.performance.successRate}%</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Uptime</p>
              <p className="font-bold">
                {formatDuration(metrics?.performance.uptime || 0)}
              </p>
            </div>
          </div>
        </Card>
      </div>

      <Tabs defaultValue="monitoring">
        <TabsList>
          <TabsTrigger value="monitoring">Live Monitoring</TabsTrigger>
          <TabsTrigger value="performance">Performance History</TabsTrigger>
          <TabsTrigger value="jobs">Job History</TabsTrigger>
          <TabsTrigger value="earnings">Earnings</TabsTrigger>
        </TabsList>

        <TabsContent value="monitoring" className="mt-6">
          <ResourceMonitor metrics={metrics} />
        </TabsContent>

        <TabsContent value="performance" className="mt-6">
          <PerformanceHistory />
        </TabsContent>

        <TabsContent value="jobs" className="mt-6">
          <JobHistory />
        </TabsContent>

        <TabsContent value="earnings" className="mt-6">
          <EarningsChart />
        </TabsContent>
      </Tabs>
    </div>
  );
}
