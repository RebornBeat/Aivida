import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  Card,
  Tabs,
  TabsList,
  TabsTrigger,
  TabsContent,
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { formatBytes, formatDuration } from "@/utils/format";

interface JobMetrics {
  timestamp: Date;
  jobType: "training" | "inference" | "annotation";
  successRate: number;
  processingTime: number;
  resourceEfficiency: number;
  credits: number;
}

interface PerformanceMetrics {
  timestamp: Date;
  cpuEfficiency: number;
  gpuEfficiency: number;
  memoryEfficiency: number;
  networkEfficiency: number;
  overallScore: number;
}

interface WorkerStats {
  totalJobsProcessed: number;
  avgProcessingTime: number;
  avgResourceEfficiency: number;
  totalCreditsEarned: number;
  uptimePercentage: number;
}

export function PerformanceHistory() {
  const [timeRange, setTimeRange] = useState<"24h" | "7d" | "30d" | "all">(
    "24h",
  );
  const [jobMetrics, setJobMetrics] = useState<JobMetrics[]>([]);
  const [performanceMetrics, setPerformanceMetrics] = useState<
    PerformanceMetrics[]
  >([]);
  const [workerStats, setWorkerStats] = useState<WorkerStats | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true);
        const [jobs, performance, stats] = await Promise.all([
          invoke("get_job_metrics_history", { timeRange }),
          invoke("get_performance_metrics_history", { timeRange }),
          invoke("get_worker_stats"),
        ]);

        setJobMetrics(jobs as JobMetrics[]);
        setPerformanceMetrics(performance as PerformanceMetrics[]);
        setWorkerStats(stats as WorkerStats);
        setError(null);
      } catch (err) {
        setError("Failed to fetch performance history");
        console.error("Performance history error:", err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [timeRange]);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload) return null;

    return (
      <div className="bg-white p-3 border rounded-lg shadow-lg">
        <p className="text-sm font-medium">
          {new Date(label).toLocaleDateString("en-US", {
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
          })}
        </p>
        {payload.map((entry: any) => (
          <p key={entry.name} className="text-sm">
            <span
              className="inline-block w-3 h-3 rounded-full mr-2"
              style={{ backgroundColor: entry.color }}
            />
            {entry.name}:{" "}
            {typeof entry.value === "number"
              ? entry.value.toFixed(2)
              : entry.value}
            {entry.unit || "%"}
          </p>
        ))}
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Overview Stats */}
      {workerStats && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
          <Card className="p-4">
            <p className="text-sm text-gray-500">Total Jobs</p>
            <p className="text-2xl font-bold">
              {workerStats.totalJobsProcessed}
            </p>
          </Card>
          <Card className="p-4">
            <p className="text-sm text-gray-500">Avg Processing Time</p>
            <p className="text-2xl font-bold">
              {formatDuration(workerStats.avgProcessingTime)}
            </p>
          </Card>
          <Card className="p-4">
            <p className="text-sm text-gray-500">Resource Efficiency</p>
            <p className="text-2xl font-bold">
              {workerStats.avgResourceEfficiency.toFixed(1)}%
            </p>
          </Card>
          <Card className="p-4">
            <p className="text-sm text-gray-500">Total Credits</p>
            <p className="text-2xl font-bold">
              {workerStats.totalCreditsEarned}
            </p>
          </Card>
          <Card className="p-4">
            <p className="text-sm text-gray-500">Uptime</p>
            <p className="text-2xl font-bold">
              {workerStats.uptimePercentage.toFixed(1)}%
            </p>
          </Card>
        </div>
      )}

      {/* Time Range Selector */}
      <div className="flex justify-end">
        <Select
          value={timeRange}
          onValueChange={(value: "24h" | "7d" | "30d" | "all") =>
            setTimeRange(value)
          }
        >
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="Select time range" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="24h">Last 24 Hours</SelectItem>
            <SelectItem value="7d">Last 7 Days</SelectItem>
            <SelectItem value="30d">Last 30 Days</SelectItem>
            <SelectItem value="all">All Time</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <Tabs defaultValue="efficiency">
        <TabsList>
          <TabsTrigger value="efficiency">Resource Efficiency</TabsTrigger>
          <TabsTrigger value="jobs">Job Performance</TabsTrigger>
          <TabsTrigger value="credits">Credit History</TabsTrigger>
        </TabsList>

        <TabsContent value="efficiency" className="mt-6">
          <Card className="p-4">
            <h3 className="text-lg font-semibold mb-4">
              Resource Efficiency Over Time
            </h3>
            <div className="h-[400px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={performanceMetrics}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="timestamp"
                    tickFormatter={(timestamp) =>
                      new Date(timestamp).toLocaleDateString()
                    }
                  />
                  <YAxis domain={[0, 100]} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="cpuEfficiency"
                    stroke="#3b82f6"
                    name="CPU"
                  />
                  <Line
                    type="monotone"
                    dataKey="gpuEfficiency"
                    stroke="#ef4444"
                    name="GPU"
                  />
                  <Line
                    type="monotone"
                    dataKey="memoryEfficiency"
                    stroke="#10b981"
                    name="Memory"
                  />
                  <Line
                    type="monotone"
                    dataKey="networkEfficiency"
                    stroke="#8b5cf6"
                    name="Network"
                  />
                  <Line
                    type="monotone"
                    dataKey="overallScore"
                    stroke="#f59e0b"
                    name="Overall"
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="jobs" className="mt-6">
          <Card className="p-4">
            <h3 className="text-lg font-semibold mb-4">
              Job Performance Metrics
            </h3>
            <div className="h-[400px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={jobMetrics}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="timestamp"
                    tickFormatter={(timestamp) =>
                      new Date(timestamp).toLocaleDateString()
                    }
                  />
                  <YAxis yAxisId="left" domain={[0, 100]} />
                  <YAxis
                    yAxisId="right"
                    orientation="right"
                    domain={[0, "auto"]}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Bar
                    yAxisId="left"
                    dataKey="successRate"
                    name="Success Rate"
                    fill="#10b981"
                  />
                  <Bar
                    yAxisId="right"
                    dataKey="processingTime"
                    name="Processing Time (ms)"
                    fill="#3b82f6"
                  />
                  <Bar
                    yAxisId="left"
                    dataKey="resourceEfficiency"
                    name="Resource Efficiency"
                    fill="#8b5cf6"
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>

          {/* Job Type Distribution */}
          <Card className="p-4 mt-4">
            <h3 className="text-lg font-semibold mb-4">Job Distribution</h3>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={jobMetrics}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="timestamp"
                    tickFormatter={(timestamp) =>
                      new Date(timestamp).toLocaleDateString()
                    }
                  />
                  <YAxis />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="training"
                    stackId="1"
                    stroke="#3b82f6"
                    fill="#3b82f6"
                    name="Training"
                  />
                  <Area
                    type="monotone"
                    dataKey="inference"
                    stackId="1"
                    stroke="#10b981"
                    fill="#10b981"
                    name="Inference"
                  />
                  <Area
                    type="monotone"
                    dataKey="annotation"
                    stackId="1"
                    stroke="#8b5cf6"
                    fill="#8b5cf6"
                    name="Annotation"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="credits" className="mt-6">
          <Card className="p-4">
            <h3 className="text-lg font-semibold mb-4">
              Credit Earnings History
            </h3>
            <div className="h-[400px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={jobMetrics}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="timestamp"
                    tickFormatter={(timestamp) =>
                      new Date(timestamp).toLocaleDateString()
                    }
                  />
                  <YAxis />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="credits"
                    stroke="#f59e0b"
                    fill="#f59e0b"
                    name="Credits Earned"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
