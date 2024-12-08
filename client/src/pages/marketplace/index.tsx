import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  Search,
  Filter,
  BarChart,
  Activity,
  Users,
  Clock,
  TrendingUp,
} from "lucide-react";
import { formatBytes, formatDuration } from "../../utils/format";
import { AnnotationJobCard } from "./components/annotation-job-card";
import { JobFilters } from "./components/job-filters";
import { ActiveJobs } from "./components/active-jobs";
import { CompletedJobs } from "./components/completed-jobs";
import { useAuth } from "@/contexts/auth-context";
import type { JobStats, AnnotationJob } from "../../utils/types";

export function JobMarketplace() {
  const { user } = useAuth();
  const [jobs, setJobs] = useState<AnnotationJob[]>([]);
  const [jobStats, setJobStats] = useState<JobStats | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState({
    minReward: 0,
    maxReward: null as number | null,
    dataType: [] as string[],
    annotationType: [] as string[],
    deadline: null as Date | null,
  });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState("overview");

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [availableJobs, stats] = await Promise.all([
          invoke("get_available_annotation_jobs", { filters }),
          invoke("get_job_statistics"),
        ]);
        setJobs(availableJobs as AnnotationJob[]);
        setJobStats(stats as JobStats);
        setError(null);
      } catch (err) {
        setError("Failed to fetch marketplace data");
        console.error("Marketplace data fetch error:", err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, [filters]);

  const filteredJobs = jobs.filter(
    (job) =>
      job.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      job.description.toLowerCase().includes(searchTerm.toLowerCase()),
  );

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-[500px]">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-primary">Job Marketplace</h1>
        <div className="flex items-center space-x-2">
          <div className="relative w-64">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-secondary h-4 w-4" />
            <input
              type="search"
              placeholder="Search jobs..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-9 pr-4 py-2 border border-border rounded-md bg-background focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
            />
          </div>
          <button
            onClick={() => setShowFilters(true)}
            className="inline-flex items-center px-4 py-2 border border-border rounded-md bg-background hover:bg-secondary-hover transition-colors"
          >
            <Filter className="h-4 w-4 mr-2" />
            Filters
          </button>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="bg-danger-light border-l-4 border-danger p-4 rounded-md">
          <div className="flex">
            <div className="flex-shrink-0">
              <AlertTriangle className="h-5 w-5 text-danger" />
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-danger">Error</h3>
              <p className="text-sm text-danger mt-1">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* System Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        {jobStats && renderSystemOverview(jobStats)}
      </div>

      {/* Tabs */}
      <div className="border-b border-border">
        <nav className="-mb-px flex space-x-8">
          {["available", "inProgress", "completed"].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`
                py-4 px-1 border-b-2 font-medium text-sm
                ${
                  activeTab === tab
                    ? "border-primary text-primary"
                    : "border-transparent text-secondary hover:text-primary hover:border-secondary"
                }
              `}
            >
              {tab === "available" && "Available Jobs"}
              {tab === "inProgress" && "My Active Jobs"}
              {tab === "completed" && "Completed Jobs"}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="mt-6">
        {activeTab === "available" &&
          (filteredJobs.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredJobs.map((job) => (
                <AnnotationJobCard
                  key={job.id}
                  job={job}
                  onAccept={async (jobId) => {
                    try {
                      await invoke("accept_annotation_job", { jobId });
                      setJobs(jobs.filter((j) => j.id !== jobId));
                    } catch (err) {
                      console.error("Failed to accept job:", err);
                    }
                  }}
                />
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-secondary">
              No jobs available matching your criteria
            </div>
          ))}

        {activeTab === "inProgress" && <ActiveJobs />}
        {activeTab === "completed" && <CompletedJobs />}
      </div>

      {/* Job Filters Dialog */}
      {showFilters && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center">
          <div className="bg-background rounded-lg max-w-lg w-full mx-4">
            <JobFilters
              filters={filters}
              onFiltersChange={setFilters}
              onClose={() => setShowFilters(false)}
            />
          </div>
        </div>
      )}
    </div>
  );
}

function renderSystemOverview(stats: JobStats) {
  const overviewCards = [
    {
      title: "Training Jobs",
      icon: <BarChart className="h-5 w-5 text-info" />,
      stats: [
        {
          label: "Active Jobs",
          value: `${stats.training.activeJobs}/${stats.training.totalJobs}`,
        },
        { label: "Success Rate", value: `${stats.training.successRate}%` },
        { label: "Active Workers", value: stats.training.activeWorkers },
        {
          label: "24h Earnings",
          value: `${stats.training.earnings.last24h} credits`,
          className: "text-success",
        },
      ],
      resources: [
        { label: "CPU", value: `${stats.training.totalResourcesUsed.cpu}%` },
        { label: "GPU", value: `${stats.training.totalResourcesUsed.gpu}%` },
        {
          label: "Memory",
          value: formatBytes(stats.training.totalResourcesUsed.memory),
        },
      ],
    },
    {
      title: "Inference Jobs",
      icon: <Activity className="h-5 w-5 text-accent" />,
      stats: [
        {
          label: "Active Jobs",
          value: `${stats.inference.activeJobs}/${stats.inference.totalJobs}`,
        },
        { label: "Throughput", value: `${stats.inference.throughput}/s` },
        { label: "Avg. Latency", value: `${stats.inference.averageLatency}ms` },
        {
          label: "24h Earnings",
          value: `${stats.inference.earnings.last24h} credits`,
          className: "text-success",
        },
      ],
      resources: [
        { label: "CPU", value: `${stats.inference.totalResourcesUsed.cpu}%` },
        { label: "GPU", value: `${stats.inference.totalResourcesUsed.gpu}%` },
        {
          label: "Memory",
          value: formatBytes(stats.inference.totalResourcesUsed.memory),
        },
      ],
    },
    {
      title: "Annotation Jobs",
      icon: <Users className="h-5 w-5 text-success" />,
      stats: [
        { label: "Available Jobs", value: stats.annotation.availableJobs },
        { label: "Active Jobs", value: stats.annotation.activeJobs },
        { label: "Active Workers", value: stats.annotation.activeWorkers },
        {
          label: "Completion Rate",
          value: `${stats.annotation.completionRate}%`,
        },
        { label: "Accuracy", value: `${stats.annotation.averageAccuracy}%` },
        {
          label: "24h Earnings",
          value: `${stats.annotation.earnings.last24h} credits`,
          className: "text-success",
        },
      ],
    },
  ];

  return overviewCards.map((card, index) => (
    <div
      key={index}
      className="bg-background rounded-lg border border-border p-6 shadow-sm"
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold">{card.title}</h3>
        {card.icon}
      </div>
      <div className="space-y-3">
        {card.stats.map((stat, i) => (
          <div key={i} className="flex justify-between items-center">
            <span className="text-secondary text-sm">{stat.label}</span>
            <span className={`font-medium ${stat.className || ""}`}>
              {stat.value}
            </span>
          </div>
        ))}
        {card.resources && (
          <div className="mt-2 pt-2 border-t border-border">
            <div className="text-sm text-secondary">Resource Usage</div>
            <div className="grid grid-cols-3 gap-2 mt-1">
              {card.resources.map((resource, i) => (
                <div key={i} className="text-center">
                  <div className="text-xs text-secondary">{resource.label}</div>
                  <div className="font-medium">{resource.value}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  ));
}
