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
  Input,
  Badge,
  Progress,
} from "@/components/ui";
import { formatBytes, formatDuration } from "@/utils/format";
import { Search, SlidersHorizontal } from "lucide-react";

interface BaseJobHistory {
  id: string;
  startTime: Date;
  endTime?: Date;
  status: "completed" | "failed" | "processing";
  resourceUsage: {
    cpu: number;
    memory: number;
    gpu?: number;
  };
  credits: number;
  errorMessage?: string;
}

interface TrainingJobHistory extends BaseJobHistory {
  type: "training";
  totalSamples: number;
  processedSamples: number;
  modelType: string;
  batchSize: number;
  learningRate: number;
  epochs: number;
  currentEpoch?: number;
}

interface InferenceJobHistory extends BaseJobHistory {
  type: "inference";
  totalSequences: number;
  processedSequences: number;
  modelName: string;
  batchSize: number;
  averageLatency: number;
  throughput: number;
}

interface AnnotationJobHistory extends BaseJobHistory {
  type: "annotation";
  totalItems: number;
  annotatedItems: number;
  dataType: string;
  accuracy: number;
  requester: string;
}

type JobHistoryItem =
  | TrainingJobHistory
  | InferenceJobHistory
  | AnnotationJobHistory;

export function JobHistory() {
  const [jobs, setJobs] = useState<JobHistoryItem[]>([]);
  const [filteredJobs, setFilteredJobs] = useState<JobHistoryItem[]>([]);
  const [selectedType, setSelectedType] = useState<
    "all" | "training" | "inference" | "annotation"
  >("all");
  const [timeRange, setTimeRange] = useState<"24h" | "7d" | "30d" | "all">(
    "7d",
  );
  const [searchQuery, setSearchQuery] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<"date" | "credits" | "duration">("date");

  useEffect(() => {
    const fetchJobs = async () => {
      try {
        setIsLoading(true);
        const jobHistory = (await invoke("get_job_history", {
          timeRange,
          jobType: selectedType,
        })) as JobHistoryItem[];
        setJobs(jobHistory);
        applyFilters(jobHistory);
      } catch (err) {
        setError("Failed to fetch job history");
        console.error("Job history error:", err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchJobs();
  }, [timeRange, selectedType]);

  const applyFilters = (jobList: JobHistoryItem[]) => {
    let filtered = [...jobList];

    // Apply job type filter
    if (selectedType !== "all") {
      filtered = filtered.filter((job) => job.type === selectedType);
    }

    // Apply search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter((job) => {
        if (job.type === "training") {
          return job.modelType.toLowerCase().includes(query);
        } else if (job.type === "inference") {
          return job.modelName.toLowerCase().includes(query);
        } else {
          return (
            job.dataType.toLowerCase().includes(query) ||
            job.requester.toLowerCase().includes(query)
          );
        }
      });
    }

    // Apply sorting
    filtered.sort((a, b) => {
      switch (sortBy) {
        case "date":
          return (
            new Date(b.startTime).getTime() - new Date(a.startTime).getTime()
          );
        case "credits":
          return b.credits - a.credits;
        case "duration":
          const aDuration = a.endTime
            ? new Date(a.endTime).getTime() - new Date(a.startTime).getTime()
            : 0;
          const bDuration = b.endTime
            ? new Date(b.endTime).getTime() - new Date(b.startTime).getTime()
            : 0;
          return bDuration - aDuration;
        default:
          return 0;
      }
    });

    setFilteredJobs(filtered);
  };

  useEffect(() => {
    applyFilters(jobs);
  }, [searchQuery, sortBy]);

  const renderJobDetails = (job: JobHistoryItem) => {
    const statusColors = {
      completed: "bg-green-100 text-green-800",
      failed: "bg-red-100 text-red-800",
      processing: "bg-blue-100 text-blue-800",
    };

    const baseDetails = (
      <>
        <div className="flex justify-between items-start mb-4">
          <div>
            <h3 className="font-medium">{job.id.slice(0, 8)}</h3>
            <p className="text-sm text-gray-500">
              Started: {new Date(job.startTime).toLocaleString()}
            </p>
            {job.endTime && (
              <p className="text-sm text-gray-500">
                Duration:{" "}
                {formatDuration(
                  new Date(job.endTime).getTime() -
                    new Date(job.startTime).getTime(),
                )}
              </p>
            )}
          </div>
          <Badge className={statusColors[job.status]}>{job.status}</Badge>
        </div>

        <div className="grid grid-cols-3 gap-2 mb-4 text-sm">
          <div>
            <p className="text-gray-500">CPU Usage</p>
            <p>{job.resourceUsage.cpu}%</p>
          </div>
          <div>
            <p className="text-gray-500">Memory</p>
            <p>{formatBytes(job.resourceUsage.memory)}</p>
          </div>
          {job.resourceUsage.gpu && (
            <div>
              <p className="text-gray-500">GPU Usage</p>
              <p>{job.resourceUsage.gpu}%</p>
            </div>
          )}
        </div>

        <div className="flex justify-between items-center text-sm">
          <span className="text-gray-500">Credits Earned</span>
          <span className="font-medium text-green-600">{job.credits}</span>
        </div>
      </>
    );

    switch (job.type) {
      case "training":
        return (
          <Card className="p-4">
            {baseDetails}
            <div className="mt-4 space-y-2">
              <div className="flex justify-between text-sm">
                <span>Progress</span>
                <span>
                  {job.processedSamples} / {job.totalSamples} samples
                </span>
              </div>
              <Progress
                value={(job.processedSamples / job.totalSamples) * 100}
              />
              <div className="grid grid-cols-2 gap-4 mt-2 text-sm">
                <div>
                  <p className="text-gray-500">Model Type</p>
                  <p>{job.modelType}</p>
                </div>
                <div>
                  <p className="text-gray-500">Current Epoch</p>
                  <p>
                    {job.currentEpoch || 0} / {job.epochs}
                  </p>
                </div>
                <div>
                  <p className="text-gray-500">Batch Size</p>
                  <p>{job.batchSize}</p>
                </div>
                <div>
                  <p className="text-gray-500">Learning Rate</p>
                  <p>{job.learningRate}</p>
                </div>
              </div>
            </div>
          </Card>
        );

      case "inference":
        return (
          <Card className="p-4">
            {baseDetails}
            <div className="mt-4 space-y-2">
              <div className="flex justify-between text-sm">
                <span>Progress</span>
                <span>
                  {job.processedSequences} / {job.totalSequences} sequences
                </span>
              </div>
              <Progress
                value={(job.processedSequences / job.totalSequences) * 100}
              />
              <div className="grid grid-cols-2 gap-4 mt-2 text-sm">
                <div>
                  <p className="text-gray-500">Model Name</p>
                  <p>{job.modelName}</p>
                </div>
                <div>
                  <p className="text-gray-500">Batch Size</p>
                  <p>{job.batchSize}</p>
                </div>
                <div>
                  <p className="text-gray-500">Avg. Latency</p>
                  <p>{job.averageLatency.toFixed(2)}ms</p>
                </div>
                <div>
                  <p className="text-gray-500">Throughput</p>
                  <p>{job.throughput.toFixed(2)}/s</p>
                </div>
              </div>
            </div>
          </Card>
        );

      case "annotation":
        return (
          <Card className="p-4">
            {baseDetails}
            <div className="mt-4 space-y-2">
              <div className="flex justify-between text-sm">
                <span>Progress</span>
                <span>
                  {job.annotatedItems} / {job.totalItems} items
                </span>
              </div>
              <Progress value={(job.annotatedItems / job.totalItems) * 100} />
              <div className="grid grid-cols-2 gap-4 mt-2 text-sm">
                <div>
                  <p className="text-gray-500">Data Type</p>
                  <p>{job.dataType}</p>
                </div>
                <div>
                  <p className="text-gray-500">Requester</p>
                  <p>{job.requester}</p>
                </div>
                <div>
                  <p className="text-gray-500">Accuracy</p>
                  <p>{job.accuracy}%</p>
                </div>
              </div>
            </div>
          </Card>
        );
    }
  };

  return (
    <div className="space-y-6">
      {/* Filters and Controls */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
          <Input
            type="search"
            placeholder="Search jobs..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-9"
          />
        </div>
        <div className="flex gap-2">
          <Select
            value={timeRange}
            onValueChange={(value: "24h" | "7d" | "30d" | "all") =>
              setTimeRange(value)
            }
          >
            <SelectTrigger className="w-[140px]">
              <SelectValue placeholder="Time range" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="24h">Last 24 Hours</SelectItem>
              <SelectItem value="7d">Last 7 Days</SelectItem>
              <SelectItem value="30d">Last 30 Days</SelectItem>
              <SelectItem value="all">All Time</SelectItem>
            </SelectContent>
          </Select>

          <Select
            value={sortBy}
            onValueChange={(value: "date" | "credits" | "duration") =>
              setSortBy(value)
            }
          >
            <SelectTrigger className="w-[140px]">
              <SelectValue placeholder="Sort by" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="date">Date</SelectItem>
              <SelectItem value="credits">Credits</SelectItem>
              <SelectItem value="duration">Duration</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Job Type Tabs */}
      <Tabs
        defaultValue="all"
        onValueChange={(value) => setSelectedType(value as typeof selectedType)}
      >
        <TabsList>
          <TabsTrigger value="all">All Jobs</TabsTrigger>
          <TabsTrigger value="training">Training</TabsTrigger>
          <TabsTrigger value="inference">Inference</TabsTrigger>
          <TabsTrigger value="annotation">Annotation</TabsTrigger>
        </TabsList>

        <TabsContent value="all" className="mt-6">
          <div className="space-y-4">
            {isLoading ? (
              <div className="flex justify-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
              </div>
            ) : filteredJobs.length > 0 ? (
              filteredJobs.map((job) => (
                <div key={job.id}>{renderJobDetails(job)}</div>
              ))
            ) : (
              <div className="text-center py-8 text-gray-500">
                No jobs found matching your criteria
              </div>
            )}
          </div>
        </TabsContent>

        {/* Individual job type tabs will automatically show filtered content */}
        <TabsContent value="training" className="mt-6">
          <div className="space-y-4">
            {filteredJobs
              .filter((job) => job.type === "training")
              .map((job) => (
                <div key={job.id}>{renderJobDetails(job)}</div>
              ))}
          </div>
        </TabsContent>

        <TabsContent value="inference" className="mt-6">
          <div className="space-y-4">
            {filteredJobs
              .filter((job) => job.type === "inference")
              .map((job) => (
                <div key={job.id}>{renderJobDetails(job)}</div>
              ))}
          </div>
        </TabsContent>

        <TabsContent value="annotation" className="mt-6">
          <div className="space-y-4">
            {filteredJobs
              .filter((job) => job.type === "annotation")
              .map((job) => (
                <div key={job.id}>{renderJobDetails(job)}</div>
              ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
