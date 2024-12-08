export interface WorkerStatus {
  available: boolean;
  credits?: number;
  currentLoad?: {
    cpu: number;
    memory: number;
    gpu?: number;
  };
}

export interface ResourceMetrics {
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

export interface JobStats {
  training: {
    totalJobs: number;
    activeJobs: number;
    activeWorkers: number;
    totalResourcesUsed: {
      cpu: number;
      gpu: number;
      memory: number;
    };
    earnings: {
      last24h: number;
      allTime: number;
    };
    averageCompletionTime: number;
    successRate: number;
  };
  inference: {
    totalJobs: number;
    activeJobs: number;
    activeWorkers: number;
    totalResourcesUsed: {
      cpu: number;
      gpu: number;
      memory: number;
    };
    earnings: {
      last24h: number;
      allTime: number;
    };
    throughput: number;
    averageLatency: number;
  };
  annotation: {
    totalJobs: number;
    activeJobs: number;
    availableJobs: number;
    activeWorkers: number;
    completionRate: number;
    averageAccuracy: number;
    earnings: {
      last24h: number;
      allTime: number;
    };
  };
}

export interface JobFilters {
  minReward: number;
  maxReward: number | null;
  dataType: string[];
  annotationType: string[];
  deadline: Date | null;
}

export interface JobFiltersProps {
  filters: JobFilters;
  onFiltersChange: (filters: JobFilters) => void;
  onClose: () => void;
}
