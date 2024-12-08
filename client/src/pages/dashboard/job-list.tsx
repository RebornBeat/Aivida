interface BaseJob {
  id: string;
  type: "training" | "inference" | "annotation";
  status: "pending" | "processing" | "completed" | "failed";
  startTime?: Date;
  estimatedCompletion?: Date;
}

interface TrainingJob extends BaseJob {
  type: "training";
  totalSamples: number;
  processedSamples: number;
  resourceUsage: {
    cpu: number;
    memory: number;
    gpu?: number;
  };
}

interface InferenceJob extends BaseJob {
  type: "inference";
  totalSequences: number;
  processedSequences: number;
  resourceUsage: {
    cpu: number;
    memory: number;
    gpu?: number;
  };
}

interface AnnotationJob extends BaseJob {
  type: "annotation";
  totalItems: number;
  annotatedItems: number;
  reward: number;
  deadline?: Date;
}

type Job = TrainingJob | InferenceJob | AnnotationJob;

interface JobListProps {
  jobs: Job[];
}

export function JobList({ jobs }: JobListProps) {
  if (jobs.length === 0) {
    return (
      <div className="text-center py-8 text-secondary">No active jobs</div>
    );
  }

  return (
    <div className="space-y-4">
      {jobs.map((job) => (
        <JobCard key={job.id} job={job} />
      ))}
    </div>
  );
}

function JobCard({ job }: { job: Job }) {
  const statusColors = {
    pending: "bg-yellow-100 text-yellow-800",
    processing: "bg-blue-100 text-blue-800",
    completed: "bg-green-100 text-green-800",
    failed: "bg-red-100 text-red-800",
  };

  const renderProgress = () => {
    switch (job.type) {
      case "training":
        return (
          <>
            <div>
              <p className="text-secondary">Progress</p>
              <p className="font-medium">
                {job.processedSamples} / {job.totalSamples} samples
              </p>
            </div>
            <div>
              <p className="text-secondary">CPU Usage</p>
              <p className="font-medium">{job.resourceUsage.cpu}%</p>
            </div>
            {job.resourceUsage.gpu && (
              <div>
                <p className="text-secondary">GPU Usage</p>
                <p className="font-medium">{job.resourceUsage.gpu}%</p>
              </div>
            )}
          </>
        );
      case "inference":
        return (
          <>
            <div>
              <p className="text-secondary">Progress</p>
              <p className="font-medium">
                {job.processedSequences} / {job.totalSequences} sequences
              </p>
            </div>
            <div>
              <p className="text-secondary">Resource Usage</p>
              <p className="font-medium">{job.resourceUsage.cpu}% CPU</p>
            </div>
          </>
        );
      case "annotation":
        return (
          <>
            <div>
              <p className="text-secondary">Progress</p>
              <p className="font-medium">
                {job.annotatedItems} / {job.totalItems} items
              </p>
            </div>
            <div>
              <p className="text-secondary">Reward</p>
              <p className="font-medium">{job.reward} credits</p>
            </div>
            {job.deadline && (
              <div>
                <p className="text-secondary">Deadline</p>
                <p className="font-medium">
                  {job.deadline.toLocaleDateString()}
                </p>
              </div>
            )}
          </>
        );
    }
  };

  return (
    <div className="rounded-lg border border-border bg-background p-4 hover:shadow-lg transition-shadow">
      <div className="flex justify-between items-start">
        <div>
          <h3 className="font-semibold">Job #{job.id.slice(0, 8)}</h3>
          <p className="text-sm text-secondary">{job.type.toUpperCase()}</p>
        </div>
        <span
          className={`px-2 py-1 rounded-full text-xs ${statusColors[job.status]}`}
        >
          {job.status}
        </span>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-2 text-sm">
        {renderProgress()}
      </div>

      {job.startTime && (
        <div className="mt-2 text-sm text-secondary">
          Started: {job.startTime.toLocaleString()}
        </div>
      )}
      {job.estimatedCompletion && job.status === "processing" && (
        <div className="mt-1 text-sm text-secondary">
          Est. Completion: {job.estimatedCompletion.toLocaleString()}
        </div>
      )}
    </div>
  );
}
