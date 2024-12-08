import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { CheckCircle, Clock } from "lucide-react";
import type { AnnotationJob } from "@/types";

export function CompletedJobs() {
  const [jobs, setJobs] = useState<AnnotationJob[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchCompletedJobs = async () => {
      try {
        setIsLoading(true);
        const completedJobs = (await invoke(
          "get_completed_annotation_jobs",
        )) as AnnotationJob[];
        setJobs(completedJobs);
      } catch (err) {
        console.error("Failed to fetch completed jobs:", err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchCompletedJobs();
  }, []);

  if (isLoading) {
    return (
      <div className="flex justify-center py-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (jobs.length === 0) {
    return (
      <div className="text-center py-8 text-secondary">
        No completed jobs found
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {jobs.map((job) => (
        <div
          key={job.id}
          className="bg-background rounded-lg border border-border p-4 shadow-sm hover:shadow-md transition-shadow"
        >
          <div className="space-y-4">
            <div className="flex justify-between items-start">
              <div>
                <h3 className="font-semibold text-primary">{job.title}</h3>
                <p className="text-sm text-secondary">{job.description}</p>
              </div>
              <CheckCircle className="h-5 w-5 text-success" />
            </div>

            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <p className="text-secondary">Items Completed</p>
                <p className="font-medium text-primary">{job.totalItems}</p>
              </div>
              <div>
                <p className="text-secondary">Earnings</p>
                <p className="font-medium text-success">{job.reward} credits</p>
              </div>
            </div>

            <div className="flex items-center text-sm text-secondary">
              <Clock className="h-4 w-4 mr-1" />
              Completed on {new Date(job.completedAt).toLocaleDateString()}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
