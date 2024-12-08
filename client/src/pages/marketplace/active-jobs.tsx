import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Card, Button, Progress } from "@/components/ui";
import { Clock, Check } from "lucide-react";
import type { AnnotationJob } from "@/types";

export function ActiveJobs() {
  const [jobs, setJobs] = useState<AnnotationJob[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchActiveJobs();
  }, []);

  const fetchActiveJobs = async () => {
    try {
      setIsLoading(true);
      const activeJobs = (await invoke(
        "get_active_annotation_jobs",
      )) as AnnotationJob[];
      setJobs(activeJobs);
    } catch (err) {
      console.error("Failed to fetch active jobs:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmitJob = async (jobId: string) => {
    try {
      await invoke("submit_annotation_job", { jobId });
      // Refresh jobs list
      fetchActiveJobs();
    } catch (err) {
      console.error("Failed to submit job:", err);
    }
  };

  if (isLoading) {
    return (
      <div className="flex justify-center py-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (jobs.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">No active jobs found</div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {jobs.map((job) => (
        <Card key={job.id} className="p-4">
          <div className="space-y-4">
            <div className="flex justify-between items-start">
              <div>
                <h3 className="font-semibold">{job.title}</h3>
                <p className="text-sm text-gray-500">{job.description}</p>
              </div>
              <span className="text-green-600 font-medium">
                {job.reward} credits
              </span>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-gray-500">Progress</span>
                <span>
                  {job.annotatedItems} / {job.totalItems} items
                </span>
              </div>
              <Progress
                value={(job.annotatedItems / job.totalItems) * 100}
                className="h-2"
              />
            </div>

            <div className="flex justify-between items-center text-sm">
              <div className="flex items-center text-gray-500">
                <Clock className="h-4 w-4 mr-1" />
                {new Date(job.deadline).toLocaleDateString()}
              </div>
              <Button
                onClick={() => handleSubmitJob(job.id)}
                disabled={job.annotatedItems < job.totalItems}
              >
                <Check className="h-4 w-4 mr-2" />
                Submit Job
              </Button>
            </div>
          </div>
        </Card>
      ))}
    </div>
  );
}
