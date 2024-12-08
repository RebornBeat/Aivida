import { Card, Button } from "@/components/ui";
import { Clock, Tag, FileText } from "lucide-react";
import type { AnnotationJob } from "@/types";

interface AnnotationJobCardProps {
  job: AnnotationJob;
  onAccept: (jobId: string) => Promise<void>;
}

export function AnnotationJobCard({ job, onAccept }: AnnotationJobCardProps) {
  const handleAccept = async () => {
    // Show confirmation dialog
    if (window.confirm("Are you sure you want to accept this job?")) {
      await onAccept(job.id);
    }
  };

  const timeLeft = job.deadline
    ? new Date(job.deadline).getTime() - Date.now()
    : null;
  const daysLeft = timeLeft
    ? Math.ceil(timeLeft / (1000 * 60 * 60 * 24))
    : null;

  return (
    <Card className="p-4 hover:shadow-lg transition-shadow">
      <div className="space-y-4">
        <div>
          <h3 className="font-semibold text-lg">{job.title}</h3>
          <p className="text-sm text-gray-500 mt-1">{job.description}</p>
        </div>

        <div className="grid grid-cols-2 gap-4 text-sm">
          <div className="flex items-center">
            <Tag className="h-4 w-4 mr-2 text-gray-400" />
            <span>{job.reward} credits</span>
          </div>

          <div className="flex items-center">
            <FileText className="h-4 w-4 mr-2 text-gray-400" />
            <span>{job.totalItems} items</span>
          </div>

          {daysLeft && (
            <div className="flex items-center col-span-2">
              <Clock className="h-4 w-4 mr-2 text-gray-400" />
              <span className={daysLeft < 3 ? "text-red-500" : "text-gray-600"}>
                {daysLeft} days left
              </span>
            </div>
          )}
        </div>

        <div className="flex flex-wrap gap-2">
          {job.tags.map((tag) => (
            <span
              key={tag}
              className="px-2 py-1 bg-gray-100 rounded-full text-xs text-gray-600"
            >
              {tag}
            </span>
          ))}
        </div>

        <Button onClick={handleAccept} className="w-full">
          Accept Job
        </Button>
      </div>
    </Card>
  );
}
