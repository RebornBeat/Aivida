import type { JobFilters, JobFiltersProps } from "../../utils/types";

export function JobFilters({
  filters,
  onFiltersChange,
  onClose,
}: JobFiltersProps) {
  const dataTypes = ["image", "text", "audio", "video"];
  const annotationTypes = [
    "classification",
    "segmentation",
    "boundingBox",
    "transcription",
  ];

  return (
    <div className="bg-background p-6 rounded-lg w-full max-w-[425px]">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-lg font-semibold">Filter Jobs</h2>
      </div>

      <div className="space-y-4 py-4">
        {/* Reward Range */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-primary">
              Min Reward
            </label>
            <input
              type="number"
              value={filters.minReward}
              onChange={(e) =>
                onFiltersChange({
                  ...filters,
                  minReward: Number(e.target.value),
                })
              }
              className="w-full px-3 py-2 rounded-md border border-border bg-background text-primary focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium text-primary">
              Max Reward
            </label>
            <input
              type="number"
              value={filters.maxReward || ""}
              onChange={(e) =>
                onFiltersChange({
                  ...filters,
                  maxReward: e.target.value ? Number(e.target.value) : null,
                })
              }
              className="w-full px-3 py-2 rounded-md border border-border bg-background text-primary focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
            />
          </div>
        </div>

        {/* Data Type Select */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-primary">Data Type</label>
          <div className="relative">
            <select
              multiple
              value={filters.dataType}
              onChange={(e) => {
                const selected = Array.from(e.target.selectedOptions).map(
                  (option) => option.value,
                );
                onFiltersChange({
                  ...filters,
                  dataType: selected,
                });
              }}
              className="w-full px-3 py-2 rounded-md border border-border bg-background text-primary focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
            >
              {dataTypes.map((type) => (
                <option key={type} value={type} className="py-1">
                  {type.charAt(0).toUpperCase() + type.slice(1)}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Annotation Type Select */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-primary">
            Annotation Type
          </label>
          <div className="relative">
            <select
              multiple
              value={filters.annotationType}
              onChange={(e) => {
                const selected = Array.from(e.target.selectedOptions).map(
                  (option) => option.value,
                );
                onFiltersChange({
                  ...filters,
                  annotationType: selected,
                });
              }}
              className="w-full px-3 py-2 rounded-md border border-border bg-background text-primary focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
            >
              {annotationTypes.map((type) => (
                <option key={type} value={type} className="py-1">
                  {type.charAt(0).toUpperCase() + type.slice(1)}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="flex justify-end space-x-2 mt-6">
        <button
          onClick={() => {
            onFiltersChange({
              minReward: 0,
              maxReward: null,
              dataType: [],
              annotationType: [],
              deadline: null,
            });
            onClose();
          }}
          className="px-4 py-2 rounded-md border border-border bg-background text-primary hover:bg-secondary-hover transition-colors"
        >
          Reset
        </button>
        <button
          onClick={onClose}
          className="px-4 py-2 rounded-md bg-primary text-white hover:bg-primary-hover transition-colors"
        >
          Apply Filters
        </button>
      </div>
    </div>
  );
}
