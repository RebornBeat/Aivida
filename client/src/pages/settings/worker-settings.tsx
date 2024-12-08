import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  Card,
  Button,
  Switch,
  Slider,
  Form,
  FormField,
  FormItem,
  FormLabel,
  FormControl,
  FormDescription,
  Alert,
  AlertTitle,
  AlertDescription,
  Separator
} from "@/components/ui";
import {
  Cpu,
  Gpu,
  HardDrive,
  Power,
  Battery,
  Gauge,
  Clock,
  Activity
} from "lucide-react";
import { z } from "zod";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { formatBytes } from "@/utils/format";

interface WorkerSettingsProps {
  onError: (error: string) => void;
  onSuccess: (message: string) => void;
}

interface WorkerConfig {
  isAutomatic: boolean;
  cpuLimit: number;
  gpuLimit: number;
  memoryLimit: number;
  maxJobs: number;
  powerMode: 'performance' | 'balanced' | 'efficiency';
  schedulingPreference: 'roundRobin' | 'priorityBased' | 'resourceAware';
  autoSchedule: {
    enabled: boolean;
    startTime: string;
    endTime: string;
    daysOfWeek: number[];
  };
  thermalLimit: number;
  idleTimeout: number;
}

const configSchema = z.object({
  cpuLimit: z.number().min(0).max(100),
  gpuLimit: z.number().min(0).max(100),
  memoryLimit: z.number().min(0).max(100),
  maxJobs: z.number().min(1).max(50),
  powerMode: z.enum(['performance', 'balanced', 'efficiency']),
  schedulingPreference: z.enum(['roundRobin', 'priorityBased', 'resourceAware']),
  thermalLimit: z.number().min(60).max(95),
  idleTimeout: z.number().min(0).max(60),
});

export function WorkerSettings({ onError, onSuccess }: WorkerSettingsProps) {
  const [isRegistered, setIsRegistered] = useState(false);
  const [isActive, setIsActive] = useState(false);
  const [workerConfig, setWorkerConfig] = useState<WorkerConfig | null>(null);
  const [systemCapabilities, setSystemCapabilities] = useState<{
    cpu: { cores: number; threads: number };
    gpu: { memory: number; model: string } | null;
    memory: number;
  } | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const form = useForm<z.infer<typeof configSchema>>({
    resolver: zodResolver(configSchema),
    defaultValues: {
      cpuLimit: 80,
      gpuLimit: 90,
      memoryLimit: 75,
      maxJobs: 5,
      powerMode: 'balanced' as const,
      schedulingPreference: 'resourceAware' as const,
      thermalLimit: 85,
      idleTimeout: 10,
    },
  });

  useEffect(() => {
    checkWorkerStatus();
  }, []);

  const checkWorkerStatus = async () => {
    try {
      setIsLoading(true);
      const [status, config, capabilities] = await Promise.all([
        invoke('get_worker_status'),
        invoke('get_worker_config'),
        invoke('get_system_capabilities'),
      ]);

      setIsRegistered(status as boolean);
      setWorkerConfig(config as WorkerConfig);
      setSystemCapabilities(capabilities as typeof systemCapabilities);

      if (config) {
        form.reset({
          cpuLimit: (config as WorkerConfig).cpuLimit,
          gpuLimit: (config as WorkerConfig).gpuLimit,
          memoryLimit: (config as WorkerConfig).memoryLimit,
          maxJobs: (config as WorkerConfig).maxJobs,
          powerMode: (config as WorkerConfig).powerMode,
          schedulingPreference: (config as WorkerConfig).schedulingPreference,
          thermalLimit: (config as WorkerConfig).thermalLimit,
          idleTimeout: (config as WorkerConfig).idleTimeout,
        });
      }
    } catch (err) {
      onError('Failed to fetch worker status');
      console.error('Worker status error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleWorkerActive = async () => {
    try {
      await invoke('toggle_worker_active');
      setIsActive(!isActive);
      onSuccess(`Worker ${isActive ? 'deactivated' : 'activated'} successfully`);
    } catch (err) {
      onError('Failed to toggle worker status');
    }
  };

  const onSubmit = async (data: z.infer<typeof configSchema>) => {
    try {
      await invoke('update_worker_config', { config: data });
      onSuccess('Worker configuration updated successfully');
    } catch (err) {
      onError('Failed to update worker configuration');
    }
  };

  if (!isRegistered) {
    return (
      <Card className="p-6">
        <Alert>
          <AlertTitle>Not Registered</AlertTitle>
          <AlertDescription>
            You need to register as a worker before configuring worker settings.
            Go to the Worker Dashboard to register.
          </AlertDescription>
        </Alert>
      </Card>
    );
  }

  if (isLoading) {
    return (
      <div className="flex justify-center py-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Worker Status */}
      <Card className="p-6">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <h3 className="text-lg font-semibold">Worker Status</h3>
            <p className="text-sm text-gray-500">
              Enable or disable worker functionality
            </p>
          </div>
          <Switch
            checked={isActive}
            onCheckedChange={toggleWorkerActive}
          />
        </div>
      </Card>

      {/* System Capabilities */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">System Capabilities</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="flex items-center space-x-3">
            <Cpu className="h-5 w-5 text-blue-500" />
            <div>
              <p className="font-medium">CPU</p>
              <p className="text-sm text-gray-500">
                {systemCapabilities?.cpu.cores} cores / {systemCapabilities?.cpu.threads} threads
              </p>
            </div>
          </div>

          {systemCapabilities?.gpu && (
            <div className="flex items-center space-x-3">
              <Gpu className="h-5 w-5 text-green-500" />
              <div>
                <p className="font-medium">GPU</p>
                <p className="text-sm text-gray-500">
                  {systemCapabilities.gpu.model} ({formatBytes(systemCapabilities.gpu.memory)})
                </p>
              </div>
            </div>
          )}

          <div className="flex items-center space-x-3">
            <HardDrive className="h-5 w-5 text-purple-500" />
            <div>
              <p className="font-medium">Memory</p>
              <p className="text-sm text-gray-500">
                {formatBytes(systemCapabilities?.memory || 0)}
              </p>
            </div>
          </div>
        </div>
      </Card>

      {/* Worker Configuration */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">Resource Configuration</h3>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
            {/* Resource Limits */}
            <div className="space-y-4">
              <FormField
                control={form.control}
                name="cpuLimit"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>CPU Usage Limit</FormLabel>
                    <FormControl>
                      <div className="flex items-center space-x-3">
                        <Slider
                          value={[field.value]}
                          onValueChange={([value]) => field.onChange(value)}
                          max={100}
                          step={5}
                          className="flex-1"
                        />
                        <span className="w-12 text-right">{field.value}%</span>
                      </div>
                    </FormControl>
                    <FormDescription>
                      Maximum CPU usage allowed for worker tasks
                    </FormDescription>
                  </FormItem>
                )}
              />

              {systemCapabilities?.gpu && (
                <FormField
                  control={form.control}
                  name="gpuLimit"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>GPU Usage Limit</FormLabel>
                      <FormControl>
                        <div className="flex items-center space-x-3">
                          <Slider
                            value={[field.value]}
                            onValueChange={([value]) => field.onChange(value)}
                            max={100}
                            step={5}
                            className="flex-1"
                          />
                          <span className="w-12 text-right">{field.value}%</span>
                        </div>
                      </FormControl>
                      <FormDescription>
                        Maximum GPU usage allowed for worker tasks
                      </FormDescription>
                    </FormItem>
                  )}
                />
              )}

              <FormField
                control={form.control}
                name="memoryLimit"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Memory Usage Limit</FormLabel>
                    <FormControl>
                      <div className="flex items-center space-x-3">
                        <Slider
                          value={[field.value]}
                          onValueChange={([value]) => field.onChange(value)}
                          max={100}
                          step={5}
                          className="flex-1"
                        />
                        <span className="w-12 text-right">{field.value}%</span>
                      </div>
                    </FormControl>
                    <FormDescription>
                      Maximum memory usage allowed for worker tasks
                    </FormDescription>
                  </FormItem>
                )}
              />
            </div>

            <Separator />

            {/* Performance Settings */}
            <div className="space-y-4">
              <FormField
                control={form.control}
                name="powerMode"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Power Mode</FormLabel>
                    <div className="grid grid-cols-3 gap-4">
                      {['efficiency', 'balanced', 'performance'].map((mode) => (
                        <Card
                          key={mode}
                          className={`p-4 cursor-pointer hover:border-primary transition-colors ${
                            field.value === mode ? 'border-primary bg-primary/5' : ''
                          }`}
                          onClick={() => field.onChange(mode)}
                        >
                          <div className="flex flex-col items-center space-y-2">
                            {mode === 'efficiency' && <Battery className="h-5 w-5 text-green-500" />}
                            {mode === 'balanced' && <Gauge className="h-5 w-5 text-blue-500" />}
                            {mode === 'performance' && <Power className="h-5 w-5 text-red-500" />}
                            <span className="capitalize">{mode}</span>
                          </div>
                        </Card>
                      ))}
                    </div>
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="schedulingPreference"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Scheduling Preference</FormLabel>
                    <div className="grid grid-cols-3 gap-4">
                      {[
                        { value: 'roundRobin', icon: Clock, label: 'Round Robin' },
                        { value: 'priorityBased', icon: Activity, label: 'Priority Based' },
                        { value: 'resourceAware', icon: Gauge, label: 'Resource Aware' },
                      ].map(({ value, icon: Icon, label }) => (
                        <Card
                          key={value}
                          className={`p-4 cursor-pointer hover:border-primary transition-colors ${
                            field.value === value ? 'border-primary bg-primary/5' : ''
                          }`}
                          onClick={() => field.onChange(value)}
                        >
                          <div className="flex flex-col items-center space-y-2">
                            <Icon className="h-5 w-5 text-primary" />
                            <span>{label}</span>
                          </div>
                        </Card>
                      ))}
                    </div>
                  </FormItem>
                )}
              />
            </div>

            <Separator />

            {/* Additional Settings */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <FormField
                control={form.control}
                name="maxJobs"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Maximum Concurrent Jobs</FormLabel>
                    <FormControl>
                      <div className="flex items-center space-x-3">
                        <Slider
                          value={[field.value]}
                          onValueChange={([value]) => field.onChange(value)}
                          min={1}
                          max={50}
                          step={1}
                          className="flex-1"
                        />
                        <span className="w-12 text-right">{field.value}</span>
                      </div>
                    </FormControl>
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="thermalLimit"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Thermal Limit (°C)</FormLabel>
                    <FormControl>
                      <div className="flex items-center space-x-3">
                        <Slider
                          value={[field.value]}
                          onValueChange={([value]) => field.onChange(value)}
                          min={60}
                          max={95}
                          step={1}
                          className="flex-1"
                        />
                        <span className="w-12 text-right">{field.value}°C</span>
                      </div>
                    </FormControl>
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="idleTimeout"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Idle Timeout (minutes)</FormLabel>
                    <FormControl>
                      <div className="flex items-center space-x-3">
                        <Slider
                          value={[field.value]}
                          onValueChange={([value]) => field.onChange(value)}
                          min={0}
                          max={60}
                          step={5}
                          className="flex-1"
                        />
                        <span className="w-12 text-right">{field.value}m</span>
                      </div>
                    </FormControl>
                  </FormItem>
                )}
              />
            </div>

            <Button type="submit" className="w-full">
              Save Configuration
            </Button>
          </form>
        </Form>
      </Card>
    </div>
  );
}
