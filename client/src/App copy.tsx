import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Routes, Route, useNavigate } from "react-router-dom";
import { Layout, Button, Card, Input, Alert, Tabs, Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, Form, FormField, FormItem, FormLabel, FormControl, Select, SelectTrigger, SelectValue, SelectContent, SelectItem, Progress, AlertTitle, AlertDescription,} from "@/components/ui";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Area, AreaChart } from 'recharts';
import {
  ChartLine,
  Users,
  Briefcase,
  Cpu,
  ActivitySquare,
  Settings,
  LogOut,
  Loader2,
} from "lucide-react";

// Types
interface ResourceMetrics {
  cpuUtilization: number;
  gpuUtilization: number;
  memoryUsed: number;
  networkBandwidth: {
    upload: number;
    download: number;
  };
}

interface WorkerInfo {
  id: string;
  status: "available" | "busy" | "offline";
  metrics: ResourceMetrics;
  capabilities: {
    gpu: Array<{ model: string; memory: number }>;
    cpu: { cores: number; threads: number };
  };
}

interface Job {
  id: string;
  title: string;
  requirements: {
    minGpuMemory: number;
    minCpuCores: number;
    minMemory: number;
  };
  status: "pending" | "processing" | "completed" | "failed";
  cost: number;
}

// Main App Component
function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    if (!isAuthenticated) {
      navigate("/login");
    }
  }, [isAuthenticated]);

  return (
    <div className="min-h-screen bg-gray-50">
      <Routes>
        <Route path="/login" element={<AuthPages mode="login" />} />
        <Route path="/signup" element={<AuthPages mode="signup" />} />
        <Route path="/" element={<MainLayout />}>
          <Route index element={<Dashboard />} />
          <Route path="marketplace" element={<JobMarketplace />} />
          <Route path="worker" element={<WorkerDashboard />} />
          <Route path="settings" element={<Settings />} />
        </Route>
      </Routes>
    </div>
  );
}

// Authentication Pages
function AuthPages({ mode }: { mode: "login" | "signup" }) {
  const [formData, setFormData] = useState({
    email: "",
    password: "",
    confirmPassword: "",
  });
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      if (mode === "signup") {
        await invoke("register_user", formData);
      } else {
        await invoke("login_user", formData);
      }
    } catch (err) {
      setError(err.toString());
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center">
      <Card className="w-full max-w-md p-6">
        <h2 className="text-2xl font-bold mb-6">
          {mode === "login" ? "Login" : "Create Account"}
        </h2>
        {error && (
          <Alert variant="destructive" className="mb-4">
            {error}
          </Alert>
        )}
        <form onSubmit={handleSubmit} className="space-y-4">
          <Input
            type="email"
            placeholder="Email"
            value={formData.email}
            onChange={(e) =>
              setFormData((prev) => ({ ...prev, email: e.target.value }))
            }
          />
          <Input
            type="password"
            placeholder="Password"
            value={formData.password}
            onChange={(e) =>
              setFormData((prev) => ({ ...prev, password: e.target.value }))
            }
          />
          {mode === "signup" && (
            <Input
              type="password"
              placeholder="Confirm Password"
              value={formData.confirmPassword}
              onChange={(e) =>
                setFormData((prev) => ({
                  ...prev,
                  confirmPassword: e.target.value,
                }))
              }
            />
          )}
          <Button type="submit" className="w-full">
            {mode === "login" ? "Login" : "Sign Up"}
          </Button>
        </form>
      </Card>
    </div>
  );
}

// Main Layout
function MainLayout() {
  const menuItems = [
    { icon: <ChartLine />, label: "Dashboard", path: "/" },
    { icon: <Briefcase />, label: "Job Marketplace", path: "/marketplace" },
    { icon: <Cpu />, label: "Worker Dashboard", path: "/worker" },
    { icon: <Settings />, label: "Settings", path: "/settings" },
  ];

  return (
    <div className="flex h-screen">
      <aside className="w-64 bg-white border-r border-gray-200 p-4">
        <div className="flex items-center space-x-2 mb-8">
          <img src="/logo.svg" alt="Aivida" className="h-8 w-8" />
          <span className="text-xl font-bold">Aivida</span>
        </div>
        <nav className="space-y-2">
          {menuItems.map((item) => (
            <NavItem key={item.path} {...item} />
          ))}
        </nav>
      </aside>
      <main className="flex-1 overflow-auto p-6">
        <Outlet />
      </main>
    </div>
  );
}

// Dashboard
function Dashboard() {
  const [metrics, setMetrics] = useState<ResourceMetrics | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [metricsData, jobsData] = await Promise.all([
          invoke("get_resources"),
          invoke("get_active_jobs"),
        ]);
        setMetrics(metricsData as ResourceMetrics);
        setJobs(jobsData as Job[]);
      } catch (err) {
        console.error("Failed to fetch dashboard data:", err);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Dashboard</h1>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <MetricCard
          title="CPU Usage"
          value={`${metrics?.cpuUtilization.toFixed(1)}%`}
          icon={<Cpu />}
        />
        <MetricCard
          title="GPU Usage"
          value={`${metrics?.gpuUtilization.toFixed(1)}%`}
          icon={<ActivitySquare />}
        />
        <MetricCard
          title="Memory Used"
          value={formatBytes(metrics?.memoryUsed || 0)}
          icon={<ChartLine />}
        />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="p-4">
          <h3 className="text-lg font-semibold mb-4">Active Jobs</h3>
          <div className="space-y-2">
            {jobs.map((job) => (
              <JobCard key={job.id} job={job} />
            ))}
          </div>
        </Card>

        <Card className="p-4">
          <h3 className="text-lg font-semibold mb-4">System Health</h3>
          <SystemHealth metrics={metrics} />
        </Card>
      </div>
    </div>
  );
}

// Job Marketplace
function JobMarketplace() {
  const [availableJobs, setAvailableJobs] = useState<Job[]>([]);
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    const fetchJobs = async () => {
      try {
        const jobs = await invoke("get_available_jobs");
        setAvailableJobs(jobs as Job[]);
      } catch (err) {
        console.error("Failed to fetch jobs:", err);
      }
    };

    fetchJobs();
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">Job Marketplace</h1>
        <Button
          onClick={() => {
            /* Open job creation modal */
          }}
        >
          Post New Job
        </Button>
      </div>

      <Input
        type="search"
        placeholder="Search jobs..."
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        className="max-w-md"
      />

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {availableJobs
          .filter((job) =>
            job.title.toLowerCase().includes(searchTerm.toLowerCase()),
          )
          .map((job) => (
            <JobListingCard key={job.id} job={job} />
          ))}
      </div>
    </div>
  );
}

// Worker Dashboard
function WorkerDashboard() {
  const [workerInfo, setWorkerInfo] = useState<WorkerInfo | null>(null);
  const [isRegistered, setIsRegistered] = useState(false);

  useEffect(() => {
    const checkRegistration = async () => {
      try {
        const info = await invoke("get_worker_info");
        setWorkerInfo(info as WorkerInfo);
        setIsRegistered(true);
      } catch (err) {
        setIsRegistered(false);
      }
    };

    checkRegistration();
  }, []);

  if (!isRegistered) {
    return <WorkerRegistration />;
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Worker Dashboard</h1>

      <Tabs defaultValue="overview">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="jobs">Active Jobs</TabsTrigger>
          <TabsTrigger value="earnings">Earnings</TabsTrigger>
        </TabsList>

        <TabsContent value="overview">
          <WorkerOverview info={workerInfo} />
        </TabsContent>

        <TabsContent value="jobs">
          <WorkerJobs />
        </TabsContent>

        <TabsContent value="earnings">
          <WorkerEarnings />
        </TabsContent>
      </Tabs>
    </div>
  );
}

// Helper Components

function MetricCard({
  title,
  value,
  icon,
  trend
}: {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  trend?: { value: number; label: string; }
}) {
  return (
    <Card className="p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div className="p-2 bg-primary/10 rounded-lg">
            {icon}
          </div>
          <div>
            <p className="text-sm text-gray-500">{title}</p>
            <h3 className="text-2xl font-bold">{value}</h3>
          </div>
        </div>
        {trend && (
          <div className={`text-sm ${trend.value >= 0 ? 'text-green-500' : 'text-red-500'}`}>
            {trend.value >= 0 ? '↑' : '↓'} {Math.abs(trend.value)}%
            <p className="text-gray-500 text-xs">{trend.label}</p>
          </div>
        )}
      </div>
    </Card>
  );
}

function JobCard({ job }: { job: Job }) {
  const statusColors = {
    pending: 'bg-yellow-100 text-yellow-800',
    processing: 'bg-blue-100 text-blue-800',
    completed: 'bg-green-100 text-green-800',
    failed: 'bg-red-100 text-red-800',
  };

  return (
    <Card className="p-4 hover:shadow-lg transition-shadow">
      <div className="flex justify-between items-start">
        <div>
          <h3 className="font-semibold">{job.title}</h3>
          <p className="text-sm text-gray-500">ID: {job.id.slice(0, 8)}</p>
        </div>
        <span className={`px-2 py-1 rounded-full text-xs ${statusColors[job.status]}`}>
          {job.status}
        </span>
      </div>
      <div className="mt-4 grid grid-cols-3 gap-2 text-sm">
        <div>
          <p className="text-gray-500">GPU Memory</p>
          <p>{job.requirements.minGpuMemory}GB</p>
        </div>
        <div>
          <p className="text-gray-500">CPU Cores</p>
          <p>{job.requirements.minCpuCores}</p>
        </div>
        <div>
          <p className="text-gray-500">Cost</p>
          <p>{job.cost} credits</p>
        </div>
      </div>
    </Card>
  );
}

function SystemHealth({ metrics }: { metrics: ResourceMetrics | null }) {
  if (!metrics) return <div>Loading metrics...</div>;

  const data = [
    { name: 'CPU', value: metrics.cpuUtilization },
    { name: 'GPU', value: metrics.gpuUtilization },
    { name: 'Memory', value: (metrics.memoryUsed / 1024 / 1024 / 1024) },
    { name: 'Network', value: metrics.networkBandwidth.upload / 1024 / 1024 },
  ];

  return (
    <div className="space-y-4">
      {data.map((item) => (
        <div key={item.name} className="space-y-2">
          <div className="flex justify-between">
            <span>{item.name}</span>
            <span>{item.value.toFixed(1)}{item.name === 'Memory' ? 'GB' : '%'}</span>
          </div>
          <Progress value={item.value} />
        </div>
      ))}
    </div>
  );
}

// Resource Monitoring Visualizations

function ResourceMonitor() {
  const [data, setData] = useState<ResourceMetrics[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const metrics = await invoke('get_resource_metrics');
        setData(prev => [...prev, metrics as ResourceMetrics].slice(-30));
        setError(null);
      } catch (err) {
        setError(err.toString());
      } finally {
        setIsLoading(false);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 1000);
    return () => clearInterval(interval);
  }, []);

  if (isLoading) return <LoadingSpinner />;
  if (error) return <ErrorAlert message={error} />;

  return (
    <div className="space-y-6">
      <Card className="p-4">
        <h3 className="text-lg font-semibold mb-4">CPU & GPU Usage</h3>
        <LineChart width={600} height={200} data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="timestamp" />
          <YAxis />
          <Tooltip />
          <Line type="monotone" dataKey="cpuUtilization" stroke="#8884d8" name="CPU" />
          <Line type="monotone" dataKey="gpuUtilization" stroke="#82ca9d" name="GPU" />
        </LineChart>
      </Card>

      <Card className="p-4">
        <h3 className="text-lg font-semibold mb-4">Memory Usage</h3>
        <AreaChart width={600} height={200} data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="timestamp" />
          <YAxis />
          <Tooltip />
          <Area type="monotone" dataKey="memoryUsed" stroke="#8884d8" fill="#8884d8" />
        </AreaChart>
      </Card>
    </div>
  );
}

// Job Submission Modal

function JobSubmissionModal({
  isOpen,
  onClose
}: {
  isOpen: boolean;
  onClose: () => void;
}) {
  const [formData, setFormData] = useState({
    title: '',
    minGpuMemory: 0,
    minCpuCores: 0,
    minMemory: 0,
    securityLevel: 'standard',
    maxCost: 0,
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(null);

    try {
      await invoke('submit_computation_job', { jobData: formData });
      onClose();
    } catch (err) {
      setError(err.toString());
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Submit New Job</DialogTitle>
          <DialogDescription>
            Specify the requirements for your computation job.
          </DialogDescription>
        </DialogHeader>

        {error && (
          <Alert variant="destructive">
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <FormField
            name="title"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Job Title</FormLabel>
                <FormControl>
                  <Input {...field} />
                </FormControl>
              </FormItem>
            )}
          />

          <div className="grid grid-cols-2 gap-4">
            <FormField
              name="minGpuMemory"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Min GPU Memory (GB)</FormLabel>
                  <FormControl>
                    <Input type="number" {...field} />
                  </FormControl>
                </FormItem>
              )}
            />

            <FormField
              name="minCpuCores"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Min CPU Cores</FormLabel>
                  <FormControl>
                    <Input type="number" {...field} />
                  </FormControl>
                </FormItem>
              )}
            />
          </div>

          <FormField
            name="securityLevel"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Security Level</FormLabel>
                <Select
                  onValueChange={field.onChange}
                  defaultValue={field.value}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select security level" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="basic">Basic</SelectItem>
                    <SelectItem value="standard">Standard</SelectItem>
                    <SelectItem value="maximum">Maximum (P2P)</SelectItem>
                  </SelectContent>
                </Select>
              </FormItem>
            )}
          />

          <FormField
            name="maxCost"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Maximum Cost (Credits)</FormLabel>
                <FormControl>
                  <Input type="number" {...field} />
                </FormControl>
              </FormItem>
            )}
          />

          <div className="flex justify-end space-x-2">
            <Button variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" disabled={isSubmitting}>
              {isSubmitting && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Submit Job
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}

// Worker Registration Flow

function WorkerRegistration() {
  const [step, setStep] = useState(1);
  const [formData, setFormData] = useState({
    capabilities: {
      gpu: [],
      cpu: { cores: 0, threads: 0 },
      memory: 0,
    },
    securityLevel: 'standard',
    connectionMode: 'standard',
  });

  const steps = [
    {
      title: 'Hardware Detection',
      component: <HardwareDetection onNext={detectHardware} />,
    },
    {
      title: 'Security Settings',
      component: <SecuritySettings formData={formData} setFormData={setFormData} />,
    },
    {
      title: 'Verification',
      component: <WorkerVerification formData={formData} onSubmit={registerWorker} />,
    },
  ];

  async function detectHardware() {
    try {
      const hardware = await invoke('detect_hardware');
      setFormData(prev => ({ ...prev, capabilities: hardware }));
      setStep(2);
    } catch (err) {
      // Handle error
    }
  }

  async function registerWorker() {
    try {
      await invoke('register_worker', { workerInfo: formData });
      // Handle success
    } catch (err) {
      // Handle error
    }
  }

  return (
    <Card className="max-w-2xl mx-auto p-6">
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <h2 className="text-2xl font-bold">Worker Registration</h2>
          <div className="text-sm text-gray-500">
            Step {step} of {steps.length}
          </div>
        </div>

        <Progress value={(step / steps.length) * 100} />

        {steps[step - 1].component}
      </div>
    </Card>
  );
}

// Error Handling Components

function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center p-4">
      <Loader2 className="h-8 w-8 animate-spin" />
    </div>
  );
}

function ErrorAlert({ message }: { message: string }) {
  return (
    <Alert variant="destructive">
      <AlertTitle>Error</AlertTitle>
      <AlertDescription>{message}</AlertDescription>
    </Alert>
  );
}

export default App;
