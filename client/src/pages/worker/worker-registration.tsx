import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  Card,
  Button,
  Progress,
  Alert,
  AlertTitle,
  AlertDescription,
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui";
import {
  Cpu,
  Gpu,
  HardDrive,
  Network,
  Shield,
  CheckCircle,
  XCircle,
} from "lucide-react";
import { formatBytes } from "@/utils/format";

interface WorkerRegistrationProps {
  onRegistered: () => void;
}

interface HardwareCapabilities {
  cpu: {
    cores: number;
    threads: number;
    architecture: string;
    frequency: number;
  };
  gpu: Array<{
    model: string;
    memory: number;
    computeCapability: number;
  }>;
  memory: number;
  network: {
    bandwidth: number;
  };
}

interface RegistrationStep {
  title: string;
  description: string;
  icon: React.ReactNode;
}

export function WorkerRegistration({ onRegistered }: WorkerRegistrationProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hardwareCapabilities, setHardwareCapabilities] =
    useState<HardwareCapabilities | null>(null);
  const [selectedConfig, setSelectedConfig] = useState({
    securityLevel: "standard" as "basic" | "standard" | "maximum",
    connectionMode: "standard" as "standard" | "p2p",
  });

  const steps: RegistrationStep[] = [
    {
      title: "Hardware Detection",
      description: "Scanning system hardware capabilities",
      icon: <Cpu className="h-6 w-6" />,
    },
    {
      title: "Resource Configuration",
      description: "Configure resource allocation",
      icon: <HardDrive className="h-6 w-6" />,
    },
    {
      title: "Network Setup",
      description: "Configure network settings",
      icon: <Network className="h-6 w-6" />,
    },
    {
      title: "Security Configuration",
      description: "Set security preferences",
      icon: <Shield className="h-6 w-6" />,
    },
  ];

  const detectHardware = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const capabilities = (await invoke(
        "detect_hardware",
      )) as HardwareCapabilities;
      setHardwareCapabilities(capabilities);
      setCurrentStep(1);
    } catch (err) {
      setError("Failed to detect hardware capabilities");
      console.error("Hardware detection error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const completeRegistration = async () => {
    try {
      setIsLoading(true);
      setError(null);

      if (!hardwareCapabilities) {
        throw new Error("Hardware capabilities not detected");
      }

      await invoke("register_worker", {
        capabilities: hardwareCapabilities,
        config: {
          securityLevel: selectedConfig.securityLevel,
          connectionMode: selectedConfig.connectionMode,
        },
      });

      onRegistered();
    } catch (err) {
      setError("Failed to complete worker registration");
      console.error("Registration error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const renderHardwareDetails = () => {
    if (!hardwareCapabilities) return null;

    return (
      <div className="space-y-4">
        {/* CPU Information */}
        <div className="flex items-start space-x-4">
          <Cpu className="h-5 w-5 text-blue-500 mt-1" />
          <div>
            <h4 className="font-medium">CPU</h4>
            <p className="text-sm text-gray-500">
              {hardwareCapabilities.cpu.cores} cores /{" "}
              {hardwareCapabilities.cpu.threads} threads
              <br />
              {hardwareCapabilities.cpu.architecture} @{" "}
              {hardwareCapabilities.cpu.frequency}GHz
            </p>
          </div>
        </div>

        {/* GPU Information */}
        {hardwareCapabilities.gpu.length > 0 && (
          <div className="flex items-start space-x-4">
            <Gpu className="h-5 w-5 text-green-500 mt-1" />
            <div>
              <h4 className="font-medium">GPU</h4>
              {hardwareCapabilities.gpu.map((gpu, index) => (
                <p key={index} className="text-sm text-gray-500">
                  {gpu.model} - {formatBytes(gpu.memory)} VRAM
                  <br />
                  Compute Capability: {gpu.computeCapability}
                </p>
              ))}
            </div>
          </div>
        )}

        {/* Memory Information */}
        <div className="flex items-start space-x-4">
          <HardDrive className="h-5 w-5 text-purple-500 mt-1" />
          <div>
            <h4 className="font-medium">Memory</h4>
            <p className="text-sm text-gray-500">
              {formatBytes(hardwareCapabilities.memory)} Available
            </p>
          </div>
        </div>

        {/* Network Information */}
        <div className="flex items-start space-x-4">
          <Network className="h-5 w-5 text-orange-500 mt-1" />
          <div>
            <h4 className="font-medium">Network</h4>
            <p className="text-sm text-gray-500">
              {formatBytes(hardwareCapabilities.network.bandwidth)}/s Bandwidth
            </p>
          </div>
        </div>
      </div>
    );
  };

  const renderCurrentStep = () => {
    switch (currentStep) {
      case 0:
        return (
          <div className="space-y-4">
            <p className="text-gray-600">
              We'll scan your system to detect available hardware resources.
              This information will be used to optimize job assignments.
            </p>
            <Button
              onClick={detectHardware}
              isLoading={isLoading}
              className="w-full"
            >
              Start Hardware Detection
            </Button>
          </div>
        );

      case 1:
        return (
          <div className="space-y-4">
            <p className="text-gray-600">
              Review detected hardware capabilities and configure resource
              allocation.
            </p>
            {renderHardwareDetails()}
            <Button onClick={() => setCurrentStep(2)} className="w-full">
              Continue
            </Button>
          </div>
        );

      case 2:
        return (
          <div className="space-y-4">
            <p className="text-gray-600">
              Configure how your worker connects to the network.
            </p>
            <Select
              value={selectedConfig.connectionMode}
              onValueChange={(value: "standard" | "p2p") =>
                setSelectedConfig((prev) => ({
                  ...prev,
                  connectionMode: value,
                }))
              }
            >
              <SelectTrigger>
                <SelectValue placeholder="Select connection mode" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="standard">
                  Standard (Server Coordinated)
                </SelectItem>
                <SelectItem value="p2p">P2P (Direct Connection)</SelectItem>
              </SelectContent>
            </Select>
            <Button onClick={() => setCurrentStep(3)} className="w-full">
              Continue
            </Button>
          </div>
        );

      case 3:
        return (
          <div className="space-y-4">
            <p className="text-gray-600">
              Choose your security level. Higher security levels enable
              additional privacy features but may affect performance.
            </p>
            <Select
              value={selectedConfig.securityLevel}
              onValueChange={(value: "basic" | "standard" | "maximum") =>
                setSelectedConfig((prev) => ({ ...prev, securityLevel: value }))
              }
            >
              <SelectTrigger>
                <SelectValue placeholder="Select security level" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="basic">Basic (Minimal Security)</SelectItem>
                <SelectItem value="standard">Standard (Recommended)</SelectItem>
                <SelectItem value="maximum">
                  Maximum (Enhanced Privacy)
                </SelectItem>
              </SelectContent>
            </Select>
            <Button
              onClick={completeRegistration}
              isLoading={isLoading}
              className="w-full"
            >
              Complete Registration
            </Button>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <Card className="max-w-2xl mx-auto p-6">
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <h2 className="text-2xl font-bold">Worker Registration</h2>
          <div className="text-sm text-gray-500">
            Step {currentStep + 1} of {steps.length}
          </div>
        </div>

        <Progress
          value={(currentStep / (steps.length - 1)) * 100}
          className="h-2"
        />

        {error && (
          <Alert variant="destructive">
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {steps.map((step, index) => (
            <div
              key={index}
              className={`flex flex-col items-center p-4 rounded-lg border ${
                index === currentStep
                  ? "border-primary bg-primary/5"
                  : index < currentStep
                    ? "border-green-500 bg-green-50"
                    : "border-gray-200"
              }`}
            >
              {index < currentStep ? (
                <CheckCircle className="h-6 w-6 text-green-500" />
              ) : index === currentStep ? (
                step.icon
              ) : (
                <div className="h-6 w-6 rounded-full border-2 border-gray-300" />
              )}
              <h3 className="mt-2 text-sm font-medium">{step.title}</h3>
              <p className="mt-1 text-xs text-gray-500">{step.description}</p>
            </div>
          ))}
        </div>

        <div className="mt-6">{renderCurrentStep()}</div>
      </div>
    </Card>
  );
}
