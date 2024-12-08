import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  Bell,
  AlertCircle,
  Zap,
  Shield,
  Clock,
  Settings,
  Radio,
  Cpu,
  DollarSign,
  Loader2,
  ChevronDown,
  Check
} from "lucide-react";
import { z } from "zod";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";

interface NotificationSettingsProps {
  onError: (error: string) => void;
  onSuccess: (message: string) => void;
}

interface NotificationChannel {
  type: 'system' | 'email';
  enabled: boolean;
  status: 'active' | 'inactive';
}

const notificationSchema = z.object({
  channels: z.object({
    system: z.boolean(),
    email: z.boolean(),
  }),
  preferences: z.object({
    jobUpdates: z.boolean(),
    securityAlerts: z.boolean(),
    performanceAlerts: z.boolean(),
    workerStatus: z.boolean(),
    systemUpdates: z.boolean(),
    earnings: z.boolean(),
  }),
  thresholds: z.object({
    performanceThreshold: z.number().min(0).max(100),
    earningsThreshold: z.number().min(0),
    resourceThreshold: z.number().min(0).max(100),
  }),
  schedule: z.object({
    quietHoursStart: z.string(),
    quietHoursEnd: z.string(),
    timezone: z.string(),
  }),
});

type NotificationPreference = {
  id: string;
  title: string;
  description: string;
  icon: React.ReactNode;
};

export function NotificationSettings({ onError, onSuccess }: NotificationSettingsProps) {
  const [channels, setChannels] = useState<NotificationChannel[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isTesting, setIsTesting] = useState(false);
  const [showThresholdSelect, setShowThresholdSelect] = useState("");
  const [showTimezoneSelect, setShowTimezoneSelect] = useState(false);

  const form = useForm<z.infer<typeof notificationSchema>>({
    resolver: zodResolver(notificationSchema),
    defaultValues: {
      channels: {
        system: true,
        email: false,
      },
      preferences: {
        jobUpdates: true,
        securityAlerts: true,
        performanceAlerts: true,
        workerStatus: true,
        systemUpdates: true,
        earnings: true,
      },
      thresholds: {
        performanceThreshold: 80,
        earningsThreshold: 100,
        resourceThreshold: 90,
      },
      schedule: {
        quietHoursStart: "22:00",
        quietHoursEnd: "07:00",
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      },
    },
  });

  const notificationPreferences: NotificationPreference[] = [
    {
      id: 'jobUpdates',
      title: 'Job Updates',
      description: 'Notifications about job status changes and completions',
      icon: <Clock className="h-5 w-5 text-blue-500" />,
    },
    {
      id: 'securityAlerts',
      title: 'Security Alerts',
      description: 'Important security-related notifications',
      icon: <Shield className="h-5 w-5 text-danger" />,
    },
    {
      id: 'performanceAlerts',
      title: 'Performance Alerts',
      description: 'Alerts about system performance issues',
      icon: <Zap className="h-5 w-5 text-warning" />,
    },
    {
      id: 'workerStatus',
      title: 'Worker Status',
      description: 'Updates about worker availability and status changes',
      icon: <Radio className="h-5 w-5 text-success" />,
    },
    {
      id: 'systemUpdates',
      title: 'System Updates',
      description: 'Important system updates and maintenance notifications',
      icon: <Settings className="h-5 w-5 text-purple-500" />,
    },
    {
      id: 'earnings',
      title: 'Earnings',
      description: 'Notifications about credit earnings and payouts',
      icon: <DollarSign className="h-5 w-5 text-emerald-500" />,
    },
  ];

  useEffect(() => {
    fetchNotificationSettings();
  }, []);

  const fetchNotificationSettings = async () => {
    try {
      setIsLoading(true);
      const [settings, channelStatus] = await Promise.all([
        invoke('get_notification_settings'),
        invoke('get_notification_channels'),
      ]);

      form.reset(settings as z.infer<typeof notificationSchema>);
      setChannels(channelStatus as NotificationChannel[]);
    } catch (err) {
      onError('Failed to fetch notification settings');
      console.error('Notification settings error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const testNotifications = async () => {
    try {
      setIsTesting(true);
      await invoke('test_notifications');
      onSuccess('Test notification sent successfully');
    } catch (err) {
      onError('Failed to send test notification');
    } finally {
      setIsTesting(false);
    }
  };

  const onSubmit = async (data: z.infer<typeof notificationSchema>) => {
    try {
      await invoke('update_notification_settings', { settings: data });
      onSuccess('Notification settings updated successfully');
    } catch (err) {
      onError('Failed to update notification settings');
    }
  };

  if (isLoading) {
    return (
      <div className="flex justify-center py-8">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Notification Channels */}
      <div className="rounded-lg border border-border bg-background p-6 shadow-sm">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-lg font-semibold">Notification Channels</h3>
            <p className="text-secondary text-sm">
              Configure how you receive notifications
            </p>
          </div>
          <button
            onClick={testNotifications}
            disabled={isTesting}
            className="inline-flex items-center justify-center rounded-md border border-border bg-background px-4 py-2 text-sm font-medium transition-colors hover:bg-secondary-hover disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Bell className="mr-2 h-4 w-4" />
            Test Notifications
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {channels.map((channel) => (
            <div
              key={channel.type}
              className="flex items-center justify-between p-4 rounded-lg border border-border"
            >
              <div className="flex items-center space-x-3">
                <div className={`p-2 rounded-full ${
                  channel.status === 'active' ? 'bg-success-light' : 'bg-secondary'
                }`}>
                  {channel.type === 'system' ? (
                    <Bell className={`h-5 w-5 ${
                      channel.status === 'active' ? 'text-success' : 'text-secondary'
                    }`} />
                  ) : (
                    <AlertCircle className={`h-5 w-5 ${
                      channel.status === 'active' ? 'text-success' : 'text-secondary'
                    }`} />
                  )}
                </div>
                <div>
                  <p className="font-medium capitalize">{channel.type}</p>
                  <p className="text-secondary text-sm capitalize">{channel.status}</p>
                </div>
              </div>

              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  className="sr-only peer"
                  checked={channel.enabled}
                  onChange={async (e) => {
                    try {
                      await invoke('toggle_notification_channel', {
                        channel: channel.type,
                        enabled: e.target.checked,
                      });
                      setChannels(channels.map(c =>
                        c.type === channel.type
                          ? { ...c, enabled: e.target.checked }
                          : c
                      ));
                    } catch (err) {
                      onError(`Failed to toggle ${channel.type} notifications`);
                    }
                  }}
                />
                <div className="w-11 h-6 bg-secondary peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-border after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary" />
              </label>
            </div>
          ))}
        </div>
      </div>

      {/* Notification Preferences */}
      <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
        <div className="rounded-lg border border-border bg-background p-6 shadow-sm">
          <h3 className="text-lg font-semibold mb-6">Notification Preferences</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {notificationPreferences.map((pref) => (
              <div key={pref.id} className="flex items-center justify-between rounded-lg border border-border p-4">
                <div className="flex items-center space-x-3">
                  {pref.icon}
                  <div>
                    <p className="font-medium">{pref.title}</p>
                    <p className="text-sm text-secondary">{pref.description}</p>
                  </div>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    className="sr-only peer"
                    {...form.register(`preferences.${pref.id as keyof typeof form.getValues().preferences}`)}
                  />
                  <div className="w-11 h-6 bg-secondary peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-border after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary" />
                </label>
              </div>
            ))}
          </div>
        </div>

        {/* Thresholds */}
        <div className="rounded-lg border border-border bg-background p-6 shadow-sm">
          <h4 className="text-sm font-medium mb-4">Alert Thresholds</h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Repeatable select pattern for thresholds */}
            {['performanceThreshold', 'earningsThreshold', 'resourceThreshold'].map((threshold) => (
              <div key={threshold} className="space-y-2">
                <label className="text-sm font-medium">
                  {threshold === 'performanceThreshold' ? 'Performance Threshold (%)' :
                   threshold === 'earningsThreshold' ? 'Earnings Threshold (credits)' :
                   'Resource Usage Threshold (%)'}
                </label>
                <div className="relative">
                  <button
                    type="button"
                    onClick={() => setShowThresholdSelect(showThresholdSelect === threshold ? '' : threshold)}
                    className="flex h-10 w-full items-center justify-between rounded-md border border-border bg-background px-3 py-2 text-sm"
                  >
                    <span>{form.getValues(`thresholds.${threshold}`)}</span>
                    <ChevronDown className="h-4 w-4 opacity-50" />
                  </button>
                  {showThresholdSelect === threshold && (
                    <div className="absolute mt-1 w-full z-10 rounded-md border border-border bg-background shadow-lg">
                      <div className="p-1">
                        {(threshold === 'earningsThreshold' ? [50, 100, 200, 500, 1000] :
                          threshold === 'performanceThreshold' ? [60, 70, 80, 90, 95] :
                          [70, 80, 85, 90, 95]).map((value) => (
                          <button
                            key={value}
                            type="button"
                            className="flex w-full items-center px-2 py-1.5 text-sm hover:bg-secondary rounded-sm"
                            onClick={() => {
                              form.setValue(`thresholds.${threshold}`, value);
                              setShowThresholdSelect('');
                            }}
                          >
                            {value}{threshold === 'earningsThreshold' ? ' credits' : '%'}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Quiet Hours */}
        <div className="rounded-lg border border-border bg-background p-6 shadow-sm">
          <h4 className="text-sm font-medium mb-4">Quiet Hours</h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Start Time</label>
              <input
                type="time"
                className="flex h-10 w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                {...form.register('schedule.quietHoursStart')}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">End Time</label>
              <input
                type="time"
                className="flex h-10 w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                {...form.register('schedule.quietHoursEnd')}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Timezone</label>
              <div className="relative">
                <button
                  type="button"
                  onClick={() => setShowTimezoneSelect(!showTimezoneSelect)}
                  className="flex h-10 w-full items-center justify-between rounded-md border border-border bg-background px-3 py-2 text-sm"
                >
                  <span>{form.getValues('schedule.timezone')}</span>
                  <ChevronDown className="h-4 w-4 opacity-50" />
                </button>
                {showTimezoneSelect && (
                  <div className="absolute mt-1 w-full z-10 rounded-md border border-border bg-background shadow-lg max-h-60 overflow-auto">
                    <div className="p-1">
                      {Intl.supportedValuesOf('timeZone').map((tz) => (
                        <button
                          key={tz}
                          type="button"
                          className="flex w-full items-center px-2 py-1.5 text-sm hover:bg-secondary rounded-sm"
                          onClick={() => {
                            form.setValue('schedule.timezone', tz);
                            setShowTimezoneSelect(false);
                          }}
                        >
                          {tz}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        <button
          type="submit"
          className="w-full bg-primary text-white rounded-md px-4 py-2 text-sm font-medium hover:bg-primary-dark transition-colors"
        >
          Save Notification Settings
        </button>
      </form>
    </div>
  );
}
