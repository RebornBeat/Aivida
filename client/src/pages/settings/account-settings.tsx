import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  Card,
  Button,
  Input,
  Form,
  FormField,
  FormItem,
  FormLabel,
  FormControl,
  FormMessage,
  Alert,
  AlertTitle,
  AlertDescription,
  Separator,
} from "@/components/ui";
import { User, Mail, Lock, AlertTriangle } from "lucide-react";
import { useAuth } from "@/contexts/auth-context";
import { z } from "zod";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";

interface AccountSettingsProps {
  onError: (error: string) => void;
  onSuccess: (message: string) => void;
}

const profileSchema = z.object({
  username: z.string().min(3).max(50),
  email: z.string().email(),
});

const passwordSchema = z
  .object({
    currentPassword: z.string().min(1, "Current password is required"),
    newPassword: z
      .string()
      .min(8, "Password must be at least 8 characters")
      .regex(
        /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/,
        "Password must include uppercase, lowercase, number and special character",
      ),
    confirmPassword: z.string(),
  })
  .refine((data) => data.newPassword === data.confirmPassword, {
    message: "Passwords don't match",
    path: ["confirmPassword"],
  });

export function AccountSettings({ onError, onSuccess }: AccountSettingsProps) {
  const { user } = useAuth();
  const [accountInfo, setAccountInfo] = useState<{
    username: string;
    email: string;
    createdAt: string;
    lastLogin: string;
    totalJobsCompleted: number;
    totalCreditsEarned: number;
  } | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isDeletingAccount, setIsDeletingAccount] = useState(false);

  const profileForm = useForm({
    resolver: zodResolver(profileSchema),
    defaultValues: {
      username: "",
      email: "",
    },
  });

  const passwordForm = useForm({
    resolver: zodResolver(passwordSchema),
    defaultValues: {
      currentPassword: "",
      newPassword: "",
      confirmPassword: "",
    },
  });

  useEffect(() => {
    fetchAccountInfo();
  }, []);

  const fetchAccountInfo = async () => {
    try {
      setIsLoading(true);
      const info = (await invoke("get_account_info")) as typeof accountInfo;
      setAccountInfo(info);

      // Update form defaults
      profileForm.reset({
        username: info?.username || "",
        email: info?.email || "",
      });
    } catch (err) {
      onError("Failed to fetch account information");
      console.error("Account info fetch error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const onUpdateProfile = async (data: z.infer<typeof profileSchema>) => {
    try {
      await invoke("update_profile", data);
      onSuccess("Profile updated successfully");
      await fetchAccountInfo(); // Refresh account info
    } catch (err) {
      onError("Failed to update profile");
      console.error("Profile update error:", err);
    }
  };

  const onUpdatePassword = async (data: z.infer<typeof passwordSchema>) => {
    try {
      await invoke("update_password", {
        currentPassword: data.currentPassword,
        newPassword: data.newPassword,
      });
      onSuccess("Password updated successfully");
      passwordForm.reset(); // Clear password form
    } catch (err) {
      onError("Failed to update password");
      console.error("Password update error:", err);
    }
  };

  const handleDeleteAccount = async () => {
    if (
      !window.confirm(
        "Are you sure you want to delete your account? This action cannot be undone.",
      )
    ) {
      return;
    }

    try {
      setIsDeletingAccount(true);
      await invoke("delete_account");
      // Handle successful deletion (probably logout and redirect)
    } catch (err) {
      onError("Failed to delete account");
      console.error("Account deletion error:", err);
    } finally {
      setIsDeletingAccount(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex justify-center py-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Account Overview */}
      <Card className="p-6">
        <div className="flex items-center space-x-4">
          <div className="p-3 bg-primary/10 rounded-full">
            <User className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h2 className="text-xl font-semibold">{accountInfo?.username}</h2>
            <p className="text-sm text-gray-500">
              Member since{" "}
              {new Date(accountInfo?.createdAt || "").toLocaleDateString()}
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
          <div>
            <p className="text-sm text-gray-500">Total Jobs Completed</p>
            <p className="text-lg font-semibold">
              {accountInfo?.totalJobsCompleted}
            </p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Total Credits Earned</p>
            <p className="text-lg font-semibold">
              {accountInfo?.totalCreditsEarned}
            </p>
          </div>
        </div>
      </Card>

      {/* Profile Settings */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">Profile Settings</h3>
        <Form {...profileForm}>
          <form
            onSubmit={profileForm.handleSubmit(onUpdateProfile)}
            className="space-y-4"
          >
            <FormField
              control={profileForm.control}
              name="username"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Username</FormLabel>
                  <FormControl>
                    <Input {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={profileForm.control}
              name="email"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Email</FormLabel>
                  <FormControl>
                    <Input {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <Button type="submit" disabled={!profileForm.formState.isDirty}>
              Update Profile
            </Button>
          </form>
        </Form>
      </Card>

      {/* Password Settings */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">Change Password</h3>
        <Form {...passwordForm}>
          <form
            onSubmit={passwordForm.handleSubmit(onUpdatePassword)}
            className="space-y-4"
          >
            <FormField
              control={passwordForm.control}
              name="currentPassword"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Current Password</FormLabel>
                  <FormControl>
                    <Input type="password" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={passwordForm.control}
              name="newPassword"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>New Password</FormLabel>
                  <FormControl>
                    <Input type="password" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={passwordForm.control}
              name="confirmPassword"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Confirm New Password</FormLabel>
                  <FormControl>
                    <Input type="password" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <Button type="submit">Update Password</Button>
          </form>
        </Form>
      </Card>

      {/* Danger Zone */}
      <Card className="p-6 border-red-200">
        <h3 className="text-lg font-semibold text-red-600 flex items-center">
          <AlertTriangle className="h-5 w-5 mr-2" />
          Danger Zone
        </h3>
        <p className="text-sm text-gray-500 mt-2">
          Once you delete your account, there is no going back. Please be
          certain.
        </p>
        <Button
          variant="destructive"
          className="mt-4"
          onClick={handleDeleteAccount}
          disabled={isDeletingAccount}
        >
          Delete Account
        </Button>
      </Card>
    </div>
  );
}
