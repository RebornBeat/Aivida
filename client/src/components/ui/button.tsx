import { cn } from "./utils";
import { Loader2 } from "lucide-react";
import React from "react";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "outline" | "destructive";
  size?: "default" | "sm" | "lg";
  isLoading?: boolean;
}

export function Button({
  className,
  variant = "default",
  size = "default",
  isLoading,
  children,
  disabled,
  ...props
}: ButtonProps) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center rounded-md font-medium transition-colors",
        "disabled:opacity-50 disabled:pointer-events-none",
        {
          "bg-primary text-white hover:bg-primary/90": variant === "default",
          "border border-gray-200 hover:bg-gray-100": variant === "outline",
          "bg-red-500 text-white hover:bg-red-600": variant === "destructive",
          "h-9 px-4 py-2": size === "default",
          "h-8 px-3": size === "sm",
          "h-10 px-8": size === "lg",
        },
        className,
      )}
      disabled={isLoading || disabled}
      {...props}
    >
      {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
      {children}
    </button>
  );
}
