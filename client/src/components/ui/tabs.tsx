import { cn } from "./utils";
import React from "react";

interface TabsProps extends React.HTMLAttributes<HTMLDivElement> {}

export function Tabs({ className, ...props }: TabsProps) {
  return <div className={cn("space-y-4", className)} {...props} />;
}

export function TabsList({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("flex space-x-1 rounded-lg bg-gray-100 p-1", className)}
      {...props}
    />
  );
}

export function TabsTrigger({
  className,
  selected,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { selected?: boolean }) {
  return (
    <button
      className={cn(
        "rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
        {
          "bg-white shadow": selected,
          "text-gray-600 hover:text-gray-900": !selected,
        },
        className,
      )}
      {...props}
    />
  );
}

export function TabsContent({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("mt-2", className)} {...props} />;
}
