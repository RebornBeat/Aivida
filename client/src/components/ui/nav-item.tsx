import { Link, useLocation } from "react-router-dom";
import { cn } from "./utils";

interface NavItemProps {
  icon: React.ReactNode;
  label: string;
  path: string;
}

export function NavItem({ icon, label, path }: NavItemProps) {
  const location = useLocation();
  const isActive = location.pathname === path;

  return (
    <Link
      to={path}
      className={cn(
        "flex items-center space-x-3 px-3 py-2 rounded-lg transition-colors",
        isActive
          ? "bg-primary/10 text-primary"
          : "text-gray-600 hover:bg-gray-100",
      )}
    >
      <span
        className={cn("h-5 w-5", isActive ? "text-primary" : "text-gray-500")}
      >
        {icon}
      </span>
      <span className="font-medium">{label}</span>
    </Link>
  );
}
