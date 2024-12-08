import { Routes, Route } from "react-router-dom";
import { Login, Signup } from "./pages/auth";
import { Dashboard } from "./pages/dashboard";
import { JobMarketplace } from "./pages/marketplace";
import { WorkerDashboard } from "./pages/worker";
import { Settings } from "./pages/settings";
import { ProtectedRoute } from "./components/auth/protected-route";
import { AuthProvider, useAuth } from "./components/auth/auth-context";
import { Outlet, Link, useLocation } from "react-router-dom";
import {
  ChartLine,
  Briefcase,
  Cpu,
  Settings as SettingsIcon,
  LogOut,
} from "lucide-react";

interface MenuItem {
  icon: React.ReactNode;
  label: string;
  path: string;
}

function App() {
  return (
    <AuthProvider>
      <div className="min-h-screen bg-background">
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route
            path="/"
            element={
              <ProtectedRoute>
                <MainLayout />
              </ProtectedRoute>
            }
          >
            <Route index element={<Dashboard />} />
            <Route path="marketplace" element={<JobMarketplace />} />
            <Route path="worker" element={<WorkerDashboard />} />
            <Route path="settings" element={<Settings />} />
          </Route>
        </Routes>
      </div>
    </AuthProvider>
  );
}

function MainLayout() {
  const { logout } = useAuth();
  const menuItems: MenuItem[] = [
    { icon: <ChartLine className="h-5 w-5" />, label: "Dashboard", path: "/" },
    {
      icon: <Briefcase className="h-5 w-5" />,
      label: "Job Marketplace",
      path: "/marketplace",
    },
    {
      icon: <Cpu className="h-5 w-5" />,
      label: "Worker Dashboard",
      path: "/worker",
    },
    {
      icon: <SettingsIcon className="h-5 w-5" />,
      label: "Settings",
      path: "/settings",
    },
  ];

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <aside className="w-64 bg-background border-r border-border p-4 flex flex-col">
        <div className="flex items-center space-x-2 mb-8">
          <img src="/logo.svg" alt="Aivida" className="h-8 w-8" />
          <span className="text-xl font-bold">Aivida</span>
        </div>

        <nav className="space-y-2 flex-1">
          {menuItems.map((item) => (
            <NavItem key={item.path} {...item} />
          ))}
        </nav>

        <div className="pt-4 border-t border-border mt-auto">
          <button
            onClick={logout}
            className="flex items-center space-x-2 text-danger hover:bg-danger-light w-full p-2 rounded-lg transition-colors"
          >
            <LogOut className="h-5 w-5" />
            <span>Logout</span>
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto p-6 bg-background-secondary">
        <Outlet />
      </main>
    </div>
  );
}

function NavItem({ icon, label, path }: MenuItem) {
  const location = useLocation();
  const isActive = location.pathname === path;

  return (
    <Link
      to={path}
      className={`
        flex items-center space-x-2 px-3 py-2 rounded-lg transition-colors
        ${
          isActive
            ? "bg-primary/10 text-primary"
            : "text-secondary hover:bg-secondary hover:text-primary"
        }
      `}
    >
      <span className={`${isActive ? "text-primary" : "text-secondary"}`}>
        {icon}
      </span>
      <span className="font-medium">{label}</span>
    </Link>
  );
}

export default App;
