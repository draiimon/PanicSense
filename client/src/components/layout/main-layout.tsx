import { ReactNode } from "react";
import { Sidebar } from "@/components/sidebar";
import { useLocation } from "wouter";
import { useAuth } from "@/context/auth-context";
import { Button } from "@/components/ui/button";
import { LogOut } from "lucide-react";

interface MainLayoutProps {
  children: ReactNode;
}

export function MainLayout({ children }: MainLayoutProps) {
  const [location] = useLocation();
  const { user, logout } = useAuth();
  const isAuthPage = location.includes('/login') || location.includes('/signup');

  const handleLogout = () => {
    logout();
  };

  if (isAuthPage) {
    return <>{children}</>;
  }

  return (
    <div className="flex h-screen overflow-hidden flex-col">
      <div className="bg-white shadow-sm z-10 flex justify-between items-center px-4 py-2">
        <h1 className="text-xl font-semibold">PanicSense PH</h1>
        {user && (
          <div className="flex items-center gap-4">
            <span className="text-sm text-gray-600">{user.fullName}</span>
            <Button 
              variant="ghost" 
              size="sm" 
              onClick={handleLogout}
              className="text-gray-600 hover:text-gray-900"
            >
              <LogOut className="h-4 w-4 mr-2" />
              Logout
            </Button>
          </div>
        )}
      </div>
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />
        <div className="flex-1 overflow-auto pl-0 lg:pl-64">
          <main className="px-4 sm:px-6 lg:px-8 py-8 flex-grow">
            {children}
          </main>
        </div>
      </div>
    </div>
  );
}