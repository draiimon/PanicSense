import { ReactNode } from "react";
import { Sidebar } from "@/components/sidebar";
import { useLocation } from "wouter";

interface MainLayoutProps {
  children: ReactNode;
}

export function MainLayout({ children }: MainLayoutProps) {
  const [location] = useLocation();
  const isAuthPage = location.includes('/login') || location.includes('/signup');

  if (isAuthPage) {
    return <>{children}</>;
  }

  return (
    <div className="flex h-screen overflow-hidden flex-col">
      <Sidebar />
      <div className="flex-1 overflow-auto pl-0 lg:pl-64">
        <main className="px-4 sm:px-6 lg:px-8 py-8 flex-grow">
          {children}
        </main>
      </div>
    </div>
  );
}