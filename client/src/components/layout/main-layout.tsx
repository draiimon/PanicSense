import { ReactNode } from "react";
import { Sidebar } from "@/components/sidebar";

interface MainLayoutProps {
  children: ReactNode;
}

export function MainLayout({ children }: MainLayoutProps) {
  return (
    <div className="flex h-screen overflow-hidden flex-col">
      <div className="bg-white shadow-sm z-10 flex justify-between items-center px-4 py-2">
        <h1 className="text-xl font-semibold">PanicSense PH</h1>
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