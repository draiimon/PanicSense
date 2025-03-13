import { ReactNode } from "react";
import { Sidebar } from "@/components/sidebar";

interface MainLayoutProps {
  children: ReactNode;
}

export function MainLayout({ children }: MainLayoutProps) {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <div className="flex-1 overflow-auto pl-0 lg:pl-64">
        {/* Header for mobile */}
        <header className="bg-white shadow-sm sticky top-0 z-20 lg:hidden">
          <div className="px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
            <div className="flex-1"></div>
            <div className="flex items-center space-x-4">
              <div className="relative">
                <button className="p-2 rounded-full bg-slate-100 hover:bg-slate-200">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-5 w-5 text-slate-600"
                    viewBox="0 0 20 20"
                    fill="currentColor"
                  >
                    <path d="M10 2a6 6 0 00-6 6v3.586l-.707.707A1 1 0 004 14h12a1 1 0 00.707-1.707L16 11.586V8a6 6 0 00-6-6zM10 18a3 3 0 01-3-3h6a3 3 0 01-3 3z" />
                  </svg>
                </button>
                <div className="absolute top-0 right-0 h-3 w-3 rounded-full bg-red-500 border-2 border-white"></div>
              </div>
              <div className="flex items-center space-x-2">
                <span className="inline-block h-9 w-9 rounded-full bg-slate-200 overflow-hidden">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-full w-full text-slate-400"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                    />
                  </svg>
                </span>
              </div>
            </div>
          </div>
        </header>

        {/* Main content */}
        <main className="px-4 sm:px-6 lg:px-8 py-8">{children}</main>
      </div>
    </div>
  );
}
