
import React from 'react';
import { Link } from 'wouter';

export function Footer() {
  return (
    <footer className="mt-auto py-6 border-t border-gray-100">
      <div className="container mx-auto px-4">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="mb-4 md:mb-0">
            <p className="text-sm text-slate-500">© 2023 PanicSense PH. All rights reserved.</p>
          </div>
          
          <div className="flex space-x-6">
            <Link href="/dashboard">
              <a className="text-sm text-slate-500 hover:text-slate-700">Dashboard</a>
            </Link>
            <Link href="/about">
              <a className="text-sm text-slate-500 hover:text-slate-700">About</a>
            </Link>
            <Link href="/privacy">
              <a className="text-sm text-slate-500 hover:text-slate-700">Privacy</a>
            </Link>
            <Link href="/terms">
              <a className="text-sm text-slate-500 hover:text-slate-700">Terms</a>
            </Link>
          </div>
        </div>
      </div>
    </footer>
  );
}
