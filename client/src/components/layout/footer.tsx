
import React from 'react';
import { Link } from 'wouter';

export function Footer() {
  return (
    <footer className="bg-white border-t border-gray-200 py-6 mt-auto">
      <div className="container mx-auto px-4">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="mb-4 md:mb-0">
            <p className="text-sm text-slate-500">
              © {new Date().getFullYear()} Disaster Sentiment Analysis System
            </p>
          </div>
          <div className="flex space-x-6">
            <Link href="/about" className="text-sm text-slate-500 hover:text-slate-700">
              About
            </Link>
            <a 
              href="mailto:contact@example.com" 
              className="text-sm text-slate-500 hover:text-slate-700"
            >
              Contact
            </a>
            <a 
              href="https://replit.com" 
              target="_blank" 
              rel="noopener noreferrer" 
              className="text-sm text-slate-500 hover:text-slate-700"
            >
              Powered by Replit
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
