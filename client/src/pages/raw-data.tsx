// UploadProvider.tsx
import { createContext, useContext, useState } from 'react';

interface UploadContext {
  isUploading: boolean;
  setIsUploading: (isUploading: boolean) => void;
}

const UploadContext = createContext<UploadContext | null>(null);

export const UploadProvider = ({ children }: { children: React.ReactNode }) => {
  const [isUploading, setIsUploading] = useState(false);

  const value: UploadContext = { isUploading, setIsUploading };

  return (
    <UploadContext.Provider value={value}>
      {children}
    </UploadContext.Provider>
  );
};

export const useUploadContext = () => {
  const context = useContext(UploadContext);
  if (context === null) {
    throw new Error('useUploadContext must be used within an UploadProvider');
  }
  return context;
};

// App.tsx
import { UploadProvider } from './UploadProvider';
import RawData from './raw-data';

function App() {
  return (
    <UploadProvider>
      <RawData />
    </UploadProvider>
  );
}

export default App;

// file-uploader.tsx
import React, { useState } from 'react';
import { useUploadContext } from './UploadProvider';

const FileUploader = ({ onSuccess }: { onSuccess: () => void }) => {
  const [file, setFile] = useState<File | null>(null);
  const { setIsUploading } = useUploadContext();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFile(e.target.files?.[0]);
  };

  const handleUpload = async () => {
    if (file) {
      setIsUploading(true);
      try {
        const formData = new FormData();
        formData.append('file', file);
        const response = await fetch('/api/upload', { 
          method: 'POST',
          body: formData,
        });
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        onSuccess();
      } catch (error) {
        console.error('Upload failed:', error);
      } finally {
        setIsUploading(false);
      }
    }
  };

  return (
    <div>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload} disabled={!file}>
        Upload
      </button>
    </div>
  );
};

export default FileUploader;

// raw-data.tsx
import { useState } from "react";
import { useDisasterContext } from "@/context/disaster-context";
import { DataTable } from "@/components/data/data-table";
import { FileUploader } from "@/components/file-uploader";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

export default function RawData() {
  const { 
    sentimentPosts, 
    analyzedFiles, 
    isLoadingSentimentPosts,
    isLoadingAnalyzedFiles
  } = useDisasterContext();
  const [selectedFileId, setSelectedFileId] = useState<string>("all");

  // Filter posts by file ID if selected
  const filteredPosts = selectedFileId === "all" 
    ? sentimentPosts 
    : sentimentPosts.filter(post => post.fileId === parseInt(selectedFileId));

  const isLoading = isLoadingSentimentPosts || isLoadingAnalyzedFiles;

  return (
    <div className="space-y-6">
      {/* Raw Data Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-800">Raw Data</h1>
          <p className="mt-1 text-sm text-slate-500">
            View and filter all analyzed sentiment data
          </p>
        </div>
        <FileUploader 
          className="mt-4 sm:mt-0"
          onSuccess={() => {
            // The disaster context will handle refetching data
          }}
        />
      </div>

      {/* Filter Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center space-y-4 sm:space-y-0 sm:space-x-4 bg-white p-4 rounded-lg shadow">
        <div className="flex-grow">
          <label className="block text-sm font-medium text-slate-700 mb-1">
            Filter by Dataset
          </label>
          <Select
            value={selectedFileId}
            onValueChange={setSelectedFileId}
          >
            <SelectTrigger className="w-full">
              <SelectValue placeholder="All datasets" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All datasets</SelectItem>
              {analyzedFiles.map((file) => (
                <SelectItem key={file.id} value={file.id.toString()}>
                  {file.originalName} ({file.recordCount} records)
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="text-sm text-slate-500 flex items-center">
          {isLoading ? (
            <div className="flex items-center">
              <svg className="animate-spin h-4 w-4 mr-2 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Loading data...
            </div>
          ) : (
            <span>
              Showing {filteredPosts.length} of {sentimentPosts.length} total records
            </span>
          )}
        </div>
      </div>

      {/* Data Table */}
      <DataTable 
        data={filteredPosts}
        title={selectedFileId === "all" ? "All Sentiment Data" : `Data from ${analyzedFiles.find(f => f.id.toString() === selectedFileId)?.originalName || "Selected File"}`}
        description="Raw sentiment analysis results with filtering capabilities"
      />
    </div>
  );
}