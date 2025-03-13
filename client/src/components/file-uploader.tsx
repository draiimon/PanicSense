import { useState, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { useUpload } from '@/context/upload-context';

interface FileUploaderProps {
  onUploadSuccess?: (data: any) => void;
}

export default function FileUploader({ onUploadSuccess }: FileUploaderProps) {
  const { isUploading, uploadProgress, uploadError, uploadFile } = useUpload();
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();

    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();

    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file: File) => {
    if (file.name.toLowerCase().endsWith('.csv')) {
      setSelectedFile(file);
    } else {
      alert('Please upload a CSV file only.');
    }
  };

  const onButtonClick = () => {
    inputRef.current?.click();
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    try {
      const result = await uploadFile(selectedFile);
      if (onUploadSuccess) {
        onUploadSuccess(result);
      }
      setSelectedFile(null);
    } catch (error) {
      console.error('Upload failed:', error);
    }
  };

  return (
    <div className="w-full">
      <Card className={`p-6 border-2 border-dashed ${dragActive ? 'border-primary bg-primary/5' : 'border-border'} rounded-lg`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}>

        <div className="flex flex-col items-center justify-center py-4 text-center">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 text-muted-foreground mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>

          <h3 className="font-medium text-lg mb-1">Upload a CSV file</h3>
          <p className="text-sm text-muted-foreground mb-4">Drag and drop your file here, or click to select</p>

          <Input
            ref={inputRef}
            type="file"
            accept=".csv"
            onChange={handleChange}
            className="hidden"
          />

          <Button 
            variant="outline" 
            onClick={onButtonClick}
            disabled={isUploading}
          >
            Select File
          </Button>
        </div>
      </Card>

      {selectedFile && (
        <div className="mt-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 text-sm">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-green-500" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              <span className="font-medium">{selectedFile.name}</span>
              <span className="text-muted-foreground">({(selectedFile.size / 1024).toFixed(2)} KB)</span>
            </div>

            <Button 
              onClick={handleUpload}
              disabled={isUploading}
            >
              {isUploading ? 'Uploading...' : 'Upload'}
            </Button>
          </div>

          {isUploading && (
            <Progress value={uploadProgress} className="mt-2" />
          )}
        </div>
      )}

      {uploadError && (
        <Alert variant="destructive" className="mt-4">
          <AlertDescription>{uploadError}</AlertDescription>
        </Alert>
      )}
    </div>
  );
}