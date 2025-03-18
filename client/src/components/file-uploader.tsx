import { AnimatePresence, motion } from "framer-motion";
import { useDisasterContext } from "@/context/disaster-context";
import { Loader2, CheckCircle, AlertCircle } from "lucide-react";
import { FileUploaderButton } from "./file-uploader-button";
import { UploadProgressModal } from "./upload-progress-modal";

interface FileUploaderProps {
  onSuccess?: (data: any) => void;
  className?: string;
}

export function FileUploader({ onSuccess, className }: FileUploaderProps) {
  return (
    <>
      <FileUploaderButton onSuccess={onSuccess} className={className} />
      <UploadProgressModal />
    </>
  );
}