import { AnimatePresence, motion } from "framer-motion";
import { useDisasterContext } from "@/context/disaster-context";
import { Loader2, CheckCircle, AlertCircle } from "lucide-react";
import { HybridFileUploaderButton } from "./hybrid-file-uploader-button";

interface HybridFileUploaderProps {
  onSuccess?: (data: any) => void;
  className?: string;
  id?: string;
}

export function HybridFileUploader({ onSuccess, className, id }: HybridFileUploaderProps) {
  return <HybridFileUploaderButton id={id} onSuccess={onSuccess} className={className} />;
}