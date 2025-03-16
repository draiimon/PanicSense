
import React from "react";
import { cn } from "@/lib/utils";

export const RippleContainer = ({
  className,
  children,
}: {
  className?: string;
  children: React.ReactNode;
}) => {
  return (
    <div className={cn("relative overflow-hidden", className)}>
      {children}
      <div className="absolute inset-0">
        <div className="absolute inset-0 animate-ripple-slow rounded-full bg-white/10" />
      </div>
    </div>
  );
};
