import { Card, CardContent, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { AlertTriangle, Home, ArrowLeft } from "lucide-react";
import { Link } from "wouter";
import { motion } from "framer-motion";

export default function NotFound() {
  // Handle go back safely
  const handleGoBack = () => {
    window.history.back();
  };

  return (
    <div className="w-full">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-md mx-auto"
      >
        <Card className="border-none shadow-lg overflow-hidden bg-white/90 backdrop-blur-sm">
          <div className="absolute top-0 left-0 right-0 h-1.5 bg-gradient-to-r from-red-500 via-amber-500 to-red-500"></div>
          
          <CardContent className="pt-8 px-8">
            <div className="flex flex-col items-center text-center mb-6">
              <div className="mb-4 w-20 h-20 rounded-full bg-red-50 flex items-center justify-center">
                <motion.div
                  animate={{ rotate: [0, 5, -5, 0] }}
                  transition={{ duration: 2, repeat: Infinity }}
                >
                  <AlertTriangle className="h-10 w-10 text-red-500" />
                </motion.div>
              </div>
              
              <h1 className="text-3xl font-bold bg-gradient-to-br from-red-600 to-amber-600 bg-clip-text text-transparent">
                404 - Page Not Found
              </h1>
              
              <p className="mt-4 text-slate-600 max-w-sm">
                The page you are looking for might have been removed, 
                had its name changed, or is temporarily unavailable.
              </p>
            </div>
          </CardContent>
          
          <CardFooter className="justify-center gap-3 pb-8 flex-wrap">
            <Button
              variant="outline"
              className="gap-2"
              onClick={handleGoBack}
            >
              <ArrowLeft className="h-4 w-4" />
              Go Back
            </Button>
            
            <Link href="/dashboard">
              <Button className="gap-2">
                <Home className="h-4 w-4" />
                Go to Dashboard
              </Button>
            </Link>
          </CardFooter>
        </Card>
      </motion.div>
    </div>
  );
}
