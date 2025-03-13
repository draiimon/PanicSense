
import { motion } from "framer-motion";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

const fadeIn = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 }
};

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.2
    }
  }
};

export default function About() {
  return (
    <motion.div 
      initial="hidden"
      animate="visible"
      variants={staggerContainer}
      className="container mx-auto py-8 space-y-8"
    >
      <motion.h1 
        variants={fadeIn}
        className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-indigo-800 text-center mb-8"
      >
        About PanicSense PH
      </motion.h1>
      
      <motion.div variants={fadeIn}>
        <Card className="mb-8 overflow-hidden border-0 shadow-lg bg-gradient-to-br from-white to-blue-50">
          <CardHeader className="bg-gradient-to-r from-blue-50 to-indigo-50 border-b">
            <CardTitle className="text-2xl text-blue-800 font-bold">Our Mission</CardTitle>
            <CardDescription className="text-indigo-600 font-medium">Empowering communities through disaster sentiment analysis</CardDescription>
          </CardHeader>
          <CardContent className="p-6">
            <motion.p 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
              className="mb-4 text-gray-700 leading-relaxed"
            >
              PanicSense PH is dedicated to providing real-time disaster sentiment analysis to help communities
              and emergency responders better understand public reactions during crisis situations. By analyzing
              social media posts and other text sources, we identify patterns of panic, fear, resilience, and
              other emotions to guide more effective disaster response.
            </motion.p>
          </CardContent>
        </Card>
      </motion.div>
      
      <motion.div 
        variants={staggerContainer}
        className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8"
      >
        <motion.div variants={fadeIn}>
          <Card className="h-full border-0 shadow-lg bg-gradient-to-br from-white to-blue-50 hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1">
            <CardHeader className="bg-gradient-to-r from-blue-50 to-indigo-50 border-b">
              <CardTitle className="text-xl text-blue-800 font-bold">Our Team</CardTitle>
              <CardDescription className="text-indigo-600 font-medium">The founders behind PanicSense PH</CardDescription>
            </CardHeader>
            <CardContent className="p-6">
              <div className="space-y-6">
                <motion.div 
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.3 }}
                >
                  <h3 className="font-bold text-indigo-800">Castillo, Mark Andrei R.</h3>
                  <p className="text-sm text-blue-600 font-medium">Co-Founder & Lead Developer</p>
                  <p className="mt-2 text-gray-700">Expert software architect and backend specialist focusing on system optimization and AI integration.</p>
                </motion.div>
                
                <motion.div 
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.5 }}
                >
                  <h3 className="font-bold text-indigo-800">Garcia, Ivahnn</h3>
                  <p className="text-sm text-blue-600 font-medium">Co-Founder & UI/UX Lead</p>
                  <p className="mt-2 text-gray-700">Frontend engineering specialist with expertise in creating intuitive, responsive disaster management interfaces.</p>
                </motion.div>
                
                <motion.div 
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.7 }}
                >
                  <h3 className="font-bold text-indigo-800">Gatdula, Julia Daphne Ngan</h3>
                  <p className="text-sm text-blue-600 font-medium">Co-Founder & Data Science Lead</p>
                  <p className="mt-2 text-gray-700">Natural language processing expert specializing in bilingual sentiment analysis and disaster response data models.</p>
                </motion.div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
        
        <motion.div variants={fadeIn}>
          <Card className="h-full border-0 shadow-lg bg-gradient-to-br from-white to-blue-50 hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1">
            <CardHeader className="bg-gradient-to-r from-blue-50 to-indigo-50 border-b">
              <CardTitle className="text-xl text-blue-800 font-bold">Our Technology</CardTitle>
              <CardDescription className="text-indigo-600 font-medium">State-of-the-art sentiment analysis for disaster response</CardDescription>
            </CardHeader>
            <CardContent className="p-6">
              <motion.p 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="mb-4 text-gray-700 leading-relaxed"
              >
                PanicSense PH employs cutting-edge transformer-based deep learning models specifically fine-tuned for disaster response scenarios. Our proprietary neural architecture combines BERT-based contextual understanding with domain-specific attention mechanisms to accurately process disaster communications.
              </motion.p>
              <motion.p 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 }}
                className="mb-4 text-gray-700 leading-relaxed"
              >
                Our multi-modal sentiment analysis engine features cross-lingual capabilities, processing both English and Filipino content with state-of-the-art accuracy. The system employs advanced NLP techniques including contextual embeddings, aspect-based sentiment analysis, and emotion-specific feature extraction.
              </motion.p>
              <motion.p 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.8 }}
                className="text-gray-700 leading-relaxed"
              >
                The platform's advanced classification system identifies five distinct emotional categories: Panic, Fear/Anxiety, Disbelief, Resilience, and Neutral—providing emergency responders with nuanced, actionable intelligence during crisis situations.
              </motion.p>
            </CardContent>
          </Card>
        </motion.div>
      </motion.div>
      
      <motion.div variants={fadeIn}>
        <Card className="border-0 shadow-lg bg-gradient-to-br from-white to-blue-50 hover:shadow-xl transition-all duration-300">
          <CardHeader className="bg-gradient-to-r from-blue-50 to-indigo-50 border-b">
            <CardTitle className="text-xl text-blue-800 font-bold">Contact Us</CardTitle>
            <CardDescription className="text-indigo-600 font-medium">Get in touch with the PanicSense PH team</CardDescription>
          </CardHeader>
          <CardContent className="p-6">
            <motion.div 
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.5 }}
              className="flex items-center space-x-2 text-indigo-700"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
              </svg>
              <p className="font-medium">admin@panicsense.ph</p>
            </motion.div>
          </CardContent>
        </Card>
      </motion.div>
    </motion.div>
  );
}
