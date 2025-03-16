import React from 'react';
import { motion } from "framer-motion";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselNext,
  CarouselPrevious,
} from "@/components/ui/carousel";

export default function About() {
  const founders = [
    {
      name: "Mark Andrei R. Castillo",
      role: "Core System Architecture & Machine Learning",
      description: "Leads the development of our advanced ML pipelines and system architecture"
    },
    {
      name: "Ivahnn B. Garcia",
      role: "Frontend Development & User Experience",
      description: "Creates intuitive and responsive user interfaces for seamless interaction"
    },
    {
      name: "Julia Daphne Ngan-Gatdula",
      role: "Data Resources & Information Engineering",
      description: "Manages data infrastructure and information processing systems"
    }
  ];

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.3
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.6
      }
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-purple-900 text-white py-12">
      <motion.div 
        initial="hidden"
        animate="visible"
        variants={containerVariants}
        className="container mx-auto px-4 space-y-16"
      >
        {/* Hero Section */}
        <motion.div 
          variants={itemVariants}
          className="text-center space-y-6"
        >
          <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-blue-400 via-indigo-400 to-purple-400 bg-clip-text text-transparent">
            PanicSense PH
          </h1>
          <p className="text-xl text-blue-200 max-w-2xl mx-auto">
            Revolutionizing Disaster Response Through Advanced Sentiment Analysis
          </p>
        </motion.div>

        {/* Founders Carousel */}
        <motion.div
          variants={itemVariants}
          className="py-12"
        >
          <h2 className="text-3xl font-bold text-center mb-12 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            Meet Our Visionary Team
          </h2>
          <div className="relative px-12">
            <Carousel
              opts={{
                align: "start",
                loop: true,
              }}
              className="w-full max-w-5xl mx-auto"
            >
              <CarouselContent>
                {founders.map((founder, index) => (
                  <CarouselItem key={index} className="md:basis-1/2 lg:basis-1/3">
                    <motion.div
                      whileHover={{ scale: 1.05 }}
                      className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 h-full border border-white/20 hover:border-blue-400/50 transition-all duration-300"
                    >
                      <div className="aspect-square bg-gradient-to-br from-blue-600/20 to-purple-600/20 rounded-xl flex items-center justify-center mb-4">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-16 h-16 text-blue-400">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 1 1-7.5 0 3.75 3.75 0 0 1 7.5 0ZM4.501 20.118a7.5 7.5 0 0 1 14.998 0A17.933 17.933 0 0 1 12 21.75c-2.676 0-5.216-.584-7.499-1.632Z" />
                        </svg>
                      </div>
                      <h3 className="text-xl font-bold text-blue-300 mb-2">{founder.name}</h3>
                      <p className="text-blue-200 mb-3">{founder.role}</p>
                      <p className="text-sm text-blue-100/80">{founder.description}</p>
                    </motion.div>
                  </CarouselItem>
                ))}
              </CarouselContent>
              <CarouselPrevious className="bg-white/10 border-white/20 text-white hover:bg-white/20" />
              <CarouselNext className="bg-white/10 border-white/20 text-white hover:bg-white/20" />
            </Carousel>
          </div>
        </motion.div>

        {/* Technology Stack */}
        <motion.div
          variants={itemVariants}
          className="grid md:grid-cols-2 gap-8 max-w-6xl mx-auto"
        >
          <motion.div 
            whileHover={{ scale: 1.02 }}
            className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 border border-white/20"
          >
            <h3 className="text-2xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              Advanced Technology Stack
            </h3>
            <ul className="space-y-3 text-blue-200">
              <li className="flex items-center space-x-2">
                <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                </svg>
                <span>Deep Learning NLP Models</span>
              </li>
              <li className="flex items-center space-x-2">
                <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                </svg>
                <span>Real-time Data Processing</span>
              </li>
              <li className="flex items-center space-x-2">
                <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                </svg>
                <span>Cross-lingual Sentiment Analysis</span>
              </li>
            </ul>
          </motion.div>

          <motion.div 
            whileHover={{ scale: 1.02 }}
            className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 border border-white/20"
          >
            <h3 className="text-2xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              Impact & Innovation
            </h3>
            <ul className="space-y-3 text-blue-200">
              <li className="flex items-center space-x-2">
                <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                </svg>
                <span>Enhanced Disaster Response</span>
              </li>
              <li className="flex items-center space-x-2">
                <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                </svg>
                <span>Data-Driven Decision Making</span>
              </li>
              <li className="flex items-center space-x-2">
                <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                </svg>
                <span>Community-Centric Solutions</span>
              </li>
            </ul>
          </motion.div>
        </motion.div>

        {/* System Description */}
        <motion.div
          variants={itemVariants}
          className="max-w-4xl mx-auto space-y-8 text-center"
        >
          <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            About PanicSense PH
          </h2>
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="prose prose-invert max-w-none space-y-6"
          >
            <p className="text-lg text-blue-200 leading-relaxed">
              PanicSense PH revolutionizes disaster response through cutting-edge deep learning and natural language processing technologies. Our system performs real-time sentiment analysis during crisis events, utilizing a sophisticated neural architecture that combines transformer-based models with custom attention mechanisms.
            </p>
            <p className="text-lg text-blue-200 leading-relaxed">
              Our platform employs advanced transfer learning techniques and fine-tuned language models capable of processing both English and Filipino text with state-of-the-art accuracy. Our innovative cross-lingual embedding approach ensures contextual understanding of cultural nuances and colloquial expressions across multiple Filipino dialects.
            </p>
            <p className="text-lg text-blue-200 leading-relaxed">
              Through ensemble methods and reinforcement learning algorithms, the system dynamically categorizes emotional states into five distinct classifications: Panic, Fear/Anxiety, Disbelief, Resilience, and Neutral. This granular emotion mapping provides crucial decision support for emergency response coordination and resource allocation during disaster events.
            </p>
          </motion.div>
        </motion.div>
      </motion.div>
    </div>
  );
}