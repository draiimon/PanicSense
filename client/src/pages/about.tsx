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
    <div className="relative min-h-screen overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0 bg-slate-900">
        <div className="absolute inset-0 bg-gradient-to-br from-blue-900/50 via-indigo-900/50 to-purple-900/50" />
        <motion.div
          initial={{ opacity: 0.3 }}
          animate={{
            opacity: [0.3, 0.5, 0.3],
            rotate: [0, 360]
          }}
          transition={{
            duration: 15,
            repeat: Infinity,
            ease: "linear"
          }}
          className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-blue-500/10 via-transparent to-transparent"
        />
        <motion.div
          initial={{ opacity: 0.2 }}
          animate={{
            opacity: [0.2, 0.4, 0.2],
            rotate: [360, 0]
          }}
          transition={{
            duration: 20,
            repeat: Infinity,
            ease: "linear"
          }}
          className="absolute inset-0 bg-[radial-gradient(circle_at_70%_30%,_var(--tw-gradient-stops))] from-purple-500/10 via-transparent to-transparent"
        />
      </div>

      <motion.div 
        initial="hidden"
        animate="visible"
        variants={containerVariants}
        className="relative container mx-auto px-4 space-y-16 py-12"
      >
        {/* Hero Section */}
        <motion.div 
          variants={itemVariants}
          className="text-center space-y-6"
        >
          <h1 className="text-6xl md:text-7xl font-bold bg-gradient-to-r from-blue-400 via-indigo-400 to-purple-400 bg-clip-text text-transparent">
            PanicSense PH
          </h1>
          <p className="text-2xl text-blue-200 max-w-3xl mx-auto leading-relaxed">
            Revolutionizing Disaster Response Through Advanced Sentiment Analysis
          </p>
        </motion.div>

        {/* Founders Carousel */}
        <motion.div
          variants={itemVariants}
          className="py-16"
        >
          <h2 className="text-4xl font-bold text-center mb-12 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
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
                    <div className="p-1">
                      <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.2 }}
                        className="group relative bg-white/5 backdrop-blur-xl rounded-2xl p-6 h-full border border-white/10 hover:border-blue-400/50 hover:bg-white/10 transition-all duration-500"
                      >
                        <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-blue-500/20 to-purple-500/20 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                        <div className="relative">
                          <div className="aspect-square bg-gradient-to-br from-blue-600/20 to-purple-600/20 rounded-xl flex items-center justify-center mb-4 group-hover:from-blue-600/30 group-hover:to-purple-600/30 transition-all duration-500">
                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-16 h-16 text-blue-400 group-hover:text-blue-300 transition-colors duration-500">
                              <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 1 1-7.5 0 3.75 3.75 0 0 1 7.5 0ZM4.501 20.118a7.5 7.5 0 0 1 14.998 0A17.933 17.933 0 0 1 12 21.75c-2.676 0-5.216-.584-7.499-1.632Z" />
                            </svg>
                          </div>
                          <h3 className="text-xl font-bold text-blue-300 mb-2 group-hover:text-blue-200 transition-colors duration-500">{founder.name}</h3>
                          <p className="text-blue-200 mb-3 group-hover:text-blue-100 transition-colors duration-500">{founder.role}</p>
                          <p className="text-sm text-blue-100/80 group-hover:text-blue-100 transition-colors duration-500">{founder.description}</p>
                        </div>
                      </motion.div>
                    </div>
                  </CarouselItem>
                ))}
              </CarouselContent>
              <CarouselPrevious className="bg-white/10 border-white/20 text-white hover:bg-white/20 transition-colors duration-300" />
              <CarouselNext className="bg-white/10 border-white/20 text-white hover:bg-white/20 transition-colors duration-300" />
            </Carousel>
          </div>
        </motion.div>

        {/* Technology Stack */}
        <motion.div
          variants={itemVariants}
          className="grid md:grid-cols-2 gap-8 max-w-6xl mx-auto"
        >
          <motion.div 
            className="group bg-white/5 backdrop-blur-xl rounded-2xl p-8 border border-white/10 hover:border-blue-400/50 hover:bg-white/10 transition-all duration-500"
          >
            <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-blue-500/20 to-purple-500/20 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            <h3 className="relative text-2xl font-bold mb-6 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              Advanced Technology Stack
            </h3>
            <ul className="relative space-y-4 text-blue-200">
              <li className="flex items-center space-x-3 group/item">
                <span className="flex-shrink-0 w-8 h-8 flex items-center justify-center rounded-lg bg-blue-500/20 group-hover/item:bg-blue-500/30 transition-colors duration-300">
                  <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                  </svg>
                </span>
                <span className="group-hover/item:text-blue-100 transition-colors duration-300">Deep Learning NLP Models</span>
              </li>
              <li className="flex items-center space-x-3 group/item">
                <span className="flex-shrink-0 w-8 h-8 flex items-center justify-center rounded-lg bg-blue-500/20 group-hover/item:bg-blue-500/30 transition-colors duration-300">
                  <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                  </svg>
                </span>
                <span className="group-hover/item:text-blue-100 transition-colors duration-300">Real-time Data Processing</span>
              </li>
              <li className="flex items-center space-x-3 group/item">
                <span className="flex-shrink-0 w-8 h-8 flex items-center justify-center rounded-lg bg-blue-500/20 group-hover/item:bg-blue-500/30 transition-colors duration-300">
                  <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                  </svg>
                </span>
                <span className="group-hover/item:text-blue-100 transition-colors duration-300">Multilingual Sentiment Analysis</span>
              </li>
            </ul>
          </motion.div>

          <motion.div 
            className="group bg-white/5 backdrop-blur-xl rounded-2xl p-8 border border-white/10 hover:border-blue-400/50 hover:bg-white/10 transition-all duration-500"
          >
            <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-blue-500/20 to-purple-500/20 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            <h3 className="relative text-2xl font-bold mb-6 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              Impact & Innovation
            </h3>
            <ul className="relative space-y-4 text-blue-200">
              <li className="flex items-center space-x-3 group/item">
                <span className="flex-shrink-0 w-8 h-8 flex items-center justify-center rounded-lg bg-blue-500/20 group-hover/item:bg-blue-500/30 transition-colors duration-300">
                  <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                  </svg>
                </span>
                <span className="group-hover/item:text-blue-100 transition-colors duration-300">Enhanced Disaster Response</span>
              </li>
              <li className="flex items-center space-x-3 group/item">
                <span className="flex-shrink-0 w-8 h-8 flex items-center justify-center rounded-lg bg-blue-500/20 group-hover/item:bg-blue-500/30 transition-colors duration-300">
                  <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                  </svg>
                </span>
                <span className="group-hover/item:text-blue-100 transition-colors duration-300">Data-Driven Decision Making</span>
              </li>
              <li className="flex items-center space-x-3 group/item">
                <span className="flex-shrink-0 w-8 h-8 flex items-center justify-center rounded-lg bg-blue-500/20 group-hover/item:bg-blue-500/30 transition-colors duration-300">
                  <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                  </svg>
                </span>
                <span className="group-hover/item:text-blue-100 transition-colors duration-300">Community-Centric Solutions</span>
              </li>
            </ul>
          </motion.div>
        </motion.div>

        {/* System Description */}
        <motion.div
          variants={itemVariants}
          className="relative max-w-4xl mx-auto space-y-8 text-center"
        >
          <div className="absolute inset-0 bg-gradient-to-r from-blue-500/5 via-purple-500/5 to-blue-500/5 rounded-3xl blur-3xl" />
          <h2 className="relative text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent mb-8">
            About PanicSense PH
          </h2>
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="relative prose prose-invert max-w-none space-y-6"
          >
            <p className="text-lg text-blue-200 leading-relaxed">
              PanicSense PH revolutionizes disaster response through cutting-edge deep learning and natural language processing technologies. Our system performs real-time sentiment analysis during crisis events, utilizing a sophisticated neural architecture that combines transformer-based models with custom attention mechanisms.
            </p>
            <p className="text-lg text-blue-200 leading-relaxed">
              Our platform employs advanced transfer learning techniques and fine-tuned language models capable of processing both English and Filipino text with state-of-the-art accuracy. Our innovative multilingual approach ensures contextual understanding of cultural nuances and colloquial expressions across multiple Filipino dialects.
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