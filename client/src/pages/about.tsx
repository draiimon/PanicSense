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
      image: "/attached_assets/drei.jpg",
      description: "Leads the development of our advanced ML pipelines and system architecture"
    },
    {
      name: "Ivahnn B. Garcia",
      role: "Frontend Development & User Experience",
      image: "/attached_assets/van.jpg",
      description: "Creates intuitive and responsive user interfaces for seamless interaction"
    },
    {
      name: "Julia Daphne Ngan-Gatdula",
      role: "Data Resources & Information Engineering",
      image: "/attached_assets/julia.jpg",
      description: "Manages data infrastructure and information processing systems"
    }
  ];

  return (
    <div className="relative min-h-screen w-full flex flex-col items-center justify-center bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-blue-100 via-indigo-50 to-white">
      {/* Animated Background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute inset-0 bg-slate-900" />
        <div className="absolute inset-0 bg-gradient-to-br from-blue-900/50 via-indigo-900/50 to-purple-900/50" />
      </div>

      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
        className="relative w-full space-y-16 py-12 px-4"
      >
        {/* Hero Section */}
        <motion.div 
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="text-center space-y-6"
        >
          <h1 className="text-6xl md:text-7xl font-bold">
            <span className="bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 bg-clip-text text-transparent">
              PanicSense PH
            </span>
          </h1>
          <p className="text-2xl md:text-3xl text-blue-200 max-w-3xl mx-auto leading-relaxed">
            Revolutionizing Disaster Response Through
            <br />
            <span className="font-semibold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              Advanced Sentiment Analysis
            </span>
          </p>
        </motion.div>

        {/* Founders Carousel */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="w-full max-w-6xl mx-auto"
        >
          <h2 className="text-4xl font-bold text-center mb-12 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            Meet Our Visionary Team
          </h2>
          <div className="relative">
            <Carousel
              opts={{
                align: "start",
                loop: true,
              }}
              className="w-full"
            >
              <CarouselContent>
                {founders.map((founder, index) => (
                  <CarouselItem key={index} className="md:basis-1/2 lg:basis-1/3 pl-4">
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="group relative bg-white/5 backdrop-blur-xl p-6 h-full border-0 hover:bg-white/10 transition-all duration-300"
                    >
                      <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-purple-500/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                      <div className="relative">
                        <div className="aspect-square bg-gradient-to-br from-blue-600/20 to-purple-600/20 flex items-center justify-center mb-4">
                          <img src={founder.image} alt={founder.name} className="w-full h-full object-cover"/>
                        </div>
                        <h3 className="text-xl font-bold text-blue-300 mb-2">{founder.name}</h3>
                        <p className="text-blue-200 mb-3">{founder.role}</p>
                        <p className="text-sm text-blue-100/80">{founder.description}</p>
                      </div>
                    </motion.div>
                  </CarouselItem>
                ))}
              </CarouselContent>
              <div className="absolute -left-12 top-1/2 -translate-y-1/2 z-10">
                <CarouselPrevious className="bg-white/10 border-0 text-white hover:bg-white/20" />
              </div>
              <div className="absolute -right-12 top-1/2 -translate-y-1/2 z-10">
                <CarouselNext className="bg-white/10 border-0 text-white hover:bg-white/20" />
              </div>
            </Carousel>
          </div>
        </motion.div>

        {/* Technology Stack & Features */}
        <div className="grid md:grid-cols-2 gap-8 max-w-6xl mx-auto px-4">
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
            className="bg-white/5 backdrop-blur-xl p-8"
          >
            <h3 className="text-2xl font-bold mb-6 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              Advanced Technology Stack
            </h3>
            <ul className="space-y-4 text-blue-200">
              <li className="flex items-center space-x-3">
                <span className="w-8 h-8 flex items-center justify-center bg-blue-500/20">
                  <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                  </svg>
                </span>
                <span>Deep Learning NLP Models</span>
              </li>
              <li className="flex items-center space-x-3">
                <span className="w-8 h-8 flex items-center justify-center bg-blue-500/20">
                  <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                  </svg>
                </span>
                <span>Real-time Data Processing</span>
              </li>
              <li className="flex items-center space-x-3">
                <span className="w-8 h-8 flex items-center justify-center bg-blue-500/20">
                  <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                  </svg>
                </span>
                <span>Multilingual Sentiment Analysis</span>
              </li>
            </ul>
          </motion.div>

          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
            className="bg-white/5 backdrop-blur-xl p-8"
          >
            <h3 className="text-2xl font-bold mb-6 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              Impact & Innovation
            </h3>
            <ul className="space-y-4 text-blue-200">
              <li className="flex items-center space-x-3">
                <span className="w-8 h-8 flex items-center justify-center bg-blue-500/20">
                  <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                  </svg>
                </span>
                <span>Enhanced Disaster Response</span>
              </li>
              <li className="flex items-center space-x-3">
                <span className="w-8 h-8 flex items-center justify-center bg-blue-500/20">
                  <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                  </svg>
                </span>
                <span>Data-Driven Decision Making</span>
              </li>
              <li className="flex items-center space-x-3">
                <span className="w-8 h-8 flex items-center justify-center bg-blue-500/20">
                  <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                  </svg>
                </span>
                <span>Community-Centric Solutions</span>
              </li>
            </ul>
          </motion.div>
        </div>

        {/* About Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="w-full max-w-4xl mx-auto px-4 space-y-8 text-center"
        >
          <h2 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            About PanicSense PH
          </h2>
          <div className="prose prose-invert max-w-none space-y-6">
            <p className="text-lg text-blue-200 leading-relaxed">
              PanicSense PH revolutionizes disaster response through cutting-edge deep learning and natural language processing technologies. Our system performs real-time sentiment analysis during crisis events, utilizing a sophisticated neural architecture that combines transformer-based models with custom attention mechanisms.
            </p>
            <p className="text-lg text-blue-200 leading-relaxed">
              Our platform employs advanced transfer learning techniques and fine-tuned language models capable of processing both English and Filipino text with state-of-the-art accuracy. Our innovative multilingual approach ensures contextual understanding of cultural nuances and colloquial expressions across multiple Filipino dialects.
            </p>
            <p className="text-lg text-blue-200 leading-relaxed">
              Through ensemble methods and reinforcement learning algorithms, the system dynamically categorizes emotional states into five distinct classifications: Panic, Fear/Anxiety, Disbelief, Resilience, and Neutral. This granular emotion mapping provides crucial decision support for emergency response coordination and resource allocation during disaster events.
            </p>
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
}