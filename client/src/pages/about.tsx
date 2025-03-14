import React from 'react';
import { motion } from "framer-motion";

export default function AboutPage() {
  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="space-y-8"
    >
      <div className="bg-gradient-to-r from-slate-800 to-slate-900 rounded-xl p-8 shadow-lg">
        <h1 className="text-3xl font-bold text-white mb-4">About PanicSense PH</h1>
        <div className="prose prose-invert max-w-none">
          <p className="text-slate-300 text-lg">
            PanicSense PH harnesses cutting-edge deep learning and natural language processing technologies to perform real-time sentiment analysis during crisis events. Our system employs a sophisticated neural architecture that combines transformer-based models with custom attention mechanisms specifically optimized for disaster-related content.
          </p>
          <p className="text-slate-300 text-lg">
            The platform implements advanced transfer learning techniques and fine-tuned language models that can process both English and Filipino text with state-of-the-art accuracy. Our novel cross-lingual embedding approach enables contextual understanding of cultural nuances and colloquial expressions across multiple Filipino dialects.
          </p>
          <p className="text-slate-300 text-lg">
            Utilizing ensemble methods and reinforcement learning algorithms, the system dynamically categorizes emotional states into five distinct classifications: Panic, Fear/Anxiety, Disbelief, Resilience, and Neutral. This granular emotion mapping provides critical decision support for emergency response coordination and resource allocation during disaster events.
          </p>
        </div>
      </div>

      <motion.div 
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.2, duration: 0.5 }}
        className="bg-white rounded-xl p-8 shadow-lg"
      >
        <h2 className="text-2xl font-bold text-slate-800 mb-6">THE FOUNDERS</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="space-y-4">
            <div className="aspect-square bg-slate-200 rounded-xl flex items-center justify-center">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-24 h-24 text-slate-400">
                <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 1 1-7.5 0 3.75 3.75 0 0 1 7.5 0ZM4.501 20.118a7.5 7.5 0 0 1 14.998 0A17.933 17.933 0 0 1 12 21.75c-2.676 0-5.216-.584-7.499-1.632Z" />
              </svg>
            </div>
            <h3 className="text-xl font-bold text-slate-800">
              Mark Andrei R. Castillo
            </h3>
            <p className="text-slate-600">Core System Architecture &amp; Machine Learning</p>
          </div>
          <div className="space-y-4">
            <div className="aspect-square bg-slate-200 rounded-xl flex items-center justify-center">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-24 h-24 text-slate-400">
                <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 1 1-7.5 0 3.75 3.75 0 0 1 7.5 0ZM4.501 20.118a7.5 7.5 0 0 1 14.998 0A17.933 17.933 0 0 1 12 21.75c-2.676 0-5.216-.584-7.499-1.632Z" />
              </svg>
            </div>
            <h3 className="text-xl font-bold text-slate-800">Ivahnn B. Garcia</h3>
            <p className="text-slate-600">Frontend Development &amp; User Experience</p>
          </div>
          <div className="space-y-4">
            <div className="aspect-square bg-slate-200 rounded-xl flex items-center justify-center">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-24 h-24 text-slate-400">
                <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 1 1-7.5 0 3.75 3.75 0 0 1 7.5 0ZM4.501 20.118a7.5 7.5 0 0 1 14.998 0A17.933 17.933 0 0 1 12 21.75c-2.676 0-5.216-.584-7.499-1.632Z" />
              </svg>
            </div>
            <h3 className="text-xl font-bold text-slate-800">Julia Daphne Ngan-Gatdula</h3>
            <p className="text-slate-600">Data Resources &amp; Information Engineering</p>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}