
import { motion } from "framer-motion";
import { useTheme } from "@/context/theme-context";

export default function About() {
  const { theme } = useTheme();
  
  const fadeIn = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { 
        duration: 0.8,
        ease: "easeOut"
      }
    }
  };
  
  const staggerChildren = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.3
      }
    }
  };

  return (
    <div className="max-w-5xl mx-auto py-10 px-4 sm:px-6">
      <motion.div
        initial="hidden"
        animate="visible"
        variants={fadeIn}
        className="text-center mb-14"
      >
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight bg-gradient-to-r from-purple-600 via-blue-500 to-green-400 bg-clip-text text-transparent pb-2">
          About PanicSense PH
        </h1>
        <p className="mt-4 text-xl text-gray-600 dark:text-gray-300">
          Advanced sentiment analysis for disaster response
        </p>
      </motion.div>

      <motion.div 
        initial="hidden"
        animate="visible"
        variants={staggerChildren}
        className="grid grid-cols-1 md:grid-cols-2 gap-10 mb-16"
      >
        <motion.div variants={fadeIn} className="rounded-2xl overflow-hidden shadow-xl transform hover:scale-105 transition-transform duration-300">
          <div className={`${theme === 'dark' ? 'bg-slate-800' : 'bg-white'} p-8 h-full`}>
            <h2 className="text-2xl font-bold mb-4 text-blue-600 dark:text-blue-400">Our Mission</h2>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
              PanicSense PH is dedicated to transforming disaster response through cutting-edge sentiment analysis, providing critical insights that save lives and optimize emergency resource allocation.
            </p>
          </div>
        </motion.div>

        <motion.div variants={fadeIn} className="rounded-2xl overflow-hidden shadow-xl transform hover:scale-105 transition-transform duration-300">
          <div className={`${theme === 'dark' ? 'bg-slate-800' : 'bg-white'} p-8 h-full`}>
            <h2 className="text-2xl font-bold mb-4 text-blue-600 dark:text-blue-400">Our Impact</h2>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
              By providing real-time sentiment analysis during disasters, we help emergency responders prioritize areas of greatest need and emotional distress, creating a more effective and compassionate response system.
            </p>
          </div>
        </motion.div>
      </motion.div>

      <motion.div 
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.4 }}
        className={`${theme === 'dark' ? 'bg-slate-800' : 'bg-white'} rounded-2xl shadow-xl p-8 mb-16`}
      >
        <h2 className="text-3xl font-bold mb-6 text-center text-purple-600 dark:text-purple-400">Advanced Technology</h2>
        <p className="text-lg text-gray-700 dark:text-gray-300 leading-relaxed mb-6">
          PanicSense PH leverages state-of-the-art transformer-based neural networks and deep learning architectures to analyze text data in real-time. Our proprietary ensemble model combines:
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-5 bg-gradient-to-br from-blue-50 to-purple-50 dark:from-slate-700 dark:to-slate-800">
            <h3 className="font-bold text-xl mb-2 text-blue-700 dark:text-blue-300">Bidirectional Encoder Representations</h3>
            <p className="text-gray-700 dark:text-gray-300">
              Contextual language understanding with transformer architecture for nuanced semantic interpretation of disaster-related communications.
            </p>
          </div>
          <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-5 bg-gradient-to-br from-green-50 to-blue-50 dark:from-slate-700 dark:to-slate-800">
            <h3 className="font-bold text-xl mb-2 text-green-700 dark:text-green-300">Multi-Head Attention Mechanisms</h3>
            <p className="text-gray-700 dark:text-gray-300">
              Parallel attention processing that captures complex relationships between words and phrases in both English and Filipino contexts.
            </p>
          </div>
          <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-5 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-slate-700 dark:to-slate-800">
            <h3 className="font-bold text-xl mb-2 text-purple-700 dark:text-purple-300">Cross-Lingual Transfer Learning</h3>
            <p className="text-gray-700 dark:text-gray-300">
              Zero-shot and few-shot learning capabilities that generalize across languages, recognizing cultural nuances and local expressions.
            </p>
          </div>
          <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-5 bg-gradient-to-br from-pink-50 to-orange-50 dark:from-slate-700 dark:to-slate-800">
            <h3 className="font-bold text-xl mb-2 text-pink-700 dark:text-pink-300">Dynamic Temporal Analysis</h3>
            <p className="text-gray-700 dark:text-gray-300">
              Recurrent neural components that track sentiment evolution throughout disaster events, providing trend analysis and predictive insights.
            </p>
          </div>
        </div>
      </motion.div>

      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1, delay: 0.8 }}
        className="mb-16"
      >
        <h2 className="text-3xl font-bold mb-8 text-center text-green-600 dark:text-green-400">Sentiment Classification Framework</h2>
        <div className="overflow-hidden rounded-xl shadow-lg">
          <div className={`${theme === 'dark' ? 'bg-slate-800' : 'bg-gray-50'} p-6`}>
            <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
              {[
                { name: "Panic", color: "from-red-500 to-orange-500", description: "Emergency distress signals requiring immediate intervention" },
                { name: "Fear/Anxiety", color: "from-orange-400 to-amber-400", description: "Heightened uncertainty and concern requiring reassurance" },
                { name: "Disbelief", color: "from-purple-400 to-indigo-400", description: "Cognitive disconnect from traumatic situations indicating shock" },
                { name: "Resilience", color: "from-green-400 to-emerald-400", description: "Adaptive coping mechanisms and community strength" },
                { name: "Neutral", color: "from-blue-400 to-sky-400", description: "Informational content with minimal emotional signaling" }
              ].map((sentiment, i) => (
                <motion.div 
                  key={i}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 1 + (i * 0.2) }}
                  className="rounded-lg overflow-hidden shadow-md"
                >
                  <div className={`h-2 bg-gradient-to-r ${sentiment.color}`}></div>
                  <div className={`p-4 ${theme === 'dark' ? 'bg-slate-700' : 'bg-white'}`}>
                    <h3 className="font-bold text-lg mb-2">{sentiment.name}</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-300">{sentiment.description}</p>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      </motion.div>

      <motion.section
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 1.2 }}
        className={`${theme === 'dark' ? 'bg-slate-800' : 'bg-white'} rounded-2xl shadow-xl p-8`}
      >
        <h2 className="text-3xl font-bold mb-8 text-center text-orange-600 dark:text-orange-400">The Founders</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="text-center">
            <div className="w-32 h-32 mx-auto bg-gradient-to-br from-blue-400 to-indigo-600 rounded-full flex items-center justify-center mb-4">
              <p className="text-white font-bold text-lg">MC</p>
            </div>
            <h3 className="text-xl font-bold">Castillo, Mark Andrei R.</h3>
            <p className="text-gray-600 dark:text-gray-300 mt-2"><strong>Lead ML Engineer</strong></p>
            <p className="text-gray-600 dark:text-gray-300 mt-1">Neural architecture design and model optimization</p>
          </div>
          
          <div className="text-center">
            <div className="w-32 h-32 mx-auto bg-gradient-to-br from-purple-400 to-pink-600 rounded-full flex items-center justify-center mb-4">
              <p className="text-white font-bold text-lg">IG</p>
            </div>
            <h3 className="text-xl font-bold">Garcia, Ivahnn</h3>
            <p className="text-gray-600 dark:text-gray-300 mt-2">Frontend Engineering</p>
            <p className="text-gray-600 dark:text-gray-300 mt-1">User experience and interactive visualizations</p>
          </div>
          
          <div className="text-center">
            <div className="w-32 h-32 mx-auto bg-gradient-to-br from-green-400 to-teal-600 rounded-full flex items-center justify-center mb-4">
              <p className="text-white font-bold text-lg">JG</p>
            </div>
            <h3 className="text-xl font-bold">Gatdula, Julia Daphne Ngan</h3>
            <p className="text-gray-600 dark:text-gray-300 mt-2">Data Science</p>
            <p className="text-gray-600 dark:text-gray-300 mt-1">Data processing and analytical framework</p>
          </div>
        </div>
      </motion.section>
    </div>
  );
}
