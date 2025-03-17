import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useIsMobile } from '@/hooks/use-mobile';

interface TeamMember {
  id: number;
  name: string;
  role: string;
  image: string;
  description: string;
}

const teamMembers: TeamMember[] = [
  {
    id: 1,
    name: "Mark Andrei",
    role: "Lead Developer",
    image: "https://i.postimg.cc/s9Md8RYv/drei.png",
    description: "Specializes in AI integration and backend development."
  },
  {
    id: 2,
    name: "Julia",
    role: "UX Designer",
    image: "https://i.postimg.cc/bxpbndZ/julia.png",
    description: "Creates intuitive user experiences and visual designs."
  },
  {
    id: 3,
    name: "Ivahnn",
    role: "Data Scientist",
    image: "https://i.postimg.cc/ZRM8FLdD/van.png",
    description: "Expert in machine learning and sentiment analysis algorithms."
  }
];

export function TeamCarousel() {
  const [activeIndex, setActiveIndex] = useState(0);
  const isMobile = useIsMobile();
  
  // Auto-rotation
  useEffect(() => {
    // Only auto-rotate on mobile
    if (!isMobile) return;
    
    const timer = setInterval(() => {
      setActiveIndex((prev) => (prev + 1) % teamMembers.length);
    }, 3000); // 3 seconds interval
    
    return () => clearInterval(timer);
  }, [isMobile]);
  
  const slideVariants = {
    enter: (direction: number) => {
      return {
        x: direction > 0 ? '100%' : '-100%',
        opacity: 0
      };
    },
    center: {
      x: 0,
      opacity: 1
    },
    exit: (direction: number) => {
      return {
        x: direction < 0 ? '100%' : '-100%',
        opacity: 0
      };
    }
  };

  const handleDotClick = (index: number) => {
    setActiveIndex(index);
  };

  return (
    <div className="relative overflow-hidden rounded-lg bg-gradient-to-br from-blue-50 to-indigo-50 p-6 shadow-md">
      <h2 className="text-2xl font-bold text-center mb-8 text-blue-800">Our Team</h2>
      
      <div className="relative h-[420px] md:h-[320px]">
        <AnimatePresence initial={false} custom={1}>
          <motion.div
            key={activeIndex}
            custom={1}
            variants={slideVariants}
            initial="enter"
            animate="center"
            exit="exit"
            transition={{
              x: { type: "spring", stiffness: 300, damping: 30 },
              opacity: { duration: 0.5 }
            }}
            className="absolute w-full"
          >
            <div className="flex flex-col md:flex-row items-center justify-center gap-6 p-4">
              <div className="relative">
                <div className="w-48 h-48 overflow-hidden rounded-full border-4 border-white shadow-lg">
                  <img 
                    src={teamMembers[activeIndex].image} 
                    alt={teamMembers[activeIndex].name}
                    className="w-full h-full object-cover"
                  />
                </div>
                <motion.div 
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ delay: 0.3, duration: 0.4 }}
                  className="absolute -bottom-2 -right-2 bg-blue-600 text-white rounded-full px-3 py-1 text-sm font-semibold shadow-md"
                >
                  {teamMembers[activeIndex].role}
                </motion.div>
              </div>
              
              <div className="text-center md:text-left md:max-w-md">
                <motion.h3 
                  initial={{ y: 20, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ delay: 0.2, duration: 0.4 }}
                  className="text-xl font-bold text-blue-800 mb-2"
                >
                  {teamMembers[activeIndex].name}
                </motion.h3>
                <motion.p 
                  initial={{ y: 20, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ delay: 0.4, duration: 0.4 }}
                  className="text-gray-600"
                >
                  {teamMembers[activeIndex].description}
                </motion.p>
              </div>
            </div>
          </motion.div>
        </AnimatePresence>
      </div>
      
      {/* Dots indicator */}
      <div className="flex justify-center mt-4 gap-2">
        {teamMembers.map((_, index) => (
          <button
            key={index}
            onClick={() => handleDotClick(index)}
            className={`w-2 h-2 rounded-full transition-all ${
              index === activeIndex 
                ? 'bg-blue-600 w-6' 
                : 'bg-gray-300 hover:bg-gray-400'
            }`}
            aria-label={`View team member ${index + 1}`}
          />
        ))}
      </div>
    </div>
  );
}