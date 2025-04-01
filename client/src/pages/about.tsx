import React from "react";
import { motion } from "framer-motion";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselNext,
  CarouselPrevious,
} from "@/components/ui/carousel";
import { useIsMobile } from "@/hooks/use-mobile";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { BrainCircuit, Users, Lightbulb, Webhook, Globe, BarChart } from "lucide-react";

export default function About() {
  const [api, setApi] = React.useState<any>();
  const [currentSlide, setCurrentSlide] = React.useState(0);
  const isMobile = useIsMobile();
  const [isPaused, setIsPaused] = React.useState(false); // Added state for pause

  // Hook to track slide changes
  React.useEffect(() => {
    if (!api) return;

    // Set up slide change detection
    const handleSelect = () => {
      const selectedIndex = api.selectedScrollSnap();
      setCurrentSlide(selectedIndex);
    };

    api.on("select", handleSelect);
    return () => {
      api.off("select", handleSelect);
    };
  }, [api]);

  // Auto-rotate carousel on mobile, with pause functionality
  React.useEffect(() => {
    if (!isMobile || !api) return;

    let interval: NodeJS.Timeout;
    if (!isPaused) {
      interval = setInterval(() => {
        api.scrollNext();
      }, 3000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isMobile, api, isPaused]); // isPaused added to dependency array

  const founders = [
    {
      name: "Mark Andrei R. Castillo",
      role: "Core System Architecture & Machine Learning",
      image: "https://raw.githubusercontent.com/draiimon/PanicSense/main/client/public/images/drei.jpg",
      description: "Leads the development of our advanced ML pipelines and system architecture"
    },
    {
      name: "Ivahnn B. Garcia",
      role: "Frontend Development & User Experience",
      image: "https://raw.githubusercontent.com/draiimon/PanicSense/main/client/public/images/van.jpg",
      description: "Creates intuitive and responsive user interfaces for seamless interaction"
    },
    {
      name: "Julia Daphne Ngan-Gatdula",
      role: "Data Resources & Information Engineering",
      image: "https://raw.githubusercontent.com/draiimon/PanicSense/main/client/public/images/julia.jpg",
      description: "Manages data infrastructure and information processing systems"
    }
  ];

  // Animation variants for staggered animation
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };
  
  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { type: "spring", stiffness: 100 }
    }
  };

  return (
    <div className="relative min-h-screen w-full flex flex-col items-center justify-center">
      {/* Animated Background EXACTLY like dashboard */}
      <div className="fixed inset-0 -z-10 bg-gradient-to-b from-violet-50 to-pink-50 overflow-hidden">
        {/* More vibrant animated gradient overlay - CSS Animation */}
        <div
          className="absolute inset-0 bg-gradient-to-r from-purple-500/15 via-teal-500/10 to-rose-500/15 animate-gradient"
          style={{ backgroundSize: "200% 200%" }}
        />

        {/* Enhanced animated patterns with more vibrant colors */}
        <div className="absolute inset-0 opacity-15 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIGZpbGw9IiM1MDUwRjAiIGZpbGwtb3BhY2l0eT0iMC41Ij48cGF0aCBkPSJNMzYgMzR2Nmg2di02aC02em02IDZ2Nmg2di02aC02em0tMTIgMGg2djZoLTZ2LTZ6bTEyIDBoNnY2aC02di02eiIvPjwvZz48L2c+PC9zdmc+')]"></div>

        {/* Additional decorative elements */}
        <div className="absolute inset-0 opacity-10 bg-[radial-gradient(circle_at_center,rgba(120,80,255,0.8)_0%,transparent_70%)]"></div>

        {/* More colorful floating elements - USING CSS ANIMATIONS */}
        <div
          className="absolute h-72 w-72 rounded-full bg-purple-500/25 filter blur-3xl animate-float-1 will-change-transform"
          style={{ top: "15%", left: "8%" }}
        />

        <div
          className="absolute h-64 w-64 rounded-full bg-teal-500/20 filter blur-3xl animate-float-2 will-change-transform"
          style={{ bottom: "15%", right: "15%" }}
        />

        <div
          className="absolute h-52 w-52 rounded-full bg-purple-500/25 filter blur-3xl animate-float-3 will-change-transform"
          style={{ top: "45%", right: "20%" }}
        />
        
        {/* Additional floating elements for more color - USING CSS ANIMATIONS */}
        <div
          className="absolute h-48 w-48 rounded-full bg-pink-500/20 filter blur-3xl animate-float-4 will-change-transform"
          style={{ top: "65%", left: "25%" }}
        />

        <div
          className="absolute h-40 w-40 rounded-full bg-yellow-400/15 filter blur-3xl animate-float-5 will-change-transform"
          style={{ top: "30%", left: "40%" }}
        />

        {/* Shimmer effect with CSS */}
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent"></div>
      </div>

      <motion.div
        initial="hidden"
        animate="visible"
        variants={containerVariants}
        className="relative w-full space-y-16 py-12 px-4 max-w-7xl mx-auto"
      >
        {/* Hero Section */}
        <motion.div
          variants={itemVariants}
          className="text-center space-y-6"
        >
          <h1 className="text-6xl md:text-7xl font-bold">
            <span className="bg-gradient-to-r from-violet-700 via-indigo-600 to-blue-600 bg-clip-text text-transparent">
              PanicSense PH
            </span>
          </h1>
          <p className="text-2xl md:text-3xl text-slate-700 max-w-3xl mx-auto leading-relaxed">
            Revolutionizing Disaster Response Through
            <br />
            <span className="font-semibold bg-gradient-to-r from-violet-600 to-blue-600 bg-clip-text text-transparent">
              Advanced Sentiment Analysis
            </span>
          </p>
        </motion.div>

        {/* Founders Carousel */}
        <motion.div
          variants={itemVariants}
          className="w-full max-w-6xl mx-auto"
        >
          <Card className="overflow-hidden shadow-lg border-0 bg-gradient-to-r from-purple-100/90 to-blue-100/90 backdrop-blur-sm">
            <CardHeader className="bg-gradient-to-r from-purple-500/90 to-blue-500/90 border-b border-gray-200/40 pb-4">
              <div className="flex justify-between items-center">
                <div>
                  <CardTitle className="text-2xl font-semibold text-white flex items-center gap-2">
                    <Users className="h-6 w-6 text-white" />
                    Meet Our Visionary Team
                  </CardTitle>
                  <CardDescription className="text-white/80">
                    The passionate developers behind PanicSense PH
                  </CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="p-6 bg-white/80 backdrop-blur-sm">
              <div className="relative">
                <Carousel
                  opts={{
                    align: "start",
                    loop: true,
                  }}
                  setApi={setApi}
                  className="w-full overflow-hidden"
                >
                  <CarouselContent>
                    {founders.map((founder, index) => (
                      <CarouselItem key={index} className="md:basis-1/2 lg:basis-1/3">
                        <motion.div
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: index * 0.1 }}
                          className="group relative bg-gradient-to-br from-white/80 to-violet-50/80 p-6 rounded-xl h-full border border-violet-100/50 hover:shadow-lg transition-all duration-300"
                          onMouseEnter={() => setIsPaused(true)}
                          onMouseLeave={() => setIsPaused(false)}
                        >
                          <div className="absolute inset-0 bg-gradient-to-r from-violet-100/40 to-blue-100/40 opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-xl" />
                          <div className="relative">
                            <div className="aspect-square bg-gradient-to-br from-violet-100 to-indigo-100 rounded-xl flex items-center justify-center mb-4 overflow-hidden shadow-md">
                              <img src={founder.image} alt={founder.name} className="w-full h-full object-cover rounded-xl"/>
                            </div>
                            <h3 className="text-xl font-bold text-slate-800 mb-2">{founder.name}</h3>
                            <p className="text-violet-700 mb-3 font-medium">{founder.role}</p>
                            <p className="text-sm text-slate-600">{founder.description}</p>
                          </div>
                        </motion.div>
                      </CarouselItem>
                    ))}
                  </CarouselContent>
                  {/* Navigation buttons */}
                  <div className="absolute -left-4 top-1/2 -translate-y-1/2 z-10 hidden md:block">
                    <CarouselPrevious className="bg-violet-100 hover:bg-violet-200 border-0 text-violet-700 rounded-full shadow-md" />
                  </div>
                  <div className="absolute -right-4 top-1/2 -translate-y-1/2 z-10 hidden md:block">
                    <CarouselNext className="bg-violet-100 hover:bg-violet-200 border-0 text-violet-700 rounded-full shadow-md" />
                  </div>

                  {/* Mobile indicator dots */}
                  <div className="flex justify-center gap-2 mt-6 md:hidden">
                    {[0, 1, 2].map((index) => (
                      <button
                        key={index}
                        onClick={() => {
                          if (api) {
                            api.scrollTo(index);
                            setCurrentSlide(index);
                          }
                        }}
                        className={`w-2 h-2 rounded-full transition-all ${
                          currentSlide === index
                            ? "bg-violet-600 w-4"
                            : "bg-violet-300"
                        }`}
                        aria-label={`Go to slide ${index + 1}`}
                      />
                    ))}
                  </div>
                </Carousel>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Technology Stack & Features */}
        <div className="grid md:grid-cols-2 gap-8">
          <motion.div
            variants={itemVariants}
          >
            <Card className="h-full overflow-hidden shadow-lg border-0 bg-gradient-to-r from-blue-100/90 to-teal-100/90 backdrop-blur-sm">
              <CardHeader className="bg-gradient-to-r from-blue-500/90 to-teal-500/90 border-b border-gray-200/40 pb-4">
                <div className="flex justify-between items-center">
                  <div>
                    <CardTitle className="text-lg font-semibold text-white flex items-center gap-2">
                      <BrainCircuit className="h-5 w-5 text-white" />
                      Advanced Technology Stack
                    </CardTitle>
                    <CardDescription className="text-white/80">
                      Cutting-edge technologies powering our platform
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="p-6 bg-white/80 backdrop-blur-sm">
                <ul className="space-y-4">
                  <li className="flex items-center space-x-4">
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center shadow-sm">
                      <Webhook className="h-4 w-4 text-blue-600" />
                    </div>
                    <div>
                      <p className="font-medium text-slate-800">Deep Learning NLP Models</p>
                      <p className="text-sm text-slate-600 mt-1">Transformer-based architecture with custom attention mechanisms</p>
                    </div>
                  </li>
                  <li className="flex items-center space-x-4">
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center shadow-sm">
                      <BarChart className="h-4 w-4 text-blue-600" />
                    </div>
                    <div>
                      <p className="font-medium text-slate-800">Real-time Data Processing</p>
                      <p className="text-sm text-slate-600 mt-1">High-performance streaming analytics for immediate insights</p>
                    </div>
                  </li>
                  <li className="flex items-center space-x-4">
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center shadow-sm">
                      <Globe className="h-4 w-4 text-blue-600" />
                    </div>
                    <div>
                      <p className="font-medium text-slate-800">Multilingual Sentiment Analysis</p>
                      <p className="text-sm text-slate-600 mt-1">Support for both English and Filipino text with contextual understanding</p>
                    </div>
                  </li>
                </ul>
              </CardContent>
            </Card>
          </motion.div>

          <motion.div
            variants={itemVariants}
          >
            <Card className="h-full overflow-hidden shadow-lg border-0 bg-gradient-to-r from-pink-100/90 to-rose-100/90 backdrop-blur-sm">
              <CardHeader className="bg-gradient-to-r from-pink-500/90 to-rose-500/90 border-b border-gray-200/40 pb-4">
                <div className="flex justify-between items-center">
                  <div>
                    <CardTitle className="text-lg font-semibold text-white flex items-center gap-2">
                      <Lightbulb className="h-5 w-5 text-white" />
                      Impact & Innovation
                    </CardTitle>
                    <CardDescription className="text-white/80">
                      How our solution makes a difference
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="p-6 bg-white/80 backdrop-blur-sm">
                <ul className="space-y-4">
                  <li className="flex items-center space-x-4">
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-pink-100 flex items-center justify-center shadow-sm">
                      <svg className="h-4 w-4 text-pink-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                      </svg>
                    </div>
                    <div>
                      <p className="font-medium text-slate-800">Enhanced Disaster Response</p>
                      <p className="text-sm text-slate-600 mt-1">Improved resource allocation based on real-time emotional analysis</p>
                    </div>
                  </li>
                  <li className="flex items-center space-x-4">
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-pink-100 flex items-center justify-center shadow-sm">
                      <svg className="h-4 w-4 text-pink-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                      </svg>
                    </div>
                    <div>
                      <p className="font-medium text-slate-800">Data-Driven Decision Making</p>
                      <p className="text-sm text-slate-600 mt-1">Actionable insights from social media analysis during crises</p>
                    </div>
                  </li>
                  <li className="flex items-center space-x-4">
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-pink-100 flex items-center justify-center shadow-sm">
                      <svg className="h-4 w-4 text-pink-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                      </svg>
                    </div>
                    <div>
                      <p className="font-medium text-slate-800">Community-Centric Solutions</p>
                      <p className="text-sm text-slate-600 mt-1">Focused on addressing the specific needs of Filipino communities</p>
                    </div>
                  </li>
                </ul>
              </CardContent>
            </Card>
          </motion.div>
        </div>

        {/* About Section */}
        <motion.div
          variants={itemVariants}
        >
          <Card className="overflow-hidden shadow-lg border-0 bg-gradient-to-r from-violet-100/90 to-indigo-100/90 backdrop-blur-sm">
            <CardHeader className="bg-gradient-to-r from-violet-500/90 to-indigo-500/90 border-b border-gray-200/40 pb-4">
              <div className="flex justify-between items-center">
                <div>
                  <CardTitle className="text-lg font-semibold text-white flex items-center gap-2">
                    <Users className="h-5 w-5 text-white" />
                    About PanicSense PH
                  </CardTitle>
                  <CardDescription className="text-white/80">
                    The vision and mission behind our platform
                  </CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="p-6 bg-white/80 backdrop-blur-sm">
              <div className="prose max-w-none">
                <p className="text-slate-700 leading-relaxed mb-4">
                  PanicSense PH revolutionizes disaster response through
                  cutting-edge deep learning and natural language processing
                  technologies. Our system performs real-time sentiment analysis
                  during crisis events, utilizing a sophisticated neural
                  architecture that combines transformer-based models with custom
                  attention mechanisms.
                </p>
                <p className="text-slate-700 leading-relaxed mb-4">
                  Our platform employs advanced transfer learning techniques and
                  fine-tuned language models capable of processing both English and
                  Filipino text with state-of-the-art accuracy. Our innovative
                  multilingual approach ensures contextual understanding of cultural
                  nuances and colloquial expressions across multiple Filipino
                  dialects.
                </p>
                <p className="text-slate-700 leading-relaxed">
                  Through ensemble methods and reinforcement learning algorithms,
                  the system dynamically categorizes emotional states into five
                  distinct classifications: Panic, Fear/Anxiety, Disbelief,
                  Resilience, and Neutral. This granular emotion mapping provides
                  crucial decision support for emergency response coordination and
                  resource allocation during disaster events.
                </p>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </motion.div>
    </div>
  );
}
