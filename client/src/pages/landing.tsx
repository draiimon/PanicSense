import React, { useState } from 'react';
import { Link } from 'wouter';
import { motion } from 'framer-motion';
import { ChevronRight, X, FileText, BarChart3, AlertTriangle, MapPin, Clock, Database, ArrowRight, Info, CheckSquare } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

const fadeIn = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.6 } }
};

const Tutorial = ({ onClose }: { onClose: () => void }) => {
  const [currentStep, setCurrentStep] = useState(0);
  
  const steps = [
    {
      title: "Pag-upload ng Data",
      description: "Mag-upload ng CSV file na naglalaman ng mga text tungkol sa disaster upang makapagsimula.",
      icon: <FileText size={24} />,
      image: "/images/tutorial-1.png"
    },
    {
      title: "Pag-analyze ng Sentiment",
      description: "Ang system ay awtomatikong susuriin ang damdamin at klasipikasyon ng bawat mensahe.",
      icon: <BarChart3 size={24} />,
      image: "/images/tutorial-2.png"
    },
    {
      title: "Geographic Analysis",
      description: "Tingnan kung saan nangyari ang mga disaster at i-plot sa mapa ng Pilipinas.",
      icon: <MapPin size={24} />,
      image: "/images/tutorial-3.png"
    },
    {
      title: "Real-time Monitoring",
      description: "Subaybayan ang mga bagong ulat ng disaster sa real-time para sa mabilis na pagtugon.",
      icon: <Clock size={24} />,
      image: "/images/tutorial-4.png"
    }
  ];
  
  const nextStep = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      onClose();
    }
  };
  
  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };
  
  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <motion.div 
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.9 }}
        className="bg-card text-card-foreground rounded-lg shadow-lg max-w-3xl w-full relative overflow-hidden"
      >
        <button 
          onClick={onClose}
          className="absolute top-4 right-4 p-1 rounded-full bg-muted/20 hover:bg-muted transition-colors z-10"
        >
          <X size={20} />
        </button>
        
        <div className="flex flex-col md:flex-row h-[500px]">
          <div className="w-full md:w-1/2 bg-gradient-to-br from-blue-600 to-indigo-800 p-6 text-white relative">
            <div className="absolute top-4 left-4 flex space-x-1">
              {steps.map((_, index) => (
                <div 
                  key={index}
                  className={`w-2 h-2 rounded-full ${currentStep === index ? 'bg-white' : 'bg-white/30'}`}
                />
              ))}
            </div>
            
            <div className="h-full flex flex-col justify-center">
              <div className="mb-4 p-3 bg-white/10 rounded-full w-fit">
                {steps[currentStep].icon}
              </div>
              <h3 className="text-2xl font-bold mb-2">{steps[currentStep].title}</h3>
              <p className="text-white/80 mb-6">{steps[currentStep].description}</p>
              
              <div className="flex space-x-3 mt-auto">
                {currentStep > 0 && (
                  <Button 
                    variant="outline" 
                    className="border-white/20 text-white hover:bg-white/10"
                    onClick={prevStep}
                  >
                    Nakaraang Hakbang
                  </Button>
                )}
                <Button 
                  onClick={nextStep}
                  className="bg-white text-blue-700 hover:bg-white/90"
                >
                  {currentStep === steps.length - 1 ? 'Simulan' : 'Susunod na Hakbang'}
                  <ChevronRight className="ml-1 h-4 w-4" />
                </Button>
              </div>
            </div>
          </div>
          
          <div className="w-full md:w-1/2 flex items-center justify-center p-6 bg-gradient-to-b from-muted/50 to-transparent">
            {/* Always use logo as fallback */}
            <img 
              src="/images/PANICSENSE PH.png" 
              alt={steps[currentStep].title}
              className="max-w-full max-h-[320px] rounded-lg shadow-lg"
            />
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default function LandingPage() {
  const [showTutorial, setShowTutorial] = useState(false);
  
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-100 to-slate-200 dark:from-gray-900 dark:to-gray-950">
      {/* Header */}
      <header className="w-full py-4 px-6 bg-white dark:bg-gray-900 shadow-sm">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center space-x-2">
            <img src="/images/PANICSENSE PH.png" alt="PanicSense PH Logo" className="h-10" />
            <h1 className="text-xl font-bold text-blue-700 dark:text-blue-400">PanicSense PH</h1>
          </div>
          <div className="flex items-center space-x-4">
            <Link href="/dashboard">
              <Button>
                Mag-sign in
              </Button>
            </Link>
          </div>
        </div>
      </header>
      
      {/* Hero Section */}
      <section className="relative py-20 overflow-hidden">
        <div className="absolute inset-0 z-0 bg-gradient-to-br from-blue-100/50 to-indigo-100/50 dark:from-blue-950/30 dark:to-indigo-950/30" />
        
        <div className="max-w-7xl mx-auto px-6 relative z-10">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <motion.div 
              initial="hidden"
              animate="visible"
              variants={fadeIn}
            >
              <h1 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-4">
                Advanced Disaster <span className="text-blue-600 dark:text-blue-400">Monitoring & Analysis</span>
              </h1>
              <p className="text-lg text-gray-700 dark:text-gray-300 mb-8">
                Real-time na pag-monitor ng mga disaster sa Pilipinas gamit ang advanced NLP at sentiment analysis para sa mas mabilis na pagtugon.
              </p>
              <div className="flex flex-col sm:flex-row space-y-3 sm:space-y-0 sm:space-x-4">
                <Button size="lg" className="px-8">
                  <Link href="/dashboard">
                    Simulan Ngayon
                  </Link>
                </Button>
                <Button 
                  variant="outline" 
                  size="lg"
                  onClick={() => setShowTutorial(true)}
                >
                  Gabay sa Paggamit
                </Button>
              </div>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              className="bg-white dark:bg-gray-800 rounded-xl shadow-xl overflow-hidden"
            >
              <img 
                src="/images/PANICSENSE PH.png" 
                alt="PanicSense PH Dashboard Preview" 
                className="w-full h-auto"
              />
            </motion.div>
          </div>
        </div>
      </section>
      
      {/* Features Section */}
      <section className="py-16 bg-white dark:bg-gray-900">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">Mga Tampok na Feature</h2>
            <p className="text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
              Ang PanicSense PH ay nagbibigay ng comprehensive suite ng tools upang mas mahusay na maintindihan at ma-monitor ang mga disaster sa Pilipinas.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            <Card className="border-0 shadow-md hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="bg-blue-100 dark:bg-blue-900/30 p-3 rounded-full w-fit mb-4">
                  <BarChart3 className="h-6 w-6 text-blue-600 dark:text-blue-400" />
                </div>
                <CardTitle>Sentiment Analysis</CardTitle>
                <CardDescription>
                  Awtomatikong pagsusuri ng emosyon at damdamin sa mga post tungkol sa disaster.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Gamit ang advanced AI models, nakikilala ng system ang panic, fear, at iba pang emosyon sa mga text.
                </p>
              </CardContent>
            </Card>
            
            <Card className="border-0 shadow-md hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="bg-red-100 dark:bg-red-900/30 p-3 rounded-full w-fit mb-4">
                  <AlertTriangle className="h-6 w-6 text-red-600 dark:text-red-400" />
                </div>
                <CardTitle>Disaster Classification</CardTitle>
                <CardDescription>
                  Awtomatikong pag-identify ng uri ng kalamidad o emergency.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Maaaring i-classify ang mga post bilang sunog, baha, lindol, at iba pang uri ng disaster.
                </p>
              </CardContent>
            </Card>
            
            <Card className="border-0 shadow-md hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="bg-green-100 dark:bg-green-900/30 p-3 rounded-full w-fit mb-4">
                  <MapPin className="h-6 w-6 text-green-600 dark:text-green-400" />
                </div>
                <CardTitle>Geographic Analysis</CardTitle>
                <CardDescription>
                  Visual na representasyon ng mga disaster location sa mapa.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Tingnan kung saan nangyayari ang mga disaster at i-plot sa mapa ng Pilipinas para sa mas mabuting pagtugon.
                </p>
              </CardContent>
            </Card>
            
            <Card className="border-0 shadow-md hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="bg-purple-100 dark:bg-purple-900/30 p-3 rounded-full w-fit mb-4">
                  <Clock className="h-6 w-6 text-purple-600 dark:text-purple-400" />
                </div>
                <CardTitle>Real-time Monitoring</CardTitle>
                <CardDescription>
                  Live monitoring ng mga disaster reports mula sa iba't ibang sources.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Makakuha ng up-to-date na impormasyon tungkol sa mga nangyayaring disaster para sa agarang pagtugon.
                </p>
              </CardContent>
            </Card>
            
            <Card className="border-0 shadow-md hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="bg-orange-100 dark:bg-orange-900/30 p-3 rounded-full w-fit mb-4">
                  <Database className="h-6 w-6 text-orange-600 dark:text-orange-400" />
                </div>
                <CardTitle>Data Collection & Storage</CardTitle>
                <CardDescription>
                  Secure at scalable na storage ng disaster data para sa analysis.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Ligtas na storage ng data na may advanced na filtering at searching capabilities.
                </p>
              </CardContent>
            </Card>
            
            <Card className="border-0 shadow-md hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="bg-indigo-100 dark:bg-indigo-900/30 p-3 rounded-full w-fit mb-4">
                  <Info className="h-6 w-6 text-indigo-600 dark:text-indigo-400" />
                </div>
                <CardTitle>Multilingual Support</CardTitle>
                <CardDescription>
                  Suporta para sa Filipino, English, at iba pang rehiyonal na wika.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Ang system ay maaaring mag-process ng text sa iba't ibang wika na ginagamit sa Pilipinas.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>
      
      {/* Call to Action */}
      <section className="py-20 bg-gradient-to-r from-blue-600 to-indigo-700 text-white">
        <div className="max-w-7xl mx-auto px-6 text-center">
          <h2 className="text-3xl font-bold mb-6">Handa na bang Gumamit ng PanicSense PH?</h2>
          <p className="text-xl text-white/80 mb-8 max-w-2xl mx-auto">
            I-explore ang aming platform upang makita kung paano makakatulong ang advanced analytics para sa disaster response.
          </p>
          <div className="flex flex-col sm:flex-row justify-center space-y-4 sm:space-y-0 sm:space-x-4">
            <Link href="/dashboard">
              <Button size="lg" className="bg-white text-blue-700 hover:bg-white/90">
                Mag-sign in sa Dashboard
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </Link>
            <Button 
              variant="outline" 
              size="lg" 
              className="border-white text-white hover:bg-white/10"
              onClick={() => setShowTutorial(true)}
            >
              Tingnan ang Tutorial
            </Button>
          </div>
        </div>
      </section>
      
      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div className="col-span-1 md:col-span-2">
              <div className="flex items-center space-x-2 mb-4">
                <img src="/images/PANICSENSE PH.png" alt="PanicSense PH Logo" className="h-8" />
                <h3 className="text-lg font-bold">PanicSense PH</h3>
              </div>
              <p className="text-gray-400 mb-4">
                Advanced disaster monitoring and community resilience platform for the Philippines.
              </p>
            </div>
            
            <div>
              <h4 className="font-semibold text-lg mb-4">Links</h4>
              <ul className="space-y-2">
                <li><Link href="/dashboard" className="text-gray-400 hover:text-white">Dashboard</Link></li>
                <li><Link href="/about" className="text-gray-400 hover:text-white">About</Link></li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-semibold text-lg mb-4">Contact</h4>
              <p className="text-gray-400">Email: info@panicsenseph.com</p>
            </div>
          </div>
          
          <div className="border-t border-gray-800 mt-12 pt-8 flex flex-col md:flex-row justify-between items-center">
            <p className="text-gray-500 text-sm">© 2025 PanicSense PH. All rights reserved.</p>
            <div className="mt-4 md:mt-0">
              <p className="text-gray-500 text-sm">
                Developed by Team PanicSense PH
              </p>
            </div>
          </div>
        </div>
      </footer>
      
      {/* Tutorial Modal */}
      {showTutorial && <Tutorial onClose={() => setShowTutorial(false)} />}
    </div>
  );
}