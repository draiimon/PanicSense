import { useLocation } from "wouter";
import { Button } from "@/components/ui/button";

export function Header() {
  const [location] = useLocation();

  return (
    <header className="bg-white border-b border-gray-200 py-4">
      <div className="container mx-auto px-4">
        <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
          <div>
            <h1 className="text-2xl font-bold text-slate-800">Disaster Sentiment Analysis</h1>
            <p className="text-sm text-slate-500">
              Created by{" "}
              <span className="font-bold">Castillo, Mark Andrei R.</span>,{" "}
              Garcia, Ivahnn, and{" "}
              Gatdula, Julia Daphne Ngan
            </p>
          </div>

          {location !== '/login' && location !== '/signup' && (
            <nav className="flex space-x-4">
              <Button
                variant={location === '/dashboard' ? 'default' : 'ghost'}
                onClick={() => location !== '/dashboard' && window.location.assign('/dashboard')}
              >
                Dashboard
              </Button>
              <Button
                variant={location === '/emotion-analysis' ? 'default' : 'ghost'}
                onClick={() => location !== '/emotion-analysis' && window.location.assign('/emotion-analysis')}
              >
                Emotion Analysis
              </Button>
              <Button
                variant={location === '/timeline' ? 'default' : 'ghost'}
                onClick={() => location !== '/timeline' && window.location.assign('/timeline')}
              >
                Timeline
              </Button>
              <Button
                variant={location === '/comparison' ? 'default' : 'ghost'}
                onClick={() => location !== '/comparison' && window.location.assign('/comparison')}
              >
                Comparison
              </Button>
            </nav>
          )}
        </div>
      </div>
    </header>
  );
}
