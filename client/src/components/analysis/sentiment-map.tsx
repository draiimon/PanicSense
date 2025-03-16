import { useRef, useEffect, useState } from 'react';
import 'leaflet/dist/leaflet.css';
import { Region } from '@/lib/types';
import { getSentimentColor, getDisasterTypeColor } from '@/lib/colors';
import { Button } from '@/components/ui/button';
import { Globe, ZoomIn, ZoomOut } from 'lucide-react';

interface SentimentMapProps {
  regions: Region[];
  onRegionSelect?: (region: Region) => void;
  mapType?: 'disaster' | 'emotion';
  view?: 'standard' | 'satellite';
  showMarkers?: boolean;
}

export function SentimentMap({ 
  regions, 
  onRegionSelect,
  mapType = 'disaster',
  view = 'standard',
  showMarkers = true
}: SentimentMapProps) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<any>(null);
  const markersRef = useRef<any[]>([]);
  const [hoveredRegion, setHoveredRegion] = useState<Region | null>(null);

  useEffect(() => {
    if (typeof window === 'undefined' || !mapRef.current || mapInstanceRef.current) return;

    import('leaflet').then((L) => {
      if (!mapRef.current) return;

      // Initialize map
      const map = L.map(mapRef.current).setView([12.8797, 121.7740], 6);
      mapInstanceRef.current = map;

      // Add tile layer based on view type
      const tileLayer = view === 'satellite' 
        ? L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}')
        : L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png');

      tileLayer.addTo(map);

      // Add markers for each region
      regions.forEach((region) => {
        const color = mapType === 'disaster' 
          ? getDisasterTypeColor(region.disasterType || 'Unknown')
          : getSentimentColor(region.sentiment);

        // Create circle marker
        const circle = L.circle(region.coordinates, {
          color,
          fillColor: color,
          fillOpacity: 0.5,
          weight: 2,
          radius: Math.max(region.intensity * 20000, 15000)
        });

        // Create popup content
        const popupContent = `
          <div style="min-width: 200px;">
            <h3 style="font-weight: bold; margin-bottom: 8px;">${region.name}</h3>
            <p style="margin: 4px 0;">Sentiment: ${region.sentiment}</p>
            ${region.disasterType ? `<p style="margin: 4px 0;">Disaster: ${region.disasterType}</p>` : ''}
            <p style="margin: 4px 0;">Intensity: ${region.intensity}</p>
          </div>
        `;

        // Bind popup to circle
        circle.bindPopup(popupContent, {
          offset: [0, -10],
          closeButton: false
        });

        // Add hover events
        circle.on('mouseover', () => {
          circle.setStyle({ 
            weight: 3,
            fillOpacity: 0.7 
          });
          circle.openPopup();
          setHoveredRegion(region);
        });

        circle.on('mouseout', () => {
          circle.setStyle({ 
            weight: 2,
            fillOpacity: 0.5,
            color
          });
          circle.closePopup();
          setHoveredRegion(null);
        });

        circle.addTo(map);
        markersRef.current.push(circle);
      });

      return () => {
        map.remove();
        mapInstanceRef.current = null;
        markersRef.current = [];
      };
    });
  }, [regions, mapType, view]);

  return (
    <div className="relative h-full w-full overflow-hidden">
      {/* Zoom controls - Now has better contrast and positioning */}
      <div className="absolute top-4 left-4 z-[1000] flex flex-col gap-2">
        <Button
          size="icon"
          variant="secondary"
          className="h-8 w-8 bg-white shadow-lg hover:bg-slate-50 border border-slate-200"
          onClick={() => mapInstanceRef.current?.zoomIn(2)} // Zoom in with larger steps
          title="Zoom In"
        >
          <ZoomIn className="h-4 w-4" />
        </Button>
        <Button
          size="icon"
          variant="secondary"
          className="h-8 w-8 bg-white shadow-lg hover:bg-slate-50 border border-slate-200"
          onClick={() => mapInstanceRef.current?.zoomOut(1)} // Zoom out
          title="Zoom Out"
        >
          <ZoomOut className="h-4 w-4" />
        </Button>
        <Button
          size="icon"
          variant="secondary"
          className="h-8 w-8 bg-white shadow-lg hover:bg-slate-50 border border-slate-200"
          onClick={() => mapInstanceRef.current?.fitBounds([[4.566667, 116.928406], [21.120611, 126.604393]])} // Reset to default view
          title="Fit Philippines"
        >
          <Globe className="h-4 w-4" />
        </Button>
      </div>

      {/* Map Container */}
      <div
        ref={mapRef}
        className="absolute inset-0 bg-slate-100"
        style={{ width: '100%', height: '100%' }}
      />

      {/* Status Bar */}
      <div className="absolute bottom-0 left-0 right-0 p-2 bg-white/90 backdrop-blur-sm border-t border-slate-200 z-[500]">
        <div className="flex items-center justify-between text-xs text-slate-700 font-medium">
          <div className="flex items-center gap-2">
            <Globe className="h-4 w-4 text-blue-500" />
            <span>{hoveredRegion ? hoveredRegion.name : 'Philippine Region Map'}</span>
          </div>
          <div>
            {regions.length > 0 ? (
              <span>Showing {showMarkers ? regions.length : 0} affected areas</span>
            ) : (
              <span>No impact data available</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}