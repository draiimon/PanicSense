import { useEffect, useRef, useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { getSentimentColor, getDisasterTypeColor } from '@/lib/colors';
import { Globe } from 'lucide-react';
import 'leaflet/dist/leaflet.css';

// Philippine map bounds
const PH_BOUNDS = {
  northEast: { lat: 21.120611, lng: 126.604393 }, // Northern most point of Batanes to Eastern most point
  southWest: { lat: 4.566667, lng: 116.928406 }   // Southern tip of Tawi-Tawi to Western most point
};

// Center of Philippines
const PH_CENTER: [number, number] = [12.8797, 121.7740];

interface Region {
  name: string;
  coordinates: [number, number];
  sentiment: string;
  disasterType?: string;
  intensity: number;
}

interface SentimentMapProps {
  regions: Region[];
  onRegionSelect?: (region: Region) => void;
  colorBy?: 'sentiment' | 'disasterType';
}

export function SentimentMap({ regions, onRegionSelect, colorBy = 'disasterType' }: SentimentMapProps) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<any>(null);
  const markersRef = useRef<any[]>([]);
  const [hoveredRegion, setHoveredRegion] = useState<Region | null>(null);

  useEffect(() => {
    if (typeof window === 'undefined' || !mapRef.current || mapInstanceRef.current) return;

    const initMap = async () => {
      const L = await import('leaflet');

      if (!mapRef.current) return;

      // Initialize map with Philippines bounds
      mapInstanceRef.current = L.map(mapRef.current, {
        zoomControl: false,
        attributionControl: false,
        maxBounds: [
          [PH_BOUNDS.southWest.lat - 1, PH_BOUNDS.southWest.lng - 1],
          [PH_BOUNDS.northEast.lat + 1, PH_BOUNDS.northEast.lng + 1]
        ],
        minZoom: 5,
        maxZoom: 12
      }).setView(PH_CENTER, 6);

      // Add base tile layer
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        bounds: [
          [PH_BOUNDS.southWest.lat, PH_BOUNDS.southWest.lng],
          [PH_BOUNDS.northEast.lat, PH_BOUNDS.northEast.lng]
        ]
      }).addTo(mapInstanceRef.current);

      // Keep map within bounds
      mapInstanceRef.current.on('drag', () => {
        mapInstanceRef.current.panInsideBounds([
          [PH_BOUNDS.southWest.lat, PH_BOUNDS.southWest.lng],
          [PH_BOUNDS.northEast.lat, PH_BOUNDS.northEast.lng]
        ], { animate: false });
      });
    };

    initMap();

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, []);

  // Update markers when regions change
  useEffect(() => {
    if (!mapInstanceRef.current || typeof window === 'undefined') return;

    const updateMarkers = async () => {
      const L = await import('leaflet');

      // Clear previous markers
      markersRef.current.forEach(marker => marker.remove());
      markersRef.current = [];

      // Add new markers
      regions.forEach(region => {
        const color = colorBy === 'disasterType' && region.disasterType
          ? getDisasterTypeColor(region.disasterType)
          : getSentimentColor(region.sentiment);

        // Scale radius based on intensity
        const baseRadius = 15000; // Base radius in meters
        const radius = baseRadius + (region.intensity / 100) * 30000;

        const circle = L.circle(region.coordinates, {
          color,
          fillColor: color,
          fillOpacity: 0.6,
          radius,
          weight: 2,
        }).addTo(mapInstanceRef.current);

        // Create custom popup
        const popupContent = `
          <div class="p-3 font-sans">
            <div class="flex items-center justify-between mb-2">
              <h3 class="text-sm font-semibold text-slate-800">${region.name}</h3>
              <span class="text-xs px-2 py-1 bg-slate-100 rounded-full text-slate-600">
                ${Math.round(region.intensity)}% Impact
              </span>
            </div>
            <div class="space-y-1.5">
              <div class="flex items-center gap-2">
                <span class="w-2 h-2 rounded-full" style="background-color: ${getSentimentColor(region.sentiment)}"></span>
                <span class="text-xs text-slate-600">${region.sentiment}</span>
              </div>
              ${region.disasterType ? `
                <div class="flex items-center gap-2">
                  <span class="w-2 h-2 rounded-full" style="background-color: ${getDisasterTypeColor(region.disasterType)}"></span>
                  <span class="text-xs text-slate-600">${region.disasterType}</span>
                </div>
              ` : ''}
            </div>
          </div>
        `;

        circle.bindPopup(popupContent, {
          maxWidth: 300,
          className: 'custom-popup'
        });

        // Enhanced interaction
        circle.on('mouseover', () => {
          circle.setStyle({ 
            weight: 3,
            fillOpacity: 0.8 
          });
          setHoveredRegion(region);
        });

        circle.on('mouseout', () => {
          circle.setStyle({ 
            weight: 2,
            fillOpacity: 0.6 
          });
          setHoveredRegion(null);
        });

        // Handle click event
        circle.on('click', () => {
          if (onRegionSelect) {
            onRegionSelect(region);
          }
        });

        markersRef.current.push(circle);
      });

      // If no regions, show a message
      if (regions.length === 0) {
        const message = L.popup()
          .setLatLng(PH_CENTER)
          .setContent(`
            <div class="p-4 text-center">
              <span class="text-sm text-slate-600">No geographic data available</span>
            </div>
          `)
          .openOn(mapInstanceRef.current);

        markersRef.current.push(message);
      }
    };

    updateMarkers();
  }, [regions, onRegionSelect, colorBy]);

  return (
    <Card className="bg-white rounded-lg shadow-md border border-slate-200">
      <CardContent className="p-0 overflow-hidden relative">
        {/* Map Container */}
        <div 
          ref={mapRef} 
          className="h-[600px] w-full bg-slate-50"
        />

        {/* Footer Stats */}
        <div className="p-3 border-t border-slate-200 bg-slate-50 flex items-center justify-between text-xs text-slate-600">
          <div className="flex items-center gap-2">
            <Globe className="h-4 w-4 text-slate-400" />
            <span>Philippine Region Map</span>
          </div>
          <div>
            {regions.length > 0 ? (
              <span>Showing {regions.length} affected areas</span>
            ) : (
              <span>No impact data available</span>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}