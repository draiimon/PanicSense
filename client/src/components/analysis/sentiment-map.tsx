import { useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { getSentimentColor, getDisasterTypeColor } from '@/lib/colors';
import 'leaflet/dist/leaflet.css';

interface Region {
  name: string;
  coordinates: [number, number]; // [latitude, longitude]
  sentiment: string;
  disasterType?: string;
  intensity: number; // 0-100
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

  useEffect(() => {
    // Skip if leaflet is not available (SSR) or map is already initialized
    if (typeof window === 'undefined' || !mapRef.current || mapInstanceRef.current) return;

    // Dynamically import Leaflet - needed for client-side only use
    import('leaflet').then((L) => {
      // Center map on Philippines
      mapInstanceRef.current = L.map(mapRef.current).setView([12.8797, 121.7740], 6);

      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      }).addTo(mapInstanceRef.current);
    });

    // Cleanup function
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

    import('leaflet').then((L) => {
      // Clear previous markers
      markersRef.current.forEach(marker => marker.remove());
      markersRef.current = [];

      // Add new markers
      regions.forEach(region => {
        // Choose color based on the colorBy preference
        const color = colorBy === 'disasterType' && region.disasterType
          ? getDisasterTypeColor(region.disasterType)
          : getSentimentColor(region.sentiment);

        // Calculate radius based on intensity (min 10, max 50)
        const radius = 10 + (region.intensity / 100) * 40;

        const circle = L.circle(region.coordinates, {
          color,
          fillColor: color,
          fillOpacity: 0.5,
          radius: radius * 1000 // Convert to meters
        }).addTo(mapInstanceRef.current);

        // Add a popup with different content based on colorBy
        let popupContent = `<strong>${region.name}</strong><br>`;
        popupContent += `Sentiment: ${region.sentiment}<br>`;
        
        if (region.disasterType) {
          popupContent += `Disaster Type: ${region.disasterType}<br>`;
        }
        
        popupContent += `Intensity: ${region.intensity.toFixed(1)}%`;
        
        circle.bindPopup(popupContent);

        // Handle click event
        if (onRegionSelect) {
          circle.on('click', () => {
            onRegionSelect(region);
          });
        }

        markersRef.current.push(circle);
      });

      // If no regions, show a message
      if (regions.length === 0) {
        const message = L.popup()
          .setLatLng([12.8797, 121.7740])
          .setContent('No sentiment data available for regions')
          .openOn(mapInstanceRef.current);

        markersRef.current.push(message);
      }
    });
  }, [regions, onRegionSelect, colorBy]);

  return (
    <Card className="bg-white rounded-lg shadow">
      <CardHeader className="p-5 border-b border-gray-200">
        <CardTitle className="text-lg font-medium text-slate-800">
          {colorBy === 'disasterType' ? 'Disaster Impact Map' : 'Sentiment Map'}
        </CardTitle>
        <CardDescription className="text-sm text-slate-500">
          {colorBy === 'disasterType' 
            ? 'Disaster type impact visualization' 
            : 'Regions colored by dominant emotion'}
        </CardDescription>
      </CardHeader>
      <CardContent className="p-5">
        <div 
          ref={mapRef} 
          className="h-[500px] w-full bg-slate-50 rounded-lg"
        />
      </CardContent>
    </Card>
  );
}