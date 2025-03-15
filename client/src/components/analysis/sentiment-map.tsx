import { useEffect, useRef, useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { getSentimentColor, getDisasterTypeColor } from '@/lib/colors';
import { Button } from '@/components/ui/button';
import { Globe, Map, Layers } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import 'leaflet/dist/leaflet.css';

// Philippine map bounds
const PH_BOUNDS = {
  northEast: [21.120611, 126.604393], // Northern most point of Batanes to Eastern most point
  southWest: [4.566667, 116.928406]   // Southern tip of Tawi-Tawi to Western most point
};

// Center of Philippines
const PH_CENTER = [12.8797, 121.7740];

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
  const [mapView, setMapView] = useState<'standard' | 'satellite'>('standard');
  const [mapZoom, setMapZoom] = useState(6);
  const [hoveredRegion, setHoveredRegion] = useState<Region | null>(null);

  useEffect(() => {
    if (typeof window === 'undefined' || !mapRef.current || mapInstanceRef.current) return;

    import('leaflet').then((L) => {
      if (!mapRef.current) return;

      // Initialize map with Philippines bounds
      mapInstanceRef.current = L.map(mapRef.current, {
        zoomControl: false,
        attributionControl: false,
        maxBounds: [
          [PH_BOUNDS.southWest[0] - 1, PH_BOUNDS.southWest[1] - 1], // Add padding
          [PH_BOUNDS.northEast[0] + 1, PH_BOUNDS.northEast[1] + 1]
        ],
        minZoom: 5,
        maxZoom: 12
      }).setView(PH_CENTER, mapZoom);

      // Add base tile layer
      const tileLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        bounds: [
          [PH_BOUNDS.southWest[0], PH_BOUNDS.southWest[1]],
          [PH_BOUNDS.northEast[0], PH_BOUNDS.northEast[1]]
        ]
      }).addTo(mapInstanceRef.current);

      // Add zoom change handler
      mapInstanceRef.current.on('zoomend', () => {
        setMapZoom(mapInstanceRef.current.getZoom());
      });

      // Keep map within bounds
      mapInstanceRef.current.on('drag', () => {
        mapInstanceRef.current.panInsideBounds([
          [PH_BOUNDS.southWest[0], PH_BOUNDS.southWest[1]],
          [PH_BOUNDS.northEast[0], PH_BOUNDS.northEast[1]]
        ], { animate: false });
      });
    });

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, []);

  // Update map tiles when view changes
  useEffect(() => {
    if (!mapInstanceRef.current || typeof window === 'undefined') return;

    import('leaflet').then((L) => {
      // Remove existing tile layers
      mapInstanceRef.current.eachLayer((layer: any) => {
        if (layer instanceof L.TileLayer) {
          mapInstanceRef.current.removeLayer(layer);
        }
      });

      // Add new tile layer based on selected view
      if (mapView === 'standard') {
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
          attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(mapInstanceRef.current);
      } else {
        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
          attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
        }).addTo(mapInstanceRef.current);
      }
    });
  }, [mapView]);

  // Update markers when regions change
  useEffect(() => {
    if (!mapInstanceRef.current || typeof window === 'undefined') return;

    import('leaflet').then((L) => {
      // Clear previous markers
      markersRef.current.forEach(marker => marker.remove());
      markersRef.current = [];

      // Add new markers
      regions.forEach(region => {
        const color = colorBy === 'disasterType' && region.disasterType
          ? getDisasterTypeColor(region.disasterType)
          : getSentimentColor(region.sentiment);

        const radius = 10 + (region.intensity / 100) * 40;

        const circle = L.circle(region.coordinates, {
          color,
          fillColor: color,
          fillOpacity: 0.7,
          radius: radius * 1000, // Convert to meters
          weight: 2,
        }).addTo(mapInstanceRef.current);

        // Create custom popup
        let popupContent = `
          <div style="font-family: system-ui, sans-serif; padding: 8px;">
            <h3 style="margin: 0 0 8px 0; color: #1e293b; font-size: 16px; font-weight: 600;">${region.name}</h3>
            <div style="display: flex; flex-direction: column; gap: 4px;">
              <div style="display: flex; align-items: center; gap: 6px;">
                <span style="width: 8px; height: 8px; border-radius: 50%; background-color: ${getSentimentColor(region.sentiment)}"></span>
                <span style="font-size: 14px; color: #4b5563;">Dominant Sentiment: ${region.sentiment}</span>
              </div>
        `;

        if (region.disasterType) {
          popupContent += `
              <div style="display: flex; align-items: center; gap: 6px;">
                <span style="width: 8px; height: 8px; border-radius: 50%; background-color: ${getDisasterTypeColor(region.disasterType)}"></span>
                <span style="font-size: 14px; color: #4b5563;">Disaster Type: ${region.disasterType}</span>
              </div>
          `;
        }

        popupContent += `
              <div style="margin-top: 4px;">
                <span style="font-size: 14px; color: #4b5563;">Impact Intensity: 
                  <span style="font-weight: 600;">${region.intensity.toFixed(1)}%</span>
                </span>
              </div>
            </div>
          </div>
        `;

        circle.bindPopup(popupContent, {
          maxWidth: 300,
          className: 'custom-popup'
        });

        // Enhanced interaction
        circle.on('mouseover', () => {
          circle.setStyle({ weight: 3, fillOpacity: 0.85 });
          setHoveredRegion(region);
        });

        circle.on('mouseout', () => {
          circle.setStyle({ weight: 2, fillOpacity: 0.7 });
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
          .setLatLng([12.8797, 121.7740])
          .setContent(`
            <div style="font-family: system-ui, sans-serif; padding: 8px; text-align: center;">
              <span style="font-size: 14px; color: #4b5563;">No geographic data available for this region</span>
            </div>
          `)
          .openOn(mapInstanceRef.current);

        markersRef.current.push(message);
      }
    });
  }, [regions, onRegionSelect, colorBy]);

  return (
    <Card className="bg-white rounded-lg shadow-md border border-slate-200">
      <CardContent className="p-0 overflow-hidden relative">
        {/* Control buttons container with higher z-index */}
        <div className="absolute top-4 right-4 z-[1000] flex items-start gap-4">
          {/* Map Type Toggle */}
          <div className="bg-white rounded-lg shadow-lg p-1 flex">
            <Button
              size="sm"
              variant={mapView === 'standard' ? 'default' : 'outline'}
              className="rounded-r-none px-3 h-8"
              onClick={() => setMapView('standard')}
            >
              <Map className="h-4 w-4 mr-1" />
              <span className="text-xs">Standard</span>
            </Button>
            <Button
              size="sm"
              variant={mapView === 'satellite' ? 'default' : 'outline'}
              className="rounded-l-none px-3 h-8"
              onClick={() => setMapView('satellite')}
            >
              <Layers className="h-4 w-4 mr-1" />
              <span className="text-xs">Satellite</span>
            </Button>
          </div>
          {/* Zoom controls */}
          <div className="flex flex-col gap-2">
            <Button 
              size="icon" 
              variant="secondary"
              className="h-8 w-8 bg-white shadow-lg hover:bg-slate-50"
              onClick={() => mapInstanceRef.current?.zoomIn()}
            >
              <span className="text-lg font-medium">+</span>
            </Button>
            <Button 
              size="icon" 
              variant="secondary"
              className="h-8 w-8 bg-white shadow-lg hover:bg-slate-50"
              onClick={() => mapInstanceRef.current?.zoomOut()}
            >
              <span className="text-lg font-medium">−</span>
            </Button>
          </div>v>

          {/* Map Type Toggle */}
          <div className="bg-white rounded-lg shadow-lg p-1 flex">
            <Button
              size="sm"
              variant={mapView === 'standard' ? 'default' : 'outline'}
              className="rounded-r-none px-3 h-8"
              onClick={() => setMapView('standard')}
            >
              <Map className="h-4 w-4 mr-1" />
              <span className="text-xs">Standard</span>
            </Button>
            <Button
              size="sm"
              variant={mapView === 'satellite' ? 'default' : 'outline'}
              className="rounded-l-none px-3 h-8"
              onClick={() => setMapView('satellite')}
            >
              <Layers className="h-4 w-4 mr-1" />
              <span className="text-xs">Satellite</span>
            </Button>
          </div>
        </div>

        {/* Map Container */}
        <div 
          ref={mapRef} 
          className="h-[500px] w-full bg-slate-50"
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