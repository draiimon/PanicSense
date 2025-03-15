import { useEffect, useRef, useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { getSentimentColor, getDisasterTypeColor } from '@/lib/colors';
import { Button } from '@/components/ui/button';
import { Globe, Map, Layers } from 'lucide-react';
import 'leaflet/dist/leaflet.css';

// Philippine map bounds - Tightly focused on the archipelago
const PH_BOUNDS = {
  northEast: [21.120611, 126.604393], // Northern most point of Batanes
  southWest: [4.566667, 116.928406]   // Southern tip of Tawi-Tawi
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
  mapType?: 'disaster' | 'emotion';
  view?: 'standard' | 'satellite';
}

export function SentimentMap({ 
  regions, 
  onRegionSelect, 
  colorBy = 'disasterType',
  view = 'standard' 
}: SentimentMapProps) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<any>(null);
  const markersRef = useRef<any[]>([]);
  const [mapZoom, setMapZoom] = useState(6);
  const [hoveredRegion, setHoveredRegion] = useState<Region | null>(null);

  useEffect(() => {
    if (typeof window === 'undefined' || !mapRef.current || mapInstanceRef.current) return;

    import('leaflet').then((L) => {
      if (!mapRef.current) return;

      // Initialize map with exact Philippines bounds
      mapInstanceRef.current = L.map(mapRef.current, {
        zoomControl: false,
        attributionControl: false,
        maxBounds: [
          [PH_BOUNDS.southWest[0], PH_BOUNDS.southWest[1]],
          [PH_BOUNDS.northEast[0], PH_BOUNDS.northEast[1]]
        ],
        minZoom: 5.5,
        maxZoom: 12,
        maxBoundsViscosity: 1.0 // Prevents dragging outside bounds
      }).setView(PH_CENTER, mapZoom);

      // Add base tile layer with noWrap option
      updateTileLayer(L, view);

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

  // Function to update tile layer with correct options
  const updateTileLayer = (L: any, mapView: string) => {
    if (!mapInstanceRef.current) return;

    // Remove existing tile layers
    mapInstanceRef.current.eachLayer((layer: any) => {
      if (layer instanceof L.TileLayer) {
        mapInstanceRef.current.removeLayer(layer);
      }
    });

    // Common tile layer options
    const tileOptions = {
      noWrap: true,
      bounds: [
        [PH_BOUNDS.southWest[0], PH_BOUNDS.southWest[1]],
        [PH_BOUNDS.northEast[0], PH_BOUNDS.northEast[1]]
      ]
    };

    // Add new tile layer based on selected view
    if (mapView === 'standard') {
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        ...tileOptions,
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      }).addTo(mapInstanceRef.current);
    } else {
      L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        ...tileOptions,
        attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
      }).addTo(mapInstanceRef.current);
    }
  };

  // Update map tiles when view changes
  useEffect(() => {
    if (!mapInstanceRef.current || typeof window === 'undefined') return;
    import('leaflet').then((L) => updateTileLayer(L, view));
  }, [view]);

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
          radius: radius * 1000,
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
          .setLatLng(PH_CENTER)
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
    <Card className="bg-white border-none shadow-none h-full">
      <CardContent className="p-0 h-full overflow-hidden relative">
        {/* Zoom controls */}
        <div className="absolute top-4 left-4 z-[1000] flex flex-col gap-2">
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
        </div>

        {/* Map Container */}
        <div
          ref={mapRef}
          className="h-full w-full bg-slate-50"
          style={{ minHeight: '600px' }}
        />

        {/* Footer Stats */}
        <div className="absolute bottom-0 left-0 right-0 p-3 bg-white/80 backdrop-blur-sm border-t border-slate-200">
          <div className="flex items-center justify-between text-xs text-slate-600">
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
        </div>
      </CardContent>
    </Card>
  );
}