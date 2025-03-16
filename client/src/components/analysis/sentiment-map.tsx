import { useEffect, useRef, useState } from 'react';
import { Card } from '@/components/ui/card';
import { getSentimentColor, getDisasterTypeColor } from '@/lib/colors';
import { Button } from '@/components/ui/button';
import { Globe, ZoomIn, ZoomOut } from 'lucide-react';
import 'leaflet/dist/leaflet.css';

// Philippine map bounds - Tightly focused on the archipelago
const PH_BOUNDS = {
  northEast: [21.120611, 126.604393], // Northern most point of Batanes
  southWest: [4.566667, 116.928406]   // Southern tip of Tawi-Tawi
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
  const [mapZoom, setMapZoom] = useState(6);
  const [hoveredRegion, setHoveredRegion] = useState<Region | null>(null);
  const popupRef = useRef<any>(null);

  useEffect(() => {
    if (typeof window === 'undefined' || !mapRef.current || mapInstanceRef.current) return;

    import('leaflet').then((L) => {
      if (!mapRef.current) return;

      // Initialize map with exact Philippines bounds and improved zoom
      mapInstanceRef.current = L.map(mapRef.current, {
        zoomControl: false,
        attributionControl: false,
        maxBounds: [
          [0, 110],
          [25, 130]
        ],
        minZoom: 5,      // Lower minimum zoom
        maxZoom: 18,     // Higher maximum zoom
        maxBoundsViscosity: 0.8, // Slightly less strict bounds
        scrollWheelZoom: true,
        dragging: true,
        zoomDelta: 0.5,  // Larger zoom steps
        zoomSnap: 0.25,
        doubleClickZoom: true, // Enable double-click zoom
        touchZoom: true, // Better mobile touch zoom
        bounceAtZoomLimits: true // Bounce effect at zoom limits
      }).fitBounds([
        [4.566667, 116.928406],
        [21.120611, 126.604393]
      ]);

      // Add base tile layer with noWrap option
      updateTileLayer(L, view);

      // Keep map within bounds and prevent white edges
      mapInstanceRef.current.on('moveend', () => {
        const newZoom = mapInstanceRef.current.getZoom();
        setMapZoom(newZoom);
      });

      // Handle resize
      const handleResize = () => {
        if (mapInstanceRef.current) {
          mapInstanceRef.current.invalidateSize();
        }
      };

      window.addEventListener('resize', handleResize);
      return () => {
        window.removeEventListener('resize', handleResize);
        if (mapInstanceRef.current) {
          mapInstanceRef.current.remove();
          mapInstanceRef.current = null;
        }
      };
    });
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

    // Common tile layer options with improved zoom
    const tileOptions = {
      noWrap: true,
      bounds: [
        [PH_BOUNDS.southWest[0], PH_BOUNDS.southWest[1]],
        [PH_BOUNDS.northEast[0], PH_BOUNDS.northEast[1]]
      ],
      minZoom: 5,
      maxZoom: 18, // Higher maximum zoom level
      detectRetina: true // Better resolution on high-DPI screens
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

  // Clear existing markers and create new ones
  const updateMarkers = async () => {
    if (!mapInstanceRef.current || typeof window === 'undefined') return;

    const L = await import('leaflet');

    // Clear previous markers and popup
    markersRef.current.forEach(marker => {
      if (marker) marker.remove();
    });
    markersRef.current = [];

    if (popupRef.current) {
      popupRef.current.remove();
      popupRef.current = null;
    }

    // Add new markers if showMarkers is true
    if (showMarkers) {
      regions.forEach(region => {
        const color = mapType === 'disaster' && region.disasterType
          ? getDisasterTypeColor(region.disasterType)
          : getSentimentColor(region.sentiment);

        // Even smaller base radius with water ripple effect
        const baseRadius = 3 + (region.intensity / 100) * 15; // Smaller base radius
        const radius = baseRadius * Math.pow(1.2, mapZoom - 6);

        // Create main circle (smaller and more transparent)
        const circle = L.circle(region.coordinates, {
          color,
          fillColor: color,
          fillOpacity: 0.3, // More transparent
          radius: radius * 1000,
          weight: 1, // Even thinner border
          className: 'pulse-circle' // Add pulse animation class
        }).addTo(mapInstanceRef.current);
        
        // Create outer ripple effect circle
        const outerRipple = L.circle(region.coordinates, {
          color,
          fillColor: 'transparent',
          fillOpacity: 0,
          radius: radius * 1500,
          weight: 0.8,
          opacity: 0.4,
          className: 'ripple-circle' // Add ripple animation class
        }).addTo(mapInstanceRef.current);
        
        // Add to markers for cleanup
        markersRef.current.push(outerRipple);

        // Create custom popup with improved styling
        const popupContent = `
          <div class="p-3 font-sans">
            <h3 class="text-lg font-semibold text-slate-800 mb-2">${region.name}</h3>
            <div class="space-y-2">
              <div class="flex items-center gap-2">
                <span class="w-3 h-3 rounded-full" style="background-color: ${getSentimentColor(region.sentiment)}"></span>
                <span class="text-sm text-slate-600">Sentiment: ${region.sentiment}</span>
              </div>
              ${region.disasterType ? `
                <div class="flex items-center gap-2">
                  <span class="w-3 h-3 rounded-full" style="background-color: ${getDisasterTypeColor(region.disasterType)}"></span>
                  <span class="text-sm text-slate-600">Disaster: ${region.disasterType}</span>
                </div>
              ` : ''}
              <div class="mt-2 pt-2 border-t border-slate-200">
                <span class="text-sm font-medium text-slate-700">
                  Impact Level: ${region.intensity.toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        `;

        circle.bindPopup(popupContent, {
          maxWidth: 300,
          className: 'custom-popup',
          closeButton: false,
          offset: [0, -10]
        });

        // Enhanced hover interactions with adjusted values
        circle.on('mouseover', () => {
          circle.setStyle({ 
            weight: 2.5, 
            fillOpacity: 0.65
          });
          circle.setRadius(radius * 1050);
          setHoveredRegion(region);
          circle.openPopup();
        });

        circle.on('mouseout', () => {
          circle.setStyle({ 
            weight: 1, 
            fillOpacity: 0.3
          });
          circle.setRadius(radius * 1000);
          setHoveredRegion(null);
          circle.closePopup();
        });

        circle.on('click', () => {
          if (onRegionSelect) {
            onRegionSelect(region);
          }
        });

        markersRef.current.push(circle);
      });
    } else {
      // Show message when markers are hidden
      popupRef.current = L.popup()
        .setLatLng(PH_CENTER)
        .setContent(`
          <div class="p-4 text-center">
            <span class="text-sm text-slate-600">Map markers are currently hidden</span>
          </div>
        `)
        .openOn(mapInstanceRef.current);
      markersRef.current.push(popupRef.current);
    }
  };

  // Update markers when regions, mapType, mapZoom, or showMarkers changes
  useEffect(() => {
    updateMarkers();
  }, [regions, mapType, mapZoom, showMarkers]);

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
          onClick={() => mapInstanceRef.current?.fitBounds([
            [4.566667, 116.928406],
            [21.120611, 126.604393]
          ])} // Reset to default view
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