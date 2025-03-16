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
  // Define our marker types
  interface MarkerWithAnimation {
    circle: any;
    pulseCircle: any;
    animationRef: { current: number | null };
  }

  type MarkerRef = any | MarkerWithAnimation;

  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<any>(null);
  const markersRef = useRef<MarkerRef[]>([]);
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

    // Clear previous markers, animations, and popup
    markersRef.current.forEach(marker => {
      if (typeof marker === 'object' && marker !== null && 'circle' in marker) {
        marker.circle.remove();
      } else if (marker) {
        marker.remove();
      }
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

        // Scale radius based on area coverage - match actual geographic area size
        // For Philippine regions, typical coverage is around 0.5-2km for cities/municipalities
        let areaRadius = 0;

        // Specific size adjustments based on location name
        if (region.name.includes("Province") || region.name.includes("Region")) {
          areaRadius = 10; // Province level
        } else if (region.name.includes("City")) {
          areaRadius = 2; // City level
        } else if (region.name.includes("Municipality") || region.name.includes("Town")) {
          areaRadius = 1.5; // Municipality level
        } else if (region.name.includes("Barangay") || region.name.includes("Street") || region.name.includes("Village")) {
          areaRadius = 0.8; // Specific location level
        } else {
          areaRadius = 1; // Default for other specific places
        }

        // Add subtle variation based on intensity
        const intensityFactor = 1 + (region.intensity / 200); // Subtle intensity impact
        const radius = areaRadius * intensityFactor;

        // Create main marker circle - smaller and transparent
        const circle = L.circle(region.coordinates, {
          color,
          fillColor: color,
          fillOpacity: 0.4, // Transparency but still visible
          radius: radius * 1000, // Smaller scale
          weight: 1.5, // Thin border
          className: 'main-marker',
        }).addTo(mapInstanceRef.current);

        // Store marker reference only
        markersRef.current.push({ circle });

        // Create custom popup with better styling and layout
        const popupContent = `
          <div class="p-4 font-sans shadow-sm rounded-lg" style="min-width: 200px">
            <h3 class="text-lg font-bold text-slate-800 mb-3 border-b pb-2" style="color: ${color}">${region.name}</h3>
            <div class="space-y-3">
              <div class="flex items-center gap-2">
                <span class="w-4 h-4 rounded-full flex-shrink-0" style="background-color: ${getSentimentColor(region.sentiment)}"></span>
                <span class="text-sm font-medium text-slate-700">
                  <strong>Sentiment:</strong> ${region.sentiment}
                </span>
              </div>
              ${region.disasterType ? `
                <div class="flex items-center gap-2">
                  <span class="w-4 h-4 rounded-full flex-shrink-0" style="background-color: ${getDisasterTypeColor(region.disasterType)}"></span>
                  <span class="text-sm font-medium text-slate-700">
                    <strong>Disaster:</strong> ${region.disasterType}
                  </span>
                </div>
              ` : ''}
              <div class="mt-3 pt-2 border-t border-slate-200">
                <div class="flex items-center justify-between">
                  <span class="text-sm font-medium text-slate-700">Impact Level:</span>
                  <span class="text-sm font-bold" style="color: ${color}">${region.intensity.toFixed(1)}%</span>
                </div>
                <div class="w-full bg-slate-200 rounded-full h-2 mt-1 overflow-hidden">
                  <div class="h-full rounded-full" style="background-color: ${color}; width: ${region.intensity}%"></div>
                </div>
              </div>
            </div>
          </div>
        `;

        circle.bindPopup(popupContent, {
          maxWidth: 300,
          className: 'custom-popup',
          closeButton: false,
          offset: [0, -10], // Reduced offset for closer pinning
          autoPan: true,
          autoPanPadding: [10, 10], // Tighter padding
          keepInView: true // Keep popup in view when panning
        });

        let isSelected = false;
        let isHovered = false;
        let hoverTimeout: NodeJS.Timeout;

        circle.on('mouseover', (e) => {
          isHovered = true;
          clearTimeout(hoverTimeout);
          
          circle.setStyle({ 
            weight: 3,
            fillOpacity: 0.8,
            color: '#FFFFFF'
          });

          // Pin popup exactly to mouse position
          const popup = circle.getPopup();
          if (popup) {
            popup.setLatLng(e.latlng);
            popup.openOn(mapInstanceRef.current);
          }

          circle.bringToFront();
          setHoveredRegion(region);
        });

        circle.on('mousemove', (e) => {
          if (isHovered) {
            // Keep popup open and update its position
            const popup = circle.getPopup();
            if (popup) {
              popup.setLatLng(e.latlng);
            }
          }
        });

        circle.on('mouseout', () => {
          isHovered = false;
          // Add delay before closing popup
          hoverTimeout = setTimeout(() => {
            if (!isHovered) {
              circle.setStyle({ 
                weight: 2,
                fillOpacity: 0.5,
                color
              });
              circle.closePopup();
              setHoveredRegion(null);
            }
          }, 100); // Small delay to prevent flickering
        });

        circle.on('click', () => {
          if (onRegionSelect) {
            onRegionSelect(region);
          }
        });
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