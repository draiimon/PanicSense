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
      if (typeof marker === 'object' && marker !== null) {
        // If marker is our complex object with animation
        if ('circle' in marker && 'pulseCircle' in marker && 'animationRef' in marker) {
          // Cancel animation frame if exists
          if (marker.animationRef && marker.animationRef.current) {
            cancelAnimationFrame(marker.animationRef.current);
          }
          // Remove circles from map
          if (marker.circle) marker.circle.remove();
          if (marker.pulseCircle) marker.pulseCircle.remove();
        } else {
          // Simple marker
          marker.remove();
        }
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
        if (region.name.includes("Manila") || region.name.includes("Makati") || 
            region.name.includes("Pasig") || region.name.includes("Taguig")) {
          areaRadius = 3; // Small city areas
        } else if (region.name.includes("Cebu") || region.name.includes("Davao") ||
                   region.name.includes("Baguio")) {
          areaRadius = 6; // Medium city areas
        } else if (region.name.includes("Province") || region.name.includes("Region")) {
          areaRadius = 20; // Province level - larger area
        } else {
          areaRadius = 4; // Default for municipalities
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

        // Create animated pulse circle with same initial size as main marker
        const pulseCircle = L.circle(region.coordinates, {
          color: 'rgba(255,255,255,0.3)', // Subtle border
          fillColor: color,
          fillOpacity: 0.3, // More visible starting opacity
          radius: radius * 1000, // Start at same size as main marker
          weight: 1,
          className: 'pulse-marker',
        }).addTo(mapInstanceRef.current);

        // Add ping animation with better, smoother effect - MUCH SLOWER
        const animatePulse = () => {
          let size = radius * 1000; // Start at main circle size
          let opacity = 0.3; 
          let growing = true;
          let maxSize = radius * 3000; // Larger maximum size for more dramatic effect
          let growthRate = radius * 30; // MUCH slower growth rate

          const expandPulse = () => {
            if (growing) {
              // Growing phase
              size += growthRate;
              opacity -= 0.006; // Slower fade out for smoother effect

              if (size >= maxSize) {
                growing = false;
              }
            } else {
              // Reset phase
              size = radius * 1000;  // Reset to main marker size
              opacity = 0.3;  // Reset opacity
              growing = true;
            }

            pulseCircle.setRadius(size);
            pulseCircle.setStyle({ 
              fillOpacity: Math.max(0.01, opacity), // Keep minimum opacity visible
              opacity: Math.max(0.05, opacity * 1.5) // Keep borders slightly more visible
            });

            // Schedule next frame
            animationRef.current = requestAnimationFrame(expandPulse);
          };

          // Start animation
          expandPulse();
        };

        // Store animation reference for cleanup
        const animationRef = { current: null as number | null };
        animatePulse();

        // Store animation ref for cleanup
        // Store marker with animation data
        markersRef.current.push({ circle, pulseCircle, animationRef });

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
          offset: [0, -15], // Greater offset to position better
          autoPan: true, // Auto-pan to make popup visible
          autoPanPadding: [20, 20] // Padding for auto-pan
        });

        // Enhanced hover interactions with adjusted values for smaller circles
        circle.on('mouseover', () => {
          circle.setStyle({ 
            weight: 3.5,
            fillOpacity: 0.85,
            color: '#FFFFFF'
          });
          const newRadius = radius * 2000; // Even larger for better visibility
          circle.setRadius(newRadius);
          setHoveredRegion(region);
          
          // Ensure popup stays visible during animation
          setTimeout(() => {
            circle.openPopup();
            circle.bringToFront();
          }, 50);

          // Update popup position continuously during animation
          const updatePopup = () => {
            if (circle._popup && circle._popup.isOpen()) {
              circle._popup.setLatLng(circle.getLatLng());
            }
          };
          circle.on('radius', updatePopup);
        });

        circle.on('mouseout', () => {
          circle.setStyle({ 
            weight: 2,
            fillOpacity: 0.5,
            color
          });
          circle.setRadius(radius * 1200);
          circle.off('radius');

          // Keep popup open for a moment to allow user to read
          setTimeout(() => {
            if (!circle._map) return; // Check if marker is still on map
            setHoveredRegion(null);
            circle.closePopup();
          }, 500); // Small delay to keep popup visible for better UX
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