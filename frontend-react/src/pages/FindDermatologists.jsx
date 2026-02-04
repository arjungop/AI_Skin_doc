import { useState, useEffect, useRef } from 'react';
import { MapPin, Navigation, Phone, Clock, Star, ExternalLink } from 'lucide-react';
import { Card } from '../components/Card';
import { motion } from 'framer-motion';

const GOOGLE_MAPS_API_KEY = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;

export default function FindDermatologists() {
  const [userLocation, setUserLocation] = useState(null);
  const [hospitals, setHospitals] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedHospital, setSelectedHospital] = useState(null);
  const mapRef = useRef(null);
  const googleMapRef = useRef(null);
  const markersRef = useRef([]);

  useEffect(() => {
    // Load Google Maps script
    const script = document.createElement('script');
    script.src = `https://maps.googleapis.com/maps/api/js?key=${GOOGLE_MAPS_API_KEY}&libraries=places`;
    script.async = true;
    script.defer = true;
    document.head.appendChild(script);

    script.onload = () => {
      getUserLocation();
    };

    return () => {
      if (document.head.contains(script)) {
        document.head.removeChild(script);
      }
    };
  }, []);

  const getUserLocation = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const location = {
            lat: position.coords.latitude,
            lng: position.coords.longitude,
          };
          setUserLocation(location);
          initMap(location);
          searchNearbyDermatologists(location);
        },
        (error) => {
          console.error('Error getting location:', error);
          // Default to a major city if location access denied
          const defaultLocation = { lat: 51.5074, lng: -0.1278 }; // London
          setUserLocation(defaultLocation);
          initMap(defaultLocation);
          searchNearbyDermatologists(defaultLocation);
        }
      );
    }
  };

  const initMap = (location) => {
    if (!mapRef.current || !window.google) return;

    googleMapRef.current = new window.google.maps.Map(mapRef.current, {
      center: location,
      zoom: 13,
      styles: [
        {
          featureType: 'all',
          elementType: 'geometry',
          stylers: [{ color: '#f5f5f5' }],
        },
        {
          featureType: 'all',
          elementType: 'labels.text.fill',
          stylers: [{ color: '#121212' }],
        },
        {
          featureType: 'water',
          elementType: 'geometry',
          stylers: [{ color: '#0F766E' }, { lightness: 80 }],
        },
        {
          featureType: 'road',
          elementType: 'geometry',
          stylers: [{ color: '#ffffff' }],
        },
      ],
    });

    // Add user location marker
    new window.google.maps.Marker({
      position: location,
      map: googleMapRef.current,
      icon: {
        path: window.google.maps.SymbolPath.CIRCLE,
        scale: 10,
        fillColor: '#2563EB',
        fillOpacity: 1,
        strokeColor: '#ffffff',
        strokeWeight: 3,
      },
      title: 'Your Location',
    });
  };

  const searchNearbyDermatologists = (location) => {
    if (!window.google) return;

    const service = new window.google.maps.places.PlacesService(googleMapRef.current);
    const request = {
      location: location,
      radius: 5000, // 5km radius
      type: 'hospital',
      keyword: 'dermatology dermatologist skin clinic',
    };

    service.nearbySearch(request, (results, status) => {
      if (status === window.google.maps.places.PlacesServiceStatus.OK) {
        // Get detailed info for each place
        const detailsPromises = results.slice(0, 10).map((place) => {
          return new Promise((resolve) => {
            service.getDetails(
              {
                placeId: place.place_id,
                fields: ['name', 'formatted_address', 'formatted_phone_number', 'rating', 'opening_hours', 'geometry', 'website', 'photos'],
              },
              (details, status) => {
                if (status === window.google.maps.places.PlacesServiceStatus.OK) {
                  resolve({
                    ...details,
                    distance: calculateDistance(location, details.geometry.location),
                  });
                } else {
                  resolve(null);
                }
              }
            );
          });
        });

        Promise.all(detailsPromises).then((detailedResults) => {
          const validResults = detailedResults.filter((r) => r !== null);
          setHospitals(validResults);
          addMarkersToMap(validResults);
          setLoading(false);
        });
      } else {
        setError('Could not find nearby dermatologists');
        setLoading(false);
      }
    });
  };

  const addMarkersToMap = (places) => {
    // Clear existing markers
    markersRef.current.forEach((marker) => marker.setMap(null));
    markersRef.current = [];

    places.forEach((place, index) => {
      const marker = new window.google.maps.Marker({
        position: place.geometry.location,
        map: googleMapRef.current,
        label: {
          text: (index + 1).toString(),
          color: '#ffffff',
          fontWeight: 'bold',
        },
        icon: {
          path: window.google.maps.SymbolPath.CIRCLE,
          scale: 15,
          fillColor: '#0F766E',
          fillOpacity: 1,
          strokeColor: '#ffffff',
          strokeWeight: 2,
        },
        title: place.name,
      });

      marker.addListener('click', () => {
        setSelectedHospital(place);
        googleMapRef.current.panTo(place.geometry.location);
      });

      markersRef.current.push(marker);
    });
  };

  const calculateDistance = (from, to) => {
    const lat1 = from.lat;
    const lon1 = from.lng;
    const lat2 = to.lat();
    const lon2 = to.lng();

    const R = 6371; // Radius of Earth in km
    const dLat = ((lat2 - lat1) * Math.PI) / 180;
    const dLon = ((lon2 - lon1) * Math.PI) / 180;
    const a =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos((lat1 * Math.PI) / 180) * Math.cos((lat2 * Math.PI) / 180) * Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return (R * c).toFixed(1);
  };

  const getDirections = (place) => {
    const url = `https://www.google.com/maps/dir/?api=1&destination=${place.geometry.location.lat()},${place.geometry.location.lng()}`;
    window.open(url, '_blank');
  };

  return (
    <div className="min-h-screen bg-surface p-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-4xl font-bold text-text-primary mb-2">
          Find <span className="text-accent-medical">Dermatologists</span> Nearby
        </h1>
        <p className="text-text-secondary">
          Locate the nearest skin clinics and dermatology hospitals in your area
        </p>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Map Section */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
          className="lg:col-span-2"
        >
          <Card variant="glass" className="overflow-hidden h-[600px]">
            <div ref={mapRef} className="w-full h-full" />
          </Card>
        </motion.div>

        {/* Results List */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="space-y-4 max-h-[600px] overflow-y-auto"
        >
          {loading ? (
            <Card variant="ceramic" className="p-6">
              <div className="animate-pulse space-y-4">
                <div className="h-4 bg-border-light rounded w-3/4" />
                <div className="h-4 bg-border-light rounded w-1/2" />
              </div>
            </Card>
          ) : error ? (
            <Card variant="ceramic" className="p-6">
              <p className="text-red-500">{error}</p>
            </Card>
          ) : (
            hospitals.map((hospital, index) => (
              <Card
                key={index}
                variant={selectedHospital === hospital ? 'glass' : 'ceramic'}
                className={`p-4 cursor-pointer transition-all hover:shadow-float ${
                  selectedHospital === hospital ? 'ring-2 ring-accent-medical' : ''
                }`}
                onClick={() => {
                  setSelectedHospital(hospital);
                  googleMapRef.current.panTo(hospital.geometry.location);
                }}
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 rounded-full bg-accent-medical text-white flex items-center justify-center font-mono text-sm font-bold">
                      {index + 1}
                    </div>
                    <div>
                      <h3 className="font-semibold text-text-primary">{hospital.name}</h3>
                      {hospital.rating && (
                        <div className="flex items-center gap-1 text-sm text-text-secondary mt-1">
                          <Star className="w-4 h-4 fill-yellow-400 text-yellow-400" />
                          <span className="font-mono">{hospital.rating}</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                <div className="space-y-2 text-sm">
                  <div className="flex items-start gap-2 text-text-secondary">
                    <MapPin className="w-4 h-4 mt-0.5 text-accent-medical flex-shrink-0" />
                    <span>{hospital.formatted_address}</span>
                  </div>

                  <div className="flex items-center gap-2 text-text-secondary">
                    <Navigation className="w-4 h-4 text-accent-ai" />
                    <span className="font-mono">{hospital.distance} km away</span>
                  </div>

                  {hospital.formatted_phone_number && (
                    <div className="flex items-center gap-2 text-text-secondary">
                      <Phone className="w-4 h-4 text-accent-medical" />
                      <a
                        href={`tel:${hospital.formatted_phone_number}`}
                        className="hover:text-accent-medical transition-colors"
                      >
                        {hospital.formatted_phone_number}
                      </a>
                    </div>
                  )}

                  {hospital.opening_hours?.weekday_text && (
                    <div className="flex items-start gap-2 text-text-secondary">
                      <Clock className="w-4 h-4 mt-0.5 text-accent-ai" />
                      <span>
                        {hospital.opening_hours.open_now ? (
                          <span className="text-green-600 font-semibold">Open Now</span>
                        ) : (
                          <span className="text-red-600 font-semibold">Closed</span>
                        )}
                      </span>
                    </div>
                  )}
                </div>

                <div className="flex gap-2 mt-4">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      getDirections(hospital);
                    }}
                    className="flex-1 px-3 py-2 bg-accent-medical text-white rounded-lg hover:bg-accent-medical/90 transition-colors text-sm font-semibold"
                  >
                    Get Directions
                  </button>
                  {hospital.website && (
                    <a
                      href={hospital.website}
                      target="_blank"
                      rel="noopener noreferrer"
                      onClick={(e) => e.stopPropagation()}
                      className="px-3 py-2 border border-border-medium rounded-lg hover:bg-border-light transition-colors"
                    >
                      <ExternalLink className="w-4 h-4 text-accent-ai" />
                    </a>
                  )}
                </div>
              </Card>
            ))
          )}
        </motion.div>
      </div>
    </div>
  );
}
