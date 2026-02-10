import { useState, useEffect, useRef } from 'react';
import { MapPin, Navigation, Phone, Clock, Star, ExternalLink, Loader2 } from 'lucide-react';
import { Card } from '../components/Card';
import { motion } from 'framer-motion';
import { Loader } from '@googlemaps/js-api-loader';

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
    const init = async () => {
      const loader = new Loader({
        apiKey: GOOGLE_MAPS_API_KEY,
        version: "weekly",
        libraries: ["places"]
      });

      try {
        await loader.load();
        getUserLocation();
      } catch (e) {
        console.error('Error loading Google Maps API:', e);
        setError('Failed to load Google Maps. Please check your internet connection.');
        setLoading(false);
      }
    };

    init();
  }, []);

  const getUserLocation = () => {
    // 1. Try Geolocation API (High Accuracy)
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
          console.warn('Geolocation denied or error:', error);
          // 2. Fallback to IP-based location
          fetchIPLocation();
        },
        { timeout: 10000 }
      );
    } else {
      // Browser doesn't support Geolocation
      fetchIPLocation();
    }
  };

  const fetchIPLocation = async () => {
    try {
      const res = await fetch('https://ipapi.co/json/');
      const data = await res.json();
      if (data.latitude && data.longitude) {
        const location = { lat: parseFloat(data.latitude), lng: parseFloat(data.longitude) };
        console.log('Using IP-based location:', data.city);
        setUserLocation(location);
        initMap(location);
        searchNearbyDermatologists(location);
      } else {
        throw new Error('Invalid IP location data');
      }
    } catch (err) {
      console.error('IP location failed:', err);
      // 3. Last Resort Fallback (London)
      const defaultLocation = { lat: 51.5074, lng: -0.1278 };
      setUserLocation(defaultLocation);
      initMap(defaultLocation);
      searchNearbyDermatologists(defaultLocation);
      setError('Could not access your location. Showing default results for London.');
    }
  };

  const initMap = (location) => {
    if (!mapRef.current || !window.google) return;

    googleMapRef.current = new window.google.maps.Map(mapRef.current, {
      center: location,
      zoom: 13,
      disableDefaultUI: true,
      styles: [
        {
          featureType: 'all',
          elementType: 'geometry',
          stylers: [{ color: '#f8fafc' }],
        },
        {
          featureType: 'all',
          elementType: 'labels.text.fill',
          stylers: [{ color: '#64748b' }],
        },
        {
          featureType: 'all',
          elementType: 'labels.text.stroke',
          stylers: [{ color: '#ffffff' }, { visibility: 'on' }, { weight: 3 }],
        },
        {
          featureType: 'water',
          elementType: 'geometry',
          stylers: [{ color: '#e0f2fe' }],
        },
        {
          featureType: 'water',
          elementType: 'labels.text.fill',
          stylers: [{ color: '#0ea5e9' }],
        },
        {
          featureType: 'road',
          elementType: 'geometry',
          stylers: [{ color: '#ffffff' }, { visibility: 'simplified' }],
        },
        {
          featureType: 'poi.medical',
          elementType: 'geometry',
          stylers: [{ color: '#f0fdf4' }],
        },
        {
          featureType: 'poi.park',
          elementType: 'geometry',
          stylers: [{ color: '#f0fdf4' }],
        },
      ],
    });

    // Add user location marker
    new window.google.maps.Marker({
      position: location,
      map: googleMapRef.current,
      icon: {
        path: window.google.maps.SymbolPath.CIRCLE,
        scale: 8,
        fillColor: '#3b82f6',
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
    if (markersRef.current) {
      markersRef.current.forEach((marker) => marker.setMap(null));
    }
    markersRef.current = [];

    places.forEach((place, index) => {
      const marker = new window.google.maps.Marker({
        position: place.geometry.location,
        map: googleMapRef.current,
        label: {
          text: (index + 1).toString(),
          color: '#ffffff',
          fontWeight: 'bold',
          fontSize: '12px',
        },
        icon: {
          path: window.google.maps.SymbolPath.CIRCLE,
          scale: 14,
          fillColor: '#059669', // Emerald-600
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
    <div className="min-h-screen pb-12">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8 max-w-7xl mx-auto pt-6"
      >
        <h1 className="text-4xl font-bold text-slate-900 mb-2 tracking-tight">
          Find <span className="text-emerald-600">Dermatologists</span> Nearby
        </h1>
        <p className="text-slate-500 text-lg">
          Locate the nearest skin clinics and dermatology hospitals in your area
        </p>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 max-w-7xl mx-auto h-[600px]">
        {/* Map Section */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
          className="lg:col-span-2 h-full"
        >
          <Card className="overflow-hidden h-full p-0 border-slate-200">
            <div ref={mapRef} className="w-full h-full" />
          </Card>
        </motion.div>

        {/* Results List */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="space-y-4 h-full overflow-y-auto pr-1"
        >
          {loading ? (
            <Card className="p-8 flex items-center justify-center h-40">
              <div className="flex flex-col items-center gap-3 text-slate-400">
                <Loader2 className="animate-spin text-emerald-500" size={32} />
                <p>Searching nearby...</p>
              </div>
            </Card>
          ) : error ? (
            <Card className="p-6 border-rose-100 bg-rose-50">
              <p className="text-rose-600 font-medium">{error}</p>
              <p className="text-xs text-rose-500 mt-2">Try allowing location access or check your connection.</p>
            </Card>
          ) : (
            hospitals.map((hospital, index) => (
              <Card
                key={index}
                className={`p-5 cursor-pointer transition-all hover:shadow-soft-lg group ${selectedHospital === hospital ? 'ring-2 ring-emerald-500 bg-emerald-50/30' : 'hover:border-emerald-200'
                  }`}
                onClick={() => {
                  setSelectedHospital(hospital);
                  googleMapRef.current.panTo(hospital.geometry.location);
                }}
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 rounded-full bg-emerald-600 text-white flex items-center justify-center font-bold text-sm shadow-sm flex-shrink-0">
                      {index + 1}
                    </div>
                    <div>
                      <h3 className="font-bold text-slate-900 leading-tight group-hover:text-emerald-700 transition-colors">{hospital.name}</h3>
                      {hospital.rating && (
                        <div className="flex items-center gap-1 text-sm text-slate-500 mt-1">
                          <Star className="w-3.5 h-3.5 fill-amber-400 text-amber-400" />
                          <span className="font-semibold">{hospital.rating}</span>
                          <span className="text-slate-400">({hospital.user_ratings_total || 0})</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                <div className="space-y-2 text-sm pl-11">
                  <div className="flex items-start gap-2 text-slate-500">
                    <MapPin className="w-4 h-4 mt-0.5 text-emerald-500 flex-shrink-0" />
                    <span className="line-clamp-2">{hospital.formatted_address}</span>
                  </div>

                  <div className="flex items-center gap-2 text-slate-500">
                    <Navigation className="w-4 h-4 text-sky-500" />
                    <span className="font-medium text-slate-700">{hospital.distance} km away</span>
                  </div>

                  {hospital.formatted_phone_number && (
                    <div className="flex items-center gap-2 text-slate-500">
                      <Phone className="w-4 h-4 text-emerald-500" />
                      <a
                        href={`tel:${hospital.formatted_phone_number}`}
                        className="hover:text-emerald-600 transition-colors hover:underline"
                        onClick={e => e.stopPropagation()}
                      >
                        {hospital.formatted_phone_number}
                      </a>
                    </div>
                  )}

                  {hospital.opening_hours?.weekday_text && (
                    <div className="flex items-start gap-2 text-slate-500">
                      <Clock className="w-4 h-4 mt-0.5 text-slate-400" />
                      <span>
                        {hospital.opening_hours.open_now ? (
                          <span className="text-emerald-600 font-bold text-xs bg-emerald-50 px-2 py-0.5 rounded-full border border-emerald-100">Open Now</span>
                        ) : (
                          <span className="text-rose-600 font-bold text-xs bg-rose-50 px-2 py-0.5 rounded-full border border-rose-100">Closed</span>
                        )}
                      </span>
                    </div>
                  )}
                </div>

                <div className="flex gap-2 mt-4 pl-11">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      getDirections(hospital);
                    }}
                    className="flex-1 px-3 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 transition-colors text-sm font-semibold shadow-sm hover:shadow-md"
                  >
                    Get Directions
                  </button>
                  {hospital.website && (
                    <a
                      href={hospital.website}
                      target="_blank"
                      rel="noopener noreferrer"
                      onClick={(e) => e.stopPropagation()}
                      className="px-3 py-2 border border-slate-200 rounded-lg hover:bg-slate-50 transition-colors text-slate-500 hover:text-slate-900 bg-white"
                    >
                      <ExternalLink className="w-4 h-4" />
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
