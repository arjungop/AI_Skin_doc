import { useState, useEffect, useRef, useCallback } from 'react';
import { LuMapPin, LuNavigation, LuPhone, LuClock, LuExternalLink, LuLocate, LuSearch } from 'react-icons/lu';
import { Card } from '../components/Card';
import { motion } from 'framer-motion';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Default location: Coimbatore, India
const DEFAULT_LOCATION = { lat: 11.0168, lng: 76.9558 };

// Create numbered marker icon
function createNumberedIcon(number) {
  return L.divIcon({
    className: 'custom-marker',
    html: `<div style="background:#0F766E;color:#fff;border-radius:50%;width:30px;height:30px;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:13px;border:2px solid #fff;box-shadow:0 2px 6px rgba(0,0,0,0.3);">${number}</div>`,
    iconSize: [30, 30],
    iconAnchor: [15, 15],
  });
}

// Haversine distance in km
function calculateDistance(from, to) {
  const R = 6371;
  const dLat = ((to.lat - from.lat) * Math.PI) / 180;
  const dLon = ((to.lng - from.lng) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((from.lat * Math.PI) / 180) * Math.cos((to.lat * Math.PI) / 180) * Math.sin(dLon / 2) ** 2;
  return (R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))).toFixed(1);
}

// Curated database of well-known dermatology clinics in major Indian cities
const CURATED_CLINICS = [
  // Coimbatore
  { id: 'c1', name: 'Sri Ramakrishna Hospital - Dermatology', lat: 11.0225, lng: 76.9665, address: 'Saibaba Colony, Coimbatore, Tamil Nadu', phone: '+91 422 4500000', website: 'https://www.sriramakrishnahospital.com', opening_hours: 'Mon-Sat 8am-8pm', speciality: 'Dermatology', city: 'Coimbatore' },
  { id: 'c2', name: 'KMCH - Skin & Cosmetology Dept', lat: 11.0287, lng: 76.9320, address: 'Avinashi Road, Coimbatore, Tamil Nadu', phone: '+91 422 4323800', website: 'https://www.kmchhospitals.com', opening_hours: 'Mon-Sat 8am-9pm', speciality: 'Dermatology & Cosmetology', city: 'Coimbatore' },
  { id: 'c3', name: 'PSG Hospitals - Dermatology', lat: 11.0248, lng: 77.0028, address: 'Peelamedu, Coimbatore, Tamil Nadu', phone: '+91 422 2570170', website: 'https://www.psghospitals.com', opening_hours: 'Mon-Sat 8am-8pm', speciality: 'Dermatology', city: 'Coimbatore' },
  { id: 'c4', name: 'Ganga Hospital - Dermatology', lat: 11.0180, lng: 76.9613, address: 'Mettupalayam Road, Coimbatore, Tamil Nadu', phone: '+91 422 2485000', website: 'https://www.gangahospital.com', opening_hours: 'Mon-Sat 9am-7pm', speciality: 'Dermatology', city: 'Coimbatore' },
  { id: 'c5', name: 'Rasi Skin Clinic', lat: 11.0164, lng: 76.9725, address: 'RS Puram, Coimbatore, Tamil Nadu', phone: '+91 422 2541100', website: '', opening_hours: 'Mon-Sat 10am-7pm', speciality: 'Dermatology & Hair', city: 'Coimbatore' },
  { id: 'c6', name: 'Ko Cosmetic Clinic', lat: 11.0206, lng: 77.0183, address: 'Race Course, Coimbatore, Tamil Nadu', phone: '+91 422 4961188', website: 'https://www.kocosmetic.com', opening_hours: 'Mon-Sat 10am-8pm', speciality: 'Cosmetic Dermatology', city: 'Coimbatore' },
  { id: 'c7', name: 'Royal Care Super Speciality Hospital - Skin', lat: 11.0083, lng: 76.9583, address: 'Neelambur, Coimbatore, Tamil Nadu', phone: '+91 422 4277777', website: 'https://www.royalcarehospital.com', opening_hours: 'Mon-Sat 8am-8pm', speciality: 'Dermatology', city: 'Coimbatore' },
  { id: 'c8', name: 'Revive Skin And Cosmetology Clinic', lat: 10.7655, lng: 76.6602, address: 'Coimbatore Road, Palakkad, Kerala', phone: '', website: '', opening_hours: '', speciality: 'Skin & Cosmetology', city: 'Coimbatore' },
  // Chennai
  { id: 'ch1', name: 'Apollo Hospitals - Dermatology', lat: 13.0068, lng: 80.2544, address: 'Greams Road, Chennai, Tamil Nadu', phone: '+91 44 28290200', website: 'https://www.apollohospitals.com', opening_hours: 'Open 24/7', speciality: 'Dermatology', city: 'Chennai' },
  { id: 'ch2', name: 'Dr. Kamakshi Memorial Hospital - Skin', lat: 12.8438, lng: 80.2216, address: 'Pallikaranai, Chennai, Tamil Nadu', phone: '+91 44 22771000', website: 'https://www.drkamakshi.com', opening_hours: 'Mon-Sat 8am-8pm', speciality: 'Dermatology', city: 'Chennai' },
  { id: 'ch3', name: 'SkinLab - The Dermatology Clinic', lat: 13.0489, lng: 80.2522, address: 'T Nagar, Chennai, Tamil Nadu', phone: '+91 44 49008800', website: '', opening_hours: 'Mon-Sat 10am-7pm', speciality: 'Dermatology & Aesthetics', city: 'Chennai' },
  // Bangalore
  { id: 'b1', name: 'Manipal Hospital - Dermatology', lat: 12.9593, lng: 77.6484, address: 'HAL Airport Road, Bangalore, Karnataka', phone: '+91 80 25024444', website: 'https://www.manipalhospitals.com', opening_hours: 'Mon-Sat 8am-8pm', speciality: 'Dermatology', city: 'Bangalore' },
  { id: 'b2', name: 'Dr. Sheth\'s Skin Clinic', lat: 12.9705, lng: 77.6068, address: 'Koramangala, Bangalore, Karnataka', phone: '+91 80 41122233', website: '', opening_hours: 'Mon-Sat 10am-7pm', speciality: 'Dermatology', city: 'Bangalore' },
  // Mumbai
  { id: 'm1', name: 'Jaslok Hospital - Dermatology', lat: 18.9710, lng: 72.8096, address: 'Peddar Road, Mumbai, Maharashtra', phone: '+91 22 66573333', website: 'https://www.jaslokhospital.net', opening_hours: 'Mon-Sat 8am-8pm', speciality: 'Dermatology', city: 'Mumbai' },
  // Delhi
  { id: 'd1', name: 'AIIMS - Dermatology OPD', lat: 28.5672, lng: 77.2100, address: 'Ansari Nagar, New Delhi', phone: '+91 11 26588500', website: 'https://www.aiims.edu', opening_hours: 'Mon-Fri 8am-1pm', speciality: 'Dermatology', city: 'Delhi' },
  // Hyderabad
  { id: 'h1', name: 'KIMS Hospital - Dermatology', lat: 17.3616, lng: 78.4747, address: 'Secunderabad, Hyderabad, Telangana', phone: '+91 40 44885000', website: 'https://www.kimshospitals.com', opening_hours: 'Mon-Sat 8am-8pm', speciality: 'Dermatology', city: 'Hyderabad' },
];

// Search using Nominatim (OpenStreetMap) — free, no API key
async function searchNominatim(location, radiusKm = 10) {
  const { lat, lng } = location;
  const delta = radiusKm / 111; // rough km -> degrees
  const queries = ['skin clinic', 'dermatologist', 'cosmetic clinic', 'skin hospital', 'dermatology'];
  const allResults = [];
  const seenIds = new Set();

  for (const q of queries) {
    try {
      const resp = await fetch(
        `https://nominatim.openstreetmap.org/search?` +
        `q=${encodeURIComponent(q)}&format=json&limit=8` +
        `&viewbox=${lng - delta},${lat + delta},${lng + delta},${lat - delta}&bounded=1`,
        { headers: { 'User-Agent': 'SkinDoc/1.0' }, signal: AbortSignal.timeout(8000) }
      );
      if (!resp.ok) continue;
      const data = await resp.json();
      data.forEach((item) => {
        if (seenIds.has(item.place_id)) return;
        seenIds.add(item.place_id);
        allResults.push({
          id: `nom-${item.place_id}`,
          name: item.display_name.split(',')[0],
          lat: parseFloat(item.lat),
          lng: parseFloat(item.lon),
          address: item.display_name.split(',').slice(1, 4).join(',').trim(),
          phone: '',
          website: '',
          opening_hours: '',
          speciality: q,
          distance: calculateDistance(location, { lat: parseFloat(item.lat), lng: parseFloat(item.lon) }),
          source: 'live',
        });
      });
      // Rate limit: Nominatim requires ~1 req/sec
      await new Promise((r) => setTimeout(r, 1100));
    } catch {
      /* continue */
    }
  }

  return allResults;
}

// Get curated clinics near a location, with distance
function getCuratedClinics(location, radiusKm = 50) {
  return CURATED_CLINICS
    .map((c) => ({
      ...c,
      distance: calculateDistance(location, { lat: c.lat, lng: c.lng }),
      source: 'curated',
    }))
    .filter((c) => parseFloat(c.distance) <= radiusKm)
    .sort((a, b) => parseFloat(a.distance) - parseFloat(b.distance));
}

// Combined search: live results + curated, deduplicated
async function searchDermatologists(location, radiusKm = 10) {
  // Start with curated clinics (instant, always available)
  const curated = getCuratedClinics(location, radiusKm + 20);

  // Try live search in parallel
  let live = [];
  try {
    live = await searchNominatim(location, radiusKm);
  } catch {
    /* curated fallback is enough */
  }

  // Merge: live first, then curated (deduplicate by proximity — within 200m = same place)
  const merged = [...live];
  for (const c of curated) {
    const isDuplicate = merged.some((m) => {
      const d = parseFloat(calculateDistance({ lat: m.lat, lng: m.lng }, { lat: c.lat, lng: c.lng }));
      return d < 0.2; // 200m threshold
    });
    if (!isDuplicate) merged.push(c);
  }

  return merged.sort((a, b) => parseFloat(a.distance) - parseFloat(b.distance)).slice(0, 15);
}

export default function FindDermatologists() {
  const [userLocation, setUserLocation] = useState(DEFAULT_LOCATION);
  const [hospitals, setHospitals] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedHospital, setSelectedHospital] = useState(null);
  const [searchRadius, setSearchRadius] = useState(10); // km
  const [geoStatus, setGeoStatus] = useState('detecting'); // detecting | found | denied
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const markersRef = useRef([]);
  const userMarkerRef = useRef(null);

  // Initialize Leaflet map
  useEffect(() => {
    if (mapInstanceRef.current || !mapContainerRef.current) return;
    const map = L.map(mapContainerRef.current).setView([DEFAULT_LOCATION.lat, DEFAULT_LOCATION.lng], 13);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    }).addTo(map);
    mapInstanceRef.current = map;
    // Fix tiles not loading due to container size
    setTimeout(() => map.invalidateSize(), 200);
    return () => { map.remove(); mapInstanceRef.current = null; };
  }, []);

  // Update markers when hospitals change
  const updateMarkers = useCallback((places, userLoc) => {
    const map = mapInstanceRef.current;
    if (!map) return;
    // Clear old markers
    markersRef.current.forEach((m) => m.remove());
    markersRef.current = [];
    if (userMarkerRef.current) { userMarkerRef.current.remove(); userMarkerRef.current = null; }
    // User location marker
    userMarkerRef.current = L.circleMarker([userLoc.lat, userLoc.lng], {
      radius: 10, color: '#2563EB', fillColor: '#2563EB', fillOpacity: 1, weight: 3,
    }).addTo(map).bindPopup('Your Location');
    // Place markers
    places.forEach((h, i) => {
      const marker = L.marker([h.lat, h.lng], { icon: createNumberedIcon(i + 1) })
        .addTo(map)
        .bindPopup(`<strong>${h.name}</strong><br/>${h.address || ''}<br/>${h.distance} km away`);
      marker.on('click', () => setSelectedHospital(h));
      markersRef.current.push(marker);
    });
  }, []);

  // Fly map to location
  const flyTo = useCallback((loc, zoom = 14) => {
    const map = mapInstanceRef.current;
    if (map) map.flyTo([loc.lat, loc.lng], zoom, { duration: 1.2 });
  }, []);

  // On mount, get user location then search
  useEffect(() => {
    let cancelled = false;

    const doSearch = async (loc) => {
      setLoading(true);
      setError(null);
      try {
        const results = await searchDermatologists(loc, searchRadius);
        if (cancelled) return;
        if (results.length === 0) {
          setError(`No dermatologists found within ${searchRadius} km. Try increasing the search radius.`);
        }
        setHospitals(results);
        updateMarkers(results, loc);
      } catch (err) {
        if (cancelled) return;
        console.error('Search error:', err);
        setError('Search failed. Please try again.');
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    // Only request geolocation on first mount (geoStatus === 'detecting')
    // On subsequent runs (radius change), reuse existing userLocation
    if (geoStatus === 'detecting' && navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          if (cancelled) return;
          const loc = { lat: pos.coords.latitude, lng: pos.coords.longitude };
          setUserLocation(loc);
          flyTo(loc, 13);
          setGeoStatus('found');
          doSearch(loc);
        },
        () => {
          if (cancelled) return;
          setGeoStatus('denied');
          doSearch(DEFAULT_LOCATION);
        },
        { timeout: 8000 }
      );
    } else {
      // Reuse existing location for radius changes
      doSearch(userLocation);
    }

    return () => { cancelled = true; };
  }, [searchRadius]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleRelocate = () => {
    if (!navigator.geolocation) return;
    setGeoStatus('detecting');
    setError(null);
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const loc = { lat: pos.coords.latitude, lng: pos.coords.longitude };
        setUserLocation(loc);
        flyTo(loc, 13);
        setGeoStatus('found');
        setLoading(true);
        searchDermatologists(loc, searchRadius)
          .then((results) => {
            setHospitals(results);
            updateMarkers(results, loc);
            if (results.length === 0) setError(`No dermatologists found within ${searchRadius} km.`);
            else setError(null);
          })
          .catch(() => setError('Search failed.'))
          .finally(() => setLoading(false));
      },
      () => { setGeoStatus('denied'); setError(null); },
      { timeout: 8000 }
    );
  };

  const getDirections = (place) => {
    const url = `https://www.google.com/maps/dir/?api=1&destination=${place.lat},${place.lng}`;
    window.open(url, '_blank');
  };

  return (
    <div className="min-h-screen bg-surface p-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-6"
      >
        <h1 className="text-4xl font-bold text-text-primary mb-2">
          Find <span className="text-accent-medical">Dermatologists</span> Nearby
        </h1>
        <p className="text-text-secondary">
          Locate the nearest skin clinics and dermatology hospitals in your area
        </p>

        {/* Controls */}
        <div className="flex flex-wrap items-center gap-3 mt-4">
          <button
            onClick={handleRelocate}
            className="flex items-center gap-2 px-4 py-2 bg-accent-medical text-white rounded-lg hover:bg-accent-medical/90 transition-colors text-sm font-semibold"
          >
            <LuLocate className="w-4 h-4" />
            {geoStatus === 'detecting' ? 'Detecting...' : 'Use My Location'}
          </button>

          <div className="flex items-center gap-2 text-sm text-text-secondary">
            <LuSearch className="w-4 h-4" />
            <span>Radius:</span>
            <select
              value={searchRadius}
              onChange={(e) => setSearchRadius(Number(e.target.value))}
              className="px-2 py-1 rounded-lg bg-bg-secondary border border-border-medium text-text-primary text-sm"
            >
              <option value={5}>5 km</option>
              <option value={10}>10 km</option>
              <option value={20}>20 km</option>
              <option value={50}>50 km</option>
            </select>
          </div>

          {geoStatus === 'denied' && (
            <span className="text-xs text-text-tertiary">
              Showing results near Coimbatore (location access denied)
            </span>
          )}
        </div>
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
            <div ref={mapContainerRef} className="w-full h-full" style={{ height: '100%', width: '100%' }} />
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
                <p className="text-sm text-text-tertiary mt-2">Searching for dermatologists nearby...</p>
              </div>
            </Card>
          ) : error ? (
            <Card variant="ceramic" className="p-6">
              <p className="text-red-500 mb-3">{error}</p>
              <button
                onClick={handleRelocate}
                className="px-4 py-2 bg-accent-medical text-white rounded-lg hover:bg-accent-medical/90 transition-colors text-sm"
              >
                Retry
              </button>
            </Card>
          ) : hospitals.length === 0 ? (
            <Card variant="ceramic" className="p-6">
              <p className="text-text-secondary">No results found. Try a larger search radius.</p>
            </Card>
          ) : (
            hospitals.map((hospital, index) => (
              <Card
                key={hospital.id}
                variant={selectedHospital?.id === hospital.id ? 'glass' : 'ceramic'}
                className={`p-4 cursor-pointer transition-all hover:shadow-float ${
                  selectedHospital?.id === hospital.id ? 'ring-2 ring-accent-medical' : ''
                }`}
                onClick={() => {
                  setSelectedHospital(hospital);
                  flyTo({ lat: hospital.lat, lng: hospital.lng });
                }}
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 rounded-full bg-accent-medical text-white flex items-center justify-center font-mono text-sm font-bold">
                      {index + 1}
                    </div>
                    <div>
                      <h3 className="font-semibold text-text-primary">{hospital.name}</h3>
                      {hospital.speciality && (
                        <span className="text-xs text-accent-ai bg-accent-ai/10 px-2 py-0.5 rounded-full mt-1 inline-block">
                          {hospital.speciality}
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                <div className="space-y-2 text-sm">
                  {hospital.address && (
                    <div className="flex items-start gap-2 text-text-secondary">
                      <LuMapPin className="w-4 h-4 mt-0.5 text-accent-medical flex-shrink-0" />
                      <span>{hospital.address}</span>
                    </div>
                  )}

                  <div className="flex items-center gap-2 text-text-secondary">
                    <LuNavigation className="w-4 h-4 text-accent-ai" />
                    <span className="font-mono">{hospital.distance} km away</span>
                  </div>

                  {hospital.phone && (
                    <div className="flex items-center gap-2 text-text-secondary">
                      <LuPhone className="w-4 h-4 text-accent-medical" />
                      <a
                        href={`tel:${hospital.phone}`}
                        className="hover:text-accent-medical transition-colors"
                      >
                        {hospital.phone}
                      </a>
                    </div>
                  )}

                  {hospital.opening_hours && (
                    <div className="flex items-start gap-2 text-text-secondary">
                      <LuClock className="w-4 h-4 mt-0.5 text-accent-ai" />
                      <span className="text-xs">{hospital.opening_hours}</span>
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
                      <LuExternalLink className="w-4 h-4 text-accent-ai" />
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
