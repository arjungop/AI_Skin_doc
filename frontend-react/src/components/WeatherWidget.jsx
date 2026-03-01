import React, { useState, useEffect } from 'react';
import { FaSun, FaCloudRain, FaMapMarkerAlt } from 'react-icons/fa';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';
const DEFAULT_CITY = 'Coimbatore';

const WeatherWidget = () => {
    const [weather, setWeather] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [city, setCity] = useState(localStorage.getItem('user_city') || DEFAULT_CITY);
    const [locating, setLocating] = useState(false);
    const [retryCount, setRetryCount] = useState(0);

    // Fetch weather whenever city changes — with automatic retry
    useEffect(() => {
        let cancelled = false;
        const doFetch = async (retries = 2) => {
            setLoading(true);
            setError(null);
            try {
                const url = `${API_BASE}/recommendations/context?city=${encodeURIComponent(city)}`;
                const res = await fetch(url);
                if (!cancelled) {
                    if (res.ok) {
                        const data = await res.json();
                        setWeather(data);
                        if (data.city) localStorage.setItem('user_city', data.city);
                    } else if (retries > 0) {
                        // Auto-retry after a short delay (handles transient 502s)
                        await new Promise(r => setTimeout(r, 1500));
                        if (!cancelled) return doFetch(retries - 1);
                    } else {
                        const text = await res.text().catch(() => '');
                        console.error('[WeatherWidget] API error:', res.status, text);
                        setError(`API returned ${res.status}`);
                    }
                }
            } catch (err) {
                if (!cancelled && retries > 0) {
                    await new Promise(r => setTimeout(r, 1500));
                    if (!cancelled) return doFetch(retries - 1);
                }
                console.error('[WeatherWidget] fetch error:', err);
                if (!cancelled) setError(err.message);
            } finally {
                if (!cancelled) setLoading(false);
            }
        };
        doFetch();
        return () => { cancelled = true; };
    }, [city, retryCount]);

    // Try geolocation on mount (non-blocking — weather loads with default city immediately)
    useEffect(() => {
        if (localStorage.getItem('user_city')) return; // already have a saved city
        if (!('geolocation' in navigator)) return;

        navigator.geolocation.getCurrentPosition(
            (pos) => {
                const coords = `${pos.coords.latitude.toFixed(4)},${pos.coords.longitude.toFixed(4)}`;
                setCity(coords); // triggers re-fetch with actual location
            },
            () => { /* denied/error — already showing default city weather */ },
            { enableHighAccuracy: false, timeout: 8000 }
        );
    }, []);

    const handleCityChange = (e) => {
        if (e.key === 'Enter' && e.target.value.trim()) {
            const newCity = e.target.value.trim();
            localStorage.setItem('user_city', newCity);
            setCity(newCity);
        }
    };

    const handleUseMyLocation = () => {
        if (!('geolocation' in navigator)) return;
        setLocating(true);
        navigator.geolocation.getCurrentPosition(
            (pos) => {
                const coords = `${pos.coords.latitude.toFixed(4)},${pos.coords.longitude.toFixed(4)}`;
                localStorage.removeItem('user_city');
                setCity(coords);
                setLocating(false);
            },
            () => setLocating(false),
            { enableHighAccuracy: false, timeout: 8000 }
        );
    };

    if (loading || locating) return (
        <div className="animate-pulse h-32 bg-gray-100 dark:bg-slate-700 rounded-xl flex items-center justify-center text-sm text-gray-400">
            {locating ? 'Detecting your location…' : 'Loading weather…'}
        </div>
    );

    if (!weather) return (
        <div className="bg-red-50 dark:bg-red-900/30 p-4 rounded-xl text-red-600 dark:text-red-400 space-y-2">
            <p>Could not load weather{error ? `: ${error}` : '.'}</p>
            <button onClick={() => { setError(null); setRetryCount(c => c + 1); }}
                className="text-sm underline">Retry</button>
        </div>
    );

    return (
        <div className="bg-gradient-to-r from-blue-500 to-cyan-400 p-6 rounded-2xl text-white shadow-lg">
            <div className="flex justify-between items-start">
                <div>
                    <h2 className="text-2xl font-bold mb-1">{weather.city || city}</h2>
                    <p className="opacity-90 capitalize">{weather.description || 'Current conditions'}</p>
                    <div className="mt-4 text-5xl font-bold">{weather.temp_c ?? '—'}°C</div>
                </div>
                <div className="bg-white/20 p-3 rounded-lg backdrop-blur-sm">
                    {(weather.uv_index ?? 0) > 5 ? <FaSun size={30} className="text-yellow-300" /> : <FaCloudRain size={30} />}
                </div>
            </div>

            <div className="mt-6 flex justify-between text-sm bg-white/10 p-3 rounded-lg">
                <div className="flex flex-col items-center">
                    <span className="opacity-80">Humidity</span>
                    <span className="font-bold">{weather.humidity ?? '—'}%</span>
                </div>
                <div className="w-px bg-white/20"></div>
                <div className="flex flex-col items-center">
                    <span className="opacity-80">UV Index</span>
                    <span className={`font-bold ${(weather.uv_index ?? 0) > 5 ? 'text-yellow-300' : ''}`}>
                        {weather.uv_index ?? '—'}
                    </span>
                </div>
                <div className="w-px bg-white/20"></div>
                <div className="flex flex-col items-center">
                    <span className="opacity-80">Skin Alert</span>
                    <span className="font-bold">
                        {(weather.uv_index ?? 0) > 5 ? 'Wear SPF 50+' : 'Good to go'}
                    </span>
                </div>
            </div>

            <div className="mt-4 flex gap-2">
                <input
                    type="text"
                    placeholder="Change city..."
                    onKeyDown={handleCityChange}
                    className="text-black text-sm px-3 py-1 rounded flex-1 opacity-50 focus:opacity-100 transition-opacity outline-none"
                />
                <button
                    onClick={handleUseMyLocation}
                    title="Use my location"
                    className="bg-white/20 hover:bg-white/30 transition-colors px-3 py-1 rounded text-sm flex items-center gap-1"
                >
                    <FaMapMarkerAlt size={12} /> GPS
                </button>
            </div>
        </div>
    );
};

export default WeatherWidget;
