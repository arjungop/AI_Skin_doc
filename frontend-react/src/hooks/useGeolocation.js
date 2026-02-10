import { useState, useEffect } from 'react';

export function useGeolocation() {
    const [location, setLocation] = useState(null);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(false);

    const requestLocation = () => {
        setLoading(true);
        setError(null);

        // 1. Try Browser Geolocation
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    setLocation({
                        latitude: position.coords.latitude,
                        longitude: position.coords.longitude,
                    });
                    setLoading(false);
                },
                (err) => {
                    console.warn('Geolocation access denied/failed, falling back to IP location.');
                    fetchIPLocation();
                }
            );
        } else {
            fetchIPLocation();
        }
    };

    const fetchIPLocation = async () => {
        try {
            const res = await fetch('https://ipapi.co/json/');
            const data = await res.json();
            if (data.latitude && data.longitude) {
                setLocation({
                    latitude: parseFloat(data.latitude),
                    longitude: parseFloat(data.longitude)
                });
            } else {
                throw new Error('Invalid IP data');
            }
        } catch (err) {
            console.error(err);
            setError('Could not retrieve location via GPS or IP.');
        } finally {
            setLoading(false);
        }
    };

    return { location, error, loading, requestLocation };
}
