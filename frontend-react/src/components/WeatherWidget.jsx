import React, { useState, useEffect } from 'react';
import { FaSun, FaCloudRain, FaSearch, FaLeaf } from 'react-icons/fa';

const WeatherWidget = () => {
    const [weather, setWeather] = useState(null);
    const [loading, setLoading] = useState(true);
    const [city, setCity] = useState(localStorage.getItem('user_city') || 'London'); // Default or saved

    useEffect(() => {
        fetchWeather();
    }, [city]);

    const fetchWeather = async () => {
        setLoading(true);
        try {
            // Fetch from our backend adapter
            const res = await fetch(`http://127.0.0.1:8000/recommendations/context?city=${city}`);
            if (res.ok) {
                const data = await res.json();
                setWeather(data);
            }
        } catch (err) {
            console.error("Weather fetch failed", err);
        } finally {
            setLoading(false);
        }
    };

    const handleCityChange = (e) => {
        if (e.key === 'Enter') {
            setCity(e.target.value);
            localStorage.setItem('user_city', e.target.value);
        }
    };

    if (loading) return <div className="animate-pulse h-32 bg-gray-100 rounded-xl"></div>;

    if (!weather) return (
        <div className="bg-red-50 p-4 rounded-xl text-red-600">
            Could not load weather. Check backend connection.
        </div>
    );

    return (
        <div className="bg-gradient-to-r from-blue-500 to-cyan-400 p-6 rounded-2xl text-white shadow-lg">
            <div className="flex justify-between items-start">
                <div>
                    <h2 className="text-2xl font-bold mb-1">{weather.city}</h2>
                    <p className="opacity-90 capitalize">{weather.description}</p>
                    <div className="mt-4 text-5xl font-bold">{weather.temp_c}°C</div>
                </div>
                <div className="bg-white/20 p-3 rounded-lg backdrop-blur-sm">
                    {weather.uv_index > 5 ? <FaSun size={30} className="text-yellow-300" /> : <FaCloudRain size={30} />}
                </div>
            </div>

            <div className="mt-6 flex justify-between text-sm bg-white/10 p-3 rounded-lg">
                <div className="flex flex-col items-center">
                    <span className="opacity-80">Humidity</span>
                    <span className="font-bold">{weather.humidity}%</span>
                </div>
                <div className="w-px bg-white/20"></div>
                <div className="flex flex-col items-center">
                    <span className="opacity-80">UV Index</span>
                    <span className={`font-bold ${weather.uv_index > 5 ? 'text-yellow-300' : ''}`}>
                        {weather.uv_index}
                    </span>
                </div>
                <div className="w-px bg-white/20"></div>
                <div className="flex flex-col items-center">
                    <span className="opacity-80">Skin Alert</span>
                    <span className="font-bold">
                        {weather.uv_index > 5 ? 'Wear SPF 50+' : 'Good to go'}
                    </span>
                </div>
            </div>

            <div className="mt-4">
                <input
                    type="text"
                    placeholder="Change city..."
                    onKeyDown={handleCityChange}
                    className="text-black text-sm px-3 py-1 rounded w-full opacity-50 focus:opacity-100 transition-opacity outline-none"
                />
            </div>
        </div>
    );
};

export default WeatherWidget;
