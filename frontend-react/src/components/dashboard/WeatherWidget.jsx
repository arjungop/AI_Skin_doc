import { useState, useEffect } from 'react'
import { LuCloudSun, LuDroplets, LuThermometer, LuMapPin, LuLoader } from 'react-icons/lu'
import { Card, IconWrapper } from '../Card'

export default function WeatherWidget({ location }) {
    const [weather, setWeather] = useState(null)
    const [loading, setLoading] = useState(false)
    const [cityName, setCityName] = useState('')

    useEffect(() => {
        if (location?.latitude && location?.longitude) {
            fetchWeather(location.latitude, location.longitude)
            fetchCityName(location.latitude, location.longitude)
        }
    }, [location])

    const fetchWeather = async (lat, lon) => {
        setLoading(true)
        try {
            // Using Open-Meteo free API (no key required)
            const res = await fetch(
                `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,relative_humidity_2m,weather_code&timezone=auto`
            )
            const data = await res.json()
            if (data.current) {
                setWeather({
                    temp: Math.round(data.current.temperature_2m),
                    humidity: data.current.relative_humidity_2m,
                    weatherCode: data.current.weather_code
                })
            }
        } catch (err) {
            console.error('Weather fetch error:', err)
        } finally {
            setLoading(false)
        }
    }

    const fetchCityName = async (lat, lon) => {
        try {
            // Reverse geocoding with Open-Meteo geocoding API
            const res = await fetch(
                `https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lon}&format=json`
            )
            const data = await res.json()
            if (data.address) {
                const city = data.address.city || data.address.town || data.address.village || data.address.suburb || 'Your Location'
                setCityName(city)
            }
        } catch (err) {
            setCityName('Your Location')
        }
    }

    const getWeatherDescription = (code) => {
        if (code <= 3) return 'Clear skies'
        if (code <= 48) return 'Cloudy'
        if (code <= 67) return 'Rainy'
        if (code <= 77) return 'Snowy'
        return 'Stormy'
    }

    const getSkinAdvice = (temp, humidity) => {
        if (humidity > 70) return '💧 High humidity → light hydration recommended'
        if (humidity < 30) return '🧴 Low humidity → use rich moisturizer'
        if (temp > 30) return '☀️ Hot weather → stay hydrated & use SPF'
        if (temp < 10) return '❄️ Cold weather → protect with barrier cream'
        return '✓ Good conditions for your skin'
    }

    // Show prompt to enable location if not available
    if (!location) {
        return (
            <Card variant="glass" className="h-full" hover={false}>
                <div className="flex flex-col items-center justify-center h-full py-4 text-center">
                    <div className="w-12 h-12 rounded-2xl bg-white/5 flex items-center justify-center border border-white/10 mb-3">
                        <LuMapPin className="text-text-muted" size={22} />
                    </div>
                    <p className="text-sm text-text-secondary font-medium mb-1">Location Required</p>
                    <p className="text-xs text-text-muted">Enable location to see weather data</p>
                </div>
            </Card>
        )
    }

    if (loading || !weather) {
        return (
            <Card variant="glass" className="h-full" hover={false}>
                <div className="flex items-center justify-center h-full py-8">
                    <LuLoader className="animate-spin text-primary-400" size={24} />
                </div>
            </Card>
        )
    }

    return (
        <Card variant="glass" className="h-full" hover={false}>
            <div className="flex items-center justify-between mb-4">
                <div className="text-xs uppercase tracking-widest text-text-tertiary font-medium">
                    Environment
                </div>
                {cityName && (
                    <div className="flex items-center gap-1 text-xs text-text-muted">
                        <LuMapPin size={12} />
                        <span>{cityName}</span>
                    </div>
                )}
            </div>

            <div className="flex items-center gap-6">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-warning/10 flex items-center justify-center">
                        <LuThermometer className="text-warning" size={20} />
                    </div>
                    <div>
                        <span className="font-mono text-2xl font-bold text-text-primary">{weather.temp}°</span>
                        <p className="text-xs text-text-tertiary">Celsius</p>
                    </div>
                </div>

                <div className="h-12 w-px bg-white/10" />

                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-ai-500/10 flex items-center justify-center">
                        <LuDroplets className="text-ai-400" size={20} />
                    </div>
                    <div>
                        <span className="font-mono text-2xl font-bold text-text-primary">{weather.humidity}%</span>
                        <p className="text-xs text-text-tertiary">Humidity</p>
                    </div>
                </div>
            </div>

            <p className="text-sm text-text-secondary mt-4 font-medium">
                {getSkinAdvice(weather.temp, weather.humidity)}
            </p>
        </Card>
    )
}
