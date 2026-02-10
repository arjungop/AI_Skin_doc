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

    const getSkinAdvice = (temp, humidity) => {
        if (humidity > 70) return { text: 'High humidity: Light hydration recommended', color: 'text-blue-600' }
        if (humidity < 30) return { text: 'Low humidity: Use rich moisturizer', color: 'text-indigo-600' }
        if (temp > 30) return { text: 'Hot weather: Stay hydrated & use SPF', color: 'text-orange-600' }
        if (temp < 10) return { text: 'Cold weather: Protect with barrier cream', color: 'text-sky-600' }
        return { text: 'Conditions are good for your skin', color: 'text-emerald-600' }
    }

    // Show prompt to enable location if not available
    if (!location) {
        return (
            <Card className="h-full" hover={false}>
                <div className="flex flex-col items-center justify-center h-full py-4 text-center">
                    <div className="w-12 h-12 rounded-2xl bg-slate-100 flex items-center justify-center border border-slate-200 mb-3">
                        <LuMapPin className="text-slate-400" size={22} />
                    </div>
                    <p className="text-sm text-slate-500 font-medium mb-1">Location Required</p>
                    <p className="text-xs text-slate-400">Enable location to see weather data</p>
                </div>
            </Card>
        )
    }

    if (loading || !weather) {
        return (
            <Card className="h-full" hover={false}>
                <div className="flex items-center justify-center h-full py-8">
                    <LuLoader className="animate-spin text-primary-400" size={24} />
                </div>
            </Card>
        )
    }

    const advice = getSkinAdvice(weather.temp, weather.humidity)

    return (
        <Card className="h-full" hover={false}>
            <div className="flex items-center justify-between mb-6">
                <div className="text-xs uppercase tracking-widest text-slate-400 font-medium">
                    Environment
                </div>
                {cityName && (
                    <div className="flex items-center gap-1.5 text-xs text-slate-500 bg-slate-100 px-2 py-1 rounded-full">
                        <LuMapPin size={10} />
                        <span className="max-w-[100px] truncate">{cityName}</span>
                    </div>
                )}
            </div>

            <div className="flex items-center justify-between gap-4 mb-6">
                <div className="flex items-center gap-3">
                    <div className="w-12 h-12 rounded-2xl bg-orange-50 flex items-center justify-center text-orange-500">
                        <LuThermometer size={24} />
                    </div>
                    <div>
                        <span className="text-3xl font-bold text-slate-800">{weather.temp}°</span>
                        <p className="text-xs text-slate-500 font-medium uppercase tracking-wide">Celsius</p>
                    </div>
                </div>

                <div className="w-px h-10 bg-slate-100" />

                <div className="flex items-center gap-3">
                    <div className="w-12 h-12 rounded-2xl bg-blue-50 flex items-center justify-center text-blue-500">
                        <LuDroplets size={24} />
                    </div>
                    <div>
                        <span className="text-3xl font-bold text-slate-800">{weather.humidity}%</span>
                        <p className="text-xs text-slate-500 font-medium uppercase tracking-wide">Humidity</p>
                    </div>
                </div>
            </div>

            <div className={`text-sm font-medium flex items-start gap-2 ${advice.color} bg-slate-50 p-3 rounded-xl`}>
                <LuCloudSun className="w-4 h-4 mt-0.5 flex-shrink-0" />
                <span className="leading-tight">{advice.text}</span>
            </div>
        </Card>
    )
}
