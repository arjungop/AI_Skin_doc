import { useEffect, useState } from 'react'
import { LuSun, LuShield, LuMapPin, LuLoader, LuShieldAlert } from 'react-icons/lu'
import { Card } from '../Card'

export default function UVIndexWidget({ location }) {
    const [uv, setUv] = useState(null)
    const [loading, setLoading] = useState(false)

    useEffect(() => {
        if (location?.latitude && location?.longitude) {
            fetchUV(location.latitude, location.longitude)
        }
    }, [location])

    const fetchUV = async (lat, lon) => {
        setLoading(true)
        try {
            // Using Open-Meteo free API for UV data
            const res = await fetch(
                `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&daily=uv_index_max&timezone=auto&forecast_days=1`
            )
            const data = await res.json()
            if (data.daily?.uv_index_max?.[0] !== undefined) {
                setUv(Math.round(data.daily.uv_index_max[0] * 10) / 10)
            }
        } catch (err) {
            console.error('UV fetch error:', err)
            setUv(0)
        } finally {
            setLoading(false)
        }
    }

    const getRiskLevel = (val) => {
        if (val === null) return { text: 'Unknown', color: 'text-slate-400', bg: 'bg-slate-100' }
        if (val <= 2) return { text: 'Low', color: 'text-emerald-600', bg: 'bg-emerald-100' }
        if (val <= 5) return { text: 'Moderate', color: 'text-amber-500', bg: 'bg-amber-100' }
        if (val <= 7) return { text: 'High', color: 'text-orange-500', bg: 'bg-orange-100' }
        if (val <= 10) return { text: 'Very High', color: 'text-rose-600', bg: 'bg-rose-100' }
        return { text: 'Extreme', color: 'text-purple-600', bg: 'bg-purple-100' }
    }

    const getSPFRecommendation = (val) => {
        if (val === null) return 'Enable location for UV data'
        if (val <= 2) return 'Low exposure today'
        if (val <= 5) return 'SPF 30 recommended'
        if (val <= 7) return 'Wear SPF 50+'
        if (val <= 10) return 'SPF 50+ & seek shade'
        return 'Avoid sun exposure!'
    }

    const risk = getRiskLevel(uv)

    // Show prompt to enable location if not available
    if (!location) {
        return (
            <Card className="h-full" hover={false}>
                <div className="flex flex-col items-center justify-center h-full py-4 text-center">
                    <div className="w-12 h-12 rounded-2xl bg-slate-100 flex items-center justify-center border border-slate-200 mb-3">
                        <LuSun className="text-slate-400" size={22} />
                    </div>
                    <p className="text-sm text-slate-500 font-medium mb-1">Location Required</p>
                    <p className="text-xs text-slate-400">Enable location to see UV index</p>
                </div>
            </Card>
        )
    }

    if (loading) {
        return (
            <Card className="h-full" hover={false}>
                <div className="flex items-center justify-center h-full py-8">
                    <LuLoader className="animate-spin text-primary-400" size={24} />
                </div>
            </Card>
        )
    }

    return (
        <Card className="h-full" hover={false}>
            <div className="flex items-start justify-between">
                <div className="w-full">
                    <div className="text-xs uppercase tracking-widest text-slate-400 font-medium mb-4">
                        UV Index
                    </div>

                    <div className="flex items-start justify-between w-full mb-6">
                        <div className="flex flex-col">
                            <span className="text-5xl font-bold text-slate-800 tracking-tight leading-none mb-1">{uv ?? '--'}</span>
                            <span className={`inline-flex self-start px-2 py-0.5 rounded-md text-[10px] font-bold uppercase tracking-wider ${risk.bg} ${risk.color}`}>
                                {risk.text}
                            </span>
                        </div>

                        <div className={`w-12 h-12 rounded-2xl ${risk.bg} flex items-center justify-center ${risk.color} flex-shrink-0`}>
                            {uv !== null && uv > 5 ? <LuShieldAlert size={24} /> : <LuSun size={24} />}
                        </div>
                    </div>

                    <div className="flex items-center gap-2 text-sm text-slate-600 font-medium bg-slate-50 p-3 rounded-xl">
                        <LuShield className="w-4 h-4 text-emerald-500 flex-shrink-0" />
                        <span>{getSPFRecommendation(uv)}</span>
                    </div>
                </div>
            </div>
        </Card>
    )
}
