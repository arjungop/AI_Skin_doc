import { useEffect, useState } from 'react'
import { LuSun, LuShield, LuMapPin, LuLoader } from 'react-icons/lu'
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
        if (val === null) return { text: 'Unknown', color: 'text-text-muted', bg: 'bg-white/5' }
        if (val <= 2) return { text: 'Low', color: 'text-success', bg: 'bg-success/10' }
        if (val <= 5) return { text: 'Moderate', color: 'text-warning', bg: 'bg-warning/10' }
        if (val <= 7) return { text: 'High', color: 'text-orange-400', bg: 'bg-orange-400/10' }
        if (val <= 10) return { text: 'Very High', color: 'text-danger', bg: 'bg-danger/10' }
        return { text: 'Extreme', color: 'text-pink-500', bg: 'bg-pink-500/10' }
    }

    const getSPFRecommendation = (val) => {
        if (val === null) return 'Enable location for UV data'
        if (val <= 2) return '✓ Low exposure today'
        if (val <= 5) return '☀️ SPF 30 recommended'
        if (val <= 7) return '☀️ Wear SPF 50+'
        if (val <= 10) return '⚠️ SPF 50+ & seek shade'
        return '🚨 Avoid sun exposure!'
    }

    const risk = getRiskLevel(uv)

    // Show prompt to enable location if not available
    if (!location) {
        return (
            <Card variant="glass" className="h-full" hover={false}>
                <div className="flex flex-col items-center justify-center h-full py-4 text-center">
                    <div className="w-12 h-12 rounded-2xl bg-white/5 flex items-center justify-center border border-white/10 mb-3">
                        <LuSun className="text-text-muted" size={22} />
                    </div>
                    <p className="text-sm text-text-secondary font-medium mb-1">Location Required</p>
                    <p className="text-xs text-text-muted">Enable location to see UV index</p>
                </div>
            </Card>
        )
    }

    if (loading) {
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
            <div className="flex items-start justify-between">
                <div>
                    <div className="text-xs uppercase tracking-widest text-text-tertiary font-medium mb-3">
                        UV Index
                    </div>
                    <div className="flex items-baseline gap-3 mb-3">
                        <span className="font-mono text-4xl font-bold text-text-primary">{uv ?? '--'}</span>
                        <span className={`text-xs font-semibold px-2.5 py-1 rounded-full ${risk.bg} ${risk.color} uppercase tracking-wider`}>
                            {risk.text}
                        </span>
                    </div>
                    <p className="text-sm text-text-secondary font-medium">
                        {getSPFRecommendation(uv)}
                    </p>
                </div>
                <div className={`w-12 h-12 rounded-2xl ${risk.bg} flex items-center justify-center ${risk.color}`}>
                    {uv !== null && uv > 5 ? <LuSun size={24} /> : <LuShield size={24} />}
                </div>
            </div>
        </Card>
    )
}
