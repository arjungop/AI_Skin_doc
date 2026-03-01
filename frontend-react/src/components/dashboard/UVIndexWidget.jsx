import { useEffect, useState } from 'react'
import { LuSun, LuShield, LuLoader, LuShieldAlert, LuClock, LuDroplet } from 'react-icons/lu'
import { Card } from '../Card'
import { api } from '../../services/api.js'

export default function UVIndexWidget() {
    const [data, setData] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)

    useEffect(() => {
        fetchUVRisk()
    }, [])

    const fetchUVRisk = async () => {
        setLoading(true)
        setError(null)
        try {
            // Call our backend — auto-detects city from user profile
            const result = await api.getUVRisk()
            setData(result)
        } catch (err) {
            console.error('UV risk fetch error:', err)
            // Graceful fallback: show the widget with an error message
            setError(err.message || 'Could not fetch UV data')
        } finally {
            setLoading(false)
        }
    }

    const getRiskColors = (level) => {
        if (!level) return { color: 'text-slate-400', bg: 'bg-slate-100' }
        const l = level.toLowerCase()
        if (l === 'low') return { color: 'text-emerald-600', bg: 'bg-emerald-100' }
        if (l === 'moderate') return { color: 'text-amber-500', bg: 'bg-amber-100' }
        if (l === 'high') return { color: 'text-orange-500', bg: 'bg-orange-100' }
        if (l === 'very high') return { color: 'text-rose-600', bg: 'bg-rose-100' }
        return { color: 'text-purple-600', bg: 'bg-purple-100' }
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

    if (error || !data) {
        return (
            <Card className="h-full" hover={false}>
                <div className="flex flex-col items-center justify-center h-full py-4 text-center">
                    <div className="w-12 h-12 rounded-2xl bg-slate-100 flex items-center justify-center border border-slate-200 mb-3">
                        <LuSun className="text-slate-400" size={22} />
                    </div>
                    <p className="text-sm text-slate-500 font-medium mb-1">UV Data Unavailable</p>
                    <p className="text-xs text-slate-400">{error || 'Set your city in profile settings'}</p>
                    <button
                        onClick={fetchUVRisk}
                        className="mt-3 text-xs text-primary-500 hover:text-primary-600 font-semibold"
                    >
                        Retry
                    </button>
                </div>
            </Card>
        )
    }

    const risk = getRiskColors(data.risk_level)

    return (
        <Card className="h-full" hover={false}>
            <div className="flex items-start justify-between w-full">
                <div className="w-full">
                    <div className="text-xs uppercase tracking-widest text-slate-400 font-medium mb-4">
                        UV Risk · {data.city}
                    </div>

                    <div className="flex items-start justify-between w-full mb-5">
                        <div className="flex flex-col">
                            <span className="text-5xl font-bold text-slate-800 tracking-tight leading-none mb-1">
                                {data.uv_index ?? '--'}
                            </span>
                            <span className={`inline-flex self-start px-2 py-0.5 rounded-md text-[10px] font-bold uppercase tracking-wider ${risk.bg} ${risk.color}`}>
                                {data.risk_level}
                            </span>
                        </div>

                        <div className={`w-12 h-12 rounded-2xl ${risk.bg} flex items-center justify-center ${risk.color} flex-shrink-0`}>
                            {data.uv_index > 5 ? <LuShieldAlert size={24} /> : <LuSun size={24} />}
                        </div>
                    </div>

                    {/* SPF + Exposure Limit */}
                    <div className="flex items-center gap-2 text-sm text-slate-600 font-medium bg-slate-50 p-3 rounded-xl mb-3">
                        <LuShield className="w-4 h-4 text-emerald-500 flex-shrink-0" />
                        <span>{data.spf_recommendation}</span>
                    </div>

                    {/* Exposure time */}
                    {data.exposure_limit_minutes && (
                        <div className="flex items-center gap-2 text-xs text-slate-500 bg-slate-50 px-3 py-2 rounded-lg">
                            <LuClock className="w-3.5 h-3.5 text-slate-400 flex-shrink-0" />
                            <span>Max exposure: <strong>{data.exposure_limit_minutes} min</strong></span>
                        </div>
                    )}

                    {/* Fitzpatrick personalization indicator */}
                    {data.fitzpatrick_type && (
                        <div className="mt-3 text-[10px] text-slate-400 font-medium uppercase tracking-wider flex items-center gap-1.5">
                            <LuDroplet size={10} />
                            Personalized for Skin Type {data.fitzpatrick_type}
                        </div>
                    )}
                </div>
            </div>
        </Card>
    )
}
