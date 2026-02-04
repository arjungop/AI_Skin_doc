import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { api } from '../services/api'
import { useNavigate } from 'react-router-dom'
import { FaArrowRight, FaCheck, FaMars, FaVenus, FaSun, FaLeaf, FaShieldAlt } from 'react-icons/fa'

const STEPS = [
    { id: 'profile', title: 'Basic Profile', desc: 'Tell us a bit about yourself.' },
    { id: 'skin', title: 'Skin Analysis', desc: 'Understand your skin type & biology.' },
    { id: 'concerns', title: 'Key Concerns', desc: 'What do you want to improve?' },
    { id: 'lifestyle', title: 'Lifestyle', desc: 'Daily habits affecting your skin.' },
    { id: 'safety', title: 'Medical Safety', desc: 'Important history for AI safety.' },
    { id: 'location', title: 'Environment', desc: 'Analyzing local UV & Humidity.' }
]

export default function Onboarding() {
    const navigate = useNavigate()
    const [step, setStep] = useState(0)
    const [data, setData] = useState({
        age_range: '',
        gender: '',
        skin_type: '',
        sensitivity: '',
        concerns: [],
        sun_exposure: '',
        sunscreen: '',
        smoke: false,
        cancer_history: false,
        mole_check: false,
        location_city: ''
    })
    const [loading, setLoading] = useState(false)

    const handleNext = () => {
        if (step < STEPS.length - 1) setStep(step + 1)
        else submit()
    }

    const submit = async () => {
        setLoading(true)
        try {
            // Map data to backend schema
            const payload = {
                skin_type: data.skin_type,
                sensitivity_level: data.sensitivity,
                // Using concerns as goals array
                goals: JSON.stringify(data.concerns),
                // Simple logic for fitzpatrick estimate (placeholder) if needed, or leave null
                location_city: data.location_city,
                // Store lifestyle in a structured way if schema expands, currently using goals/attributes
                acne_prone: data.concerns.includes('Acne'),
                allergies: JSON.stringify({
                    sun_exposure: data.sun_exposure,
                    sunscreen: data.sunscreen,
                    cancer_history: data.cancer_history
                }) // Using allergies field for misc context for now until schema migration
            }

            await api.updateProfile(payload)
            // Force reload or redirect
            window.location.href = '/dashboard'
        } catch (err) {
            console.error(err)
            setLoading(false)
        }
    }

    const update = (key, val) => setData(prev => ({ ...prev, [key]: val }))
    const toggleConcern = (c) => {
        const list = data.concerns.includes(c)
            ? data.concerns.filter(x => x !== c)
            : [...data.concerns, c]
        update('concerns', list)
    }

    // Auto-fetch location on step 5 (Location)
    useEffect(() => {
        if (STEPS[step].id === 'location') {
            if ("geolocation" in navigator) {
                navigator.geolocation.getCurrentPosition(async (pos) => {
                    try {
                        // Reverse geocode stub or simple coordinate usage
                        // Ideally call backend to resolve, here just mocking or using coordinates
                        // For now, we'll try to fetch city from a public API or just auto-submit
                        const res = await fetch(`https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${pos.coords.latitude}&longitude=${pos.coords.longitude}&localityLanguage=en`)
                        const geo = await res.json()
                        update('location_city', geo.city || geo.locality || 'Unknown')
                    } catch {
                        update('location_city', 'Unknown')
                    }
                }, () => update('location_city', 'Denied'))
            }
        }
    }, [step])

    return (
        <div className="min-h-screen bg-[#f8fafc] flex items-center justify-center p-6">
            <div className="max-w-2xl w-full">
                {/* Progress */}
                <div className="mb-8 pl-1">
                    <div className="flex gap-2 mb-2">
                        {STEPS.map((s, i) => (
                            <div key={s.id} className={`h-1.5 flex-1 rounded-full transition-all duration-500 ${i <= step ? 'bg-slate-900' : 'bg-slate-200'}`} />
                        ))}
                    </div>
                    <motion.div
                        key={step}
                        initial={{ opacity: 0, y: 5 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="flex justify-between items-end"
                    >
                        <div>
                            <h2 className="text-2xl font-serif text-slate-900">{STEPS[step].title}</h2>
                            <p className="text-slate-500">{STEPS[step].desc}</p>
                        </div>
                        <span className="text-xs font-bold text-slate-300">STEP {step + 1}/{STEPS.length}</span>
                    </motion.div>
                </div>

                {/* Card */}
                <div className="bg-white rounded-3xl shadow-xl shadow-slate-200/50 border border-slate-100 p-8 min-h-[400px] relative overflow-hidden flex flex-col">
                    <div className="flex-1">
                        <AnimatePresence mode="wait">
                            <motion.div
                                key={step}
                                initial={{ opacity: 0, x: 20 }}
                                animate={{ opacity: 1, x: 0 }}
                                exit={{ opacity: 0, x: -20 }}
                                transition={{ duration: 0.3 }}
                                className="h-full"
                            >
                                {/* Step Content Swticher */}
                                {step === 0 && (
                                    <div className="space-y-6">
                                        <div>
                                            <label className="label">Age Range</label>
                                            <div className="grid grid-cols-2 gap-3 mt-2">
                                                {['Under 18', '18–25', '26–35', '36–50', '50+'].map(opt => (
                                                    <SelectBtn key={opt} active={data.age_range === opt} onClick={() => update('age_range', opt)}>{opt}</SelectBtn>
                                                ))}
                                            </div>
                                        </div>
                                        <div>
                                            <label className="label">Gender (Optional)</label>
                                            <div className="flex gap-4 mt-2">
                                                <SelectBtn active={data.gender === 'Female'} onClick={() => update('gender', 'Female')}><FaVenus /> Female</SelectBtn>
                                                <SelectBtn active={data.gender === 'Male'} onClick={() => update('gender', 'Male')}><FaMars /> Male</SelectBtn>
                                                <SelectBtn active={data.gender === 'Other'} onClick={() => update('gender', 'Other')}>Other</SelectBtn>
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {step === 1 && (
                                    <div className="space-y-6">
                                        <div>
                                            <label className="label">How does your skin feel usually?</label>
                                            <div className="grid grid-cols-2 gap-3 mt-2">
                                                {['Oily', 'Dry', 'Combination', 'Normal'].map(t => (
                                                    <SelectBtn key={t} active={data.skin_type === t} onClick={() => update('skin_type', t)}>{t}</SelectBtn>
                                                ))}
                                            </div>
                                        </div>
                                        <div>
                                            <label className="label">Sensitivity Level</label>
                                            <div className="flex flex-col gap-2 mt-2">
                                                <SelectBtn active={data.sensitivity === 'High'} onClick={() => update('sensitivity', 'High')}>
                                                    🔴 <b>High</b> (Reacts easily, often red)
                                                </SelectBtn>
                                                <SelectBtn active={data.sensitivity === 'Medium'} onClick={() => update('sensitivity', 'Medium')}>
                                                    🟡 <b>Medium</b> (Sometimes reacts)
                                                </SelectBtn>
                                                <SelectBtn active={data.sensitivity === 'Low'} onClick={() => update('sensitivity', 'Low')}>
                                                    🟢 <b>Low</b> (Resilient)
                                                </SelectBtn>
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {step === 2 && (
                                    <div>
                                        <label className="label mb-4 block">Select your primary concerns</label>
                                        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                                            {['Acne', 'Pigmentation', 'Wrinkles', 'Redness', 'Uneven Tone', 'Dark Circles', 'Pores', 'Dryness'].map(c => (
                                                <button
                                                    key={c}
                                                    onClick={() => toggleConcern(c)}
                                                    className={`p-4 rounded-xl text-left transition-all border ${data.concerns.includes(c) ? 'bg-slate-900 text-white border-slate-900 shadow-lg' : 'bg-slate-50 text-slate-600 border-transparent hover:bg-slate-100'}`}
                                                >
                                                    <div className="flex justify-between items-center">
                                                        <span className="font-medium">{c}</span>
                                                        {data.concerns.includes(c) && <FaCheck className="text-emerald-400" />}
                                                    </div>
                                                </button>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {step === 3 && (
                                    <div className="space-y-6">
                                        <div>
                                            <label className="label">Daily Sun Exposure</label>
                                            <div className="grid grid-cols-3 gap-3 mt-2">
                                                {['Low (Indoor)', 'Moderate', 'High (Outdoor)'].map(o => (
                                                    <SelectBtn key={o} active={data.sun_exposure === o} onClick={() => update('sun_exposure', o)}>{o}</SelectBtn>
                                                ))}
                                            </div>
                                        </div>
                                        <div>
                                            <label className="label">Sunscreen Usage</label>
                                            <div className="grid grid-cols-3 gap-3 mt-2">
                                                {['Daily', 'Sometimes', 'Never'].map(o => (
                                                    <SelectBtn key={o} active={data.sunscreen === o} onClick={() => update('sunscreen', o)}>{o}</SelectBtn>
                                                ))}
                                            </div>
                                        </div>
                                        <label className="flex items-center gap-3 p-4 bg-slate-50 rounded-xl cursor-pointer">
                                            <input type="checkbox" className="w-5 h-5 accent-slate-900" checked={data.smoke} onChange={e => update('smoke', e.target.checked)} />
                                            <span className="font-medium text-slate-700">I smoke regularly</span>
                                        </label>
                                    </div>
                                )}

                                {step === 4 && (
                                    <div className="space-y-6">
                                        <div className="p-4 bg-amber-50 border border-amber-100 rounded-xl flex gap-3 text-amber-800">
                                            <FaShieldAlt className="mt-1 flex-shrink-0" />
                                            <p className="text-sm">This helps our AI prioritize specific screenings during automated scans.</p>
                                        </div>
                                        <label className="flex items-center gap-3 p-4 bg-white border border-slate-200 rounded-xl cursor-pointer">
                                            <input type="checkbox" className="w-5 h-5 accent-slate-900" checked={data.cancer_history} onChange={e => update('cancer_history', e.target.checked)} />
                                            <div>
                                                <div className="font-medium text-slate-900">Family History of Skin Cancer</div>
                                                <div className="text-xs text-slate-500">Immediate family members only</div>
                                            </div>
                                        </label>
                                        <label className="flex items-center gap-3 p-4 bg-white border border-slate-200 rounded-xl cursor-pointer">
                                            <input type="checkbox" className="w-5 h-5 accent-slate-900" checked={data.mole_check} onChange={e => update('mole_check', e.target.checked)} />
                                            <div>
                                                <div className="font-medium text-slate-900">Active Monitoring</div>
                                                <div className="text-xs text-slate-500">I have existing moles I'm watching</div>
                                            </div>
                                        </label>
                                    </div>
                                )}

                                {step === 5 && (
                                    <div className="text-center py-10">
                                        <div className="w-20 h-20 bg-blue-50 rounded-full flex items-center justify-center mx-auto mb-6 text-blue-500 animate-pulse">
                                            <FaLeaf size={32} />
                                        </div>
                                        <h3 className="text-xl font-bold mb-2">Analyzing Local Environment...</h3>
                                        <p className="text-slate-500 mb-8">
                                            {data.location_city ? `Detected: ${data.location_city}` : 'Requesting location permission to fetch UV & Humidity data...'}
                                        </p>
                                        {!data.location_city && (
                                            <button className="text-sm text-slate-400 hover:text-slate-600 underline" onClick={() => update('location_city', 'Skipped')}>Skip Location</button>
                                        )}
                                    </div>
                                )}

                            </motion.div>
                        </AnimatePresence>
                    </div>

                    {/* Footer Controls */}
                    <div className="mt-8 pt-6 border-t border-slate-100 flex justify-end">
                        <button
                            onClick={handleNext}
                            disabled={loading}
                            className="bg-slate-900 text-white px-8 py-3 rounded-xl font-medium shadow-lg shadow-slate-900/20 hover:scale-105 active:scale-95 transition-all flex items-center gap-2"
                        >
                            {loading ? 'Finalizing...' : (step === STEPS.length - 1 ? 'Finish' : 'Next Step')}
                            {!loading && <FaArrowRight />}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    )
}

function SelectBtn({ active, children, onClick }) {
    return (
        <button
            onClick={onClick}
            className={`px-4 py-3 rounded-xl font-medium text-sm transition-all border flex items-center justify-center gap-2
        ${active
                    ? 'bg-slate-900 text-white border-slate-900 shadow-md ring-2 ring-slate-200'
                    : 'bg-white text-slate-600 border-slate-200 hover:border-slate-300 hover:bg-slate-50'
                }`}
        >
            {children}
        </button>
    )
}
