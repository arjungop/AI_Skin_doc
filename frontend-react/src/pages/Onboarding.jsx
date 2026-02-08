import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { api } from '../services/api'
import { useNavigate } from 'react-router-dom'
import { FaArrowRight, FaCheck, FaMars, FaVenus, FaLeaf, FaShieldAlt } from 'react-icons/fa'
import { LuDroplets, LuSun, LuWind } from 'react-icons/lu'
import { Card } from '../components/Card'

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
        age_range: '', gender: '', skin_type: '', sensitivity: '',
        concerns: [], sun_exposure: '', sunscreen: '',
        smoke: false, cancer_history: false, mole_check: false, location_city: ''
    })
    const [loading, setLoading] = useState(false)

    const handleNext = () => {
        if (step < STEPS.length - 1) setStep(step + 1)
        else submit()
    }

    const submit = async () => {
        setLoading(true)
        try {
            const payload = {
                skin_type: data.skin_type,
                sensitivity_level: data.sensitivity,
                goals: JSON.stringify(data.concerns),
                location_city: data.location_city,
                acne_prone: data.concerns.includes('Acne'),
                allergies: JSON.stringify({
                    sun_exposure: data.sun_exposure,
                    sunscreen: data.sunscreen,
                    cancer_history: data.cancer_history
                })
            }
            await api.updateProfile(payload)
            window.location.href = '/dashboard'
        } catch (err) {
            console.error(err); setLoading(false)
        }
    }

    const update = (key, val) => setData(prev => ({ ...prev, [key]: val }))
    const toggleConcern = (c) => {
        const list = data.concerns.includes(c) ? data.concerns.filter(x => x !== c) : [...data.concerns, c]
        update('concerns', list)
    }

    useEffect(() => {
        if (STEPS[step].id === 'location' && "geolocation" in navigator) {
            navigator.geolocation.getCurrentPosition(async (pos) => {
                try {
                    const res = await fetch(`https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${pos.coords.latitude}&longitude=${pos.coords.longitude}&localityLanguage=en`)
                    const geo = await res.json()
                    update('location_city', geo.city || geo.locality || 'Unknown')
                } catch { update('location_city', 'Unknown') }
            }, () => update('location_city', 'Denied'))
        }
    }, [step])

    return (
        <div className="min-h-screen relative flex items-center justify-center p-6 overflow-hidden">
            {/* Background Ambience */}
            <div className="fixed top-0 left-0 w-full h-full overflow-hidden pointer-events-none -z-10 bg-background">
                <div className="absolute top-1/4 left-1/4 w-[600px] h-[600px] bg-primary-500/10 rounded-full blur-[120px]" />
                <div className="absolute bottom-1/4 right-1/4 w-[500px] h-[500px] bg-accent-500/10 rounded-full blur-[150px]" />
            </div>

            <div className="max-w-2xl w-full relative z-10">
                {/* Progress */}
                <div className="mb-8 px-1">
                    <div className="flex gap-2 mb-4">
                        {STEPS.map((s, i) => (
                            <div key={s.id} className={`h-1.5 flex-1 rounded-full transition-all duration-500 ${i <= step ? 'bg-gradient-to-r from-primary-500 to-accent-500' : 'bg-white/10'}`} />
                        ))}
                    </div>
                    <motion.div
                        key={step}
                        initial={{ opacity: 0, y: 5 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="flex justify-between items-end"
                    >
                        <div>
                            <h2 className="text-3xl font-bold text-text-primary tracking-tight">{STEPS[step].title}</h2>
                            <p className="text-text-secondary mt-1">{STEPS[step].desc}</p>
                        </div>
                        <span className="text-xs font-bold text-text-tertiary bg-white/5 px-2 py-1 rounded-md border border-white/5">
                            STEP {step + 1}/{STEPS.length}
                        </span>
                    </motion.div>
                </div>

                {/* Card */}
                <Card variant="glass" className="min-h-[450px] relative overflow-hidden flex flex-col p-8 md:p-10 shadow-2xl shadow-primary-500/5">
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
                                {step === 0 && (
                                    <div className="space-y-8">
                                        <div>
                                            <label className="text-sm font-medium text-text-secondary mb-3 block">Age Range</label>
                                            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                                                {['Under 18', '18–25', '26–35', '36–50', '50+'].map(opt => (
                                                    <SelectBtn key={opt} active={data.age_range === opt} onClick={() => update('age_range', opt)}>{opt}</SelectBtn>
                                                ))}
                                            </div>
                                        </div>
                                        <div>
                                            <label className="text-sm font-medium text-text-secondary mb-3 block">Gender (Optional)</label>
                                            <div className="flex gap-4">
                                                <SelectBtn active={data.gender === 'Female'} onClick={() => update('gender', 'Female')}><FaVenus /> Female</SelectBtn>
                                                <SelectBtn active={data.gender === 'Male'} onClick={() => update('gender', 'Male')}><FaMars /> Male</SelectBtn>
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {step === 1 && (
                                    <div className="space-y-8">
                                        <div>
                                            <label className="text-sm font-medium text-text-secondary mb-3 block">How does your skin feel usually?</label>
                                            <div className="grid grid-cols-2 gap-3">
                                                {['Oily', 'Dry', 'Combination', 'Normal'].map(t => (
                                                    <SelectBtn key={t} active={data.skin_type === t} onClick={() => update('skin_type', t)}>{t}</SelectBtn>
                                                ))}
                                            </div>
                                        </div>
                                        <div>
                                            <label className="text-sm font-medium text-text-secondary mb-3 block">Sensitivity Level</label>
                                            <div className="flex flex-col gap-3">
                                                <SelectBtn active={data.sensitivity === 'High'} onClick={() => update('sensitivity', 'High')}>
                                                    <div className="flex items-center justify-between w-full">
                                                        <span>🔴 <b>High</b> (Reacts easily, often red)</span>
                                                    </div>
                                                </SelectBtn>
                                                <SelectBtn active={data.sensitivity === 'Medium'} onClick={() => update('sensitivity', 'Medium')}>
                                                    <div className="flex items-center justify-between w-full">
                                                        <span>🟡 <b>Medium</b> (Sometimes reacts)</span>
                                                    </div>
                                                </SelectBtn>
                                                <SelectBtn active={data.sensitivity === 'Low'} onClick={() => update('sensitivity', 'Low')}>
                                                    <div className="flex items-center justify-between w-full">
                                                        <span>🟢 <b>Low</b> (Resilient)</span>
                                                    </div>
                                                </SelectBtn>
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {step === 2 && (
                                    <div>
                                        <label className="text-sm font-medium text-text-secondary mb-4 block">Select your primary concerns</label>
                                        <div className="grid grid-cols-2 gap-3">
                                            {['Acne', 'Pigmentation', 'Wrinkles', 'Redness', 'Uneven Tone', 'Dark Circles', 'Pores', 'Dryness'].map(c => (
                                                <button
                                                    key={c}
                                                    onClick={() => toggleConcern(c)}
                                                    className={`p-4 rounded-xl text-left transition-all border ${data.concerns.includes(c)
                                                            ? 'bg-primary-500/20 text-text-primary border-primary-500/50 shadow-[0_0_15px_rgba(var(--color-primary-500),0.2)]'
                                                            : 'bg-white/5 text-text-secondary border-white/5 hover:bg-white/10 hover:border-white/10'
                                                        }`}
                                                >
                                                    <div className="flex justify-between items-center">
                                                        <span className="font-medium">{c}</span>
                                                        {data.concerns.includes(c) && <FaCheck className="text-primary-400" />}
                                                    </div>
                                                </button>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {step === 3 && (
                                    <div className="space-y-8">
                                        <div>
                                            <label className="text-sm font-medium text-text-secondary mb-3 block">Daily Sun Exposure</label>
                                            <div className="grid grid-cols-3 gap-3">
                                                {['Low', 'Moderate', 'High'].map(o => (
                                                    <SelectBtn key={o} active={data.sun_exposure === o} onClick={() => update('sun_exposure', o)}>{o}</SelectBtn>
                                                ))}
                                            </div>
                                        </div>
                                        <div>
                                            <label className="text-sm font-medium text-text-secondary mb-3 block">Sunscreen Usage</label>
                                            <div className="grid grid-cols-3 gap-3">
                                                {['Daily', 'Sometimes', 'Never'].map(o => (
                                                    <SelectBtn key={o} active={data.sunscreen === o} onClick={() => update('sunscreen', o)}>{o}</SelectBtn>
                                                ))}
                                            </div>
                                        </div>
                                        <label className="flex items-center gap-4 p-4 bg-white/5 border border-white/5 rounded-xl cursor-pointer hover:bg-white/10 transition-colors">
                                            <div className={`w-5 h-5 rounded border flex items-center justify-center transition-colors ${data.smoke ? 'bg-primary-500 border-primary-500' : 'border-text-muted bg-transparent'}`}>
                                                {data.smoke && <FaCheck className="text-white text-xs" />}
                                            </div>
                                            <input type="checkbox" className="hidden" checked={data.smoke} onChange={e => update('smoke', e.target.checked)} />
                                            <span className="font-medium text-text-primary">I smoke regularly</span>
                                        </label>
                                    </div>
                                )}

                                {step === 4 && (
                                    <div className="space-y-6">
                                        <div className="p-5 bg-warning/10 border border-warning/20 rounded-xl flex gap-4 text-warning">
                                            <FaShieldAlt className="mt-1 flex-shrink-0 text-xl" />
                                            <p className="text-sm leading-relaxed">This information helps our AI prioritize specific screenings during automated scans. Your privacy is paramount.</p>
                                        </div>
                                        <label className="flex items-center gap-4 p-4 bg-white/5 border border-white/5 rounded-xl cursor-pointer hover:bg-white/10 transition-colors group">
                                            <div className={`w-6 h-6 rounded border flex items-center justify-center transition-colors ${data.cancer_history ? 'bg-primary-500 border-primary-500' : 'border-text-muted bg-transparent'}`}>
                                                {data.cancer_history && <FaCheck className="text-white text-xs" />}
                                            </div>
                                            <input type="checkbox" className="hidden" checked={data.cancer_history} onChange={e => update('cancer_history', e.target.checked)} />
                                            <div>
                                                <div className="font-medium text-text-primary group-hover:text-primary-400 transition-colors">Family History of Skin Cancer</div>
                                                <div className="text-xs text-text-muted">Immediate family members only</div>
                                            </div>
                                        </label>
                                    </div>
                                )}

                                {step === 5 && (
                                    <div className="text-center py-12">
                                        <div className="relative w-24 h-24 mx-auto mb-8">
                                            <div className="absolute inset-0 bg-primary-500/20 rounded-full blur-xl animate-pulse" />
                                            <div className="relative w-full h-full bg-surface-elevated rounded-full border border-white/10 flex items-center justify-center text-primary-400">
                                                <LuSun size={40} className="animate-spin-slow" />
                                            </div>
                                        </div>
                                        <h3 className="text-2xl font-bold mb-3 text-text-primary">Analyzing Environment...</h3>
                                        <p className="text-text-secondary mb-8">
                                            {data.location_city ? <span className="text-success flex items-center justify-center gap-2"><FaCheck /> Detected: {data.location_city}</span> : 'Requesting location permission...'}
                                        </p>
                                        {!data.location_city && (
                                            <button className="text-sm text-text-muted hover:text-text-primary underline" onClick={() => update('location_city', 'Skipped')}>Skip Location</button>
                                        )}
                                    </div>
                                )}
                            </motion.div>
                        </AnimatePresence>
                    </div>

                    {/* Footer Controls */}
                    <div className="mt-10 pt-6 border-t border-white/10 flex justify-end">
                        <button
                            onClick={handleNext}
                            disabled={loading}
                            className="btn-primary px-8 py-3 shadow-lg shadow-primary-500/20 flex items-center gap-2"
                        >
                            {loading ? 'Finalizing...' : (step === STEPS.length - 1 ? 'Finish' : 'Next Step')}
                            {!loading && <FaArrowRight />}
                        </button>
                    </div>
                </Card>
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
                    ? 'bg-primary-500/20 text-text-primary border-primary-500/50 shadow-[0_0_15px_rgba(var(--color-primary-500),0.2)]'
                    : 'bg-white/5 text-text-secondary border-white/5 hover:border-white/10 hover:bg-white/10'
                }`}
        >
            {children}
        </button>
    )
}
