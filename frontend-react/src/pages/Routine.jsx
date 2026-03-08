import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
    LuCheck, LuTriangleAlert, LuPill, LuClock, LuCalendar,
    LuActivity, LuSun, LuMoon, LuChevronRight, LuFileText,
    LuSparkles, LuRefreshCw, LuWand
} from 'react-icons/lu'
import { api } from '../services/api'
import { Card, CardTitle, CardDescription, IconWrapper, CardBadge } from '../components/Card'

const SKIN_TYPES = ['oily', 'dry', 'combination', 'normal', 'sensitive']
const CONCERNS = ['acne', 'dryness', 'pigmentation', 'redness', 'aging', 'oiliness', 'dark spots', 'uneven texture']
const SENSITIVITIES = ['low', 'medium', 'high']

export default function TreatmentPlan() {
    const [activeTab, setActiveTab] = useState('ai')
    const [plans, setPlans] = useState([])
    const [loading, setLoading] = useState(true)
    const [selectedPlan, setSelectedPlan] = useState(null)
    const [steps, setSteps] = useState([])
    const [adherence, setAdherence] = useState([])
    const [sideEffectModal, setSideEffectModal] = useState({ open: false, stepId: null, stepName: '' })
    const [sideEffectText, setSideEffectText] = useState('')
    const [sideEffectSeverity, setSideEffectSeverity] = useState('mild')
    const [sideEffectFeedback, setSideEffectFeedback] = useState({ type: '', message: '' })

    // AI routine state
    const [aiSteps, setAiSteps] = useState([])
    const [aiLoading, setAiLoading] = useState(false)
    const [aiError, setAiError] = useState('')
    const [generating, setGenerating] = useState(false)
    const [patientProfile, setPatientProfile] = useState(null)
    const [usedProfile, setUsedProfile] = useState(null)  // profile returned by API after generation
    const [profile, setProfile] = useState({ skin_type: 'combination', concerns: [], sensitivity: 'medium', goals: '' })

    useEffect(() => {
        fetchPlans()
        fetchMyRoutine()
        loadPatientProfile()
    }, [])

    const loadPatientProfile = async () => {
        try {
            const data = await api.getProfile()
            setPatientProfile(data)
            // Pre-fill form with real profile
            setProfile(p => ({
                ...p,
                skin_type: data.skin_type?.toLowerCase() || p.skin_type,
                sensitivity: data.sensitivity_level?.toLowerCase() || p.sensitivity,
                goals: data.goals || p.goals,
                concerns: data.acne_prone ? [...p.concerns, 'acne'].filter((v,i,a)=>a.indexOf(v)===i) : p.concerns,
            }))
        } catch {
            // no profile set — form defaults apply
        }
    }

    const fetchMyRoutine = async () => {
        setAiLoading(true)
        try {
            const data = await api.getMyRoutine()
            setAiSteps(data?.steps || [])
        } catch {
            // no saved routine yet — that's fine
        } finally {
            setAiLoading(false)
        }
    }

    const generateRoutine = async () => {
        setGenerating(true)
        setAiError('')
        try {
            const data = await api.generateAiRoutine(profile)
            setAiSteps(data?.steps || [])
            if (data?.skin_profile) setUsedProfile(data.skin_profile)
        } catch (err) {
            setAiError(err?.message || 'AI generation failed. Make sure OPENROUTER_API_KEY is set in your .env.')
        } finally {
            setGenerating(false)
        }
    }

    const toggleConcern = (c) => {
        setProfile(p => ({
            ...p,
            concerns: p.concerns.includes(c) ? p.concerns.filter(x => x !== c) : [...p.concerns, c]
        }))
    }

    const fetchPlans = async () => {
        setLoading(true)
        try {
            const data = await api.getTreatmentPlans()
            setPlans(Array.isArray(data) ? data : [])
            // Auto-select first active plan
            const active = (data || []).find(p => p.status === 'active')
            if (active) selectPlan(active)
        } catch (err) {
            console.error('Failed to fetch plans:', err)
            setPlans([])
        } finally {
            setLoading(false)
        }
    }

    const selectPlan = async (plan) => {
        setSelectedPlan(plan)
        try {
            const stepsData = await api.getTreatmentSteps(plan.plan_id)
            setSteps(stepsData || [])

            const today = new Date().toISOString().split('T')[0]
            const adhData = await api.getTreatmentAdherence(plan.plan_id, today)
            setAdherence(adhData || [])
        } catch (err) {
            console.error('Failed to load plan details:', err)
        }
    }

    const recordAdherence = async (stepId) => {
        if (!selectedPlan) return
        // Skip if already taken today
        if (adherence.some(a => a.step_id === stepId && a.taken)) return

        try {
            await api.recordAdherence(selectedPlan.plan_id, {
                step_id: stepId,
                date: new Date().toISOString().split('T')[0],
                taken: true,
            })
            // Refresh adherence
            const today = new Date().toISOString().split('T')[0]
            const adhData = await api.getTreatmentAdherence(selectedPlan.plan_id, today)
            setAdherence(adhData || [])
        } catch (err) {
            console.error('Failed to record adherence:', err)
        }
    }

    const submitSideEffect = async () => {
        if (!selectedPlan || !sideEffectModal.stepId || !sideEffectText.trim()) return
        setSideEffectFeedback({ type: '', message: '' })
        try {
            await api.reportSideEffect(selectedPlan.plan_id, {
                step_id: sideEffectModal.stepId,
                description: sideEffectText,
                severity: sideEffectSeverity,
            })
            setSideEffectFeedback({ type: 'success', message: 'Side effect reported successfully.' })
            setTimeout(() => {
                setSideEffectModal({ open: false, stepId: null, stepName: '' })
                setSideEffectText('')
                setSideEffectSeverity('mild')
                setSideEffectFeedback({ type: '', message: '' })
            }, 1200)
        } catch (err) {
            console.error('Failed to report side effect:', err)
            setSideEffectFeedback({ type: 'error', message: 'Failed to submit. Please try again.' })
        }
    }

    const isStepTakenToday = (stepId) => {
        return adherence.some(a => a.step_id === stepId && a.taken)
    }

    const adherencePercent = steps.length > 0
        ? Math.round((steps.filter(s => isStepTakenToday(s.step_id)).length / steps.length) * 100)
        : 0

    const getFrequencyLabel = (freq) => {
        switch (freq) {
            case 'twice_daily': return 'Twice daily'
            case 'weekly': return 'Weekly'
            default: return 'Daily'
        }
    }

    const getTimeIcon = (time) => {
        switch (time) {
            case 'AM': return <LuSun size={14} className="text-amber-500" />
            case 'PM': return <LuMoon size={14} className="text-indigo-500" />
            default: return <LuClock size={14} className="text-slate-400" />
        }
    }

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-[60vh]">
                <div className="animate-pulse text-slate-400 flex items-center gap-3">
                    <LuPill size={24} className="animate-spin" />
                    Loading...
                </div>
            </div>
        )
    }

    const amSteps = aiSteps.filter(s => s.time === 'AM').sort((a, b) => a.step - b.step)
    const pmSteps = aiSteps.filter(s => s.time === 'PM').sort((a, b) => a.step - b.step)

    return (
        <div className="relative min-h-screen pb-12">
            <div className="relative z-10 max-w-5xl mx-auto space-y-8">
                {/* Header */}
                {/* Page header */}
                <div>
                    <h1 className="text-4xl md:text-5xl font-display font-bold text-slate-900 tracking-tight">My Routine</h1>
                    <p className="text-slate-500 mt-2 text-lg">AI-generated skincare & doctor treatment plans</p>
                </div>

                {/* Tabs */}
                <div className="flex gap-2 bg-slate-100 p-1 rounded-2xl w-fit">
                    {[
                        { id: 'ai', icon: <LuSparkles size={16} />, label: 'AI Routine' },
                        { id: 'plans', icon: <LuPill size={16} />, label: 'Treatment Plans' },
                    ].map(tab => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            className={`flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold transition-all ${
                                activeTab === tab.id
                                    ? 'bg-white text-slate-900 shadow-sm'
                                    : 'text-slate-500 hover:text-slate-700'
                            }`}
                        >
                            {tab.icon}{tab.label}
                        </button>
                    ))}
                </div>

                {/* ── AI Routine Tab ── */}
                <AnimatePresence mode="wait">
                {activeTab === 'ai' && (
                    <motion.div
                        key="ai-tab"
                        initial={{ opacity: 0, y: 12 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -8 }}
                        className="space-y-8"
                    >
                        {/* Skin profile form */}
                        <Card className="p-7 space-y-6" hover={false}>
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-3">
                                    <div className="w-10 h-10 rounded-xl bg-primary-50 text-primary-500 flex items-center justify-center">
                                        <LuWand size={20} />
                                    </div>
                                    <div>
                                        <CardTitle>Generate My Routine</CardTitle>
                                        <CardDescription>
                                            {patientProfile
                                                ? 'Personalised using your health profile — refine below if needed'
                                                : 'Tell us about your skin — AI does the rest'}
                                        </CardDescription>
                                    </div>
                                </div>
                                {aiSteps.length > 0 && (
                                    <span className="text-xs text-emerald-600 bg-emerald-50 px-3 py-1 rounded-full font-semibold">Routine saved</span>
                                )}
                            </div>

                            {/* Profile data chip — shows what the AI will use */}
                            {patientProfile && (
                                <div className="flex flex-wrap gap-2 p-4 bg-primary-50/60 rounded-2xl border border-primary-100">
                                    <span className="text-xs font-semibold text-primary-600 mr-1 self-center">From your profile:</span>
                                    {patientProfile.location_city && (
                                        <span className="text-xs px-2.5 py-1 bg-white border border-primary-100 rounded-full text-slate-600 font-medium">📍 {patientProfile.location_city}</span>
                                    )}
                                    {patientProfile.skin_type && (
                                        <span className="text-xs px-2.5 py-1 bg-white border border-primary-100 rounded-full text-slate-600 font-medium capitalize">🧴 {patientProfile.skin_type} skin</span>
                                    )}
                                    {patientProfile.fitzpatrick_type && (
                                        <span className="text-xs px-2.5 py-1 bg-white border border-primary-100 rounded-full text-slate-600 font-medium">Fitzpatrick {patientProfile.fitzpatrick_type}</span>
                                    )}
                                    {patientProfile.acne_prone && (
                                        <span className="text-xs px-2.5 py-1 bg-rose-50 border border-rose-100 rounded-full text-rose-600 font-medium">Acne-prone</span>
                                    )}
                                    {patientProfile.allergies && (
                                        <span className="text-xs px-2.5 py-1 bg-amber-50 border border-amber-100 rounded-full text-amber-700 font-medium">⚠ Allergies noted</span>
                                    )}
                                </div>
                            )}

                            {/* Skin type */}
                            <div>
                                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Skin Type</p>
                                <div className="flex flex-wrap gap-2">
                                    {SKIN_TYPES.map(t => (
                                        <button
                                            key={t}
                                            onClick={() => setProfile(p => ({ ...p, skin_type: t }))}
                                            className={`px-4 py-2 rounded-xl text-sm font-semibold capitalize transition-all border ${
                                                profile.skin_type === t
                                                    ? 'bg-primary-50 text-primary-600 border-primary-200'
                                                    : 'bg-white text-slate-500 border-slate-200 hover:border-slate-300'
                                            }`}
                                        >{t}</button>
                                    ))}
                                </div>
                            </div>

                            {/* Concerns */}
                            <div>
                                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Concerns <span className="normal-case font-normal">(pick all that apply)</span></p>
                                <div className="flex flex-wrap gap-2">
                                    {CONCERNS.map(c => (
                                        <button
                                            key={c}
                                            onClick={() => toggleConcern(c)}
                                            className={`px-4 py-2 rounded-xl text-sm font-semibold capitalize transition-all border ${
                                                profile.concerns.includes(c)
                                                    ? 'bg-accent-50 text-accent-600 border-accent-200'
                                                    : 'bg-white text-slate-500 border-slate-200 hover:border-slate-300'
                                            }`}
                                        >{c}</button>
                                    ))}
                                </div>
                            </div>

                            {/* Sensitivity */}
                            <div>
                                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Sensitivity</p>
                                <div className="flex gap-2">
                                    {SENSITIVITIES.map(s => (
                                        <button
                                            key={s}
                                            onClick={() => setProfile(p => ({ ...p, sensitivity: s }))}
                                            className={`flex-1 py-2.5 rounded-xl text-sm font-semibold capitalize transition-all border ${
                                                profile.sensitivity === s
                                                    ? s === 'high' ? 'bg-rose-50 text-rose-600 border-rose-200'
                                                        : s === 'medium' ? 'bg-amber-50 text-amber-600 border-amber-200'
                                                        : 'bg-emerald-50 text-emerald-600 border-emerald-200'
                                                    : 'bg-white text-slate-500 border-slate-200'
                                            }`}
                                        >{s}</button>
                                    ))}
                                </div>
                            </div>

                            {/* Goals (optional) */}
                            <div>
                                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Goal <span className="normal-case font-normal">(optional)</span></p>
                                <input
                                    value={profile.goals}
                                    onChange={e => setProfile(p => ({ ...p, goals: e.target.value }))}
                                    placeholder="e.g. clear glowing skin, reduce hyperpigmentation..."
                                    className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-slate-900 placeholder:text-slate-400 focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20 text-sm"
                                />
                            </div>

                            {aiError && (
                                <p className="text-sm text-rose-500 bg-rose-50 px-4 py-3 rounded-xl">{aiError}</p>
                            )}

                            <motion.button
                                whileHover={{ scale: 1.02 }}
                                whileTap={{ scale: 0.98 }}
                                onClick={generateRoutine}
                                disabled={generating}
                                className="w-full py-4 rounded-2xl bg-gradient-to-r from-primary-500 to-accent-500 text-white font-bold text-base shadow-lg shadow-primary-500/30 disabled:opacity-60 flex items-center justify-center gap-3"
                            >
                                {generating ? (
                                    <><LuRefreshCw size={20} className="animate-spin" /> Generating your routine...</>
                                ) : (
                                    <><LuSparkles size={20} /> {aiSteps.length > 0 ? 'Regenerate Routine' : 'Generate My Routine'}</>
                                )}
                            </motion.button>
                        </Card>

                        {/* Generated routine display */}
                        {aiSteps.length > 0 && (
                            <div className="space-y-6">
                                {usedProfile && (
                                    <div className="text-sm text-slate-500 bg-slate-50 rounded-2xl px-5 py-3.5 border border-slate-200 flex flex-wrap gap-3">
                                        <span className="font-semibold text-slate-700">Routine for {usedProfile.name}</span>
                                        {usedProfile.age && <span>· Age {usedProfile.age}</span>}
                                        {usedProfile.location && <span>· 📍 {usedProfile.location}</span>}
                                        {usedProfile.skin_type && <span>· {usedProfile.skin_type} skin</span>}
                                        {usedProfile.concerns?.length > 0 && <span>· {usedProfile.concerns.join(', ')}</span>}
                                    </div>
                                )}
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                {/* AM routine */}
                                <div className="space-y-3">
                                    <div className="flex items-center gap-2 mb-4">
                                        <div className="w-8 h-8 rounded-xl bg-amber-50 text-amber-500 flex items-center justify-center">
                                            <LuSun size={18} />
                                        </div>
                                        <h3 className="font-bold text-slate-900 text-lg">Morning Routine</h3>
                                        <span className="text-xs text-slate-400">{amSteps.length} steps</span>
                                    </div>
                                    <AnimatePresence>
                                        {amSteps.map((s, i) => (
                                            <motion.div
                                                key={s.item_id || i}
                                                initial={{ opacity: 0, x: -16 }}
                                                animate={{ opacity: 1, x: 0 }}
                                                transition={{ delay: i * 0.07 }}
                                            >
                                                <Card className="p-4" hover={false}>
                                                    <div className="flex items-start gap-3">
                                                        <div className="w-7 h-7 rounded-lg bg-amber-50 text-amber-600 flex items-center justify-center text-xs font-bold flex-shrink-0 mt-0.5">
                                                            {s.step}
                                                        </div>
                                                        <div>
                                                            <p className="font-semibold text-slate-900 text-sm">{s.product}</p>
                                                            {s.brand && <p className="text-xs text-primary-500 font-medium">{s.brand}</p>}
                                                            {s.instructions && <p className="text-xs text-slate-400 mt-1 leading-relaxed">{s.instructions}</p>}
                                                        </div>
                                                    </div>
                                                </Card>
                                            </motion.div>
                                        ))}
                                    </AnimatePresence>
                                </div>

                                {/* PM routine */}
                                <div className="space-y-3">
                                    <div className="flex items-center gap-2 mb-4">
                                        <div className="w-8 h-8 rounded-xl bg-indigo-50 text-indigo-500 flex items-center justify-center">
                                            <LuMoon size={18} />
                                        </div>
                                        <h3 className="font-bold text-slate-900 text-lg">Evening Routine</h3>
                                        <span className="text-xs text-slate-400">{pmSteps.length} steps</span>
                                    </div>
                                    <AnimatePresence>
                                        {pmSteps.map((s, i) => (
                                            <motion.div
                                                key={s.item_id || i}
                                                initial={{ opacity: 0, x: -16 }}
                                                animate={{ opacity: 1, x: 0 }}
                                                transition={{ delay: i * 0.07 }}
                                            >
                                                <Card className="p-4" hover={false}>
                                                    <div className="flex items-start gap-3">
                                                        <div className="w-7 h-7 rounded-lg bg-indigo-50 text-indigo-600 flex items-center justify-center text-xs font-bold flex-shrink-0 mt-0.5">
                                                            {s.step}
                                                        </div>
                                                        <div>
                                                            <p className="font-semibold text-slate-900 text-sm">{s.product}</p>
                                                            {s.brand && <p className="text-xs text-primary-500 font-medium">{s.brand}</p>}
                                                            {s.instructions && <p className="text-xs text-slate-400 mt-1 leading-relaxed">{s.instructions}</p>}
                                                        </div>
                                                    </div>
                                                </Card>
                                            </motion.div>
                                        ))}
                                    </AnimatePresence>
                                </div>
                            </div>
                            </div>
                        )}

                        {aiSteps.length === 0 && !generating && (
                            <div className="text-center py-12 text-slate-400">
                                <LuWand size={40} className="mx-auto mb-3 opacity-30" />
                                <p className="font-medium">No routine yet</p>
                                <p className="text-sm">Fill in your skin profile above and hit Generate.</p>
                            </div>
                        )}
                    </motion.div>
                )}

                {/* ── Treatment Plans Tab ── */}
                {activeTab === 'plans' && (
                    <motion.div
                        key="plans-tab"
                        initial={{ opacity: 0, y: 12 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -8 }}
                        className="space-y-6"
                    >
                        {plans.length === 0 ? (
                            <div className="text-center py-20 bg-slate-50 rounded-3xl border border-slate-200 border-dashed">
                                <LuPill className="mx-auto mb-4 text-slate-300" size={48} />
                                <h3 className="text-xl font-bold text-slate-900 mb-2">No Treatment Plans</h3>
                                <p className="text-slate-400 max-w-md mx-auto">
                                    Your dermatologist will prescribe a treatment plan after your consultation.
                                    Once assigned, your medications and schedule will appear here.
                                </p>
                            </div>
                        ) : (<>
                        {/* Adherence Score */}
                        <Card className="px-6 py-4 flex items-center gap-6 w-fit" hover={false}>
                            <div className="flex items-center gap-3">
                                <IconWrapper variant="primary" size="md" className="bg-emerald-50 text-emerald-500">
                                    <LuActivity size={20} />
                                </IconWrapper>
                                <div>
                                    <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Today</p>
                                    <p className="text-xl font-bold text-emerald-600">{adherencePercent}%</p>
                                </div>
                            </div>
                            <div className="w-px h-10 bg-slate-100" />
                            <div className="flex items-center gap-3">
                                <IconWrapper variant="ai" size="md" className="bg-primary-50 text-primary-500">
                                    <LuPill size={20} />
                                </IconWrapper>
                                <div>
                                    <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Medications</p>
                                    <p className="text-xl font-bold text-slate-900">{steps.length}</p>
                                </div>
                            </div>
                        </Card>

                        {/* Plan selector */}
                        {plans.length > 1 && (
                            <div className="flex gap-3 overflow-x-auto pb-2">
                                {plans.map(plan => (
                                    <motion.button
                                        key={plan.plan_id}
                                        whileHover={{ scale: 1.02 }}
                                        whileTap={{ scale: 0.98 }}
                                        onClick={() => selectPlan(plan)}
                                        className={`px-5 py-3 rounded-2xl text-sm font-semibold whitespace-nowrap transition-all border ${selectedPlan?.plan_id === plan.plan_id
                                            ? 'bg-primary-50 text-primary-600 border-primary-200'
                                            : 'bg-white text-slate-500 border-slate-200 hover:text-slate-900'
                                            }`}
                                    >
                                        {plan.diagnosis}
                                        <CardBadge variant={plan.status === 'active' ? 'success' : 'default'} className="ml-2">{plan.status}</CardBadge>
                                    </motion.button>
                                ))}
                            </div>
                        )}

                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                            {/* Medication list */}
                            <div className="lg:col-span-2 space-y-4">
                                {/* Progress bar */}
                                <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
                                    <motion.div
                                        className="h-full bg-emerald-500 rounded-full"
                                        initial={{ width: 0 }}
                                        animate={{ width: `${adherencePercent}%` }}
                                        transition={{ duration: 0.6, ease: 'easeOut' }}
                                    />
                                </div>

                                <AnimatePresence mode="popLayout">
                                    {steps.map((step, i) => {
                                        const taken = isStepTakenToday(step.step_id)
                                        return (
                                            <motion.div
                                                key={step.step_id}
                                                initial={{ opacity: 0, x: -20 }}
                                                animate={{ opacity: 1, x: 0 }}
                                        exit={{ opacity: 0, scale: 0.9 }}
                                        transition={{ delay: i * 0.06 }}
                                    >
                                        <Card
                                            className={`p-5 group ${taken ? 'bg-emerald-50/50 border-emerald-100' : ''}`}
                                            hover={!taken}
                                        >
                                            <div className="flex items-start gap-4">
                                                {/* Check button */}
                                                <motion.button
                                                    whileHover={{ scale: 1.1 }}
                                                    whileTap={{ scale: 0.9 }}
                                                    onClick={() => recordAdherence(step.step_id)}
                                                    className={`w-11 h-11 rounded-xl flex items-center justify-center flex-shrink-0 transition-all mt-0.5 ${taken
                                                        ? 'bg-emerald-500 text-white shadow-lg shadow-emerald-500/30'
                                                        : 'bg-slate-50 border border-slate-200 text-slate-400 hover:border-emerald-500/50 hover:text-emerald-500'
                                                        }`}
                                                >
                                                    <LuCheck size={20} />
                                                </motion.button>

                                                {/* Medication details */}
                                                <div className="flex-1 min-w-0">
                                                    <div className="flex items-center gap-2 mb-1">
                                                        <h3 className={`font-semibold text-lg ${taken ? 'text-emerald-700 line-through opacity-70' : 'text-slate-900'}`}>
                                                            {step.medication_name}
                                                        </h3>
                                                    </div>

                                                    {step.dosage && (
                                                        <p className="text-sm text-primary-600 font-medium mb-1">{step.dosage}</p>
                                                    )}

                                                    <div className="flex items-center gap-3 text-xs text-slate-500">
                                                        <span className="flex items-center gap-1">
                                                            {getTimeIcon(step.time_of_day)}
                                                            {step.time_of_day === 'BOTH' ? 'Morning & Evening' : step.time_of_day}
                                                        </span>
                                                        <span className="flex items-center gap-1">
                                                            <LuClock size={12} />
                                                            {getFrequencyLabel(step.frequency)}
                                                        </span>
                                                    </div>

                                                    {step.instructions && (
                                                        <p className="text-sm text-slate-400 mt-2 italic">{step.instructions}</p>
                                                    )}
                                                </div>

                                                {/* Side effect button */}
                                                <motion.button
                                                    whileHover={{ scale: 1.05 }}
                                                    onClick={() => setSideEffectModal({ open: true, stepId: step.step_id, stepName: step.medication_name })}
                                                    className="p-2 text-slate-300 hover:text-amber-500 hover:bg-amber-50 rounded-lg opacity-0 group-hover:opacity-100 transition-all"
                                                    title="Report side effect"
                                                >
                                                    <LuTriangleAlert size={18} />
                                                </motion.button>
                                            </div>
                                        </Card>
                                    </motion.div>
                                )
                                    })}
                                </AnimatePresence>

                                {steps.length === 0 && selectedPlan && (
                                    <div className="text-center py-12 text-slate-400">
                                        <LuFileText size={32} className="mx-auto mb-3 opacity-50" />
                                        <p className="font-medium">No medications in this plan yet</p>
                                        <p className="text-sm">Your doctor will add medication steps to this plan.</p>
                                    </div>
                                )}
                            </div>

                            {/* Sidebar: Plan details */}
                            <div className="space-y-6">
                                {selectedPlan && (
                                    <>
                                        <Card className="p-6" hover={false}>
                                            <div className="flex items-center gap-2 mb-4">
                                                <LuFileText className="text-primary-500" size={18} />
                                                <CardTitle>Plan Details</CardTitle>
                                            </div>
                                            <div className="space-y-3">
                                                <div>
                                                    <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Diagnosis</p>
                                                    <p className="text-slate-900 font-medium">{selectedPlan.diagnosis}</p>
                                                </div>
                                                <div>
                                                    <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Status</p>
                                                    <CardBadge variant={selectedPlan.status === 'active' ? 'success' : 'default'}>
                                                        {selectedPlan.status}
                                                    </CardBadge>
                                                </div>
                                                <div>
                                                    <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Prescribed</p>
                                                    <p className="text-slate-600 text-sm">{new Date(selectedPlan.created_at).toLocaleDateString()}</p>
                                                </div>
                                                {selectedPlan.notes && (
                                                    <div>
                                                        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Doctor's Notes</p>
                                                        <p className="text-slate-600 text-sm">{selectedPlan.notes}</p>
                                                    </div>
                                                )}
                                            </div>
                                        </Card>

                                        <Card className="p-6 bg-gradient-to-br from-primary-500 to-primary-600 text-white overflow-hidden relative border-none shadow-lg shadow-primary-500/30">
                                            <div className="relative z-10">
                                                <h3 className="text-lg font-bold mb-2">Treatment Adherence</h3>
                                                <p className="text-white/90 text-sm leading-relaxed">
                                                    Taking medications exactly as prescribed by your dermatologist is crucial for treatment success.
                                                    Report any side effects promptly.
                                                </p>
                                            </div>
                                            <div className="absolute -bottom-6 -right-6 w-32 h-32 bg-white/20 rounded-full blur-2xl" />
                                        </Card>
                                    </>
                                )}
                            </div>
                        </div>
                        </>)}
                    </motion.div>
                )}
                </AnimatePresence>

            {/* Side Effect Reporting Modal */}
            <AnimatePresence>
                {sideEffectModal.open && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm flex items-center justify-center p-4"
                        onClick={() => setSideEffectModal({ open: false, stepId: null, stepName: '' })}
                    >
                        <motion.div
                            initial={{ scale: 0.9, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            exit={{ scale: 0.9, opacity: 0 }}
                            className="bg-white rounded-3xl p-8 max-w-md w-full shadow-2xl"
                            onClick={e => e.stopPropagation()}
                        >
                            <div className="flex items-center gap-3 mb-6">
                                <div className="w-12 h-12 rounded-2xl bg-amber-50 text-amber-500 flex items-center justify-center">
                                    <LuTriangleAlert size={24} />
                                </div>
                                <div>
                                    <h3 className="text-xl font-bold text-slate-900">Report Side Effect</h3>
                                    <p className="text-sm text-slate-500">{sideEffectModal.stepName}</p>
                                </div>
                            </div>

                            <div className="space-y-4">
                                <div>
                                    <label className="block text-sm font-medium text-slate-700 mb-2">Severity</label>
                                    <div className="flex gap-2">
                                        {['mild', 'moderate', 'severe'].map(sev => (
                                            <button
                                                key={sev}
                                                onClick={() => setSideEffectSeverity(sev)}
                                                className={`flex-1 py-2.5 rounded-xl text-sm font-semibold capitalize border transition-all ${sideEffectSeverity === sev
                                                    ? sev === 'severe' ? 'bg-rose-50 text-rose-600 border-rose-200'
                                                        : sev === 'moderate' ? 'bg-amber-50 text-amber-600 border-amber-200'
                                                            : 'bg-emerald-50 text-emerald-600 border-emerald-200'
                                                    : 'bg-white text-slate-500 border-slate-200'
                                                    }`}
                                            >
                                                {sev}
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                <div>
                                    <label className="block text-sm font-medium text-slate-700 mb-2">Description</label>
                                    <textarea
                                        value={sideEffectText}
                                        onChange={e => setSideEffectText(e.target.value)}
                                        placeholder="Describe what you experienced (e.g., mild redness, burning sensation)..."
                                        className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-slate-900 placeholder:text-slate-400 focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20 text-sm resize-none"
                                        rows={3}
                                    />
                                </div>

                                <div className="flex gap-3 pt-2">
                                    <button
                                        onClick={() => { setSideEffectModal({ open: false, stepId: null, stepName: '' }); setSideEffectFeedback({ type: '', message: '' }); }}
                                        className="flex-1 py-3 rounded-xl text-sm font-semibold text-slate-500 bg-slate-50 hover:bg-slate-100 transition-colors"
                                    >
                                        Cancel
                                    </button>
                                    <button
                                        onClick={submitSideEffect}
                                        disabled={!sideEffectText.trim()}
                                        className="flex-1 py-3 rounded-xl text-sm font-semibold text-white bg-amber-500 hover:bg-amber-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-lg shadow-amber-500/30"
                                    >
                                        Submit Report
                                    </button>
                                </div>

                                {sideEffectFeedback.message && (
                                    <p role="alert" className={`text-sm text-center mt-2 font-medium ${sideEffectFeedback.type === 'success' ? 'text-emerald-600' : 'text-red-500'}`}>
                                        {sideEffectFeedback.message}
                                    </p>
                                )}
                            </div>

                            <p className="text-xs text-slate-400 mt-4 text-center">
                                Your doctor will be notified of this report.
                            </p>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>
            </div>
        </div>
    )
}
