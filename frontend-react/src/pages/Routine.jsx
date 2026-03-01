import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
    LuCheck, LuTriangleAlert, LuPill, LuClock, LuCalendar,
    LuActivity, LuSun, LuMoon, LuChevronRight, LuFileText
} from 'react-icons/lu'
import { api } from '../services/api'
import { Card, CardTitle, CardDescription, IconWrapper, CardBadge } from '../components/Card'

export default function TreatmentPlan() {
    const [plans, setPlans] = useState([])
    const [loading, setLoading] = useState(true)
    const [selectedPlan, setSelectedPlan] = useState(null)
    const [steps, setSteps] = useState([])
    const [adherence, setAdherence] = useState([])
    const [sideEffectModal, setSideEffectModal] = useState({ open: false, stepId: null, stepName: '' })
    const [sideEffectText, setSideEffectText] = useState('')
    const [sideEffectSeverity, setSideEffectSeverity] = useState('mild')
    const [sideEffectFeedback, setSideEffectFeedback] = useState({ type: '', message: '' })

    useEffect(() => { fetchPlans() }, [])

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
                    Loading treatment plans...
                </div>
            </div>
        )
    }

    // Empty state: no plans from any doctor
    if (plans.length === 0) {
        return (
            <div className="relative min-h-screen pb-12">
                <div className="relative z-10 max-w-4xl mx-auto">
                    <h1 className="text-4xl md:text-5xl font-bold text-slate-900 tracking-tight mb-2">Treatment Plan</h1>
                    <p className="text-slate-500 text-lg mb-12">Doctor-prescribed medication tracking</p>

                    <div className="text-center py-20 bg-slate-50 rounded-3xl border border-slate-200 border-dashed">
                        <LuPill className="mx-auto mb-4 text-slate-300" size={48} />
                        <h3 className="text-xl font-bold text-slate-900 mb-2">No Treatment Plans</h3>
                        <p className="text-slate-400 max-w-md mx-auto">
                            Your dermatologist will prescribe a treatment plan after your consultation.
                            Once assigned, your medications and schedule will appear here.
                        </p>
                    </div>
                </div>
            </div>
        )
    }

    return (
        <div className="relative min-h-screen pb-12">
            <div className="relative z-10 max-w-5xl mx-auto space-y-8">
                {/* Header */}
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
                    <div>
                        <h1 className="text-4xl md:text-5xl font-bold text-slate-900 tracking-tight">Treatment Plan</h1>
                        <p className="text-slate-500 mt-2 text-lg">Track your prescribed medications</p>
                    </div>

                    {/* Adherence Score */}
                    <Card className="px-6 py-4 flex items-center gap-6" hover={false}>
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
                </div>

                {/* Plan selector (if multiple) */}
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
            </div>

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
    )
}
