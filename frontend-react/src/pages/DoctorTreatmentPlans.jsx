import React, { useState, useEffect } from 'react'
import { api } from '../services/api.js'
import { motion, AnimatePresence } from 'framer-motion'
import {
    LuPlus, LuPill, LuCheck, LuActivity, LuTriangleAlert,
    LuChevronRight, LuClock, LuSun, LuMoon, LuX, LuFileText, LuUsers
} from 'react-icons/lu'
import { Card, CardTitle, CardBadge, IconWrapper } from '../components/Card'

export default function DoctorTreatmentPlans() {
    const [patients, setPatients] = useState([])
    const [selectedPatient, setSelectedPatient] = useState(null)
    const [plans, setPlans] = useState([])
    const [selectedPlan, setSelectedPlan] = useState(null)
    const [steps, setSteps] = useState([])
    const [adherence, setAdherence] = useState([])
    const [loading, setLoading] = useState(true)

    // Create Plan form
    const [showCreatePlan, setShowCreatePlan] = useState(false)
    const [newDiagnosis, setNewDiagnosis] = useState('')
    const [newPlanNotes, setNewPlanNotes] = useState('')

    // Add Step form
    const [showAddStep, setShowAddStep] = useState(false)
    const [stepForm, setStepForm] = useState({
        medication_name: '', dosage: '', frequency: 'daily',
        time_of_day: 'PM', instructions: '', step_order: 1
    })

    // Search
    const [patientSearch, setPatientSearch] = useState('')

    useEffect(() => { fetchPatients() }, [])

    const fetchPatients = async () => {
        setLoading(true)
        try {
            const pts = await api.getDoctorPatients()
            setPatients(pts || [])
        } catch (err) {
            console.error(err)
        } finally {
            setLoading(false)
        }
    }

    const selectPatient = async (patient) => {
        setSelectedPatient(patient)
        setSelectedPlan(null)
        setSteps([])
        setAdherence([])
        // Doctors see all plans — we fetch from the patient's perspective via the patient_id
        try {
            // Use getTreatmentPlans — backend filters by logged-in user, but for doctor we need all plans
            // We'll attempt to get plans for this patient
            const allPlans = await api.getTreatmentPlans()
            // Filter client-side for this patient (backend returns plans where doctor_id = current doc)
            const patientPlans = (allPlans || []).filter(p => p.patient_id === patient.patient_id)
            setPlans(patientPlans)
            if (patientPlans.length > 0) selectPlan(patientPlans[0])
        } catch (err) {
            console.error(err)
            setPlans([])
        }
    }

    const selectPlan = async (plan) => {
        setSelectedPlan(plan)
        try {
            const s = await api.getTreatmentSteps(plan.plan_id)
            setSteps(s || [])
            const a = await api.getTreatmentAdherence(plan.plan_id)
            setAdherence(a || [])
        } catch (err) {
            console.error(err)
        }
    }

    const createPlan = async (e) => {
        e.preventDefault()
        if (!newDiagnosis.trim() || !selectedPatient) return
        try {
            const plan = await api.createTreatmentPlan({
                patient_id: selectedPatient.patient_id,
                diagnosis: newDiagnosis,
                notes: newPlanNotes || null,
            })
            setPlans(prev => [plan, ...prev])
            selectPlan(plan)
            setNewDiagnosis('')
            setNewPlanNotes('')
            setShowCreatePlan(false)
        } catch (err) {
            alert('Failed to create plan: ' + (err.message || err))
        }
    }

    const addStep = async (e) => {
        e.preventDefault()
        if (!stepForm.medication_name.trim() || !selectedPlan) return
        try {
            const step = await api.addTreatmentStep(selectedPlan.plan_id, stepForm)
            setSteps(prev => [...prev, step])
            setStepForm({ medication_name: '', dosage: '', frequency: 'daily', time_of_day: 'PM', instructions: '', step_order: steps.length + 2 })
            setShowAddStep(false)
        } catch (err) {
            alert('Failed to add step: ' + (err.message || err))
        }
    }

    // Adherence stats
    const getAdherenceRate = (stepId) => {
        const records = adherence.filter(a => a.step_id === stepId)
        if (records.length === 0) return null
        const taken = records.filter(a => a.taken).length
        return Math.round((taken / records.length) * 100)
    }

    const getSideEffects = () => {
        return adherence.filter(a => a.side_effects && a.side_effects.trim())
    }

    const filteredPatients = patientSearch
        ? patients.filter(p => {
            const q = patientSearch.toLowerCase()
            return String(p.patient_id).includes(q) || (p.name && p.name.toLowerCase().includes(q))
        })
        : patients

    const getTimeIcon = (time) => {
        if (time === 'AM') return <LuSun size={14} className="text-amber-500" />
        if (time === 'PM') return <LuMoon size={14} className="text-indigo-500" />
        return <LuClock size={14} className="text-slate-400" />
    }

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-[60vh]">
                <div className="animate-pulse text-slate-400 flex items-center gap-3">
                    <LuPill size={24} className="animate-spin" />
                    Loading patients...
                </div>
            </div>
        )
    }

    return (
        <div className="min-h-screen pb-12 space-y-8">
            {/* Header */}
            <div>
                <h1 className="text-3xl font-bold text-slate-900">Treatment Plans</h1>
                <p className="text-slate-500 mt-1">Prescribe and monitor patient treatment adherence</p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                {/* Patient List - Left Panel */}
                <div className="lg:col-span-3 space-y-4">
                    <div className="relative">
                        <LuUsers className="absolute left-3 top-3 text-slate-400" size={16} />
                        <input
                            type="text"
                            placeholder="Search name or ID..."
                            value={patientSearch}
                            onChange={e => setPatientSearch(e.target.value)}
                            className="w-full bg-white border border-slate-200 rounded-xl py-2.5 pl-10 pr-4 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none focus:border-primary-500 focus:ring-4 focus:ring-primary-500/10"
                        />
                    </div>

                    <div className="space-y-2 max-h-[60vh] overflow-y-auto pr-1">
                        {filteredPatients.map(p => (
                            <motion.button
                                key={p.patient_id}
                                whileHover={{ scale: 1.01 }}
                                whileTap={{ scale: 0.99 }}
                                onClick={() => selectPatient(p)}
                                className={`w-full text-left p-4 rounded-2xl border transition-all ${selectedPatient?.patient_id === p.patient_id
                                    ? 'bg-primary-50 border-primary-200 shadow-sm'
                                    : 'bg-white border-slate-200 hover:border-slate-300'
                                    }`}
                            >
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="font-semibold text-slate-900 text-sm">{p.name || `Patient #${p.patient_id}`}</p>
                                        <p className="text-xs text-slate-400">ID: {p.patient_id} · {p.visits} visit{p.visits !== 1 ? 's' : ''}</p>
                                    </div>
                                    <LuChevronRight size={16} className="text-slate-300" />
                                </div>
                            </motion.button>
                        ))}
                        {filteredPatients.length === 0 && (
                            <p className="text-sm text-slate-400 text-center py-8">No patients found</p>
                        )}
                    </div>
                </div>

                {/* Plan Management - Right Panel */}
                <div className="lg:col-span-9 space-y-6">
                    {!selectedPatient ? (
                        <div className="text-center py-20 bg-slate-50 rounded-3xl border border-slate-200 border-dashed">
                            <LuUsers className="mx-auto mb-4 text-slate-300" size={48} />
                            <h3 className="text-xl font-bold text-slate-900 mb-2">Select a Patient</h3>
                            <p className="text-slate-400">Choose a patient from the list to view or create treatment plans.</p>
                        </div>
                    ) : (
                        <>
                            {/* Patient Header + Create Plan Button */}
                            <div className="flex items-center justify-between">
                                <div>
                                    <h2 className="text-xl font-bold text-slate-900">{selectedPatient.name || `Patient #${selectedPatient.patient_id}`}</h2>
                                    <p className="text-sm text-slate-400">{selectedPatient.name ? `ID: ${selectedPatient.patient_id} · ` : ''}{plans.length} treatment plan{plans.length !== 1 ? 's' : ''}</p>
                                </div>
                                <motion.button
                                    whileHover={{ scale: 1.02 }}
                                    whileTap={{ scale: 0.98 }}
                                    onClick={() => setShowCreatePlan(true)}
                                    className="btn btn-primary flex items-center gap-2"
                                >
                                    <LuPlus size={18} /> New Plan
                                </motion.button>
                            </div>

                            {/* Create Plan Modal */}
                            <AnimatePresence>
                                {showCreatePlan && (
                                    <motion.div
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        exit={{ opacity: 0 }}
                                        className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm flex items-center justify-center p-4"
                                        onClick={() => setShowCreatePlan(false)}
                                    >
                                        <motion.form
                                            initial={{ scale: 0.9, opacity: 0 }}
                                            animate={{ scale: 1, opacity: 1 }}
                                            exit={{ scale: 0.9, opacity: 0 }}
                                            className="bg-white rounded-3xl p-8 max-w-lg w-full shadow-2xl"
                                            onClick={e => e.stopPropagation()}
                                            onSubmit={createPlan}
                                        >
                                            <div className="flex items-center gap-3 mb-6">
                                                <IconWrapper variant="primary" size="md">
                                                    <LuFileText size={22} />
                                                </IconWrapper>
                                                <div>
                                                    <h3 className="text-xl font-bold text-slate-900">New Treatment Plan</h3>
                                                    <p className="text-sm text-slate-500">{selectedPatient.name || `Patient #${selectedPatient.patient_id}`}</p>
                                                </div>
                                            </div>

                                            <div className="space-y-4">
                                                <div>
                                                    <label className="block text-sm font-medium text-slate-700 mb-2">Diagnosis *</label>
                                                    <input
                                                        type="text"
                                                        value={newDiagnosis}
                                                        onChange={e => setNewDiagnosis(e.target.value)}
                                                        placeholder="e.g., Moderate Acne Vulgaris"
                                                        className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-slate-900 placeholder:text-slate-400 focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20 text-sm"
                                                        autoFocus
                                                        required
                                                    />
                                                </div>
                                                <div>
                                                    <label className="block text-sm font-medium text-slate-700 mb-2">Notes (optional)</label>
                                                    <textarea
                                                        value={newPlanNotes}
                                                        onChange={e => setNewPlanNotes(e.target.value)}
                                                        placeholder="Treatment approach, precautions, follow-up schedule..."
                                                        className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-slate-900 placeholder:text-slate-400 focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20 text-sm resize-none"
                                                        rows={3}
                                                    />
                                                </div>
                                                <div className="flex gap-3 pt-2">
                                                    <button type="button" onClick={() => setShowCreatePlan(false)}
                                                        className="flex-1 py-3 rounded-xl text-sm font-semibold text-slate-500 bg-slate-50 hover:bg-slate-100 transition-colors">
                                                        Cancel
                                                    </button>
                                                    <button type="submit"
                                                        className="flex-1 py-3 rounded-xl text-sm font-semibold text-white bg-primary-500 hover:bg-primary-600 transition-colors shadow-lg shadow-primary-500/30">
                                                        Create Plan
                                                    </button>
                                                </div>
                                            </div>
                                        </motion.form>
                                    </motion.div>
                                )}
                            </AnimatePresence>

                            {/* Plan Tabs */}
                            {plans.length > 0 && (
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

                            {/* Plan Detail */}
                            {selectedPlan && (
                                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                                    {/* Medication Steps */}
                                    <div className="lg:col-span-2 space-y-4">
                                        <div className="flex items-center justify-between">
                                            <h3 className="text-lg font-bold text-slate-900">Medication Steps</h3>
                                            <motion.button
                                                whileHover={{ scale: 1.02 }}
                                                whileTap={{ scale: 0.98 }}
                                                onClick={() => {
                                                    setStepForm(prev => ({ ...prev, step_order: steps.length + 1 }))
                                                    setShowAddStep(true)
                                                }}
                                                className="text-sm font-semibold text-primary-500 hover:text-primary-600 flex items-center gap-1"
                                            >
                                                <LuPlus size={16} /> Add Medication
                                            </motion.button>
                                        </div>

                                        <AnimatePresence mode="popLayout">
                                            {steps.map((step, i) => {
                                                const rate = getAdherenceRate(step.step_id)
                                                return (
                                                    <motion.div
                                                        key={step.step_id}
                                                        initial={{ opacity: 0, x: -20 }}
                                                        animate={{ opacity: 1, x: 0 }}
                                                        transition={{ delay: i * 0.05 }}
                                                    >
                                                        <Card className="p-5" hover={false}>
                                                            <div className="flex items-start justify-between">
                                                                <div className="flex items-start gap-4">
                                                                    <div className="w-10 h-10 rounded-xl bg-primary-50 text-primary-500 flex items-center justify-center font-bold text-sm flex-shrink-0">
                                                                        {step.step_order}
                                                                    </div>
                                                                    <div>
                                                                        <h4 className="font-semibold text-slate-900">{step.medication_name}</h4>
                                                                        {step.dosage && <p className="text-sm text-primary-600 font-medium">{step.dosage}</p>}
                                                                        <div className="flex items-center gap-3 mt-1 text-xs text-slate-500">
                                                                            <span className="flex items-center gap-1">
                                                                                {getTimeIcon(step.time_of_day)}
                                                                                {step.time_of_day === 'BOTH' ? 'AM & PM' : step.time_of_day}
                                                                            </span>
                                                                            <span className="flex items-center gap-1">
                                                                                <LuClock size={12} />
                                                                                {step.frequency}
                                                                            </span>
                                                                        </div>
                                                                        {step.instructions && (
                                                                            <p className="text-sm text-slate-400 mt-2 italic">{step.instructions}</p>
                                                                        )}
                                                                    </div>
                                                                </div>
                                                                {/* Adherence rate */}
                                                                <div className="text-right flex-shrink-0">
                                                                    {rate !== null ? (
                                                                        <div>
                                                                            <p className={`text-2xl font-bold ${rate >= 80 ? 'text-emerald-500' : rate >= 50 ? 'text-amber-500' : 'text-rose-500'}`}>
                                                                                {rate}%
                                                                            </p>
                                                                            <p className="text-xs text-slate-400">adherence</p>
                                                                        </div>
                                                                    ) : (
                                                                        <p className="text-xs text-slate-300">No data yet</p>
                                                                    )}
                                                                </div>
                                                            </div>
                                                        </Card>
                                                    </motion.div>
                                                )
                                            })}
                                        </AnimatePresence>

                                        {steps.length === 0 && (
                                            <div className="text-center py-12 text-slate-400 bg-slate-50 rounded-2xl border border-dashed border-slate-200">
                                                <LuPill className="mx-auto mb-3" size={32} />
                                                <p className="font-medium">No medications prescribed yet</p>
                                                <p className="text-sm">Add medication steps to this plan</p>
                                            </div>
                                        )}

                                        {/* Add Step Modal */}
                                        <AnimatePresence>
                                            {showAddStep && (
                                                <motion.div
                                                    initial={{ opacity: 0 }}
                                                    animate={{ opacity: 1 }}
                                                    exit={{ opacity: 0 }}
                                                    className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm flex items-center justify-center p-4"
                                                    onClick={() => setShowAddStep(false)}
                                                >
                                                    <motion.form
                                                        initial={{ scale: 0.9, opacity: 0 }}
                                                        animate={{ scale: 1, opacity: 1 }}
                                                        exit={{ scale: 0.9, opacity: 0 }}
                                                        className="bg-white rounded-3xl p-8 max-w-lg w-full shadow-2xl"
                                                        onClick={e => e.stopPropagation()}
                                                        onSubmit={addStep}
                                                    >
                                                        <div className="flex items-center gap-3 mb-6">
                                                            <IconWrapper variant="primary" size="md">
                                                                <LuPill size={22} />
                                                            </IconWrapper>
                                                            <div>
                                                                <h3 className="text-xl font-bold text-slate-900">Add Medication</h3>
                                                                <p className="text-sm text-slate-500">{selectedPlan.diagnosis}</p>
                                                            </div>
                                                        </div>

                                                        <div className="space-y-4">
                                                            <div>
                                                                <label className="block text-sm font-medium text-slate-700 mb-2">Medication Name *</label>
                                                                <input type="text" value={stepForm.medication_name}
                                                                    onChange={e => setStepForm(p => ({ ...p, medication_name: e.target.value }))}
                                                                    placeholder="e.g., Tretinoin" required autoFocus
                                                                    className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-slate-900 placeholder:text-slate-400 focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20 text-sm"
                                                                />
                                                            </div>
                                                            <div>
                                                                <label className="block text-sm font-medium text-slate-700 mb-2">Dosage</label>
                                                                <input type="text" value={stepForm.dosage}
                                                                    onChange={e => setStepForm(p => ({ ...p, dosage: e.target.value }))}
                                                                    placeholder="e.g., 0.025% cream, pea-sized amount"
                                                                    className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-slate-900 placeholder:text-slate-400 focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20 text-sm"
                                                                />
                                                            </div>
                                                            <div className="grid grid-cols-2 gap-4">
                                                                <div>
                                                                    <label className="block text-sm font-medium text-slate-700 mb-2">Frequency</label>
                                                                    <select value={stepForm.frequency}
                                                                        onChange={e => setStepForm(p => ({ ...p, frequency: e.target.value }))}
                                                                        className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-slate-900 text-sm focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20"
                                                                    >
                                                                        <option value="daily">Daily</option>
                                                                        <option value="twice_daily">Twice Daily</option>
                                                                        <option value="weekly">Weekly</option>
                                                                    </select>
                                                                </div>
                                                                <div>
                                                                    <label className="block text-sm font-medium text-slate-700 mb-2">Time of Day</label>
                                                                    <select value={stepForm.time_of_day}
                                                                        onChange={e => setStepForm(p => ({ ...p, time_of_day: e.target.value }))}
                                                                        className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-slate-900 text-sm focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20"
                                                                    >
                                                                        <option value="AM">Morning</option>
                                                                        <option value="PM">Evening</option>
                                                                        <option value="BOTH">Both</option>
                                                                    </select>
                                                                </div>
                                                            </div>
                                                            <div>
                                                                <label className="block text-sm font-medium text-slate-700 mb-2">Instructions</label>
                                                                <textarea value={stepForm.instructions}
                                                                    onChange={e => setStepForm(p => ({ ...p, instructions: e.target.value }))}
                                                                    placeholder="Apply thin layer after cleansing, avoid eye area..."
                                                                    className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-slate-900 placeholder:text-slate-400 focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20 text-sm resize-none"
                                                                    rows={2}
                                                                />
                                                            </div>
                                                            <div className="flex gap-3 pt-2">
                                                                <button type="button" onClick={() => setShowAddStep(false)}
                                                                    className="flex-1 py-3 rounded-xl text-sm font-semibold text-slate-500 bg-slate-50 hover:bg-slate-100 transition-colors">
                                                                    Cancel
                                                                </button>
                                                                <button type="submit"
                                                                    className="flex-1 py-3 rounded-xl text-sm font-semibold text-white bg-primary-500 hover:bg-primary-600 transition-colors shadow-lg shadow-primary-500/30">
                                                                    Add Medication
                                                                </button>
                                                            </div>
                                                        </div>
                                                    </motion.form>
                                                </motion.div>
                                            )}
                                        </AnimatePresence>
                                    </div>

                                    {/* Sidebar: Plan Info + Side Effects */}
                                    <div className="space-y-6">
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
                                                    <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Created</p>
                                                    <p className="text-slate-600 text-sm">{new Date(selectedPlan.created_at).toLocaleDateString()}</p>
                                                </div>
                                                {selectedPlan.notes && (
                                                    <div>
                                                        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Notes</p>
                                                        <p className="text-slate-600 text-sm">{selectedPlan.notes}</p>
                                                    </div>
                                                )}
                                            </div>
                                        </Card>

                                        {/* Side Effect Reports */}
                                        {getSideEffects().length > 0 && (
                                            <Card className="p-6 border-amber-200 bg-amber-50/50" hover={false}>
                                                <div className="flex items-center gap-2 mb-4">
                                                    <LuTriangleAlert className="text-amber-500" size={18} />
                                                    <CardTitle>Side Effect Reports</CardTitle>
                                                </div>
                                                <div className="space-y-3">
                                                    {getSideEffects().map((a, i) => (
                                                        <div key={i} className="p-3 bg-white rounded-xl border border-amber-100">
                                                            <p className="text-sm text-slate-700">{a.side_effects}</p>
                                                            <div className="flex items-center justify-between mt-2">
                                                                <p className="text-xs text-slate-400">
                                                                    {new Date(a.date).toLocaleDateString()}
                                                                </p>
                                                                {a.notes && a.notes.startsWith('SEVERITY:') && (
                                                                    <CardBadge variant={
                                                                        a.notes.includes('severe') ? 'danger' :
                                                                            a.notes.includes('moderate') ? 'warning' : 'success'
                                                                    }>
                                                                        {a.notes.replace('SEVERITY: ', '')}
                                                                    </CardBadge>
                                                                )}
                                                            </div>
                                                        </div>
                                                    ))}
                                                </div>
                                            </Card>
                                        )}

                                        {/* Overall Adherence */}
                                        {adherence.length > 0 && (
                                            <Card className="p-6" hover={false}>
                                                <div className="flex items-center gap-2 mb-4">
                                                    <LuActivity className="text-emerald-500" size={18} />
                                                    <CardTitle>Adherence Overview</CardTitle>
                                                </div>
                                                <div className="space-y-2">
                                                    <div className="flex items-center justify-between">
                                                        <p className="text-sm text-slate-500">Total records</p>
                                                        <p className="font-bold text-slate-900">{adherence.length}</p>
                                                    </div>
                                                    <div className="flex items-center justify-between">
                                                        <p className="text-sm text-slate-500">Taken</p>
                                                        <p className="font-bold text-emerald-500">{adherence.filter(a => a.taken).length}</p>
                                                    </div>
                                                    <div className="flex items-center justify-between">
                                                        <p className="text-sm text-slate-500">Overall rate</p>
                                                        <p className="font-bold text-slate-900">
                                                            {adherence.length > 0 ? Math.round((adherence.filter(a => a.taken).length / adherence.length) * 100) : 0}%
                                                        </p>
                                                    </div>
                                                </div>
                                            </Card>
                                        )}
                                    </div>
                                </div>
                            )}

                            {/* No plans yet */}
                            {plans.length === 0 && !showCreatePlan && (
                                <div className="text-center py-16 bg-slate-50 rounded-3xl border border-slate-200 border-dashed">
                                    <LuFileText className="mx-auto mb-4 text-slate-300" size={48} />
                                    <h3 className="text-xl font-bold text-slate-900 mb-2">No Treatment Plans</h3>
                                    <p className="text-slate-400 mb-6">No plans exist for this patient yet.</p>
                                    <button onClick={() => setShowCreatePlan(true)}
                                        className="btn btn-primary inline-flex items-center gap-2">
                                        <LuPlus size={18} /> Create First Plan
                                    </button>
                                </div>
                            )}
                        </>
                    )}
                </div>
            </div>
        </div>
    )
}
