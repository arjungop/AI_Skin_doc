import { useState, useEffect } from 'react'
import { api } from '../../services/api.js'
import { motion, AnimatePresence } from 'framer-motion'
import { LuPlus, LuExternalLink, LuPackage, LuX, LuLoader } from 'react-icons/lu'

/**
 * DoctorSuggestions — displays product suggestions for a patient.
 * - Doctors see an "Add Suggestion" form + existing suggestions.
 * - Patients see the suggestions list only.
 *
 * Props:
 *   patientId  – required, the patient to show suggestions for
 *   reportId   – optional, if provided, the "Add" form will attach to this report
 *   isDoctor   – boolean, enables the add form
 */
export default function DoctorSuggestions({ patientId, reportId, isDoctor = false }) {
    const [suggestions, setSuggestions] = useState([])
    const [loading, setLoading] = useState(true)
    const [showForm, setShowForm] = useState(false)

    // Form state
    const [productName, setProductName] = useState('')
    const [productLink, setProductLink] = useState('')
    const [notes, setNotes] = useState('')
    const [submitting, setSubmitting] = useState(false)

    useEffect(() => {
        if (patientId) fetchSuggestions()
    }, [patientId])

    const fetchSuggestions = async () => {
        setLoading(true)
        try {
            const data = await api.getSuggestions(patientId)
            setSuggestions(data || [])
        } catch (err) {
            console.error('Failed to load suggestions:', err)
        } finally {
            setLoading(false)
        }
    }

    const handleSubmit = async (e) => {
        e.preventDefault()
        if (!productName.trim() || !reportId) return
        setSubmitting(true)
        try {
            const suggestion = await api.addSuggestion({
                report_id: reportId,
                product_name: productName,
                product_link: productLink || null,
                notes: notes || null,
            })
            setSuggestions(prev => [suggestion, ...prev])
            setProductName('')
            setProductLink('')
            setNotes('')
            setShowForm(false)
        } catch (err) {
            alert('Failed to add suggestion: ' + (err.message || err))
        } finally {
            setSubmitting(false)
        }
    }

    if (loading) {
        return (
            <div className="flex items-center gap-2 py-4 text-slate-400 text-sm">
                <LuLoader className="animate-spin" size={16} />
                Loading suggestions...
            </div>
        )
    }

    if (suggestions.length === 0 && !isDoctor) {
        return null // Don't show section for patients if no suggestions
    }

    return (
        <div className="space-y-4">
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <LuPackage className="text-primary-500" size={18} />
                    <h3 className="text-base font-bold text-slate-900">Doctor Recommendations</h3>
                </div>
                {isDoctor && reportId && (
                    <motion.button
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={() => setShowForm(!showForm)}
                        className="text-sm font-semibold text-primary-500 hover:text-primary-600 flex items-center gap-1"
                    >
                        {showForm ? <LuX size={16} /> : <LuPlus size={16} />}
                        {showForm ? 'Cancel' : 'Add Suggestion'}
                    </motion.button>
                )}
            </div>

            {/* Add Suggestion Form */}
            <AnimatePresence>
                {showForm && (
                    <motion.form
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        onSubmit={handleSubmit}
                        className="bg-slate-50 rounded-2xl p-5 border border-slate-200 overflow-hidden"
                    >
                        <div className="space-y-3">
                            <div>
                                <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1.5">Product Name *</label>
                                <input
                                    type="text"
                                    value={productName}
                                    onChange={e => setProductName(e.target.value)}
                                    placeholder="e.g., CeraVe Moisturizing Cream"
                                    className="w-full bg-white border border-slate-200 rounded-xl px-4 py-2.5 text-sm text-slate-900 placeholder:text-slate-400 focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20"
                                    required
                                    autoFocus
                                />
                            </div>
                            <div>
                                <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1.5">Product Link</label>
                                <input
                                    type="url"
                                    value={productLink}
                                    onChange={e => setProductLink(e.target.value)}
                                    placeholder="https://pharmacy.com/product"
                                    className="w-full bg-white border border-slate-200 rounded-xl px-4 py-2.5 text-sm text-slate-900 placeholder:text-slate-400 focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20"
                                />
                            </div>
                            <div>
                                <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1.5">Instructions / Notes</label>
                                <textarea
                                    value={notes}
                                    onChange={e => setNotes(e.target.value)}
                                    placeholder="Apply after cleansing, twice daily..."
                                    className="w-full bg-white border border-slate-200 rounded-xl px-4 py-2.5 text-sm text-slate-900 placeholder:text-slate-400 focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20 resize-none"
                                    rows={2}
                                />
                            </div>
                            <button
                                type="submit"
                                disabled={submitting || !productName.trim()}
                                className="w-full py-2.5 rounded-xl text-sm font-semibold text-white bg-primary-500 hover:bg-primary-600 disabled:opacity-50 transition-colors shadow-sm"
                            >
                                {submitting ? 'Adding...' : 'Add Recommendation'}
                            </button>
                        </div>
                    </motion.form>
                )}
            </AnimatePresence>

            {/* Suggestions List */}
            <div className="space-y-3">
                {suggestions.map((s, i) => (
                    <motion.div
                        key={s.suggestion_id || i}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.05 }}
                        className="flex items-start gap-4 bg-white p-4 rounded-2xl border border-slate-200 hover:border-slate-300 transition-colors"
                    >
                        <div className="w-10 h-10 rounded-xl bg-primary-50 text-primary-500 flex items-center justify-center flex-shrink-0">
                            <LuPackage size={18} />
                        </div>
                        <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                                <h4 className="font-semibold text-slate-900 text-sm truncate">{s.product_name}</h4>
                                {s.product_link && (
                                    <a
                                        href={s.product_link}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="text-primary-500 hover:text-primary-600 flex-shrink-0"
                                        title="Open product link"
                                    >
                                        <LuExternalLink size={14} />
                                    </a>
                                )}
                            </div>
                            {s.notes && <p className="text-sm text-slate-500 mt-1">{s.notes}</p>}
                            <p className="text-xs text-slate-400 mt-2">{new Date(s.created_at).toLocaleDateString()}</p>
                        </div>
                    </motion.div>
                ))}
                {suggestions.length === 0 && isDoctor && (
                    <div className="text-center py-6 text-slate-400 text-sm">
                        <p>No recommendations yet. Add one above.</p>
                    </div>
                )}
            </div>
        </div>
    )
}
