import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { LuPlus, LuX, LuCalendar, LuImage, LuCamera, LuChevronLeft, LuChevronRight, LuHash, LuLayoutGrid, LuScale, LuTrash2 } from 'react-icons/lu'
import { api } from '../services/api'
import { Card, CardTitle, CardDescription, IconWrapper, CardBadge } from '../components/Card'

export default function SkinJourney() {
    const [logs, setLogs] = useState([])
    const [loading, setLoading] = useState(true)
    const [showModal, setShowModal] = useState(false)
    const [viewMode, setViewMode] = useState('gallery')

    const [selectedImage, setSelectedImage] = useState(null)
    const [previewUrl, setPreviewUrl] = useState('')
    const [notes, setNotes] = useState('')
    const [tags, setTags] = useState('')
    const [selectedZones, setSelectedZones] = useState([])
    const [submitting, setSubmitting] = useState(false)

    useEffect(() => {
        fetchLogs()
    }, [])

    const handleZoneSelect = (zone) => {
        if (selectedZones.includes(zone)) {
            setSelectedZones(prev => prev.filter(z => z !== zone))
        } else {
            setSelectedZones(prev => [...prev, zone])
        }
    }

    const fetchLogs = async () => {
        setLoading(true)
        try {
            const data = await api.getJourney()
            setLogs(Array.isArray(data) ? data : [])
        } catch (err) {
            console.error('Failed to fetch logs:', err)
            setLogs([])
        } finally {
            setLoading(false)
        }
    }

    const handleDelete = async (logId) => {
        if (!confirm('Are you sure you want to delete this entry?')) return
        try {
            await api.deleteJourneyLog(logId)
            setLogs(prev => prev.filter(log => log.log_id !== logId))
        } catch (err) {
            console.error('Failed to delete log:', err)
            alert('Failed to delete entry. Please try again.')
        }
    }

    const handleImageSelect = (e) => {
        const file = e.target.files[0]
        if (file) {
            setSelectedImage(file)
            setPreviewUrl(URL.createObjectURL(file))
        }
    }

    const handleSubmit = async (e) => {
        e.preventDefault()
        setSubmitting(true)
        try {
            let finalImagePath = ''
            if (selectedImage) {
                const reader = new FileReader()
                finalImagePath = await new Promise((resolve) => {
                    reader.onload = (e) => resolve(e.target.result)
                    reader.readAsDataURL(selectedImage)
                })
            }

            const manualTags = tags.split(',').map(t => t.trim()).filter(Boolean)
            const zoneTags = selectedZones.map(z => `loc:${z.toLowerCase().replace(' ', '_')}`)
            const allTags = [...manualTags, ...zoneTags]

            const payload = {
                image_path: finalImagePath,
                notes,
                tags: JSON.stringify(allTags)
            }

            await api.addJourneyLog(payload)
            setShowModal(false)
            fetchLogs()

            setNotes('')
            setTags('')
            setSelectedZones([])
            setSelectedImage(null)
            setPreviewUrl('')
        } catch (err) {
            alert('Failed to save log')
        } finally {
            setSubmitting(false)
        }
    }

    return (
        <div className="relative min-h-screen pb-12">
            <div className="relative z-10 max-w-7xl mx-auto space-y-8">
                {/* Header */}
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 lg:pr-72">
                    <div>
                        <h1 className="text-4xl md:text-5xl font-bold text-slate-900 tracking-tight">Skin Journey</h1>
                        <p className="text-slate-500 mt-2 text-lg">Track your progress and visualize changes over time</p>
                    </div>
                    <div className="flex items-center gap-3">
                        <div className="bg-white p-1.5 rounded-2xl flex items-center border border-slate-200 shadow-sm">
                            <button
                                onClick={() => setViewMode('gallery')}
                                className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-semibold transition-all ${viewMode === 'gallery' ? 'bg-primary-50 text-primary-600 border border-primary-100' : 'text-slate-500 hover:text-slate-900'}`}
                            >
                                <LuLayoutGrid size={16} /> Gallery
                            </button>
                            <button
                                onClick={() => setViewMode('compare')}
                                className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-semibold transition-all ${viewMode === 'compare' ? 'bg-indigo-50 text-indigo-600 border border-indigo-100' : 'text-slate-500 hover:text-slate-900'}`}
                            >
                                <LuScale size={16} /> Compare
                            </button>
                        </div>
                        <motion.button
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                            onClick={() => setShowModal(true)}
                            className="btn-primary flex items-center gap-2 shadow-lg shadow-primary-500/20"
                        >
                            <LuPlus size={18} /> New Entry
                        </motion.button>
                    </div>
                </div>

                {loading ? (
                    <div className="flex items-center justify-center py-20">
                        <div className="w-10 h-10 border-4 border-slate-200 border-t-primary-500 rounded-full animate-spin" />
                    </div>
                ) : (
                    <>
                        {viewMode === 'gallery' && (
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                                <AnimatePresence>
                                    {logs.map((log, i) => (
                                        <motion.div
                                            key={log.log_id}
                                            initial={{ opacity: 0, y: 20 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            transition={{ delay: i * 0.05 }}
                                        >
                                            <Card className="overflow-hidden group p-0 pb-4 h-full" hover>
                                                <div className="relative aspect-[4/3] bg-slate-100 overflow-hidden border-b border-slate-100">
                                                    {log.image_path ? (
                                                        <img src={log.image_path} alt="Skin log" className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" />
                                                    ) : (
                                                        <div className="flex items-center justify-center h-full text-slate-300">
                                                            <LuImage size={48} className="opacity-50" />
                                                        </div>
                                                    )}
                                                    <div className="absolute top-3 left-3 bg-white/90 backdrop-blur-sm px-3 py-1 rounded-full text-xs font-bold text-slate-700 shadow-sm">
                                                        {log.created_at ? new Date(log.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) : 'Unknown date'}
                                                    </div>
                                                    <button
                                                        onClick={(e) => { e.stopPropagation(); handleDelete(log.log_id); }}
                                                        className="absolute top-3 right-3 w-8 h-8 bg-white/90 hover:bg-rose-500 backdrop-blur-sm rounded-full flex items-center justify-center text-slate-400 hover:text-white opacity-0 group-hover:opacity-100 transition-all shadow-sm"
                                                        title="Delete entry"
                                                    >
                                                        <LuTrash2 size={14} />
                                                    </button>
                                                </div>
                                                <div className="p-5">
                                                    <p className="text-slate-600 text-sm line-clamp-2 leading-relaxed">{log.notes || "No notes added."}</p>
                                                    {log.tags && (() => {
                                                        try {
                                                            const parsed = JSON.parse(log.tags)
                                                            if (Array.isArray(parsed) && parsed.length > 0) {
                                                                return (
                                                                    <div className="flex flex-wrap gap-2 mt-4">
                                                                        {parsed.map((tag, idx) => (
                                                                            <CardBadge key={idx} variant="primary" className="bg-primary-50 text-primary-700 border border-primary-100">{tag}</CardBadge>
                                                                        ))}
                                                                    </div>
                                                                )
                                                            }
                                                        } catch { /* Invalid JSON */ }
                                                        return null
                                                    })()}
                                                </div>
                                            </Card>
                                        </motion.div>
                                    ))}
                                </AnimatePresence>
                                {logs.length === 0 && (
                                    <div className="col-span-full py-20 text-center">
                                        <Card className="p-12 border-dashed border-2 border-slate-200 bg-slate-50/50 shadow-none">
                                            <div className="w-20 h-20 bg-white rounded-full flex items-center justify-center mx-auto mb-6 shadow-sm border border-slate-100">
                                                <LuCamera className="text-slate-300" size={32} />
                                            </div>
                                            <p className="text-lg font-bold text-slate-900 mb-1">No logs yet</p>
                                            <p className="text-slate-500">Start your skin journey by adding your first entry.</p>
                                        </Card>
                                    </div>
                                )}
                            </div>
                        )}

                        {viewMode === 'compare' && (
                            <Card className="p-8 min-h-[500px] flex items-center justify-center bg-slate-50/50 border-slate-200">
                                {logs.length < 2 ? (
                                    <div className="text-center text-slate-400">
                                        <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center mx-auto mb-4 shadow-sm border border-slate-100">
                                            <LuScale size={24} />
                                        </div>
                                        <p className="text-xl font-bold text-slate-900 mb-2">Not enough data</p>
                                        <p>You need at least two entries to use the comparison view.</p>
                                    </div>
                                ) : (
                                    <CompareView logs={logs} />
                                )}
                            </Card>
                        )}
                    </>
                )}

                {/* Add Modal */}
                <AnimatePresence>
                    {showModal && (
                        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-sm">
                            <motion.div
                                initial={{ opacity: 0, scale: 0.95 }}
                                animate={{ opacity: 1, scale: 1 }}
                                exit={{ opacity: 0, scale: 0.95 }}
                                className="bg-white rounded-3xl w-full max-w-lg overflow-hidden shadow-2xl ring-1 ring-slate-900/5"
                            >
                                <div className="p-6 border-b border-slate-100 flex items-center justify-between">
                                    <CardTitle className="text-xl">New Daily Log</CardTitle>
                                    <button onClick={() => setShowModal(false)} className="p-2 hover:bg-slate-100 rounded-xl transition-colors text-slate-400 hover:text-slate-600">
                                        <LuX size={20} />
                                    </button>
                                </div>

                                <form onSubmit={handleSubmit} className="p-6 space-y-6">
                                    {/* Image Upload Area */}
                                    <div className="space-y-2">
                                        <label className="block text-sm font-semibold text-slate-500 uppercase tracking-wider">Photo</label>
                                        <div
                                            className={`relative aspect-[4/3] rounded-2xl border-2 border-dashed ${previewUrl ? 'border-transparent' : 'border-slate-200 hover:border-primary-400 hover:bg-slate-50'} bg-slate-50 flex flex-col items-center justify-center transition-all cursor-pointer overflow-hidden group`}
                                            onClick={() => document.getElementById('log-file').click()}
                                        >
                                            {previewUrl ? (
                                                <>
                                                    <img src={previewUrl} alt="Preview" className="w-full h-full object-cover" />
                                                    <div className="absolute inset-0 bg-slate-900/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                                                        <p className="text-white font-medium flex items-center gap-2"><LuCamera /> Change Photo</p>
                                                    </div>
                                                </>
                                            ) : (
                                                <>
                                                    <div className="p-4 bg-white rounded-full border border-slate-200 mb-3 shadow-sm text-primary-500">
                                                        <LuCamera size={24} />
                                                    </div>
                                                    <p className="text-sm font-bold text-slate-900">Tap to upload</p>
                                                    <p className="text-xs text-slate-400 mt-1">Selfie or close-up</p>
                                                </>
                                            )}
                                            <input
                                                id="log-file"
                                                type="file"
                                                accept="image/*"
                                                className="hidden"
                                                onChange={handleImageSelect}
                                            />
                                        </div>
                                    </div>

                                    <div className="space-y-4">
                                        <div>
                                            <label className="block text-sm font-semibold text-slate-500 uppercase tracking-wider mb-2">Notes</label>
                                            <textarea
                                                value={notes}
                                                onChange={e => setNotes(e.target.value)}
                                                placeholder="How does your skin feel today?"
                                                className="w-full rounded-xl bg-slate-50 border border-slate-200 px-4 py-3 text-slate-900 placeholder:text-slate-400 focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20 transition-all text-sm min-h-[100px]"
                                            />
                                        </div>

                                        <div>
                                            <label className="block text-sm font-semibold text-slate-500 uppercase tracking-wider mb-2">Tags (comma separated)</label>
                                            <div className="relative">
                                                <LuHash className="absolute left-4 top-3.5 text-slate-400" size={16} />
                                                <input
                                                    type="text"
                                                    value={tags}
                                                    onChange={e => setTags(e.target.value)}
                                                    placeholder="acne, dry, glowing..."
                                                    className="w-full pl-10 rounded-xl bg-slate-50 border border-slate-200 px-4 py-3 text-slate-900 placeholder:text-slate-400 focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20 transition-all text-sm"
                                                />
                                            </div>
                                        </div>
                                    </div>

                                    <motion.button
                                        whileHover={{ scale: 1.01 }}
                                        whileTap={{ scale: 0.99 }}
                                        type="submit"
                                        disabled={submitting}
                                        className="w-full btn-primary py-3.5 font-bold shadow-lg shadow-primary-500/20 disabled:opacity-70 disabled:cursor-not-allowed"
                                    >
                                        {submitting ? 'Saving...' : 'Save Entry'}
                                    </motion.button>
                                </form>
                            </motion.div>
                        </div>
                    )}
                </AnimatePresence>
            </div>
        </div>
    )
}

function CompareView({ logs }) {
    const sorted = [...logs].sort((a, b) => new Date(a.created_at) - new Date(b.created_at))
    const [leftIndex, setLeftIndex] = useState(0)
    const [rightIndex, setRightIndex] = useState(sorted.length - 1)
    const [position, setPosition] = useState(50)

    return (
        <div className="w-full max-w-4xl">
            <div className="flex items-center justify-between mb-6">
                <div className="w-1/3">
                    <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Before</label>
                    <select
                        value={leftIndex}
                        onChange={(e) => setLeftIndex(Number(e.target.value))}
                        className="w-full p-3 rounded-xl bg-white border border-slate-200 text-slate-900 text-sm font-medium focus:ring-2 focus:ring-primary-500/10 focus:border-primary-500 outline-none"
                    >
                        {sorted.map((log, i) => (
                            <option key={log.log_id} value={i} disabled={i === rightIndex}>
                                {new Date(log.created_at).toLocaleDateString()}
                            </option>
                        ))}
                    </select>
                </div>
                <div className="text-slate-300 font-medium text-lg px-4">vs</div>
                <div className="w-1/3 text-right">
                    <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">After</label>
                    <select
                        value={rightIndex}
                        onChange={(e) => setRightIndex(Number(e.target.value))}
                        className="w-full p-3 rounded-xl bg-white border border-slate-200 text-slate-900 text-sm font-medium focus:ring-2 focus:ring-primary-500/10 focus:border-primary-500 outline-none"
                    >
                        {sorted.map((log, i) => (
                            <option key={log.log_id} value={i} disabled={i === leftIndex}>
                                {new Date(log.created_at).toLocaleDateString()}
                            </option>
                        ))}
                    </select>
                </div>
            </div>

            <div className="relative aspect-[16/9] w-full bg-slate-100 rounded-2xl overflow-hidden border border-slate-200 select-none cursor-ew-resize group shadow-inner"
                onMouseMove={(e) => {
                    const rect = e.currentTarget.getBoundingClientRect()
                    const x = ((e.clientX - rect.left) / rect.width) * 100
                    setPosition(Math.min(Math.max(x, 0), 100))
                }}
            >
                {/* Right Image (Background) */}
                <img src={sorted[rightIndex].image_path} className="absolute inset-0 w-full h-full object-cover" alt="After" />

                {/* Left Image (Foreground with Clip) */}
                <div
                    className="absolute inset-0 overflow-hidden border-r-4 border-white shadow-xl"
                    style={{ width: `${position}%` }}
                >
                    <img src={sorted[leftIndex].image_path} className="absolute inset-0 w-full h-full max-w-none object-cover" style={{ width: '100vw', maxWidth: 'unset' }} alt="Before" />
                </div>

                {/* Handle */}
                <div
                    className="absolute top-0 bottom-0 bg-white/30 backdrop-blur-[2px] w-1 cursor-col-resize z-10 flex items-center justify-center pointer-events-none"
                    style={{ left: `${position}%` }}
                >
                    <div className="w-10 h-10 bg-white rounded-full shadow-lg flex items-center justify-center -ml-5 border border-slate-100 text-slate-400">
                        <div className="flex gap-0.5">
                            <LuChevronLeft size={16} />
                            <LuChevronRight size={16} />
                        </div>
                    </div>
                </div>

                {/* Overlay Dates */}
                <div className="absolute top-4 left-4 bg-white/90 text-slate-900 text-xs font-bold px-3 py-1.5 rounded-lg shadow-sm border border-white/50">
                    {new Date(sorted[leftIndex].created_at).toLocaleDateString()}
                </div>
                <div className="absolute bottom-4 right-4 bg-white/90 text-slate-900 text-xs font-bold px-3 py-1.5 rounded-lg shadow-sm border border-white/50">
                    {new Date(sorted[rightIndex].created_at).toLocaleDateString()}
                </div>
            </div>
        </div>
    )
}
