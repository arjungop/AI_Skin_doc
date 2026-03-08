import { FaClock, FaCircle, FaCheckCircle } from 'react-icons/fa'
import { LuArrowRight, LuCalendarX } from 'react-icons/lu'
import { Card, CardTitle, CardBadge } from '../Card'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import React from 'react'

export default function PatientQueue({ appointments = [], patients = [], loading = false }) {
    // Map appointments to queue format
    const queue = [...appointments]
        .filter(a => ['Scheduled', 'Confirmed'].includes(a.status))
        .sort((a, b) => new Date(a.appointment_date).getTime() - new Date(b.appointment_date).getTime())
        .map(a => {
            const pat = patients.find(p => p.patient_id === a.patient_id) || {}
            const name = [pat.first_name, pat.last_name].filter(Boolean).join(' ') || pat.username || `Patient #${a.patient_id}`
            const dateObj = new Date(a.appointment_date)
            const time = dateObj.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            return {
                id: a.appointment_id,
                name,
                time,
                type: a.reason || 'Consultation',
                status: a.status // Scheduled, Confirmed, Completed, Cancelled
            }
        })

    const getStatusColor = (s) => {
        if (s === 'Scheduled' || s === 'Waiting') return 'text-amber-500 bg-amber-50'
        if (s === 'Confirmed' || s === 'Arrived') return 'text-emerald-600 bg-emerald-50'
        if (s === 'Completed') return 'text-sky-500 bg-sky-50'
        return 'text-rose-500 bg-rose-50'
    }

    const containerVars = { hidden: { opacity: 0 }, show: { opacity: 1, transition: { staggerChildren: 0.1 } } }
    const itemVars = { hidden: { opacity: 0, x: -10 }, show: { opacity: 1, x: 0 } }

    return (
        <Card className="h-full p-6 flex flex-col group/card" hover={false}>
            <div className="flex justify-between items-center mb-5">
                <div>
                    <CardTitle>Patient Queue</CardTitle>
                    <p className="text-xs text-slate-400 font-medium">Upcoming visits</p>
                </div>
                <CardBadge variant="default" className="bg-primary-50 text-primary-600 border-none font-bold">
                    {queue.length} Left
                </CardBadge>
            </div>

            <div className="space-y-3 flex-1 overflow-y-auto min-h-0 custom-scrollbar pr-1 -mr-1">
                {loading ? (
                    <div className="flex justify-center items-center h-full">
                        <div className="w-6 h-6 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
                    </div>
                ) : queue.length === 0 ? (
                    // @ts-ignore
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="h-full flex flex-col items-center justify-center text-center p-6 border-2 border-dashed border-slate-100 rounded-2xl bg-slate-50/50">
                        <div className="w-16 h-16 bg-white shadow-sm border border-slate-100 rounded-full flex items-center justify-center text-emerald-400 mb-4">
                            <FaCheckCircle size={32} />
                        </div>
                        <p className="text-slate-700 font-bold mb-1">Queue Clear</p>
                        <p className="text-slate-400 text-xs">No pending appointments for today.</p>
                    </motion.div>
                ) : (
                    // @ts-ignore
                    <motion.div variants={containerVars} initial="hidden" animate="show" className="space-y-3 pb-2">
                        {queue.map(p => (
                            // @ts-ignore
                            <motion.div
                                key={p.id}
                                variants={itemVars}
                                className="flex items-center justify-between p-3 bg-white hover:bg-slate-50 rounded-2xl transition-all cursor-pointer group border border-slate-100 hover:border-slate-200 shadow-sm hover:shadow-md"
                            >
                                <div className="flex items-center gap-3 min-w-0 pr-2">
                                    <div className="relative">
                                        <div className="h-12 w-12 min-w-[48px] rounded-2xl bg-gradient-to-br from-primary-100 to-primary-50 flex items-center justify-center font-bold text-primary-600 text-sm shadow-inner flex-shrink-0 group-hover:scale-105 transition-transform">
                                            {(p.name.split(' ')[0]?.[0] || '').toUpperCase() + (p.name.split(' ')[1]?.[0] || '').toUpperCase()}
                                        </div>
                                    </div>
                                    <div className="min-w-0">
                                        <div className="font-bold text-slate-900 text-sm group-hover:text-primary-600 transition-colors truncate">{p.name}</div>
                                        <div className="text-xs font-medium text-slate-500 truncate mt-0.5">{p.type}</div>
                                    </div>
                                </div>
                                <div className="text-right flex-shrink-0">
                                    <div className="flex items-center justify-end gap-1.5 text-xs font-bold text-slate-700 bg-slate-50 px-2 py-1 rounded-lg">
                                        <FaClock size={10} className="text-primary-400 group-hover:animate-pulse" /> {p.time}
                                    </div>
                                    <div className={`text-[9px] font-bold uppercase tracking-widest flex items-center justify-end px-2 py-1 rounded-md mt-1.5 w-fit ml-auto ${getStatusColor(p.status)}`}>
                                        {p.status}
                                    </div>
                                </div>
                            </motion.div>
                        ))}
                    </motion.div>
                )}
            </div>

            <div className="mt-5 pt-4 border-t border-slate-100 text-center">
                <Link to="/doctor/appointments" className="text-xs font-bold text-slate-400 hover:text-primary-600 uppercase tracking-widest flex items-center justify-center gap-1.5 mx-auto transition-colors group/link">
                    Full Schedule <LuArrowRight size={14} className="group-hover/link:translate-x-1 transition-transform" />
                </Link>
            </div>
        </Card>
    )
}
