import { useState } from 'react'
import { FaPlus, FaCheck } from 'react-icons/fa'
import { LuPlus, LuPill, LuChevronRight } from 'react-icons/lu'
import { Card, CardTitle, CardDescription } from '../Card'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import React from 'react'

export default function QuickPrescription() {
    const templates = [
        { id: 1, name: 'Mild Acne', meds: ['Benzoyl Peroxide 2.5%', 'Clindamycin Gel'] },
        { id: 2, name: 'Eczema Flare', meds: ['Hydrocortisone 1%', 'Moisturizer'] },
        { id: 3, name: 'Anti-Aging', meds: ['Tretinoin 0.025%', 'Sunscreen SPF 50'] },
    ]

    const [selected, setSelected] = useState(null)

    const containerVars = { hidden: { opacity: 0 }, show: { opacity: 1, transition: { staggerChildren: 0.1 } } }
    const itemVars = { hidden: { opacity: 0, y: 10 }, show: { opacity: 1, y: 0 } }

    return (
        <Card className="h-full p-6 flex flex-col group/card" hover={false}>
            <div className="flex justify-between items-center mb-5">
                <div>
                    <CardTitle>Quick Scripts</CardTitle>
                    <p className="text-xs text-slate-400 font-medium">Common templates</p>
                </div>
                <div className="w-8 h-8 rounded-full bg-slate-50 flex items-center justify-center text-primary-400 group-hover/card:bg-primary-50 transition-colors">
                    <LuPill size={14} />
                </div>
            </div>

            <motion.div variants={containerVars} initial="hidden" animate="show" className="flex-1 space-y-2.5 overflow-y-auto min-h-0 custom-scrollbar pr-1 -mr-1 pb-2">
                {templates.map(t => (
                    // @ts-ignore
                    <motion.div
                        key={t.id}
                        variants={itemVars}
                        className={`p-3 rounded-2xl cursor-pointer transition-all border group/item ${selected === t.id
                            ? 'border-primary-200 bg-primary-50 shadow-sm'
                            : 'border-slate-100 bg-white hover:border-slate-300 hover:shadow-sm'
                            }`}
                        onClick={() => setSelected(t.id)}
                    >
                        <div className="flex items-center gap-3">
                            <div className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 transition-all ${selected === t.id
                                    ? 'bg-primary-500 text-white shadow-md shadow-primary-500/30 scale-105'
                                    : 'bg-slate-50 text-slate-400 group-hover/item:bg-slate-100'
                                }`}>
                                {selected === t.id ? <FaCheck size={14} /> : <span className="font-bold text-xs">{t.name.charAt(0)}</span>}
                            </div>
                            <div className="min-w-0 flex-1">
                                <div className={`font-bold text-sm truncate transition-colors ${selected === t.id ? 'text-primary-700' : 'text-slate-800'}`}>
                                    {t.name}
                                </div>
                                <div className={`text-[11px] font-medium leading-tight mt-0.5 transition-colors ${selected === t.id ? 'text-primary-600/70' : 'text-slate-500 line-clamp-2'}`}>
                                    {t.meds.join(', ')}
                                </div>
                            </div>
                        </div>
                    </motion.div>
                ))}
            </motion.div>

            <div className="mt-5 pt-4 border-t border-slate-100">
                <Link
                    to="/doctor/treatment-plans"
                    className={`w-full py-2.5 rounded-xl text-sm font-bold transition-all flex items-center justify-center gap-2 group/btn shadow-sm ${selected
                        ? 'bg-primary-600 hover:bg-primary-700 text-white hover:shadow-md'
                        : 'bg-slate-900 hover:bg-slate-800 text-white hover:shadow-md'
                        }`}
                >
                    {selected ? 'Use Template' : 'Manage Plans'}
                    <LuChevronRight size={16} className={`transition-transform ${selected ? 'translate-x-1' : 'group-hover/btn:translate-x-1'}`} />
                </Link>
            </div>
        </Card>
    )
}
