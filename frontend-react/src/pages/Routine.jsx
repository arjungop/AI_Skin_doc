import React, { useState, useEffect, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { LuPlus, LuSun, LuMoon, LuCheck, LuTrash2, LuTrophy, LuFlame, LuX, LuSparkles } from 'react-icons/lu'
import { api } from '../services/api'
import { Card, CardTitle, CardDescription, IconWrapper, CardBadge } from '../components/Card'
import confetti from 'canvas-confetti'

// Rotating tips for variety
const SKIN_TIPS = [
    "Consistent application of sunscreen is the #1 anti-aging secret. Even on cloudy days, UV rays penetrate the skin.",
    "Apply products from thinnest to thickest consistency for best absorption.",
    "Vitamin C serums work best in the morning to protect against free radicals.",
    "Retinol increases sun sensitivity — always use it at night.",
    "Wait 1-2 minutes between products to let each layer absorb properly.",
    "Double cleansing at night removes sunscreen and makeup more effectively.",
    "Hyaluronic acid works best when applied to damp skin.",
    "Your neck and hands age faster — extend your routine to these areas!"
]

export default function Routine() {
    const [items, setItems] = useState([])
    const [loading, setLoading] = useState(true)
    const [view, setView] = useState('AM')
    const [completions, setCompletions] = useState([])
    const [streak, setStreak] = useState(0)
    const [showAdd, setShowAdd] = useState(false)
    const [newItemName, setNewItemName] = useState('')
    const [newItemTime, setNewItemTime] = useState('AM')

    // Pick a random tip based on the day (so it changes daily, not on refresh)
    const todaysTip = useMemo(() => {
        const dayOfYear = Math.floor((Date.now() - new Date(new Date().getFullYear(), 0, 0)) / 86400000)
        return SKIN_TIPS[dayOfYear % SKIN_TIPS.length]
    }, [])

    useEffect(() => {
        fetchRoutine()
        fetchCompletionsAndStreak()
    }, [])

    const fetchRoutine = async () => {
        setLoading(true)
        try {
            const data = await api.getRoutine()
            setItems(Array.isArray(data) ? data : [])
        } catch (err) {
            console.error('Failed to fetch routine:', err)
            setItems([])
        } finally {
            setLoading(false)
        }
    }

    const fetchCompletionsAndStreak = async () => {
        try {
            const today = new Date().toISOString().split('T')[0]
            const data = await api.getCompletions(today)
            setCompletions(data.map(c => c.routine_item_id))

            // Calculate real streak by checking consecutive days
            // For now, store and retrieve from localStorage as a simple solution
            // A proper implementation would query the backend for completion history
            const storedStreak = parseInt(localStorage.getItem('routine_streak') || '0')
            const lastCompletedDate = localStorage.getItem('routine_last_complete_date')
            const yesterday = new Date(Date.now() - 86400000).toISOString().split('T')[0]

            if (lastCompletedDate === yesterday) {
                // Streak continues from yesterday
                setStreak(storedStreak)
            } else if (lastCompletedDate === today) {
                // Already completed today, keep streak
                setStreak(storedStreak)
            } else {
                // Streak broken
                setStreak(0)
                localStorage.setItem('routine_streak', '0')
            }
        } catch (err) {
            console.error(err)
        }
    }

    const toggleComplete = async (itemId) => {
        const isCompleted = completions.includes(itemId)
        if (isCompleted) return

        try {
            await api.checkRoutineItem({
                routine_item_id: itemId,
                date: new Date().toISOString(),
                status: true
            })
            setCompletions(prev => [...prev, itemId])

            const currentViewItems = items.filter(i => i.time_of_day === view || i.time_of_day === 'BOTH')
            const completedCount = currentViewItems.filter(i => completions.includes(i.item_id) || i.item_id === itemId).length

            if (completedCount === currentViewItems.length && currentViewItems.length > 0) {
                // Update streak when routine is completed
                const today = new Date().toISOString().split('T')[0]
                const lastDate = localStorage.getItem('routine_last_complete_date')
                const yesterday = new Date(Date.now() - 86400000).toISOString().split('T')[0]

                let newStreak = 1
                if (lastDate === yesterday) {
                    newStreak = streak + 1
                } else if (lastDate !== today) {
                    newStreak = 1
                } else {
                    newStreak = streak
                }

                setStreak(newStreak)
                localStorage.setItem('routine_streak', newStreak.toString())
                localStorage.setItem('routine_last_complete_date', today)
                localStorage.setItem('streak_days', newStreak.toString()) // For dashboard

                // Celebrate with theme-aligned colors (Rose/Emerald)
                confetti({
                    particleCount: 100,
                    spread: 70,
                    origin: { y: 0.6 },
                    colors: view === 'AM' ? ['#F43F5E', '#FB7185', '#ffffff'] : ['#10B981', '#34D399', '#ffffff']
                })
            }
        } catch (err) {
            alert('Failed to update')
        }
    }

    const addItem = async (e) => {
        e.preventDefault()
        if (!newItemName) return
        try {
            await api.addRoutineItem({
                product_name: newItemName,
                time_of_day: newItemTime,
                step_order: items.length + 1,
                is_active: true
            })
            setNewItemName('')
            setShowAdd(false)
            fetchRoutine()
        } catch (err) {
            alert('Failed to add item')
        }
    }

    const deleteItem = async (id) => {
        if (!confirm('Remove this product?')) return
        try {
            await api.deleteRoutineItem(id)
            setItems(prev => prev.filter(i => i.item_id !== id))
        } catch (err) {
            alert('Failed')
        }
    }

    const currentItems = items.filter(i => i.time_of_day === view || i.time_of_day === 'BOTH')
    const completedToday = currentItems.filter(i => completions.includes(i.item_id)).length
    const progressPercent = currentItems.length > 0 ? Math.round((completedToday / currentItems.length) * 100) : 0

    return (
        <div className="relative min-h-screen pb-12">
            <div className="relative z-10 max-w-5xl mx-auto space-y-8">
                {/* Header with Streak */}
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 lg:pr-72">
                    <div>
                        <h1 className="text-4xl md:text-5xl font-bold text-slate-900 tracking-tight">Daily Routine</h1>
                        <p className="text-slate-500 mt-2 text-lg">Consistency is key to healthy skin</p>
                    </div>

                    <Card className="px-6 py-4 flex items-center gap-6" hover={false}>
                        <div className="flex items-center gap-3">
                            <IconWrapper variant="primary" size="md" className="bg-rose-50 text-rose-500">
                                <LuFlame size={20} />
                            </IconWrapper>
                            <div>
                                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Streak</p>
                                <p className="text-xl font-bold text-slate-900">{streak} Day{streak !== 1 ? 's' : ''}</p>
                            </div>
                        </div>
                        <div className="w-px h-10 bg-slate-100" />
                        <div className="flex items-center gap-3">
                            <IconWrapper variant="ai" size="md" className="bg-emerald-50 text-emerald-500">
                                <LuCheck size={20} />
                            </IconWrapper>
                            <div>
                                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Today</p>
                                <p className="text-xl font-bold text-emerald-600">{progressPercent}%</p>
                            </div>
                        </div>
                    </Card>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    {/* Main List */}
                    <div className="lg:col-span-2 space-y-6">
                        {/* Toggle */}
                        <div className="bg-white p-1.5 rounded-2xl flex items-center border border-slate-200 shadow-sm">
                            <button
                                onClick={() => setView('AM')}
                                className={`flex-1 flex items-center justify-center gap-2 py-3 rounded-xl text-sm font-semibold transition-all ${view === 'AM' ? 'bg-amber-50 text-amber-600 border border-amber-200' : 'text-slate-500 hover:text-slate-900'}`}
                            >
                                <LuSun size={18} />
                                Morning
                            </button>
                            <button
                                onClick={() => setView('PM')}
                                className={`flex-1 flex items-center justify-center gap-2 py-3 rounded-xl text-sm font-semibold transition-all ${view === 'PM' ? 'bg-indigo-50 text-indigo-600 border border-indigo-200' : 'text-slate-500 hover:text-slate-900'}`}
                            >
                                <LuMoon size={18} />
                                Evening
                            </button>
                        </div>

                        <div className="space-y-3">
                            <AnimatePresence mode='popLayout'>
                                {currentItems.map((item, i) => {
                                    const isDone = completions.includes(item.item_id)
                                    return (
                                        <motion.div
                                            key={item.item_id}
                                            initial={{ opacity: 0, x: -20 }}
                                            animate={{ opacity: 1, x: 0 }}
                                            exit={{ opacity: 0, scale: 0.9 }}
                                            transition={{ delay: i * 0.05 }}
                                        >
                                            <Card
                                                className={`p-4 flex items-center justify-between group ${isDone ? 'bg-emerald-50/50 border-emerald-100' : ''}`}
                                                hover={!isDone}
                                            >
                                                <div className="flex items-center gap-4">
                                                    <motion.button
                                                        whileHover={{ scale: 1.1 }}
                                                        whileTap={{ scale: 0.9 }}
                                                        onClick={() => toggleComplete(item.item_id)}
                                                        className={`w-10 h-10 rounded-xl flex items-center justify-center transition-all ${isDone ? 'bg-emerald-500 text-white shadow-lg shadow-emerald-500/30' : 'bg-slate-50 border border-slate-200 text-slate-400 hover:border-emerald-500/50 hover:text-emerald-500'}`}
                                                    >
                                                        <LuCheck size={20} />
                                                    </motion.button>
                                                    <div>
                                                        <h3 className={`font-semibold ${isDone ? 'text-emerald-700 line-through opacity-70' : 'text-slate-900'}`}>{item.product_name}</h3>
                                                        <p className="text-xs text-slate-500 capitalize">{item.time_of_day === 'BOTH' ? 'Any time' : item.time_of_day}</p>
                                                    </div>
                                                </div>
                                                <button
                                                    onClick={() => deleteItem(item.item_id)}
                                                    className="p-2 text-slate-300 hover:text-rose-500 opacity-0 group-hover:opacity-100 transition-all"
                                                >
                                                    <LuTrash2 size={18} />
                                                </button>
                                            </Card>
                                        </motion.div>
                                    )
                                })}
                            </AnimatePresence>

                            {/* Empty state */}
                            {!loading && currentItems.length === 0 && (
                                <div className="text-center py-12 text-slate-400">
                                    <LuSparkles size={32} className="mx-auto mb-3 opacity-50" />
                                    <p className="font-medium">No {view === 'AM' ? 'morning' : 'evening'} products yet</p>
                                    <p className="text-sm">Add your first product below</p>
                                </div>
                            )}

                            {/* Add New Button */}
                            {!showAdd ? (
                                <motion.button
                                    whileHover={{ scale: 1.01 }}
                                    whileTap={{ scale: 0.99 }}
                                    onClick={() => setShowAdd(true)}
                                    className="w-full py-4 border-2 border-dashed border-slate-200 rounded-2xl text-slate-400 font-medium hover:border-primary-300 hover:text-primary-600 hover:bg-primary-50 transition-all flex items-center justify-center gap-2"
                                >
                                    <LuPlus size={20} />
                                    Add Product
                                </motion.button>
                            ) : (
                                <motion.form
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className="p-4 rounded-2xl border border-slate-200 bg-white flex gap-3 shadow-soft-md"
                                    onSubmit={addItem}
                                >
                                    <input
                                        type="text"
                                        placeholder="Product name (e.g. Vitamin C Serum)"
                                        className="flex-1 rounded-xl bg-slate-50 border border-slate-200 px-4 py-3 text-slate-900 placeholder:text-slate-400 focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20 transition-all text-sm"
                                        value={newItemName}
                                        onChange={e => setNewItemName(e.target.value)}
                                        autoFocus
                                    />
                                    <select
                                        className="rounded-xl bg-slate-50 border border-slate-200 px-4 py-3 text-slate-900 text-sm focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20"
                                        value={newItemTime}
                                        onChange={e => setNewItemTime(e.target.value)}
                                    >
                                        <option value="AM">AM</option>
                                        <option value="PM">PM</option>
                                        <option value="BOTH">Both</option>
                                    </select>
                                    <button type="submit" className="btn btn-primary px-5 shadow-none">Add</button>
                                    <button type="button" onClick={() => setShowAdd(false)} className="p-3 text-slate-400 hover:bg-slate-100 rounded-xl transition-colors">
                                        <LuX size={20} />
                                    </button>
                                </motion.form>
                            )}
                        </div>
                    </div>

                    {/* Sidebar info */}
                    <div className="space-y-6">
                        <Card className="p-6 bg-gradient-to-br from-primary-500 to-primary-600 text-white overflow-hidden relative border-none shadow-lg shadow-primary-500/30">
                            <div className="relative z-10">
                                <h3 className="text-xl font-bold mb-2">Did you know?</h3>
                                <p className="text-white/90 text-sm leading-relaxed">
                                    {todaysTip}
                                </p>
                            </div>
                            <div className="absolute -bottom-6 -right-6 w-32 h-32 bg-white/20 rounded-full blur-2xl" />
                            <div className="absolute top-4 right-4 opacity-20">
                                <LuSun size={40} />
                            </div>
                        </Card>

                        <Card className="p-6" hover={false}>
                            <div className="flex items-center gap-2 mb-4">
                                <LuTrophy className="text-amber-500" size={18} />
                                <CardTitle>Milestones</CardTitle>
                            </div>
                            <div className="space-y-4">
                                <div className={`flex items-center gap-3 ${streak >= 7 ? '' : 'opacity-50'}`}>
                                    <div className={`w-10 h-10 rounded-xl flex items-center justify-center text-xl border ${streak >= 7 ? 'bg-amber-50 border-amber-200' : 'bg-slate-50 border-slate-100'}`}>🌱</div>
                                    <div className="flex-1">
                                        <p className="text-sm font-medium text-slate-900">7 Day Streak</p>
                                        {streak < 7 ? (
                                            <div className="w-full h-1.5 bg-slate-100 rounded-full mt-1.5 overflow-hidden">
                                                <div className="h-full bg-emerald-500 rounded-full transition-all" style={{ width: `${Math.min(100, (streak / 7) * 100)}%` }} />
                                            </div>
                                        ) : (
                                            <p className="text-xs text-emerald-600 font-medium">Achieved! 🎉</p>
                                        )}
                                    </div>
                                </div>
                                <div className={`flex items-center gap-3 ${streak >= 30 ? '' : 'opacity-50'}`}>
                                    <div className={`w-10 h-10 rounded-xl flex items-center justify-center text-xl border ${streak >= 30 ? 'bg-amber-50 border-amber-200' : 'bg-slate-50 border-slate-100'}`}>🌟</div>
                                    <div className="flex-1">
                                        <p className="text-sm font-medium text-slate-900">30 Day Streak</p>
                                        {streak < 30 ? (
                                            <div className="w-full h-1.5 bg-slate-100 rounded-full mt-1.5 overflow-hidden">
                                                <div className="h-full bg-amber-500 rounded-full transition-all" style={{ width: `${Math.min(100, (streak / 30) * 100)}%` }} />
                                            </div>
                                        ) : (
                                            <p className="text-xs text-amber-600 font-medium">Achieved! 🎉</p>
                                        )}
                                    </div>
                                </div>
                            </div>
                        </Card>
                    </div>
                </div>
            </div>
        </div>
    )
}
