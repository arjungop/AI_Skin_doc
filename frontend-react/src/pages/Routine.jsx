import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { LuPlus, LuSun, LuMoon, LuCheck, LuTrash2, LuTrophy, LuFlame, LuDroplets, LuX } from 'react-icons/lu'
import { api } from '../services/api'
import { Card, CardTitle, CardDescription, IconWrapper, CardBadge } from '../components/Card'
import confetti from 'canvas-confetti'

export default function Routine() {
    const [items, setItems] = useState([])
    const [loading, setLoading] = useState(true)
    const [view, setView] = useState('AM')
    const [completions, setCompletions] = useState([])
    const [streak, setStreak] = useState(0)
    const [showAdd, setShowAdd] = useState(false)
    const [newItemName, setNewItemName] = useState('')
    const [newItemTime, setNewItemTime] = useState('AM')

    useEffect(() => {
        fetchRoutine()
        fetchCompletions()
        setStreak(Math.floor(Math.random() * 5) + 3)
    }, [])

    const fetchRoutine = async () => {
        try {
            const data = await api.getRoutine()
            setItems(data)
        } catch (err) {
            console.error(err)
        } finally {
            setLoading(false)
        }
    }

    const fetchCompletions = async () => {
        try {
            const today = new Date().toISOString().split('T')[0]
            const data = await api.getCompletions(today)
            setCompletions(data.map(c => c.routine_item_id))
        } catch (err) { console.error(err) }
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
                confetti({
                    particleCount: 100,
                    spread: 70,
                    origin: { y: 0.6 },
                    colors: view === 'AM' ? ['#00D4AA', '#10B981', '#ffffff'] : ['#8B5CF6', '#A78BFA', '#ffffff']
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

    return (
        <div className="relative min-h-screen pb-12">
            {/* Ambient Background */}
            <div className="fixed inset-0 pointer-events-none z-0">
                <div className="absolute top-20 left-1/3 w-[500px] h-[500px] bg-primary-500/10 rounded-full blur-[120px] opacity-40" />
                <div className="absolute bottom-1/4 right-1/4 w-[400px] h-[400px] bg-accent-500/10 rounded-full blur-[100px] opacity-30" />
            </div>

            <div className="relative z-10 max-w-5xl mx-auto space-y-8">
                {/* Header with Streak */}
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 lg:pr-72">
                    <div>
                        <h1 className="text-4xl md:text-5xl font-bold text-text-primary tracking-tight">Daily Routine</h1>
                        <p className="text-text-tertiary mt-2 text-lg">Consistency is key to healthy skin</p>
                    </div>

                    <Card variant="glass" className="px-6 py-4 flex items-center gap-6" hover={false}>
                        <div className="flex items-center gap-3">
                            <IconWrapper variant="accent" size="md">
                                <LuFlame size={20} />
                            </IconWrapper>
                            <div>
                                <p className="text-xs font-semibold text-text-tertiary uppercase tracking-wider">Streak</p>
                                <p className="text-xl font-bold text-gradient-primary">{streak} Days</p>
                            </div>
                        </div>
                        <div className="w-px h-10 bg-white/10" />
                        <div className="flex items-center gap-3">
                            <IconWrapper variant="ai" size="md">
                                <LuDroplets size={20} />
                            </IconWrapper>
                            <div>
                                <p className="text-xs font-semibold text-text-tertiary uppercase tracking-wider">Hydration</p>
                                <p className="text-xl font-bold text-primary-400">Good</p>
                            </div>
                        </div>
                    </Card>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    {/* Main List */}
                    <div className="lg:col-span-2 space-y-6">
                        {/* Toggle */}
                        <div className="bg-surface-elevated p-1.5 rounded-2xl flex items-center border border-white/10">
                            <button
                                onClick={() => setView('AM')}
                                className={`flex-1 flex items-center justify-center gap-2 py-3 rounded-xl text-sm font-semibold transition-all ${view === 'AM' ? 'bg-primary-500/10 text-primary-400 border border-primary-500/30' : 'text-text-secondary hover:text-text-primary'}`}
                            >
                                <LuSun size={18} />
                                Morning
                            </button>
                            <button
                                onClick={() => setView('PM')}
                                className={`flex-1 flex items-center justify-center gap-2 py-3 rounded-xl text-sm font-semibold transition-all ${view === 'PM' ? 'bg-accent-500/10 text-accent-400 border border-accent-500/30' : 'text-text-secondary hover:text-text-primary'}`}
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
                                                variant={isDone ? "glass" : "elevated"}
                                                className={`p-4 flex items-center justify-between group ${isDone ? 'border-primary-500/30 bg-primary-500/5' : ''}`}
                                                hover={!isDone}
                                            >
                                                <div className="flex items-center gap-4">
                                                    <motion.button
                                                        whileHover={{ scale: 1.1 }}
                                                        whileTap={{ scale: 0.9 }}
                                                        onClick={() => toggleComplete(item.item_id)}
                                                        className={`w-10 h-10 rounded-xl flex items-center justify-center transition-all ${isDone ? 'bg-primary-500 text-white' : 'bg-white/5 border border-white/10 text-text-muted hover:border-primary-500/30 hover:text-primary-400'}`}
                                                    >
                                                        <LuCheck size={20} />
                                                    </motion.button>
                                                    <div>
                                                        <h3 className={`font-semibold ${isDone ? 'text-primary-400 line-through opacity-70' : 'text-text-primary'}`}>{item.product_name}</h3>
                                                        <p className="text-xs text-text-muted capitalize">{item.time_of_day === 'BOTH' ? 'Any time' : item.time_of_day}</p>
                                                    </div>
                                                </div>
                                                <button
                                                    onClick={() => deleteItem(item.item_id)}
                                                    className="p-2 text-text-muted hover:text-danger opacity-0 group-hover:opacity-100 transition-all"
                                                >
                                                    <LuTrash2 size={18} />
                                                </button>
                                            </Card>
                                        </motion.div>
                                    )
                                })}
                            </AnimatePresence>

                            {/* Add New Button */}
                            {!showAdd ? (
                                <motion.button
                                    whileHover={{ scale: 1.01 }}
                                    whileTap={{ scale: 0.99 }}
                                    onClick={() => setShowAdd(true)}
                                    className="w-full py-4 border-2 border-dashed border-white/10 rounded-2xl text-text-muted font-medium hover:border-primary-500/30 hover:text-primary-400 hover:bg-primary-500/5 transition-all flex items-center justify-center gap-2"
                                >
                                    <LuPlus size={20} />
                                    Add Product
                                </motion.button>
                            ) : (
                                <motion.form
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className="p-4 rounded-2xl border border-white/10 bg-surface-elevated flex gap-3"
                                    onSubmit={addItem}
                                >
                                    <input
                                        type="text"
                                        placeholder="Product name (e.g. Vitamin C Serum)"
                                        className="flex-1 rounded-xl bg-white/5 border border-white/10 px-4 py-3 text-text-primary placeholder:text-text-muted focus:border-primary-500/50 focus:ring-2 focus:ring-primary-500/20 transition-all text-sm"
                                        value={newItemName}
                                        onChange={e => setNewItemName(e.target.value)}
                                        autoFocus
                                    />
                                    <select
                                        className="rounded-xl bg-white/5 border border-white/10 px-4 py-3 text-text-primary text-sm"
                                        value={newItemTime}
                                        onChange={e => setNewItemTime(e.target.value)}
                                    >
                                        <option value="AM">AM</option>
                                        <option value="PM">PM</option>
                                        <option value="BOTH">Both</option>
                                    </select>
                                    <button type="submit" className="btn-primary px-5">Add</button>
                                    <button type="button" onClick={() => setShowAdd(false)} className="p-3 text-text-muted hover:bg-white/5 rounded-xl">
                                        <LuX size={20} />
                                    </button>
                                </motion.form>
                            )}
                        </div>
                    </div>

                    {/* Sidebar info */}
                    <div className="space-y-6">
                        <Card variant="gradient" className="p-6 text-white overflow-hidden relative">
                            <div className="relative z-10">
                                <h3 className="text-xl font-bold mb-2">Did you know?</h3>
                                <p className="text-white/80 text-sm leading-relaxed">
                                    Consistent application of sunscreen is the #1 anti-aging secret. Even on cloudy days, UV rays penetrate the skin.
                                </p>
                            </div>
                            <div className="absolute -bottom-6 -right-6 w-32 h-32 bg-white/10 rounded-full blur-2xl" />
                            <div className="absolute top-4 right-4 opacity-20">
                                <LuSun size={40} />
                            </div>
                        </Card>

                        <Card variant="glass" className="p-6" hover={false}>
                            <div className="flex items-center gap-2 mb-4">
                                <LuTrophy className="text-accent-400" size={18} />
                                <CardTitle>Achievements</CardTitle>
                            </div>
                            <div className="space-y-4">
                                <div className="flex items-center gap-3 opacity-60">
                                    <div className="w-10 h-10 rounded-xl bg-white/5 flex items-center justify-center text-xl border border-white/10">🌱</div>
                                    <div className="flex-1">
                                        <p className="text-sm font-medium text-text-primary">7 Day Streak</p>
                                        <div className="w-full h-1.5 bg-white/5 rounded-full mt-1.5 overflow-hidden">
                                            <div className="h-full bg-primary-500 w-[60%] rounded-full" />
                                        </div>
                                    </div>
                                </div>
                                <div className="flex items-center gap-3">
                                    <div className="w-10 h-10 rounded-xl bg-accent-500/10 flex items-center justify-center text-xl border border-accent-500/20">🌟</div>
                                    <div>
                                        <p className="text-sm font-medium text-text-primary">Review Routine</p>
                                        <p className="text-xs text-primary-400 font-medium">Completed</p>
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
