import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Plus, Sun, Moon, Check, Trash2, Trophy, Flame, Droplets } from 'lucide-react'
import { api } from '../services/api'
import confetti from 'canvas-confetti'

export default function Routine() {
    const [items, setItems] = useState([])
    const [loading, setLoading] = useState(true)
    const [view, setView] = useState('AM') // 'AM' | 'PM'
    const [completions, setCompletions] = useState([]) // IDs of completed items for today
    const [streak, setStreak] = useState(0)

    // Add Item State
    const [showAdd, setShowAdd] = useState(false)
    const [newItemName, setNewItemName] = useState('')
    const [newItemTime, setNewItemTime] = useState('AM')

    useEffect(() => {
        fetchRoutine()
        fetchCompletions()
        // Mock streak for demo
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
        if (isCompleted) return // For now, only allowing check, not uncheck in this simple version

        try {
            await api.checkRoutineItem({
                routine_item_id: itemId,
                date: new Date().toISOString(),
                status: true
            })
            setCompletions(prev => [...prev, itemId])

            // Trigger generic confetti if all items in current view are done
            const currentViewItems = items.filter(i => i.time_of_day === view || i.time_of_day === 'BOTH')
            const completedCount = currentViewItems.filter(i => completions.includes(i.item_id) || i.item_id === itemId).length

            if (completedCount === currentViewItems.length && currentViewItems.length > 0) {
                confetti({
                    particleCount: 100,
                    spread: 70,
                    origin: { y: 0.6 },
                    colors: view === 'AM' ? ['#f59e0b', '#fbbf24', '#ffffff'] : ['#6366f1', '#818cf8', '#ffffff']
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
        <div className="p-6 max-w-5xl mx-auto space-y-8">
            {/* Header with Streak */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
                <div>
                    <h1 className="text-3xl font-serif text-slate-900 font-medium">Daily Routine</h1>
                    <p className="text-slate-500 mt-1">Consistency is key to healthy skin.</p>
                </div>

                <div className="flex items-center gap-4 bg-white px-6 py-3 rounded-2xl shadow-sm border border-slate-100">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full bg-orange-50 flex items-center justify-center text-orange-500">
                            <Flame className="w-5 h-5 fill-current" />
                        </div>
                        <div>
                            <p className="text-xs font-bold text-slate-400 uppercase tracking-wider">Streak</p>
                            <p className="text-xl font-bold text-slate-900">{streak} Days</p>
                        </div>
                    </div>
                    <div className="w-px h-8 bg-slate-100 mx-2"></div>
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full bg-blue-50 flex items-center justify-center text-blue-500">
                            <Droplets className="w-5 h-5 fill-current" />
                        </div>
                        <div>
                            <p className="text-xs font-bold text-slate-400 uppercase tracking-wider">Hydration</p>
                            <p className="text-xl font-bold text-slate-900">Good</p>
                        </div>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

                {/* Main List */}
                <div className="lg:col-span-2 space-y-6">
                    {/* Toggle */}
                    <div className="bg-slate-100 p-1 rounded-xl flex items-center">
                        <button
                            onClick={() => setView('AM')}
                            className={`flex-1 flex items-center justify-center gap-2 py-3 rounded-lg text-sm font-bold transition-all ${view === 'AM' ? 'bg-white text-orange-500 shadow-sm' : 'text-slate-500 hover:text-slate-600'}`}
                        >
                            <Sun className="w-4 h-4" />
                            Morning
                        </button>
                        <button
                            onClick={() => setView('PM')}
                            className={`flex-1 flex items-center justify-center gap-2 py-3 rounded-lg text-sm font-bold transition-all ${view === 'PM' ? 'bg-white text-indigo-500 shadow-sm' : 'text-slate-500 hover:text-slate-600'}`}
                        >
                            <Moon className="w-4 h-4" />
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
                                        className={`group flex items-center justify-between p-4 rounded-xl border transition-all ${isDone ? 'bg-green-50 border-green-100' : 'bg-white border-slate-100 hover:border-slate-200 hover:shadow-sm'}`}
                                    >
                                        <div className="flex items-center gap-4">
                                            <button
                                                onClick={() => toggleComplete(item.item_id)}
                                                className={`w-8 h-8 rounded-full flex items-center justify-center transition-all ${isDone ? 'bg-green-500 text-white scale-110' : 'bg-slate-100 text-slate-300 hover:bg-slate-200'}`}
                                            >
                                                <Check className="w-5 h-5" />
                                            </button>
                                            <div>
                                                <h3 className={`font-medium ${isDone ? 'text-green-800 line-through opacity-70' : 'text-slate-900'}`}>{item.product_name}</h3>
                                                <p className="text-xs text-slate-400 capitalize">{item.time_of_day === 'BOTH' ? 'Any time' : item.time_of_day}</p>
                                            </div>
                                        </div>
                                        <button
                                            onClick={() => deleteItem(item.item_id)}
                                            className="p-2 text-slate-300 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-all"
                                        >
                                            <Trash2 className="w-4 h-4" />
                                        </button>
                                    </motion.div>
                                )
                            })}
                        </AnimatePresence>

                        {/* Add New Button */}
                        {!showAdd ? (
                            <button
                                onClick={() => setShowAdd(true)}
                                className="w-full py-4 border-2 border-dashed border-slate-200 rounded-xl text-slate-400 font-medium hover:border-amber-400 hover:text-amber-500 hover:bg-amber-50 transition-all flex items-center justify-center gap-2"
                            >
                                <Plus className="w-5 h-5" />
                                Add Product
                            </button>
                        ) : (
                            <motion.form
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="bg-slate-50 p-4 rounded-xl border border-slate-200 flex gap-2"
                                onSubmit={addItem}
                            >
                                <input
                                    type="text"
                                    placeholder="Product name (e.g. Vitamin C Serum)"
                                    className="flex-1 rounded-lg border-slate-300 focus:border-amber-500 focus:ring-amber-500/20 text-sm"
                                    value={newItemName}
                                    onChange={e => setNewItemName(e.target.value)}
                                    autoFocus
                                />
                                <select
                                    className="rounded-lg border-slate-300 text-sm"
                                    value={newItemTime}
                                    onChange={e => setNewItemTime(e.target.value)}
                                >
                                    <option value="AM">AM</option>
                                    <option value="PM">PM</option>
                                    <option value="BOTH">Both</option>
                                </select>
                                <button type="submit" className="bg-slate-900 text-white px-4 rounded-lg font-medium text-sm">Add</button>
                                <button type="button" onClick={() => setShowAdd(false)} className="p-2 text-slate-500 hover:bg-slate-200 rounded-lg">
                                    <X className="w-5 h-5" />
                                </button>
                            </motion.form>
                        )}
                    </div>
                </div>

                {/* Sidebar info */}
                <div className="space-y-6">
                    <div className="bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl p-6 text-white shadow-lg overflow-hidden relative">
                        <div className="relative z-10">
                            <h3 className="font-serif text-xl mb-2">Did you know?</h3>
                            <p className="text-white/80 text-sm leading-relaxed">
                                Consistent application of sunscreen is the #1 anti-aging secret. Even on cloudy days, UV rays penetrate the skin.
                            </p>
                        </div>
                        <div className="absolute -bottom-6 -right-6 w-32 h-32 bg-white/10 rounded-full blur-2xl"></div>
                        <div className="absolute top-0 right-0 p-4 opacity-20">
                            <Sun className="w-12 h-12" />
                        </div>
                    </div>

                    <div className="bg-white rounded-2xl p-6 border border-slate-100 shadow-sm">
                        <h3 className="font-medium text-slate-900 mb-4 flex items-center gap-2">
                            <Trophy className="w-4 h-4 text-amber-500" />
                            Achievements
                        </h3>
                        <div className="space-y-4">
                            <div className="flex items-center gap-3 opacity-50">
                                <div className="w-10 h-10 rounded-full bg-slate-100 flex items-center justify-center text-xl">🌱</div>
                                <div>
                                    <p className="text-sm font-medium text-slate-800">7 Day Streak</p>
                                    <div className="w-32 h-1.5 bg-slate-100 rounded-full mt-1 overflow-hidden">
                                        <div className="h-full bg-green-500 w-[60%]"></div>
                                    </div>
                                </div>
                            </div>
                            <div className="flex items-center gap-3">
                                <div className="w-10 h-10 rounded-full bg-amber-50 flex items-center justify-center text-xl">🌟</div>
                                <div>
                                    <p className="text-sm font-medium text-slate-800">Review Routine</p>
                                    <p className="text-xs text-green-600 font-medium">Completed</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

            </div>
        </div>
    )
}
