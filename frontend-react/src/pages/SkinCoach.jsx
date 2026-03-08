import React from 'react';
import { motion } from 'framer-motion';
import WeatherWidget from '../components/WeatherWidget';
import ProductSearch from '../components/ProductSearch';
import DailySkinTip from '../components/DailySkinTip';
import IngredientCompat from '../components/IngredientCompat';
import InlineDermChat from '../components/InlineDermChat';
import {
    LuSparkles, LuSearch, LuFlaskConical, LuSun, LuMessageCircle, LuZap
} from 'react-icons/lu';

const container = {
    hidden: { opacity: 0 },
    show: { opacity: 1, transition: { staggerChildren: 0.08, delayChildren: 0.1 } }
}
const item = {
    hidden: { opacity: 0, y: 24 },
    show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: [0.4, 0, 0.2, 1] } }
}

const SkinCoach = () => {
    return (
        <div className="relative min-h-screen pb-20">
            {/* Ambient background glow */}
            <div className="fixed inset-0 pointer-events-none z-0">
                <div className="absolute top-0 left-1/3 w-[600px] h-[600px] bg-violet-500/8 rounded-full blur-[150px] opacity-60" />
                <div className="absolute bottom-1/4 right-1/4 w-[500px] h-[500px] bg-indigo-500/8 rounded-full blur-[120px] opacity-50" />
                <div className="absolute top-1/2 left-0 w-[400px] h-[400px] bg-amber-500/5 rounded-full blur-[100px] opacity-40" />
            </div>

            <motion.div
                variants={container}
                initial="hidden"
                animate="show"
                className="relative z-10 max-w-6xl mx-auto space-y-10"
            >
                {/* Header */}
                <motion.header variants={item} className="text-center space-y-4 pt-4">
                    <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-violet-50 dark:bg-violet-900/30 border border-violet-200 dark:border-violet-700 rounded-full text-xs font-bold text-violet-600 dark:text-violet-400 uppercase tracking-widest">
                        <LuZap size={12} /> AI-Powered
                    </div>
                    <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight">
                        <span className="text-transparent bg-clip-text bg-gradient-to-r from-violet-600 via-indigo-600 to-purple-600 dark:from-violet-400 dark:via-indigo-400 dark:to-purple-400">
                            Your Skin Concierge
                        </span>
                    </h1>
                    <p className="text-lg text-slate-500 dark:text-slate-400 max-w-2xl mx-auto font-light">
                        Intelligent dermatology insights tailored to your unique biology and environment.
                    </p>
                </motion.header>

                {/* === Two-column hero: Chat + Environment === */}
                <div className="grid lg:grid-cols-5 gap-6">
                    {/* Inline AI Chat — takes 3/5 on desktop */}
                    <motion.div variants={item} className="lg:col-span-3">
                        <InlineDermChat />
                    </motion.div>

                    {/* Right column: weather + daily tip stacked */}
                    <motion.div variants={item} className="lg:col-span-2 flex flex-col gap-6">
                        {/* Weather & Environmental Context */}
                        <div className="relative group flex-1">
                            <div className="absolute inset-0 bg-gradient-to-r from-blue-500/15 to-cyan-500/15 rounded-3xl blur-xl transition-all duration-500 group-hover:blur-2xl opacity-70 pointer-events-none" />
                            <div className="relative bg-white/90 dark:bg-slate-800/90 backdrop-blur-xl border border-white/60 dark:border-slate-700 p-6 rounded-3xl shadow-xl h-full">
                                <div className="flex items-center gap-3 mb-4">
                                    <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-blue-400 to-cyan-500 flex items-center justify-center shadow-md shadow-blue-500/20">
                                        <LuSun className="text-white" size={18} />
                                    </div>
                                    <div>
                                        <h2 className="text-base font-bold text-slate-800 dark:text-slate-100">Environmental Analysis</h2>
                                        <p className="text-[11px] text-slate-400 dark:text-slate-500">Real-time UV & humidity impact</p>
                                    </div>
                                </div>
                                <WeatherWidget />
                            </div>
                        </div>

                        {/* Daily Skin Tip */}
                        <div className="flex-1">
                            <DailySkinTip />
                        </div>
                    </motion.div>
                </div>

                {/* === Ingredient Compatibility === */}
                <motion.section variants={item}>
                    <div className="relative group">
                        <div className="absolute inset-0 bg-gradient-to-r from-violet-500/10 to-pink-500/10 rounded-3xl blur-xl opacity-60 pointer-events-none" />
                        <div className="relative">
                            <div className="flex items-center gap-3 mb-4 px-2">
                                <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-violet-500 to-pink-500 flex items-center justify-center shadow-md shadow-violet-500/20">
                                    <LuFlaskConical className="text-white" size={18} />
                                </div>
                                <div>
                                    <h2 className="text-lg font-bold text-slate-800 dark:text-slate-100">Ingredient Compatibility</h2>
                                    <p className="text-xs text-slate-400 dark:text-slate-500">Check if your actives work together</p>
                                </div>
                            </div>
                            <IngredientCompat />
                        </div>
                    </div>
                </motion.section>

                {/* === Product Search + Philosophy === */}
                <motion.section variants={item} className="grid md:grid-cols-5 gap-6">
                    {/* Philosophy card */}
                    <div className="md:col-span-2">
                        <div className="bg-gradient-to-br from-slate-900 to-slate-800 text-white p-8 rounded-3xl shadow-2xl h-full flex flex-col justify-between relative overflow-hidden border border-slate-700">
                            <div className="relative z-10">
                                <div className="inline-flex items-center gap-1.5 px-3 py-1 bg-white/10 rounded-full text-[10px] font-bold tracking-widest uppercase text-amber-300 mb-4">
                                    <LuSparkles size={10} /> Philosophy
                                </div>
                                <h3 className="text-2xl font-bold mb-3 text-transparent bg-clip-text bg-gradient-to-r from-amber-200 to-amber-400">Adaptive Skincare</h3>
                                <p className="text-slate-300 font-light leading-relaxed">
                                    True skincare is adaptive. Our AI analyzes 8,000+ clinical formulations to find the perfect match for your skin barrier today.
                                </p>
                            </div>
                            <div className="relative z-10 mt-8 flex items-center gap-3">
                                <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                                <div className="font-mono text-xs text-emerald-400">Keyword + Tag Search Active</div>
                            </div>
                            {/* Decorative elements */}
                            <div className="absolute top-0 right-0 w-64 h-64 bg-amber-500/10 rounded-full blur-3xl -mr-16 -mt-16 pointer-events-none" />
                            <div className="absolute bottom-0 left-0 w-32 h-32 bg-violet-500/10 rounded-full blur-2xl -ml-8 -mb-8 pointer-events-none" />
                        </div>
                    </div>

                    {/* Product Search */}
                    <div className="md:col-span-3">
                        <div className="bg-white/90 dark:bg-slate-800/90 backdrop-blur-xl border border-slate-100 dark:border-slate-700 rounded-3xl shadow-xl p-6 h-full">
                            <div className="mb-5 flex items-center justify-between">
                                <div className="flex items-center gap-3">
                                    <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-emerald-400 to-teal-500 flex items-center justify-center shadow-md shadow-emerald-500/20">
                                        <LuSearch className="text-white" size={18} />
                                    </div>
                                    <h3 className="text-lg font-bold text-slate-800 dark:text-slate-100">Formulation Search</h3>
                                </div>
                                <span className="text-[10px] font-bold text-emerald-600 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-900/30 px-3 py-1 rounded-full border border-emerald-100 dark:border-emerald-700 uppercase tracking-wider">
                                    AI Agent Active
                                </span>
                            </div>
                            <ProductSearch />
                        </div>
                    </div>
                </motion.section>
            </motion.div>
        </div>
    );
};

export default SkinCoach;
