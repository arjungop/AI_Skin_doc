import React from 'react';
import WeatherWidget from '../components/WeatherWidget';
import ProductSearch from '../components/ProductSearch';

const SkinCoach = () => {
    return (
        <div className="max-w-5xl mx-auto space-y-12 animate-fade-in pb-20">
            {/* Header Section */}
            <header className="text-center space-y-4">
                <h1 className="text-4xl md:text-5xl font-serif text-transparent bg-clip-text bg-gradient-to-r from-accentGold to-amber-600 dark:from-yellow-200 dark:to-amber-500 tracking-tight drop-shadow-sm">
                    Your Personal Skin Concierge
                </h1>
                <p className="text-lg text-textLuxuryMuted max-w-2xl mx-auto font-light">
                    Intelligent dermatology insights tailored to your unique biology and environment.
                </p>
            </header>

            {/* Weather & Context Card */}
            <section className="relative group">
                <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-3xl blur-xl transition-all duration-500 group-hover:blur-2xl opacity-70"></div>
                <div className="relative bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl border border-white/50 dark:border-slate-700 p-8 rounded-3xl shadow-xl">
                    <div className="flex flex-col md:flex-row items-center justify-between gap-8">
                        <div className="flex-1 space-y-2">
                            <h2 className="text-2xl font-serif text-slate-800 dark:text-slate-100">Daily Environmental Analysis</h2>
                            <p className="text-slate-500 dark:text-slate-400 text-sm">Real-time UV & Humidity impact assessment.</p>
                        </div>
                        <div className="w-full md:w-auto">
                            <WeatherWidget />
                        </div>
                    </div>
                </div>
            </section>

            {/* Routine / Product AI Section */}
            <section className="grid md:grid-cols-5 gap-8">
                <div className="md:col-span-2 space-y-6">
                    <div className="bg-gradient-to-br from-slate-900 to-slate-800 text-white p-8 rounded-3xl shadow-2xl h-full flex flex-col justify-between relative overflow-hidden border border-slate-700">
                        <div className="relative z-10">
                            <h3 className="text-2xl font-serif mb-2 text-accentGold">The Philosophy</h3>
                            <p className="text-slate-300 font-light leading-relaxed">
                                True skincare is adaptive. Our AI analyzes 8,000+ clinical formulations to find the perfect match for your skin barrier today.
                            </p>
                        </div>
                        <div className="relative z-10 mt-8">
                            <div className="text-xs uppercase tracking-widest text-slate-500 mb-1">Powered By</div>
                            <div className="font-mono text-sm text-emerald-400">Vector Embeddings v2.1</div>
                        </div>
                        {/* Decor */}
                        <div className="absolute top-0 right-0 w-64 h-64 bg-accentGold/10 rounded-full blur-3xl -mr-16 -mt-16 pointer-events-none"></div>
                    </div>
                </div>

                <div className="md:col-span-3">
                    <div className="bg-white dark:bg-slate-800 rounded-3xl shadow-lg border border-slate-100 dark:border-slate-700 p-8 h-full">
                        <div className="mb-6 flex items-baseline justify-between">
                            <h3 className="text-2xl font-serif text-slate-800 dark:text-slate-100">Active Formulation Search</h3>
                            <span className="text-xs font-bold text-emerald-600 bg-emerald-50 px-3 py-1 rounded-full border border-emerald-100">AI AGENT ACTIVE</span>
                        </div>
                        <ProductSearch />
                    </div>
                </div>
            </section>
        </div>
    );
};

export default SkinCoach;
