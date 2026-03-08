import React, { useState, useEffect, useMemo } from 'react';
import { LuSun, LuMoon, LuSunrise, LuDroplets, LuShield, LuSparkles, LuRefreshCw } from 'react-icons/lu';

const TIPS = {
  morning: [
    { title: 'Vitamin C First', body: 'Apply Vitamin C serum on clean skin before moisturiser — it neutralises free radicals from UV before you step outside.', icon: LuSun, source: 'J Clin Aesthet Dermatol, 2017' },
    { title: 'SPF Is Non-Negotiable', body: 'Even on cloudy days, up to 80% of UV rays penetrate clouds. Apply a broad-spectrum SPF 30+ as the last step of your morning routine.', icon: LuShield, source: 'AAD Guidelines' },
    { title: 'Hydrate Before You Seal', body: 'Apply hydrating toner or hyaluronic acid on damp skin, then lock it with moisturiser. HA pulls moisture from the environment into your skin.', icon: LuDroplets, source: 'Br J Dermatol, 2019' },
    { title: 'Gentle Morning Cleanse', body: 'A water-only or micellar rinse in the morning preserves your skin\'s overnight repair lipids. Save foaming cleansers for PM.', icon: LuSunrise, source: 'Int J Dermatol, 2020' },
    { title: 'Antioxidant Layering', body: 'Pair Vitamin C with Vitamin E and ferulic acid for 8× better photoprotection than Vitamin C alone.', icon: LuSparkles, source: 'J Invest Dermatol, 2005' },
  ],
  afternoon: [
    { title: 'Reapply Sunscreen', body: 'Sunscreen degrades after ~2 hours of sun exposure. Keep a SPF powder or mist for midday touch-ups over makeup.', icon: LuSun, source: 'AAD Guidelines' },
    { title: 'Blotting Over Washing', body: 'Midday oiliness? Use blotting paper instead of washing — over-cleansing strips barrier lipids and triggers more oil production.', icon: LuDroplets, source: 'Dermatol Ther, 2018' },
    { title: 'Hydration From Within', body: 'Skin loses 300-400ml of water daily through transepidermal water loss. Drinking water won\'t fix dry skin, but dehydration makes it worse.', icon: LuDroplets, source: 'Clin Dermatol, 2010' },
    { title: 'Hands Off Your Face', body: 'Touching transfers bacteria and oils from your hands. This is one of the top triggers for perioral and jawline breakouts.', icon: LuShield, source: 'Am J Infect Control, 2015' },
  ],
  evening: [
    { title: 'Double Cleanse at Night', body: 'Oil-based cleanser first to dissolve SPF and makeup, then water-based cleanser to clear pores. This is the #1 routine upgrade for clearer skin.', icon: LuMoon, source: 'J Cosmet Dermatol, 2019' },
    { title: 'Retinoid Window', body: 'Apply retinoids 20 min after cleansing on fully dry skin to reduce irritation. Start 2×/week and build up — consistency beats concentration.', icon: LuSparkles, source: 'JAAD, 2020' },
    { title: 'Niacinamide for Barrier', body: 'Niacinamide (Vitamin B3) at 4-5% strengthens the skin barrier, reduces redness, and regulates sebum. Best used in your PM routine.', icon: LuShield, source: 'Dermatol Ther, 2014' },
    { title: 'Overnight Repair Peak', body: 'Skin cell turnover peaks between 11pm-4am. Applying actives before bed maximises absorption during this natural repair window.', icon: LuMoon, source: 'J Invest Dermatol, 2001' },
    { title: 'Slug at Night', body: 'Sealing your routine with a thin layer of petroleum jelly (slugging) reduces transepidermal water loss by 98% while you sleep.', icon: LuDroplets, source: 'Acta Derm Venereol, 2016' },
  ],
};

function getTimeOfDay() {
  const h = new Date().getHours();
  if (h >= 5 && h < 12) return 'morning';
  if (h >= 12 && h < 18) return 'afternoon';
  return 'evening';
}

const LABELS = { morning: 'Morning', afternoon: 'Afternoon', evening: 'Evening' };
const COLORS = {
  morning: 'from-amber-500/20 to-orange-500/20',
  afternoon: 'from-sky-500/20 to-blue-500/20',
  evening: 'from-indigo-500/20 to-purple-500/20',
};

export default function DailySkinTip() {
  const tod = getTimeOfDay();
  const pool = TIPS[tod];

  // Pick a tip deterministically from date + rotate on shuffle
  const dayIndex = useMemo(() => {
    const d = new Date();
    return (d.getFullYear() * 366 + d.getMonth() * 31 + d.getDate()) % pool.length;
  }, [pool.length]);

  const [index, setIndex] = useState(dayIndex);
  const [fading, setFading] = useState(false);

  const tip = pool[index % pool.length];
  const Icon = tip.icon;

  function shuffle() {
    setFading(true);
    setTimeout(() => {
      setIndex(i => (i + 1) % pool.length);
      setFading(false);
    }, 200);
  }

  return (
    <div className="relative group">
      <div className={`absolute inset-0 bg-gradient-to-r ${COLORS[tod]} rounded-3xl blur-xl opacity-60 transition-all group-hover:blur-2xl`} />
      <div className="relative bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl border border-white/50 dark:border-slate-700 p-8 rounded-3xl shadow-xl">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500/20 to-primary-600/20 flex items-center justify-center">
              <Icon className="text-primary-600 dark:text-primary-400" size={20} />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-slate-800 dark:text-slate-100">{LABELS[tod]} Tip</h3>
              <p className="text-xs text-slate-400">Evidence-based • rotates daily</p>
            </div>
          </div>
          <button onClick={shuffle} className="p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors text-slate-400 hover:text-slate-600" title="Next tip">
            <LuRefreshCw size={16} />
          </button>
        </div>
        <div className={`transition-opacity duration-200 ${fading ? 'opacity-0' : 'opacity-100'}`}>
          <h4 className="text-xl font-bold text-slate-900 dark:text-white mb-2">{tip.title}</h4>
          <p className="text-slate-600 dark:text-slate-300 leading-relaxed text-sm">{tip.body}</p>
          <p className="text-xs text-slate-400 mt-3 italic">Source: {tip.source}</p>
        </div>
      </div>
    </div>
  );
}
