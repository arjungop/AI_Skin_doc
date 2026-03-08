import React, { useState } from 'react';
import { LuCheck, LuX, LuTriangleAlert, LuInfo } from 'react-icons/lu';

const INGREDIENTS = [
  { id: 'vitc', label: 'Vitamin C', short: 'Vit C' },
  { id: 'niacinamide', label: 'Niacinamide', short: 'Nia' },
  { id: 'retinol', label: 'Retinol', short: 'Ret' },
  { id: 'aha', label: 'AHA (Glycolic)', short: 'AHA' },
  { id: 'bha', label: 'BHA (Salicylic)', short: 'BHA' },
  { id: 'ha', label: 'Hyaluronic Acid', short: 'HA' },
  { id: 'benzoyl', label: 'Benzoyl Peroxide', short: 'BP' },
  { id: 'peptides', label: 'Peptides', short: 'Pep' },
  { id: 'spf', label: 'SPF / Sunscreen', short: 'SPF' },
  { id: 'ceramides', label: 'Ceramides', short: 'Cer' },
];

// 'good' = synergy, 'bad' = conflict/avoid, 'caution' = use with care (separate AM/PM)
// Only upper triangle needed; matrix is symmetric
const COMPAT = {
  'vitc+niacinamide': 'good',
  'vitc+retinol': 'caution',
  'vitc+aha': 'caution',
  'vitc+bha': 'caution',
  'vitc+ha': 'good',
  'vitc+benzoyl': 'bad',
  'vitc+peptides': 'good',
  'vitc+spf': 'good',
  'vitc+ceramides': 'good',
  'niacinamide+retinol': 'good',
  'niacinamide+aha': 'good',
  'niacinamide+bha': 'good',
  'niacinamide+ha': 'good',
  'niacinamide+benzoyl': 'good',
  'niacinamide+peptides': 'good',
  'niacinamide+spf': 'good',
  'niacinamide+ceramides': 'good',
  'retinol+aha': 'bad',
  'retinol+bha': 'caution',
  'retinol+ha': 'good',
  'retinol+benzoyl': 'bad',
  'retinol+peptides': 'good',
  'retinol+spf': 'good',
  'retinol+ceramides': 'good',
  'aha+bha': 'caution',
  'aha+ha': 'good',
  'aha+benzoyl': 'bad',
  'aha+peptides': 'caution',
  'aha+spf': 'good',
  'aha+ceramides': 'good',
  'bha+ha': 'good',
  'bha+benzoyl': 'caution',
  'bha+peptides': 'good',
  'bha+spf': 'good',
  'bha+ceramides': 'good',
  'ha+benzoyl': 'good',
  'ha+peptides': 'good',
  'ha+spf': 'good',
  'ha+ceramides': 'good',
  'benzoyl+peptides': 'bad',
  'benzoyl+spf': 'good',
  'benzoyl+ceramides': 'good',
  'peptides+spf': 'good',
  'peptides+ceramides': 'good',
  'spf+ceramides': 'good',
};

const NOTES = {
  'vitc+retinol': 'Use Vit C in AM, Retinol in PM',
  'vitc+aha': 'Both are acidic — alternate days or separate AM/PM',
  'vitc+bha': 'Can lower pH too much — separate AM/PM',
  'vitc+benzoyl': 'Benzoyl peroxide oxidises Vitamin C, making it ineffective',
  'retinol+aha': 'High irritation risk — never layer together',
  'retinol+bha': 'OK if skin is tolerant; start with alternate nights',
  'retinol+benzoyl': 'BP deactivates retinol — use on different nights',
  'aha+bha': 'Can over-exfoliate — alternate days recommended',
  'aha+benzoyl': 'Too harsh together — high irritation risk',
  'aha+peptides': 'Low pH of AHA can denature peptides — separate routines',
  'bha+benzoyl': 'Possible for oily/acne skin — introduce slowly',
  'benzoyl+peptides': 'BP degrades most peptide bonds',
};

function getCompat(a, b) {
  if (a === b) return null;
  const key1 = `${a}+${b}`;
  const key2 = `${b}+${a}`;
  return COMPAT[key1] || COMPAT[key2] || 'good';
}

function getNote(a, b) {
  const key1 = `${a}+${b}`;
  const key2 = `${b}+${a}`;
  return NOTES[key1] || NOTES[key2] || null;
}

const CELL = {
  good:    'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400',
  caution: 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400',
  bad:     'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400',
  self:    'bg-slate-100 dark:bg-slate-700 text-slate-300 dark:text-slate-500',
};

const ICON = {
  good: LuCheck,
  caution: LuTriangleAlert,
  bad: LuX,
};

const CELL_PX = 42;
const LABEL_PX = 90;

export default function IngredientCompat() {
  const [selected, setSelected] = useState(null);

  function handleCell(a, b) {
    if (a === b) return;
    const c = getCompat(a, b);
    const note = getNote(a, b);
    const aLabel = INGREDIENTS.find(i => i.id === a)?.label;
    const bLabel = INGREDIENTS.find(i => i.id === b)?.label;
    setSelected(prev =>
      prev?.a === aLabel && prev?.b === bLabel ? null : { a: aLabel, b: bLabel, compat: c, note }
    );
  }

  const gridCols = `${LABEL_PX}px repeat(${INGREDIENTS.length}, ${CELL_PX}px)`;

  return (
    <div className="bg-white dark:bg-slate-800 rounded-3xl shadow-lg border border-slate-100 dark:border-slate-700 p-6">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-5">
        <h3 className="text-xl font-serif text-slate-800 dark:text-slate-100">Ingredient Compatibility</h3>
        <div className="flex items-center gap-3 text-xs font-medium">
          <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-emerald-400 shrink-0" />Works great</span>
          <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-amber-400 shrink-0" />Use with care</span>
          <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-red-400 shrink-0" />Avoid together</span>
        </div>
      </div>

      <div className="overflow-x-auto">
        <div style={{ minWidth: LABEL_PX + INGREDIENTS.length * CELL_PX + 8 }}>

          {/* Column headers */}
          <div className="flex" style={{ paddingLeft: LABEL_PX }}>
            {INGREDIENTS.map(ing => (
              <div
                key={ing.id}
                style={{ width: CELL_PX, minWidth: CELL_PX }}
                className="flex items-end justify-center pb-1 text-[10px] font-bold text-slate-500 dark:text-slate-400 leading-tight text-center"
              >
                {ing.short}
              </div>
            ))}
          </div>

          {/* Rows */}
          {INGREDIENTS.map(row => (
            <div key={row.id} className="flex items-center">
              {/* Row label */}
              <div
                style={{ width: LABEL_PX, minWidth: LABEL_PX }}
                className="pr-2 text-right text-[11px] font-semibold text-slate-600 dark:text-slate-300 truncate"
              >
                {row.short}
              </div>

              {/* Cells */}
              {INGREDIENTS.map(col => {
                if (row.id === col.id) {
                  return (
                    <div
                      key={col.id}
                      style={{ width: CELL_PX, minWidth: CELL_PX, height: CELL_PX }}
                      className={`m-0.5 rounded-md ${CELL.self} flex items-center justify-center text-xs`}
                    >
                      —
                    </div>
                  );
                }
                const c = getCompat(row.id, col.id);
                const CellIcon = ICON[c];
                const isActive = selected &&
                  ((selected.a === INGREDIENTS.find(i=>i.id===row.id)?.label && selected.b === INGREDIENTS.find(i=>i.id===col.id)?.label) ||
                   (selected.b === INGREDIENTS.find(i=>i.id===row.id)?.label && selected.a === INGREDIENTS.find(i=>i.id===col.id)?.label));
                return (
                  <button
                    key={col.id}
                    onClick={() => handleCell(row.id, col.id)}
                    style={{ width: CELL_PX, minWidth: CELL_PX, height: CELL_PX }}
                    className={`m-0.5 rounded-md ${CELL[c]} flex items-center justify-center transition-all ${
                      isActive ? 'ring-2 ring-offset-1 ring-slate-400 scale-110 z-10' : 'hover:scale-105 hover:ring-2 hover:ring-offset-1 hover:ring-slate-300'
                    }`}
                    title={`${row.label} + ${col.label}`}
                  >
                    <CellIcon size={13} />
                  </button>
                );
              })}
            </div>
          ))}
        </div>
      </div>

      {/* Detail panel */}
      {selected && (
        <div className={`mt-5 p-4 rounded-xl border flex items-start gap-3 ${
          selected.compat === 'good'    ? 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-700' :
          selected.compat === 'caution' ? 'bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-700' :
                                          'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-700'
        }`}>
          <LuInfo size={16} className="mt-0.5 shrink-0 text-slate-400" />
          <div>
            <p className="font-semibold text-sm text-slate-800 dark:text-slate-100">
              {selected.a} + {selected.b}
              <span className={`ml-2 text-xs font-bold ${
                selected.compat === 'good' ? 'text-emerald-600' : selected.compat === 'caution' ? 'text-amber-600' : 'text-red-600'
              }`}>
                {selected.compat === 'good' ? '✓ Compatible' : selected.compat === 'caution' ? '⚠ Use with care' : '✗ Avoid together'}
              </span>
            </p>
            <p className="text-sm text-slate-600 dark:text-slate-300 mt-1">
              {selected.note || 'These ingredients work well together and can be layered in the same routine.'}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
