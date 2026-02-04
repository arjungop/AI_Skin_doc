import { useState } from 'react';
import { FaPlus, FaCheck } from 'react-icons/fa';

export default function QuickPrescription() {
    const templates = [
        { id: 1, name: 'Mild Acne', meds: ['Benzoyl Peroxide 2.5%', 'Clindamycin Gel'] },
        { id: 2, name: 'Eczema Flare', meds: ['Hydrocortisone 1%', 'Moisturizer'] },
        { id: 3, name: 'Anti-Aging', meds: ['Tretinoin 0.025%', 'Sunscreen SPF 50'] },
    ];

    const [selected, setSelected] = useState(null);

    return (
        <div className="bg-white p-6 rounded-3xl border border-slate-100 shadow-sm h-full">
            <div className="flex justify-between items-center mb-4">
                <h3 className="font-bold text-slate-900">Quick Prescriptions</h3>
                <button className="h-8 w-8 rounded-full bg-slate-100 flex items-center justify-center text-slate-600 hover:bg-slate-200">
                    <FaPlus size={12} />
                </button>
            </div>
            <div className="space-y-3">
                {templates.map(t => (
                    <div
                        key={t.id}
                        className={`p-4 rounded-xl border cursor-pointer transition-all ${selected === t.id ? 'border-emerald-500 bg-emerald-50' : 'border-slate-100 hover:border-slate-300'}`}
                        onClick={() => setSelected(t.id)}
                    >
                        <div className="flex justify-between items-center mb-1">
                            <span className="font-bold text-sm text-slate-800">{t.name}</span>
                            {selected === t.id && <FaCheck className="text-emerald-500" size={12} />}
                        </div>
                        <div className="text-xs text-slate-500 leading-relaxed">
                            {t.meds.join(', ')}
                        </div>
                    </div>
                ))}
            </div>
            <button className="w-full mt-4 py-3 bg-slate-900 text-white rounded-xl font-bold text-sm hover:bg-slate-800 disabled:opacity-50" disabled={!selected}>
                Apply to Current Patient
            </button>
        </div>
    );
}
