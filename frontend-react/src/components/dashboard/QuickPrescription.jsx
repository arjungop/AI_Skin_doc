import { useState } from 'react'
import { FaPlus, FaCheck } from 'react-icons/fa'
import { Card, CardTitle } from '../Card'

export default function QuickPrescription() {
    const templates = [
        { id: 1, name: 'Mild Acne', meds: ['Benzoyl Peroxide 2.5%', 'Clindamycin Gel'] },
        { id: 2, name: 'Eczema Flare', meds: ['Hydrocortisone 1%', 'Moisturizer'] },
        { id: 3, name: 'Anti-Aging', meds: ['Tretinoin 0.025%', 'Sunscreen SPF 50'] },
    ]

    const [selected, setSelected] = useState(null)

    return (
        <Card className="h-full p-6" hover={false}>
            <div className="flex justify-between items-center mb-5">
                <CardTitle>Quick Prescriptions</CardTitle>
                <button className="h-8 w-8 rounded-xl bg-slate-100 border border-slate-200 flex items-center justify-center text-slate-400 hover:text-primary-500 hover:bg-primary-50 hover:border-primary-200 transition-all">
                    <FaPlus size={12} />
                </button>
            </div>
            <div className="space-y-3">
                {templates.map(t => (
                    <div
                        key={t.id}
                        className={`p-4 rounded-xl border cursor-pointer transition-all ${selected === t.id
                            ? 'border-primary-300 bg-primary-50'
                            : 'border-slate-200 bg-slate-50 hover:border-slate-300 hover:bg-slate-100'
                            }`}
                        onClick={() => setSelected(t.id)}
                    >
                        <div className="flex justify-between items-center mb-1">
                            <span className="font-semibold text-sm text-slate-900">{t.name}</span>
                            {selected === t.id && <FaCheck className="text-primary-500" size={12} />}
                        </div>
                        <div className="text-xs text-slate-400 leading-relaxed">
                            {t.meds.join(', ')}
                        </div>
                    </div>
                ))}
            </div>
            <button
                className={`w-full mt-5 py-3 rounded-xl font-semibold text-sm transition-all ${selected
                    ? 'btn-primary'
                    : 'bg-slate-100 text-slate-400 cursor-not-allowed border border-slate-200'
                    }`}
                disabled={!selected}
            >
                Apply to Current Patient
            </button>
        </Card>
    )
}
