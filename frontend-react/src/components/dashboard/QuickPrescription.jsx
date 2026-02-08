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
        <Card variant="glass" className="h-full p-6" hover={false}>
            <div className="flex justify-between items-center mb-5">
                <CardTitle>Quick Prescriptions</CardTitle>
                <button className="h-8 w-8 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center text-text-tertiary hover:text-primary-400 hover:bg-primary-500/10 hover:border-primary-500/30 transition-all">
                    <FaPlus size={12} />
                </button>
            </div>
            <div className="space-y-3">
                {templates.map(t => (
                    <div
                        key={t.id}
                        className={`p-4 rounded-xl border cursor-pointer transition-all ${selected === t.id
                                ? 'border-primary-500/50 bg-primary-500/10'
                                : 'border-white/10 bg-white/5 hover:border-white/20 hover:bg-white/10'
                            }`}
                        onClick={() => setSelected(t.id)}
                    >
                        <div className="flex justify-between items-center mb-1">
                            <span className="font-semibold text-sm text-text-primary">{t.name}</span>
                            {selected === t.id && <FaCheck className="text-primary-400" size={12} />}
                        </div>
                        <div className="text-xs text-text-tertiary leading-relaxed">
                            {t.meds.join(', ')}
                        </div>
                    </div>
                ))}
            </div>
            <button
                className={`w-full mt-5 py-3 rounded-xl font-semibold text-sm transition-all ${selected
                        ? 'btn-primary'
                        : 'bg-white/5 text-text-muted cursor-not-allowed border border-white/10'
                    }`}
                disabled={!selected}
            >
                Apply to Current Patient
            </button>
        </Card>
    )
}
