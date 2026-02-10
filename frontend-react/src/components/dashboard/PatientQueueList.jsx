import { FaClock, FaCircle } from 'react-icons/fa'
import { LuArrowRight } from 'react-icons/lu'
import { Card, CardTitle, CardBadge } from '../Card'

export default function PatientQueue() {
    const queue = [
        { id: 1, name: 'Sarah Jenkins', time: '10:00 AM', type: 'Follow-up', status: 'Waiting' },
        { id: 2, name: 'Michael Chen', time: '10:30 AM', type: 'New Patient', status: 'Arrived' },
        { id: 3, name: 'Emma Wilson', time: '11:00 AM', type: 'Consultation', status: 'Late' },
    ]

    const getStatusColor = (s) => {
        if (s === 'Waiting') return 'text-amber-500'
        if (s === 'Arrived') return 'text-primary-500'
        return 'text-rose-500'
    }

    const getStatusBg = (s) => {
        if (s === 'Waiting') return 'bg-amber-50'
        if (s === 'Arrived') return 'bg-primary-50'
        return 'bg-rose-50'
    }

    return (
        <Card className="h-full p-6" hover={false}>
            <div className="flex justify-between items-center mb-5">
                <CardTitle>Today's Queue</CardTitle>
                <CardBadge variant="default">3 Pending</CardBadge>
            </div>
            <div className="space-y-3">
                {queue.map(p => (
                    <div
                        key={p.id}
                        className="flex items-center justify-between p-3 hover:bg-slate-50 rounded-xl transition-all cursor-pointer group border border-transparent hover:border-slate-200"
                    >
                        <div className="flex items-center gap-3">
                            <div className="h-10 w-10 rounded-xl bg-slate-100 flex items-center justify-center font-bold text-slate-500 text-sm border border-slate-200">
                                {p.name.split(' ').map(n => n[0]).join('')}
                            </div>
                            <div>
                                <div className="font-semibold text-slate-900 text-sm group-hover:text-primary-500 transition-colors">{p.name}</div>
                                <div className="text-xs text-slate-400">{p.type}</div>
                            </div>
                        </div>
                        <div className="text-right">
                            <div className="flex items-center justify-end gap-1.5 text-xs font-medium text-slate-500">
                                <FaClock size={10} className="text-slate-400" /> {p.time}
                            </div>
                            <div className={`text-[10px] font-bold uppercase tracking-wider flex items-center justify-end gap-1 mt-1 ${getStatusColor(p.status)}`}>
                                <FaCircle size={5} /> {p.status}
                            </div>
                        </div>
                    </div>
                ))}
            </div>
            <div className="mt-5 pt-4 border-t border-slate-100 text-center">
                <button className="text-xs font-semibold text-slate-400 hover:text-primary-500 uppercase tracking-widest flex items-center justify-center gap-1 mx-auto transition-colors">
                    View Full Schedule <LuArrowRight size={12} />
                </button>
            </div>
        </Card>
    )
}
