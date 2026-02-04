import { FaClock, FaCircle } from 'react-icons/fa';

export default function PatientQueue() {
    const queue = [
        { id: 1, name: 'Sarah Jenkins', time: '10:00 AM', type: 'Follow-up', status: 'Waiting' },
        { id: 2, name: 'Michael Chen', time: '10:30 AM', type: 'New Patient', status: 'Arrived' },
        { id: 3, name: 'Emma Wilson', time: '11:00 AM', type: 'Consultation', status: 'Late' },
    ];

    const getStatusColor = (s) => {
        if (s === 'Waiting') return 'text-orange-500';
        if (s === 'Arrived') return 'text-emerald-500';
        return 'text-red-500';
    }

    return (
        <div className="bg-white p-6 rounded-3xl border border-slate-100 shadow-sm h-full">
            <div className="flex justify-between items-center mb-4">
                <h3 className="font-bold text-slate-900">Today's Queue</h3>
                <span className="px-3 py-1 bg-slate-100 rounded-full text-xs font-bold text-slate-600">3 Pending</span>
            </div>
            <div className="space-y-4">
                {queue.map(p => (
                    <div key={p.id} className="flex items-center justify-between p-3 hover:bg-slate-50 rounded-xl transition-colors cursor-pointer group">
                        <div className="flex items-center gap-3">
                            <div className="h-10 w-10 rounded-full bg-slate-200 flex items-center justify-center font-bold text-slate-600 text-sm">
                                {p.name.split(' ').map(n => n[0]).join('')}
                            </div>
                            <div>
                                <div className="font-bold text-slate-900 text-sm group-hover:text-blue-600 transition-colors">{p.name}</div>
                                <div className="text-xs text-slate-500">{p.type}</div>
                            </div>
                        </div>
                        <div className="text-right">
                            <div className="flex items-center justify-end gap-1 text-xs font-bold text-slate-700">
                                <FaClock size={10} className="text-slate-400" /> {p.time}
                            </div>
                            <div className={`text-[10px] font-bold uppercase tracking-wider flex items-center justify-end gap-1 ${getStatusColor(p.status)}`}>
                                <FaCircle size={6} /> {p.status}
                            </div>
                        </div>
                    </div>
                ))}
            </div>
            <div className="mt-4 pt-4 border-t border-slate-100 text-center">
                <button className="text-xs font-bold text-blue-600 hover:text-blue-700 uppercase tracking-widest">View Full Schedule</button>
            </div>
        </div>
    );
}
