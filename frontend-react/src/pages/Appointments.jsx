import { useEffect, useState } from 'react'
import { api } from '../services/api'
import { useToast } from '../components/Toast.jsx'
import { FaCalendarPlus, FaUserMd, FaClock, FaCheckCircle, FaTimesCircle, FaCheck, FaExclamationTriangle } from 'react-icons/fa'

export default function Appointments() {
  const [list, setList] = useState([])
  const [doctors, setDoctors] = useState([])
  const [doctorId, setDoctorId] = useState('')
  const [date, setDate] = useState('')
  const [slots, setSlots] = useState([])
  const [selectedSlot, setSelectedSlot] = useState('')
  const [reason, setReason] = useState('')
  const pid = parseInt(localStorage.getItem('patient_id'))
  const did = parseInt(localStorage.getItem('doctor_id'))
  const role = (localStorage.getItem('role') || '').toUpperCase()
  const { push } = useToast()
  const [availability, setAvailability] = useState([])
  const [loading, setLoading] = useState(false)

  useEffect(() => { load() }, [])
  useEffect(() => {
    if (doctorId) api.listDoctorAvailability(doctorId).then(setAvailability).catch(() => setAvailability([]))
  }, [doctorId])

  useEffect(() => {
    if (!date || availability.length === 0) { setSlots([]); return }
    const d = new Date(date)
    const weekday = d.getDay()
    const avail = availability.filter(a => a.weekday === ((weekday + 6) % 7))
    const dur = 30
    const s = []
    for (const a of avail) {
      let [h1, m1] = a.start_time.split(':').map(Number)
      let [h2, m2] = a.end_time.split(':').map(Number)
      let t1 = h1 * 60 + m1, t2 = h2 * 60 + m2
      for (let t = t1; t + dur <= t2; t += dur) {
        const hh = String(Math.floor(t / 60)).padStart(2, '0')
        const mm = String(t % 60).padStart(2, '0')
        s.push(`${hh}:${mm}`)
      }
    }
    setSlots(s)
    setSelectedSlot('')
  }, [date, availability])

  async function load() {
    setLoading(true)
    try {
      let data = await api.listAppointments()
      if (role === 'DOCTOR' && did) data = data.filter(a => a.doctor_id === did)
      else if (pid) data = data.filter(a => a.patient_id === pid)
      setList(data)
      const docs = await api.listDoctors()
      setDoctors(docs)
    } catch { }
    setLoading(false)
  }

  async function submit(e) {
    e.preventDefault()
    if (!pid) return push('Login required', 'error')
    if (!selectedSlot || !date) return push('Pick a date and slot', 'error')
    try {
      const iso = new Date(`${date}T${selectedSlot}:00`).toISOString()
      await api.createAppointment({ patient_id: pid, doctor_id: parseInt(doctorId), appointment_date: iso, reason: reason || undefined })
      push('Appointment booked', 'success')
      setDoctorId(''); setDate(''); setReason(''); load()
    } catch (err) { push(err.message || 'Booking failed', 'error') }
  }

  return (
    <div className="space-y-8 animate-fade-in pb-12">
      <header className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-serif text-slate-800 dark:text-white">Appointments</h1>
          <p className="text-slate-500">Manage your consultations and checkups.</p>
        </div>
        {role === 'PATIENT' && (
          <button className="btn-primary flex items-center gap-2 shadow-lg shadow-blue-900/20" onClick={() => document.getElementById('book-form').scrollIntoView({ behavior: 'smooth' })}>
            <FaCalendarPlus />
            New Booking
          </button>
        )}
      </header>

      {/* Booking Form Card */}
      {role === 'PATIENT' && (
        <section id="book-form" className="bg-white dark:bg-slate-800 rounded-3xl p-8 shadow-lg border border-slate-100 dark:border-slate-700 relative overflow-hidden">
          <div className="absolute top-0 right-0 w-32 h-32 bg-blue-50 dark:bg-blue-900/20 rounded-full blur-3xl -mr-10 -mt-10"></div>

          <h2 className="text-xl font-serif text-slate-800 dark:text-white mb-6 relative z-10">Book a Consultation</h2>
          <form onSubmit={submit} className="relative z-10 space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <div className="space-y-2">
                <label className="text-sm font-semibold text-slate-500 uppercase tracking-wider">Select Doctor</label>
                <div className="relative">
                  <FaUserMd className="absolute left-4 top-4 text-slate-400" />
                  <select className="pl-12 bg-slate-50 border-slate-200 focus:ring-blue-500 rounded-xl" value={doctorId} onChange={e => setDoctorId(e.target.value)} required>
                    <option value="" disabled>Choose a specialist...</option>
                    {doctors.map(d => (
                      <option key={d.doctor_id} value={d.doctor_id}>{d.username} {d.specialization ? `(${d.specialization})` : ''}</option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-semibold text-slate-500 uppercase tracking-wider">Date</label>
                <input className="bg-slate-50 border-slate-200 focus:ring-blue-500 rounded-xl" type="date" value={date} onChange={e => setDate(e.target.value)} required />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-semibold text-slate-500 uppercase tracking-wider">Reason</label>
                <input className="bg-slate-50 border-slate-200 focus:ring-blue-500 rounded-xl" placeholder="e.g. Skin rash, Checkup..." value={reason} onChange={e => setReason(e.target.value)} />
              </div>
            </div>

            {/* Slots Grid */}
            <div className="bg-slate-50 dark:bg-slate-900/50 rounded-2xl p-6">
              <label className="text-sm font-semibold text-slate-500 uppercase tracking-wider block mb-4">Available Slots</label>
              {date && slots.length > 0 ? (
                <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-3">
                  {slots.map(s => (
                    <button
                      type="button"
                      key={s}
                      onClick={() => setSelectedSlot(s)}
                      className={`py-2 px-3 rounded-lg text-sm font-medium transition-all duration-200 flex items-center justify-center gap-1
                        ${selectedSlot === s
                          ? 'bg-blue-600 text-white shadow-md scale-105'
                          : 'bg-white border border-slate-200 text-slate-600 hover:border-blue-300 hover:text-blue-600'}`}
                    >
                      <FaClock className="text-xs opacity-70" />
                      {s}
                    </button>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-slate-400 flex flex-col items-center">
                  <FaExclamationTriangle className="mb-2 text-2xl opacity-20" />
                  {date ? 'No slots available for this date.' : 'Please select a doctor and date first.'}
                </div>
              )}
            </div>

            <div className="flex justify-end">
              <button className="btn-primary px-8 py-3 text-lg shadow-xl shadow-blue-900/10" type="submit">Confirm Booking</button>
            </div>
          </form>
        </section>
      )}

      {/* Appointment List */}
      <section className="bg-white dark:bg-slate-800 rounded-3xl shadow-sm border border-slate-100 dark:border-slate-700 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="bg-slate-50/50 text-left border-b border-slate-100">
                <th className="p-6 font-serif text-slate-800">Doctor</th>
                <th className="p-6 font-serif text-slate-800">Date & Time</th>
                <th className="p-6 font-serif text-slate-800">Reason</th>
                <th className="p-6 font-serif text-slate-800">Status</th>
                <th className="p-6 font-serif text-slate-800 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-50">
              {loading && (
                <tr>
                  <td colSpan="5" className="p-12 text-center text-slate-400">Loading appointments...</td>
                </tr>
              )}
              {!loading && list.length === 0 && (
                <tr>
                  <td colSpan="5" className="p-12 text-center text-slate-400">No appointments found.</td>
                </tr>
              )}
              {list.map(a => {
                const d = doctors.find(x => x.doctor_id === a.doctor_id)
                const isPast = new Date(a.appointment_date) < new Date()
                return (
                  <tr key={a.appointment_id} className="hover:bg-slate-50/50 transition-colors">
                    <td className="p-6">
                      <div className="font-semibold text-slate-900">{d?.username || `Dr. #${a.doctor_id}`}</div>
                      <div className="text-xs text-slate-500">{d?.specialization || 'General'}</div>
                    </td>
                    <td className="p-6 text-slate-600 font-medium">
                      {new Date(a.appointment_date).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })}
                    </td>
                    <td className="p-6 text-slate-600 max-w-xs truncate" title={a.reason}>{a.reason || 'Routine Checkup'}</td>
                    <td className="p-6">
                      <StatusBadge status={a.status} />
                    </td>
                    <td className="p-6 text-right space-x-2">
                      {role === 'PATIENT' && ['Scheduled', 'Confirmed'].includes(a.status) && !isPast && (
                        <button className="px-3 py-1.5 text-xs font-semibold text-red-600 bg-red-50 hover:bg-red-100 rounded-lg transition-colors border border-red-100" onClick={async () => { await api.updateAppointmentStatus(a.appointment_id, 'Cancelled'); load() }}>
                          Cancel
                        </button>
                      )}

                      {role === 'DOCTOR' && (
                        <>
                          {a.status === 'Scheduled' && (
                            <button className="btn-ghost text-xs px-3" onClick={async () => { await api.updateAppointmentStatus(a.appointment_id, 'Confirmed'); load() }}>Confirm</button>
                          )}
                          {['Scheduled', 'Confirmed'].includes(a.status) && (
                            <button className="btn-ghost text-xs text-red-500 hover:bg-red-50 px-3" onClick={async () => { await api.updateAppointmentStatus(a.appointment_id, 'Cancelled'); load() }}>Cancel</button>
                          )}
                          {['Scheduled', 'Confirmed'].includes(a.status) && (
                            <button className="btn-primary text-xs px-3 py-1.5" onClick={async () => { await api.updateAppointmentStatus(a.appointment_id, 'Completed'); load() }}>Complete</button>
                          )}
                        </>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}

function StatusBadge({ status }) {
  const styles = {
    Scheduled: 'bg-blue-50 text-blue-700 border-blue-100',
    Confirmed: 'bg-emerald-50 text-emerald-700 border-emerald-100',
    Cancelled: 'bg-slate-100 text-slate-600 border-slate-200',
    Completed: 'bg-purple-50 text-purple-700 border-purple-100',
  }
  const icon = {
    Confirmed: <FaCheckCircle className="mr-1.5" />,
    Cancelled: <FaTimesCircle className="mr-1.5" />,
    Completed: <FaCheck className="mr-1.5" />
  }
  return (
    <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-bold border ${styles[status] || 'bg-gray-100 text-gray-600'}`}>
      {icon[status]}
      {status}
    </span>
  )
}
