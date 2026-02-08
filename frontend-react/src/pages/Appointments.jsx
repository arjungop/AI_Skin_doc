import { useEffect, useState } from 'react'
import { api } from '../services/api'
import { useToast } from '../components/Toast.jsx'
import { motion } from 'framer-motion'
import { LuCalendarPlus, LuClock, LuCheck, LuX, LuArrowRight, LuTriangleAlert, LuUser } from 'react-icons/lu'
import { Card, CardTitle, CardDescription, IconWrapper, CardBadge } from '../components/Card'

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

  const container = {
    hidden: { opacity: 0 },
    show: { opacity: 1, transition: { staggerChildren: 0.08 } }
  }

  const item = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { duration: 0.4 } }
  }

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
    <div className="relative min-h-screen">
      {/* Ambient Background */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute top-20 right-1/4 w-[500px] h-[500px] bg-primary-500/10 rounded-full blur-[120px] opacity-40" />
        <div className="absolute bottom-1/3 left-1/3 w-[400px] h-[400px] bg-accent-500/10 rounded-full blur-[100px] opacity-30" />
      </div>

      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="relative z-10 space-y-8 pb-12"
      >
        {/* Header */}
        <motion.header variants={item} className="flex flex-col md:flex-row md:items-center justify-between gap-4 lg:pr-72">
          <div>
            <h1 className="text-4xl md:text-5xl font-bold text-text-primary tracking-tight">Appointments</h1>
            <p className="text-text-tertiary mt-2 text-lg">Manage your consultations and checkups</p>
          </div>
          {role === 'PATIENT' && (
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="btn-primary flex items-center gap-2"
              onClick={() => document.getElementById('book-form').scrollIntoView({ behavior: 'smooth' })}
            >
              <LuCalendarPlus size={18} /> New Booking
            </motion.button>
          )}
        </motion.header>

        {/* Booking Form Card */}
        {role === 'PATIENT' && (
          <motion.section variants={item} id="book-form">
            <Card variant="glass" className="p-8" hover={false}>
              <div className="flex items-center gap-3 mb-6">
                <IconWrapper variant="primary">
                  <LuCalendarPlus size={20} />
                </IconWrapper>
                <CardTitle className="text-xl">Book a Consultation</CardTitle>
              </div>

              <form onSubmit={submit} className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  <div className="space-y-2">
                    <label className="text-sm font-semibold text-text-tertiary uppercase tracking-wider">Select Doctor</label>
                    <select
                      className="w-full bg-surface-elevated border border-white/10 rounded-xl px-4 py-3 text-text-primary focus:border-primary-500/50 focus:ring-2 focus:ring-primary-500/20 transition-all"
                      value={doctorId}
                      onChange={e => setDoctorId(e.target.value)}
                      required
                    >
                      <option value="" disabled>Choose a specialist...</option>
                      {doctors.map(d => (
                        <option key={d.doctor_id} value={d.doctor_id}>{d.username} {d.specialization ? `(${d.specialization})` : ''}</option>
                      ))}
                    </select>
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-semibold text-text-tertiary uppercase tracking-wider">Date</label>
                    <input
                      className="w-full bg-surface-elevated border border-white/10 rounded-xl px-4 py-3 text-text-primary focus:border-primary-500/50 focus:ring-2 focus:ring-primary-500/20 transition-all"
                      type="date"
                      value={date}
                      onChange={e => setDate(e.target.value)}
                      required
                    />
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-semibold text-text-tertiary uppercase tracking-wider">Reason</label>
                    <input
                      className="w-full bg-surface-elevated border border-white/10 rounded-xl px-4 py-3 text-text-primary placeholder:text-text-muted focus:border-primary-500/50 focus:ring-2 focus:ring-primary-500/20 transition-all"
                      placeholder="e.g. Skin rash, Checkup..."
                      value={reason}
                      onChange={e => setReason(e.target.value)}
                    />
                  </div>
                </div>

                {/* Slots Grid */}
                <div className="bg-surface-elevated/50 rounded-2xl p-6 border border-white/5">
                  <label className="text-sm font-semibold text-text-tertiary uppercase tracking-wider block mb-4">Available Slots</label>
                  {date && slots.length > 0 ? (
                    <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-3">
                      {slots.map(s => (
                        <button
                          type="button"
                          key={s}
                          onClick={() => setSelectedSlot(s)}
                          className={`py-2.5 px-3 rounded-xl text-sm font-medium transition-all flex items-center justify-center gap-1.5
                            ${selectedSlot === s
                              ? 'bg-primary-500 text-white shadow-lg shadow-primary-500/30 scale-105'
                              : 'bg-white/5 border border-white/10 text-text-secondary hover:border-primary-500/30 hover:text-primary-400'}`}
                        >
                          <LuClock size={12} className="opacity-70" />
                          {s}
                        </button>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8 text-text-muted flex flex-col items-center">
                      <LuTriangleAlert className="mb-2 text-2xl opacity-40" />
                      {date ? 'No slots available for this date.' : 'Please select a doctor and date first.'}
                    </div>
                  )}
                </div>

                <div className="flex justify-end">
                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className="btn-primary px-8 py-3 text-lg"
                    type="submit"
                  >
                    Confirm Booking <LuArrowRight className="inline ml-2" size={18} />
                  </motion.button>
                </div>
              </form>
            </Card>
          </motion.section>
        )}

        {/* Appointment List */}
        <motion.section variants={item}>
          <Card variant="glass" className="p-0 overflow-hidden" hover={false}>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="bg-white/5 text-left border-b border-white/10">
                    <th className="p-6 font-semibold text-text-secondary">Doctor</th>
                    <th className="p-6 font-semibold text-text-secondary">Date & Time</th>
                    <th className="p-6 font-semibold text-text-secondary">Reason</th>
                    <th className="p-6 font-semibold text-text-secondary">Status</th>
                    <th className="p-6 font-semibold text-text-secondary text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {loading && (
                    <tr>
                      <td colSpan="5" className="p-12 text-center text-text-muted">Loading appointments...</td>
                    </tr>
                  )}
                  {!loading && list.length === 0 && (
                    <tr>
                      <td colSpan="5" className="p-12 text-center text-text-muted">No appointments found.</td>
                    </tr>
                  )}
                  {list.map(a => {
                    const d = doctors.find(x => x.doctor_id === a.doctor_id)
                    const isPast = new Date(a.appointment_date) < new Date()
                    return (
                      <tr key={a.appointment_id} className="hover:bg-white/5 transition-colors">
                        <td className="p-6">
                          <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-primary-500/10 flex items-center justify-center">
                              <LuUser size={18} className="text-primary-400" />
                            </div>
                            <div>
                              <div className="font-semibold text-text-primary">{d?.username || `Dr. #${a.doctor_id}`}</div>
                              <div className="text-xs text-text-tertiary">{d?.specialization || 'General'}</div>
                            </div>
                          </div>
                        </td>
                        <td className="p-6 text-text-secondary font-medium">
                          {new Date(a.appointment_date).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })}
                        </td>
                        <td className="p-6 text-text-secondary max-w-xs truncate" title={a.reason}>{a.reason || 'Routine Checkup'}</td>
                        <td className="p-6">
                          <StatusBadge status={a.status} />
                        </td>
                        <td className="p-6 text-right space-x-2">
                          {role === 'PATIENT' && ['Scheduled', 'Confirmed'].includes(a.status) && !isPast && (
                            <button className="px-3 py-1.5 text-xs font-semibold text-danger bg-danger/10 hover:bg-danger/20 rounded-lg transition-colors border border-danger/20" onClick={async () => { await api.updateAppointmentStatus(a.appointment_id, 'Cancelled'); load() }}>
                              Cancel
                            </button>
                          )}

                          {role === 'DOCTOR' && (
                            <>
                              {a.status === 'Scheduled' && (
                                <button className="btn-ghost text-xs px-3" onClick={async () => { await api.updateAppointmentStatus(a.appointment_id, 'Confirmed'); load() }}>Confirm</button>
                              )}
                              {['Scheduled', 'Confirmed'].includes(a.status) && (
                                <button className="btn-ghost text-xs text-danger hover:bg-danger/10 px-3" onClick={async () => { await api.updateAppointmentStatus(a.appointment_id, 'Cancelled'); load() }}>Cancel</button>
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
          </Card>
        </motion.section>
      </motion.div>
    </div>
  )
}

function StatusBadge({ status }) {
  const styles = {
    Scheduled: 'bg-ai-500/10 text-ai-400 border-ai-500/20',
    Confirmed: 'bg-primary-500/10 text-primary-400 border-primary-500/20',
    Cancelled: 'bg-white/5 text-text-muted border-white/10',
    Completed: 'bg-accent-500/10 text-accent-400 border-accent-500/20',
  }
  const icon = {
    Confirmed: <LuCheck className="mr-1.5" size={12} />,
    Cancelled: <LuX className="mr-1.5" size={12} />,
    Completed: <LuCheck className="mr-1.5" size={12} />
  }
  return (
    <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-bold border ${styles[status] || 'bg-white/5 text-text-muted'}`}>
      {icon[status]}
      {status}
    </span>
  )
}
