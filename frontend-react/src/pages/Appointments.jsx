import { useEffect, useState } from 'react'
import { api } from '../services/api'
import { useToast } from '../components/Toast.jsx'
import { motion } from 'framer-motion'
import { LuCalendarPlus, LuClock, LuCheck, LuX, LuArrowRight, LuTriangleAlert, LuUser } from 'react-icons/lu'
import { Card, CardTitle, CardDescription, IconWrapper, CardBadge } from '../components/Card'
import ConfirmModal from '../components/ConfirmModal.jsx'

export default function Appointments() {
  const [list, setList] = useState([])
  const [doctors, setDoctors] = useState([])
  const [doctorId, setDoctorId] = useState('')
  const [date, setDate] = useState('')
  const [slots, setSlots] = useState([])
  const [bookedSlots, setBookedSlots] = useState([])
  const [selectedSlot, setSelectedSlot] = useState('')
  const [reason, setReason] = useState('')
  const pid = parseInt(localStorage.getItem('patient_id'))
  const did = parseInt(localStorage.getItem('doctor_id'))
  const role = (localStorage.getItem('role') || '').toUpperCase()
  const { push } = useToast()
  const [availability, setAvailability] = useState([])
  const [loading, setLoading] = useState(false)
  const [cancelTarget, setCancelTarget] = useState(null)

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
    if (!date || availability.length === 0) { setSlots([]); setBookedSlots([]); return }
    const d = new Date(date)
    const weekday = d.getDay()
    const avail = availability.filter(a => a.weekday === ((weekday + 6) % 7))
    const dur = 30
    const s = []
    const now = new Date()
    const isToday = d.toDateString() === now.toDateString()

    for (const a of avail) {
      let [h1, m1] = a.start_time.split(':').map(Number)
      let [h2, m2] = a.end_time.split(':').map(Number)
      let t1 = h1 * 60 + m1, t2 = h2 * 60 + m2
      for (let t = t1; t + dur <= t2; t += dur) {
        const hh = String(Math.floor(t / 60)).padStart(2, '0')
        const mm = String(t % 60).padStart(2, '0')

        // Skip past slots if booking today
        if (isToday) {
          const slotTime = new Date(now)
          slotTime.setHours(parseInt(hh, 10), parseInt(mm, 10), 0, 0)
          if (slotTime < now) continue
        }

        s.push(`${hh}:${mm}`)
      }
    }
    setSlots(s)
    setSelectedSlot('')
    if (doctorId && date) {
      api.listBookedSlots(doctorId, date).then(b => setBookedSlots(Array.isArray(b) ? b : [])).catch(() => setBookedSlots([]))
    }
  }, [date, availability, doctorId])

  async function load() {
    setLoading(true)
    try {
      let data = await api.listAppointments()
      if (pid) data = data.filter(a => a.patient_id === pid)
      setList(data || [])
      const docs = await api.listDoctors()
      setDoctors(docs || [])
    } catch (err) {
      push(err.message || 'Failed to load appointments', 'error')
      setList([])
      setDoctors([])
    }
    setLoading(false)
  }

  const [submitting, setSubmitting] = useState(false)

  async function submit(e) {
    e.preventDefault()
    if (!pid) return push('Login required', 'error')
    if (!selectedSlot || !date) return push('Pick a date and slot', 'error')
    if (submitting) return
    setSubmitting(true)
    try {
      const iso = `${date}T${selectedSlot}:00`
      await api.createAppointment({ patient_id: pid, doctor_id: parseInt(doctorId), appointment_date: iso, reason: reason || undefined })
      push('Appointment booked', 'success')
      setDoctorId(''); setDate(''); setReason(''); setSelectedSlot(''); load()
    } catch (err) { push(err.message || 'Booking failed', 'error') }
    finally { setSubmitting(false) }
  }

  return (
    <div className="relative min-h-screen">
      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="relative z-10 space-y-8 pb-12"
      >
        {/* Header */}
        <motion.header variants={item} className="flex flex-col md:flex-row md:items-center justify-between gap-4 lg:pr-72">
          <div>
            <h1 className="text-4xl md:text-5xl font-bold text-slate-900 tracking-tight">Appointments</h1>
            <p className="text-slate-500 mt-2 text-lg">Manage your consultations and checkups</p>
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
        <motion.section variants={item} id="book-form">
          <Card className="p-8" hover={false}>
            <div className="flex items-center gap-3 mb-6">
              <IconWrapper variant="primary" className="bg-primary-50 text-primary-600">
                <LuCalendarPlus size={20} />
              </IconWrapper>
              <CardTitle className="text-xl">Book a Consultation</CardTitle>
            </div>

            <form onSubmit={submit} className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <div className="space-y-2">
                  <label className="text-sm font-semibold text-slate-500 uppercase tracking-wider">Select Doctor</label>
                  <select
                    className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-slate-900 focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20 transition-all"
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
                  <label className="text-sm font-semibold text-slate-500 uppercase tracking-wider">Date</label>
                  <input
                    className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-slate-900 focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20 transition-all"
                    type="date"
                    value={date}
                    onChange={e => setDate(e.target.value)}
                    min={new Date().toISOString().split('T')[0]}
                    required
                  />
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-semibold text-slate-500 uppercase tracking-wider">Reason</label>
                  <input
                    className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-slate-900 placeholder:text-slate-400 focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20 transition-all"
                    placeholder="e.g. Skin rash, Checkup..."
                    value={reason}
                    onChange={e => setReason(e.target.value)}
                  />
                </div>
              </div>

              {/* Slots Grid */}
              <div className="bg-slate-50/50 rounded-2xl p-6 border border-slate-100">
                <label className="text-sm font-semibold text-slate-500 uppercase tracking-wider block mb-4">Available Slots</label>
                {date && slots.length > 0 ? (
                  <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-3">
                    {slots.map(s => {
                      const isTaken = bookedSlots.includes(s)
                      const isSelected = selectedSlot === s
                      return (
                        <button
                          type="button"
                          key={s}
                          disabled={isTaken}
                          onClick={() => !isTaken && setSelectedSlot(s)}
                          title={isTaken ? 'Already booked' : s}
                          className={`py-2.5 px-3 rounded-xl text-sm font-medium transition-all flex items-center justify-center gap-1.5 ${isTaken
                            ? 'bg-slate-100 border border-slate-200 text-slate-300 cursor-not-allowed line-through'
                            : isSelected
                              ? 'bg-primary-500 text-white shadow-lg shadow-primary-500/30 scale-105'
                              : 'bg-white border border-slate-200 text-slate-600 hover:border-primary-300 hover:text-primary-600'
                            }`}
                        >
                          <LuClock size={12} className={isTaken ? 'opacity-30' : isSelected ? 'opacity-70' : 'text-slate-400'} />
                          {s}
                        </button>
                      )
                    })}
                  </div>
                ) : (
                  <div className="text-center py-8 text-slate-400 flex flex-col items-center">
                    <LuTriangleAlert className="mb-2 text-2xl opacity-40" />
                    {date ? 'No slots available for this date.' : 'Please select a doctor and date first.'}
                  </div>
                )}
              </div>

              <div className="flex justify-end">
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="btn-primary px-8 py-3 text-lg disabled:opacity-50"
                  type="submit"
                  disabled={submitting}
                >
                  {submitting ? 'Booking...' : 'Confirm Booking'} {!submitting && <LuArrowRight className="inline ml-2" size={18} />}
                </motion.button>
              </div>
            </form>
          </Card>
        </motion.section>

        {/* Appointment List */}
        <motion.section variants={item}>
          <Card className="p-0 overflow-hidden border-slate-200" hover={false}>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="bg-slate-50 text-left border-b border-slate-200">
                    <th className="p-6 font-semibold text-slate-600">Doctor</th>
                    <th className="p-6 font-semibold text-slate-600">Date & Time</th>
                    <th className="p-6 font-semibold text-slate-600">Reason</th>
                    <th className="p-6 font-semibold text-slate-600">Status</th>
                    <th className="p-6 font-semibold text-slate-600 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {loading && (
                    <tr>
                      <td colSpan="5" className="p-12 text-center text-slate-500">Loading appointments...</td>
                    </tr>
                  )}
                  {!loading && list.length === 0 && (
                    <tr>
                      <td colSpan="5" className="p-12 text-center text-slate-500">No appointments found.</td>
                    </tr>
                  )}
                  {list.map(a => {
                    const d = doctors.find(x => x.doctor_id === a.doctor_id)
                    const isPast = new Date(a.appointment_date) < new Date()
                    return (
                      <tr key={a.appointment_id} className="hover:bg-slate-50/50 transition-colors">
                        <td className="p-6">
                          <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-primary-50 flex items-center justify-center border border-primary-100">
                              <LuUser size={18} className="text-primary-500" />
                            </div>
                            <div>
                              <div className="font-semibold text-slate-900">{d?.username || `Dr. #${a.doctor_id}`}</div>
                              <div className="text-xs text-slate-500">{d?.specialization || 'General'}</div>
                            </div>
                          </div>
                        </td>
                        <td className="p-6 text-slate-600 font-medium">
                          {new Date(a.appointment_date).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })}
                        </td>
                        <td className="p-6 text-slate-600 max-w-xs truncate" title={a.reason}>{a.reason || 'Routine Checkup'}</td>
                        <td className="p-6">
                          <StatusBadge status={a.status} />
                        </td>
                        <td className="p-6 text-right space-x-2">
                          {['Scheduled', 'Confirmed'].includes(a.status) && !isPast && (
                            <button className="px-3 py-1.5 text-xs font-semibold text-rose-600 bg-rose-50 hover:bg-rose-100 rounded-lg transition-colors border border-rose-100" onClick={() => setCancelTarget(a)}>
                              Cancel
                            </button>
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

      {/* Cancel confirmation modal */}
      <ConfirmModal
        open={!!cancelTarget}
        title="Cancel Appointment?"
        confirmText="Yes, Cancel Appointment"
        cancelText="Go Back"
        onClose={() => setCancelTarget(null)}
        onConfirm={async () => {
          if (!cancelTarget) return
          try {
            await api.updateAppointmentStatus(cancelTarget.appointment_id, 'Cancelled')
            push('Appointment cancelled', 'success')
          } catch (err) {
            push(err.message || 'Failed to cancel', 'error')
          }
          setCancelTarget(null)
          load()
        }}
      >
        <div className="space-y-2">
          <p className="text-slate-600">Are you sure you want to cancel your appointment on <strong>{cancelTarget ? new Date(cancelTarget.appointment_date).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }) : ''}</strong>?</p>
          {cancelTarget?.status === 'Confirmed' && (
            <p className="text-sm font-medium text-amber-600 bg-amber-50 px-3 py-2 rounded-lg">This appointment has already been confirmed by your doctor. Cancellations within 2 hours of the appointment are not allowed.</p>
          )}
        </div>
      </ConfirmModal>
    </div>
  )
}

function StatusBadge({ status }) {
  const styles = {
    Scheduled: 'bg-violet-50 text-violet-600 border-violet-100',
    Confirmed: 'bg-emerald-50 text-emerald-600 border-emerald-100',
    Cancelled: 'bg-slate-50 text-slate-500 border-slate-200',
    Completed: 'bg-sky-50 text-sky-600 border-sky-100',
  }
  const icon = {
    Confirmed: <LuCheck className="mr-1.5" size={12} />,
    Cancelled: <LuX className="mr-1.5" size={12} />,
    Completed: <LuCheck className="mr-1.5" size={12} />
  }
  return (
    <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-bold border ${styles[status] || 'bg-slate-50 text-slate-500'}`}>
      {icon[status]}
      {status}
    </span>
  )
}
