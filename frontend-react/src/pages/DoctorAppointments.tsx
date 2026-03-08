import React, { useEffect, useMemo, useState } from 'react'
import ConfirmModal from '../components/ConfirmModal.jsx'
import { api } from '../services/api.js'
import { useToast } from '../components/Toast.jsx'
import { motion, AnimatePresence } from 'framer-motion'
import { LuCheck, LuX, LuMessageSquare, LuCalendar, LuUser, LuClock, LuClipboardList, LuFilter, LuTriangleAlert } from 'react-icons/lu'
import { Card, CardTitle, CardDescription, CardBadge } from '../components/Card.jsx'
import { useNavigate } from 'react-router-dom'

type Appt = {
  appointment_id: number
  patient_id: number
  doctor_id: number
  appointment_date: string
  reason?: string
  status: 'Scheduled' | 'Confirmed' | 'Completed' | 'Cancelled'
  video_link?: string | null
}

function StatusBadge({ status }: { status: Appt['status'] }) {
  const cls =
    status === 'Confirmed' ? 'bg-primary-50 text-primary-600 border border-primary-200' :
      status === 'Completed' ? 'bg-sky-50 text-sky-600 border border-sky-200' :
        status === 'Cancelled' ? 'bg-rose-50 text-rose-600 border border-rose-200' :
          'bg-amber-50 text-amber-600 border border-amber-200'
  return <span className={`px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest ${cls}`}>{status}</span>
}

function useDoctorId() {
  const [doctorId, setDoctorId] = useState<number | null>(null)
  const [loading, setLoading] = useState(true)
  useEffect(() => {
    (async () => {
      try {
        const me = await api.me()
        const docs = await api.listDoctors('')
        const mine = (docs || []).find((d: any) => d.user_id === me.user_id)
        setDoctorId(mine?.doctor_id ?? null)
      } catch { setDoctorId(null) } finally { setLoading(false) }
    })()
  }, [])
  return { doctorId, loading }
}

export default function DoctorAppointments() {
  const { doctorId, loading } = useDoctorId()
  const { push } = useToast() as { push: (msg: string, kind?: string) => void }
  const [items, setItems] = useState<Appt[]>([])
  const [patientNames, setPatientNames] = useState<Record<number, string>>({})
  const [view, setView] = useState<'day' | 'week' | 'month' | 'all'>('all')
  const [confirm, setConfirm] = useState<{ id: number, action: 'Confirmed' | 'Completed' | 'Cancelled' } | null>(null)
  const [completeApptFor, setCompleteApptFor] = useState<Appt | null>(null)
  const [noteFor, setNoteFor] = useState<Appt | null>(null)
  const [noteText, setNoteText] = useState('')

  const [error, setError] = useState<string | null>(null)
  const [isLoadingAppts, setIsLoadingAppts] = useState(true)
  const navigate = useNavigate()

  async function load() {
    setIsLoadingAppts(true)
    try {
      setError(null)
      const [data, patients]: [Appt[], any[]] = await Promise.all([
        api.listAppointments(),
        api.listPatients('').catch(() => []),
      ])
      // Filter out invalid dates and sort
      const validData = (data || []).filter(a => !isNaN(new Date(a.appointment_date).getTime()))
      setItems(validData.sort((a, b) => new Date(a.appointment_date).getTime() - new Date(b.appointment_date).getTime()))

      const nameMap: Record<number, string> = {}
      for (const p of (patients || [])) {
        const name = [p.first_name, p.last_name].filter(Boolean).join(' ') || p.username || `Patient #${p.patient_id}`
        nameMap[p.patient_id] = name
      }
      setPatientNames(nameMap)
    } catch (err: any) { setError(err?.message || 'Failed to load appointments') }
    finally { setIsLoadingAppts(false) }
  }
  useEffect(() => { if (!loading) load() }, [loading])

  const today = new Date()
  today.setHours(0, 0, 0, 0)

  const filteredItems = useMemo(() => {
    return items.filter(a => {
      const dt = new Date(a.appointment_date)
      dt.setHours(0, 0, 0, 0)
      if (view === 'day') return dt.getTime() === today.getTime()
      if (view === 'week') {
        const t = new Date(today); const s = new Date(t); s.setDate(t.getDate() - t.getDay())
        const e = new Date(s); e.setDate(s.getDate() + 6)
        return dt >= s && dt <= e
      }
      if (view === 'month') {
        return dt.getMonth() === today.getMonth() && dt.getFullYear() === today.getFullYear()
      }
      return true
    })
  }, [items, view])

  const groupedItems = useMemo(() => {
    const byDate: Record<string, Appt[]> = {}
    for (const a of filteredItems) {
      const dt = new Date(a.appointment_date)
      const key = dt.toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' })
      if (!byDate[key]) byDate[key] = []
      byDate[key].push(a)
    }
    return byDate
  }, [filteredItems])

  async function doAction(id: number, action: 'Confirmed' | 'Completed' | 'Cancelled') {
    try {
      await api.updateAppointmentStatus(id, action)
      const label = action === 'Confirmed' ? 'confirmed' : action === 'Completed' ? 'completed' : 'cancelled'
      push(`Appointment ${label}`, 'success')
    } catch (err: any) {
      push(err?.message || `Failed to update appointment`, 'error')
    }
    setConfirm(null)
    await load()
  }

  async function addNote(appt: Appt) {
    try {
      const rooms = await api.listRooms()
      let room = (rooms || []).find((r: any) => r.patient_id === appt.patient_id && r.doctor_id === appt.doctor_id)
      if (!room) {
        room = await api.createRoom({ patient_id: appt.patient_id, doctor_id: appt.doctor_id })
      }
      await api.postMessage(room.room_id, `[Note from Doctor for appointment #${appt.appointment_id}] ${noteText}`)
      setNoteText('')
      setNoteFor(null)
      push('Note sent to patient chat', 'success')
    } catch (e: any) {
      push(e?.message || 'Failed to send note', 'error')
    }
  }

  const containerVars = { hidden: { opacity: 0 }, show: { opacity: 1, transition: { staggerChildren: 0.05 } } }
  const itemVars = { hidden: { opacity: 0, y: 10 }, show: { opacity: 1, y: 0 } }

  return (
    <div className="space-y-8 pb-12">
      {error && (
        // @ts-ignore
        <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="flex items-center gap-3 px-4 py-3 rounded-xl bg-red-50 border border-red-200 text-red-700 text-sm shadow-sm">
          <LuTriangleAlert size={18} />
          <span>{error}</span>
          <button className="ml-auto text-xs font-semibold underline" onClick={load}>Retry</button>
        </motion.div>
      )}

      {/* Header */}
      <header className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-slate-900 flex items-center gap-3">
            <LuCalendar className="text-primary-500" /> Appointments
          </h1>
          <p className="text-slate-500 mt-1">Manage your daily schedule and patient encounters.</p>
        </div>
        <div className="flex gap-1 p-1 bg-slate-100/80 rounded-xl border border-slate-200/60 shadow-inner">
          {(['day', 'week', 'month', 'all'] as const).map(v => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all ${view === v
                ? 'bg-white text-primary-600 shadow-sm border border-slate-200'
                : 'text-slate-500 hover:text-slate-700 hover:bg-slate-200/50'
                }`}
            >
              {v.charAt(0).toUpperCase() + v.slice(1)}
            </button>
          ))}
        </div>
      </header>

      {/* Content */}
      {isLoadingAppts ? (
        <div className="flex items-center justify-center min-h-[40vh]">
          <div className="animate-pulse flex items-center gap-3 text-slate-400 font-medium">
            <div className="w-5 h-5 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
            Loading appointments...
          </div>
        </div>
      ) : filteredItems.length === 0 ? (
        // @ts-ignore
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-center py-20 bg-slate-50/50 rounded-3xl border border-slate-200 border-dashed">
          <LuCalendar className="mx-auto mb-4 text-slate-300" size={48} />
          <h3 className="text-xl font-bold text-slate-900 mb-2">No Appointments Found</h3>
          <p className="text-slate-500">You have no appointments scheduled for this time period.</p>
        </motion.div>
      ) : (
        // @ts-ignore
        <motion.div variants={containerVars} initial="hidden" animate="show" className="space-y-8">
          {Object.entries(groupedItems).map(([dateLabel, appointments]) => (
            <div key={dateLabel} className="space-y-4">
              <h3 className="font-bold text-slate-800 flex items-center gap-2 sticky top-0 bg-[#FAFAFA]/90 backdrop-blur-md py-2 z-10">
                <div className="w-2 h-2 rounded-full bg-primary-500" /> {dateLabel}
                <span className="text-sm font-normal text-slate-400 ml-2">({appointments.length})</span>
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {appointments.map(a => {
                  const d = new Date(a.appointment_date)
                  return (
                    // @ts-ignore
                    <motion.div key={a.appointment_id} variants={itemVars}>
                      <Card className="p-5 h-full flex flex-col group" hover={false}>
                        <div className="flex items-start justify-between mb-4">
                          <StatusBadge status={a.status} />
                          <div className="text-slate-400 text-xs font-mono bg-slate-100 px-2 py-0.5 rounded-md">#{a.appointment_id}</div>
                        </div>

                        <div className="flex items-center gap-3 mb-4">
                          <div className="w-12 h-12 rounded-2xl bg-primary-50 text-primary-600 flex flex-col items-center justify-center flex-shrink-0">
                            <span className="text-xs font-bold uppercase">{d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }).split(' ')[1]}</span>
                            <span className="text-sm font-bold leading-none">{d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }).split(' ')[0]}</span>
                          </div>
                          <div>
                            <h4 className="font-bold text-slate-900 truncate" title={patientNames[a.patient_id]}>{patientNames[a.patient_id] || `PID #${a.patient_id}`}</h4>
                            <p className="text-sm text-slate-500 truncate flex items-center gap-1 mt-0.5">
                              <LuClipboardList size={12} /> {a.reason || 'Consultation'}
                            </p>
                          </div>
                        </div>

                        <div className="mt-auto pt-4 border-t border-slate-100 flex flex-wrap gap-2">
                          {a.status === 'Scheduled' && (
                            <button onClick={() => setConfirm({ id: a.appointment_id, action: 'Confirmed' })} className="flex-1 min-w-[100px]  bg-emerald-50 text-emerald-600 hover:bg-emerald-100 py-2 rounded-xl text-xs font-bold transition-colors flex items-center justify-center gap-1.5 focus:outline-none focus:ring-2 focus:ring-emerald-500/20">
                              <LuCheck size={14} /> Confirm
                            </button>
                          )}
                          {a.status === 'Confirmed' && (
                            <button onClick={() => setCompleteApptFor(a)} className="flex-1 min-w-[100px] bg-primary-50 text-primary-600 hover:bg-primary-100 py-2 rounded-xl text-xs font-bold transition-colors flex items-center justify-center gap-1.5 focus:outline-none focus:ring-2 focus:ring-primary-500/20">
                              <LuCheck size={14} /> Complete
                            </button>
                          )}
                          {!['Completed', 'Cancelled'].includes(a.status) && (
                            <button onClick={() => { setNoteFor(a); setNoteText('') }} className="flex-1 min-w-[100px] bg-slate-100 text-slate-600 hover:bg-slate-200 py-2 rounded-xl text-xs font-bold transition-colors flex items-center justify-center gap-1.5 focus:outline-none focus:ring-2 focus:ring-slate-500/20">
                              <LuMessageSquare size={14} /> Msg Note
                            </button>
                          )}
                          {['Scheduled', 'Confirmed'].includes(a.status) && (
                            <button onClick={() => setConfirm({ id: a.appointment_id, action: 'Cancelled' })} className="w-auto px-3 bg-red-50 text-red-600 hover:bg-red-100 py-2 rounded-xl transition-colors flex items-center justify-center focus:outline-none focus:ring-2 focus:ring-red-500/20" title="Cancel Appointment">
                              <LuX size={14} />
                            </button>
                          )}
                        </div>
                      </Card>
                    </motion.div>
                  )
                })}
              </div>
            </div>
          ))}
        </motion.div>
      )}

      {/* Modals */}
      <ConfirmModal
        open={!!confirm}
        title={`${confirm?.action === 'Confirmed' ? 'Confirm' : confirm?.action === 'Cancelled' ? 'Cancel' : 'Update'} Appointment?`}
        onClose={() => setConfirm(null)}
        onConfirm={() => confirm && doAction(confirm.id, confirm.action)}
        confirmText={confirm?.action === 'Confirmed' ? 'Confirm' : confirm?.action === 'Cancelled' ? 'Yes, Cancel Appointment' : confirm?.action}
        cancelText={confirm?.action === 'Cancelled' ? 'Go Back' : 'Cancel'}
      >
        <div className="space-y-3">
          <p className="text-slate-600 leading-relaxed">You are about to <strong>{confirm?.action === 'Confirmed' ? 'confirm' : confirm?.action === 'Cancelled' ? 'cancel' : confirm?.action?.toLowerCase()}</strong> appointment #{confirm?.id}. This action will be logged in the system.</p>
          {confirm?.action === 'Cancelled' && (
            <p className="text-sm font-medium text-rose-600 bg-rose-50 px-3 py-2 rounded-lg">Cancellations notify the patient immediately.</p>
          )}
        </div>
      </ConfirmModal>

      <ConfirmModal
        open={!!completeApptFor}
        title={`Complete Appointment?`}
        onClose={() => setCompleteApptFor(null)}
        onConfirm={async () => {
          if (completeApptFor) {
            await doAction(completeApptFor.appointment_id, 'Completed')
            setCompleteApptFor(null)
          }
        }}
        confirmText='Mark as Completed'
      >
        <div className="space-y-4">
          <p className="text-slate-600 leading-relaxed">You are marking appointment #{completeApptFor?.appointment_id} as completed.</p>
          <p className="text-sm text-slate-500">After completing, you can create a treatment plan or clinical notes from the sidebar.</p>
        </div>
      </ConfirmModal>
    </div>
  )
}
