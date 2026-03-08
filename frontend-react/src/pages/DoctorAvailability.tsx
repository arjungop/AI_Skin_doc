import React, { useEffect, useMemo, useState } from 'react'
import { api } from '../services/api.js'
import { useToast } from '../components/Toast.jsx'
import { Card, IconWrapper } from '../components/Card.jsx'
import { LuClock, LuPlus, LuTrash2, LuSave, LuCalendarClock, LuGlobe } from 'react-icons/lu'
import { motion, AnimatePresence } from 'framer-motion'
const MotionButton = motion.button as any;
const MotionDiv = motion.div as any;

type Slot = { weekday: number; start_time: string; end_time: string; timezone?: string }

const WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
const WEEKDAY_SHORT = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

function useDoctorId() {
  const [doctorId, setDoctorId] = useState<number | null>(null)
  useEffect(() => {
    (async () => {
      try {
        const me = await api.me(); const docs = await api.listDoctors('')
        const mine = (docs || []).find((d: any) => d.user_id === me.user_id)
        setDoctorId(mine?.doctor_id ?? null)
      } catch { setDoctorId(null) }
    })()
  }, [])
  return doctorId
}

export default function DoctorAvailability() {
  const doctorId = useDoctorId()
  const { push } = useToast() as { push: (msg: string, kind?: string) => void }
  const [slots, setSlots] = useState<Slot[]>([])
  const [tz, setTz] = useState<string>(Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC')
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  async function load() {
    if (!doctorId) return
    setIsLoading(true)
    try {
      setError(null)
      const rows: any[] = await api.listDoctorAvailability(doctorId)
      setSlots((rows || []).map(r => ({ weekday: r.weekday, start_time: r.start_time, end_time: r.end_time, timezone: r.timezone })))
    } catch (err: any) {
      setError(err?.message || 'Failed to load availability')
    } finally {
      setIsLoading(false)
    }
  }
  useEffect(() => { load() }, [doctorId])

  function addSlot(day: number) {
    setSlots(s => [...s, { weekday: day, start_time: '09:00', end_time: '17:00', timezone: tz }])
  }

  function removeSlot(globalIdx: number) {
    setSlots(s => s.filter((_, i) => i !== globalIdx))
  }

  function updateSlot(globalIdx: number, field: 'start_time' | 'end_time', value: string) {
    setSlots(s => s.map((p, i) => i === globalIdx ? { ...p, [field]: value } : p))
  }

  async function save() {
    if (!doctorId) return
    setSaving(true)
    try {
      await api.setDoctorAvailability(doctorId, slots.map(s => ({ ...s, timezone: tz })))
      setSaved(true)
      push('Availability saved successfully', 'success')
      setTimeout(() => setSaved(false), 2000)
    } catch (err: any) {
      push(err?.message || 'Failed to save availability', 'error')
    } finally { setSaving(false) }
  }

  // Build a map from weekday → { slot, globalIndex }[]
  const grouped = useMemo(() => {
    const m: Record<number, { slot: Slot; idx: number }[]> = {}
    slots.forEach((s, i) => { (m[s.weekday] ||= []).push({ slot: s, idx: i }) })
    return m
  }, [slots])

  const TIMEZONES = [
    'UTC', 'America/New_York', 'America/Chicago', 'America/Denver', 'America/Los_Angeles',
    'Europe/London', 'Europe/Paris', 'Asia/Kolkata', 'Asia/Tokyo', 'Australia/Sydney',
  ]

  return (
    <div className="max-w-6xl mx-auto space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">

      {/* Header Section */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <IconWrapper variant="primary" size="lg" className="shadow-lg shadow-primary-500/20">
              <LuCalendarClock size={24} className="text-primary-600" />
            </IconWrapper>
            <h1 className="text-3xl font-bold text-slate-900 tracking-tight">Availability</h1>
          </div>
          <p className="text-slate-500 pl-1">Configure your weekly recurring schedule and timezone.</p>
        </div>

        <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
          <div className="relative group">
            <LuGlobe className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 group-focus-within:text-primary-500 transition-colors" size={18} />
            <select
              className="w-full sm:w-auto appearance-none bg-white border border-slate-200 rounded-xl py-2.5 pl-10 pr-10 text-sm font-medium text-slate-700 focus:ring-4 focus:ring-primary-500/10 focus:border-primary-500 transition-all shadow-sm cursor-pointer"
              value={tz}
              onChange={e => setTz(e.target.value)}
            >
              {TIMEZONES.includes(tz) ? null : <option value={tz}>{tz}</option>}
              {TIMEZONES.map(t => <option key={t} value={t}>{t.replace('_', ' ')}</option>)}
            </select>
            <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-slate-400">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path></svg>
            </div>
          </div>

          <MotionButton
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className={`btn py-2.5 px-6 rounded-xl flex items-center justify-center gap-2 shadow-lg transition-all ${saved
              ? 'bg-emerald-500 text-white shadow-emerald-500/20'
              : 'bg-primary-600 hover:bg-primary-700 text-white shadow-primary-500/25'
              }`}
            onClick={save}
            disabled={saving}
          >
            {saving ? (
              <><span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> Saving...</>
            ) : saved ? (
              <>✓ Saved Successfully</>
            ) : (
              <><LuSave size={18} /> Save Changes</>
            )}
          </MotionButton>
        </div>
      </div>

      {error && (
        <MotionDiv initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="flex items-center gap-3 px-5 py-4 rounded-2xl bg-rose-50 border border-rose-100 text-rose-700 shadow-sm">
          <LuClock className="shrink-0" size={20} />
          <p className="font-medium text-sm">{error}</p>
          <button className="ml-auto px-3 py-1.5 bg-white rounded-lg shadow-sm text-xs font-semibold hover:bg-rose-50 transition-colors" onClick={load}>Retry</button>
        </MotionDiv>
      )}

      {/* Grid of Days */}
      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 opacity-60">
          {[...Array(6)].map((_, i) => (
            <Card key={i} className="animate-pulse h-48" padding="lg"><div /></Card>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
          {WEEKDAYS.map((dayName, di) => {
            const daySlots = grouped[di] || []
            const isActive = daySlots.length > 0

            return (
              <Card
                key={di}
                className={`relative overflow-hidden transition-all duration-300 ${isActive ? 'ring-1 ring-primary-500/20 shadow-soft-xl' : 'opacity-80 hover:opacity-100 hover:shadow-soft-lg'}`}
                padding="lg"
              >
                {/* Active Indicator Line */}
                {isActive && <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-primary-400 to-primary-600" />}

                <div className="flex items-center justify-between mb-6">
                  <div className="flex items-center gap-3">
                    <div className={`w-10 h-10 rounded-xl flex items-center justify-center font-bold text-sm ${isActive ? 'bg-primary-50 text-primary-600' : 'bg-slate-50 text-slate-400'}`}>
                      {WEEKDAY_SHORT[di]}
                    </div>
                    <div>
                      <h3 className={`font-semibold ${isActive ? 'text-slate-900' : 'text-slate-500'}`}>{dayName}</h3>
                      <p className="text-xs text-slate-400">
                        {isActive ? `${daySlots.length} slot${daySlots.length > 1 ? 's' : ''}` : 'Unavailable'}
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={() => addSlot(di)}
                    className={`p-2 rounded-lg transition-colors flex items-center justify-center ${isActive ? 'text-primary-600 hover:bg-primary-50' : 'text-slate-400 hover:text-primary-600 hover:bg-slate-50'}`}
                    title={`Add slot on ${dayName}`}
                  >
                    <LuPlus size={20} />
                  </button>
                </div>

                <div className="space-y-3 min-h-[4rem]">
                  <AnimatePresence initial={false}>
                    {daySlots.map(({ slot, idx }) => (
                      <motion.div
                        initial={{ opacity: 0, height: 0, scale: 0.95 }}
                        animate={{ opacity: 1, height: 'auto', scale: 1 }}
                        exit={{ opacity: 0, height: 0, scale: 0.95 }}
                        transition={{ duration: 0.2 }}
                        key={idx}
                      >
                        <div className="flex items-center gap-2 bg-slate-50/80 border border-slate-100 p-2 rounded-xl group hover:border-slate-200 transition-colors">
                          <LuClock className="text-slate-400 ml-1 shrink-0" size={16} />
                          <input
                            className="bg-transparent border-none focus:ring-0 text-sm font-medium text-slate-700 w-24 px-1 cursor-pointer focus:bg-white focus:shadow-sm rounded transition-colors"
                            type="time"
                            value={slot.start_time}
                            onChange={e => updateSlot(idx, 'start_time', e.target.value)}
                          />
                          <span className="text-slate-300 font-medium">-</span>
                          <input
                            className="bg-transparent border-none focus:ring-0 text-sm font-medium text-slate-700 w-24 px-1 cursor-pointer focus:bg-white focus:shadow-sm rounded transition-colors"
                            type="time"
                            value={slot.end_time}
                            onChange={e => updateSlot(idx, 'end_time', e.target.value)}
                          />
                          <button
                            className="ml-auto p-1.5 text-slate-300 hover:text-rose-500 hover:bg-rose-50 rounded-lg transition-colors opacity-0 group-hover:opacity-100 focus:opacity-100"
                            onClick={() => removeSlot(idx)}
                            title="Remove slot"
                          >
                            <LuTrash2 size={16} />
                          </button>
                        </div>
                      </motion.div>
                    ))}
                  </AnimatePresence>

                  {!isActive && (
                    <div className="h-full flex items-center justify-center border-2 border-dashed border-slate-100 rounded-xl py-6">
                      <button
                        onClick={() => addSlot(di)}
                        className="text-sm font-medium text-slate-400 hover:text-primary-500 transition-colors flex items-center gap-1.5"
                      >
                        <LuPlus size={16} /> Add hours
                      </button>
                    </div>
                  )}
                </div>
              </Card>
            )
          })}
        </div>
      )}
    </div>
  )
}

