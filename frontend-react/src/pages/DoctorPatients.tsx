import React, { useEffect, useMemo, useState } from 'react'
import { api } from '../services/api.js'
import { motion, AnimatePresence } from 'framer-motion'
import {
  LuSearch, LuUsers, LuCalendarHeart, LuMessageSquare, LuTriangleAlert,
  LuX, LuUser, LuMapPin, LuSun, LuMoon, LuScanFace, LuPill, LuSparkles,
  LuChevronRight, LuActivity,
} from 'react-icons/lu'
import { Card } from '../components/Card.jsx'
import { Link } from 'react-router-dom'

type PatientOut = { patient_id: number; name: string; visits: number; lastVisit: string | null }

type SkinProfile = {
  skin_type?: string; sensitivity_level?: string; acne_prone?: boolean;
  fitzpatrick_type?: number; allergies?: string; goals?: string; location_city?: string;
} | null

type Overview = {
  patient: { patient_id: number; user_id: number; name: string; age: number; gender: string; email: string }
  skin_profile: SkinProfile
  recent_scans: { lesion_id: number; prediction: string; created_at: string }[]
  ai_routine: { step: number; time: string; product: string; brand: string; instructions: string }[]
  active_plans: { plan_id: number; diagnosis: string; created_at: string }[]
}

function Chip({ children, color = 'slate' }: { children: React.ReactNode; color?: string }) {
  const map: Record<string, string> = {
    slate: 'bg-slate-100 text-slate-600',
    primary: 'bg-primary-50 text-primary-600',
    rose: 'bg-rose-50 text-rose-600',
    amber: 'bg-amber-50 text-amber-700',
    emerald: 'bg-emerald-50 text-emerald-700',
    indigo: 'bg-indigo-50 text-indigo-600',
  }
  return (
    <span className={`inline-flex items-center gap-1 text-xs font-semibold px-2.5 py-1 rounded-full ${map[color] ?? map.slate}`}>
      {children}
    </span>
  )
}

function PatientDrawer({ patientId, onClose }: { patientId: number; onClose: () => void }) {
  const [data, setData] = useState<Overview | null>(null)
  const [loading, setLoading] = useState(true)
  const [err, setErr] = useState('')
  const [tab, setTab] = useState<'overview' | 'scans' | 'routine' | 'plans'>('overview')

  useEffect(() => {
    ;(async () => {
      try {
        const d = await api.getPatientOverview(patientId)
        setData(d)
      } catch (e: any) {
        setErr(e?.message || 'Failed to load patient details')
      } finally {
        setLoading(false)
      }
    })()
  }, [patientId])

  const amSteps = data?.ai_routine.filter(s => s.time === 'AM') ?? []
  const pmSteps = data?.ai_routine.filter(s => s.time === 'PM') ?? []

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm flex justify-end"
      onClick={onClose}
    >
      <motion.div
        initial={{ x: '100%' }}
        animate={{ x: 0 }}
        exit={{ x: '100%' }}
        transition={{ type: 'spring', damping: 28, stiffness: 260 }}
        className="relative w-full max-w-xl h-full bg-white shadow-2xl flex flex-col overflow-hidden"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="px-6 pt-6 pb-4 border-b border-slate-100 flex items-center justify-between flex-shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-11 h-11 rounded-2xl bg-gradient-to-br from-primary-100 to-primary-50 text-primary-600 font-bold text-lg flex items-center justify-center">
              {data ? (data.patient.name.split(' ')[0]?.[0] ?? 'P').toUpperCase() + (data.patient.name.split(' ')[1]?.[0] ?? '') : '...'}
            </div>
            <div>
              <h2 className="text-lg font-bold text-slate-900">{data?.patient.name ?? 'Loading...'}</h2>
              {data && (
                <p className="text-xs text-slate-500">
                  {[data.patient.age && `Age ${data.patient.age}`, data.patient.gender].filter(Boolean).join(' · ')}
                </p>
              )}
            </div>
          </div>
          <button onClick={onClose} className="w-8 h-8 rounded-xl bg-slate-100 hover:bg-slate-200 flex items-center justify-center text-slate-500 transition-colors">
            <LuX size={16} />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 px-6 py-3 border-b border-slate-100 flex-shrink-0">
          {([
            { id: 'overview', label: 'Overview', icon: <LuUser size={13} /> },
            { id: 'scans', label: `Scans (${data?.recent_scans.length ?? 0})`, icon: <LuScanFace size={13} /> },
            { id: 'routine', label: `Routine (${data?.ai_routine.length ?? 0})`, icon: <LuSparkles size={13} /> },
            { id: 'plans', label: `Plans (${data?.active_plans.length ?? 0})`, icon: <LuPill size={13} /> },
          ] as const).map(t => (
            <button
              key={t.id}
              onClick={() => setTab(t.id as any)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                tab === t.id ? 'bg-primary-50 text-primary-600' : 'text-slate-500 hover:text-slate-700'
              }`}
            >
              {t.icon}{t.label}
            </button>
          ))}
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto p-6">
          {loading && (
            <div className="flex items-center justify-center h-48 text-slate-400">
              <div className="w-5 h-5 border-2 border-primary-500 border-t-transparent rounded-full animate-spin mr-3" />
              Loading patient data...
            </div>
          )}
          {err && (
            <div className="flex items-center gap-2 px-4 py-3 bg-rose-50 border border-rose-200 rounded-xl text-rose-700 text-sm">
              <LuTriangleAlert size={16} />{err}
            </div>
          )}

          {data && tab === 'overview' && (
            <div className="space-y-5">
              {/* Demographics */}
              <section>
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Demographics</p>
                <div className="grid grid-cols-2 gap-3">
                  {[
                    { label: 'Email', value: data.patient.email },
                    { label: 'Age', value: data.patient.age ? `${data.patient.age} yrs` : '—' },
                    { label: 'Gender', value: data.patient.gender || '—' },
                    { label: 'Patient ID', value: `#${data.patient.patient_id}` },
                  ].map(r => (
                    <div key={r.label} className="bg-slate-50 rounded-xl px-4 py-3">
                      <p className="text-xs text-slate-400 mb-0.5">{r.label}</p>
                      <p className="text-sm font-semibold text-slate-800 truncate">{r.value}</p>
                    </div>
                  ))}
                </div>
              </section>

              {/* Skin Profile */}
              {data.skin_profile ? (
                <section>
                  <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Skin Profile</p>
                  <div className="bg-slate-50 rounded-2xl p-4 space-y-3">
                    <div className="flex flex-wrap gap-2">
                      {data.skin_profile.skin_type && (
                        <Chip color="primary">🧴 {data.skin_profile.skin_type} skin</Chip>
                      )}
                      {data.skin_profile.sensitivity_level && (
                        <Chip color="amber">Sensitivity: {data.skin_profile.sensitivity_level}</Chip>
                      )}
                      {data.skin_profile.acne_prone && <Chip color="rose">Acne-prone</Chip>}
                      {data.skin_profile.fitzpatrick_type && (
                        <Chip color="slate">Fitzpatrick {data.skin_profile.fitzpatrick_type}/6</Chip>
                      )}
                      {data.skin_profile.location_city && (
                        <Chip color="slate"><LuMapPin size={11} />{data.skin_profile.location_city}</Chip>
                      )}
                    </div>
                    {data.skin_profile.allergies && (
                      <div>
                        <p className="text-xs text-amber-600 font-semibold mb-1">⚠ Allergies / Ingredients to avoid</p>
                        <p className="text-sm text-slate-700">{data.skin_profile.allergies}</p>
                      </div>
                    )}
                    {data.skin_profile.goals && (
                      <div>
                        <p className="text-xs text-slate-500 font-semibold mb-1">Goals</p>
                        <p className="text-sm text-slate-700">{data.skin_profile.goals}</p>
                      </div>
                    )}
                  </div>
                </section>
              ) : (
                <div className="bg-slate-50 rounded-2xl p-4 text-sm text-slate-400 italic">No skin profile on file</div>
              )}

              {/* Quick stats */}
              <section>
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Quick Stats</p>
                <div className="grid grid-cols-3 gap-3">
                  {[
                    { label: 'Scans', value: data.recent_scans.length, icon: <LuScanFace size={18} />, color: 'text-primary-500', bg: 'bg-primary-50' },
                    { label: 'Routine steps', value: data.ai_routine.length, icon: <LuSparkles size={18} />, color: 'text-accent-500', bg: 'bg-accent-50' },
                    { label: 'Active plans', value: data.active_plans.length, icon: <LuPill size={18} />, color: 'text-emerald-500', bg: 'bg-emerald-50' },
                  ].map(s => (
                    <div key={s.label} className="bg-slate-50 rounded-2xl p-4 flex flex-col items-center text-center">
                      <div className={`w-9 h-9 rounded-xl ${s.bg} ${s.color} flex items-center justify-center mb-2`}>{s.icon}</div>
                      <p className="text-2xl font-bold text-slate-900">{s.value}</p>
                      <p className="text-xs text-slate-400">{s.label}</p>
                    </div>
                  ))}
                </div>
              </section>
            </div>
          )}

          {data && tab === 'scans' && (
            <div className="space-y-3">
              {data.recent_scans.length === 0 ? (
                <div className="text-center py-16 text-slate-400">
                  <LuScanFace size={40} className="mx-auto mb-3 opacity-30" />
                  <p>No scan history</p>
                </div>
              ) : data.recent_scans.map((s, i) => {
                const risk = s.prediction?.toLowerCase()
                const isHigh = ['melanoma', 'basal cell carcinoma', 'squamous cell carcinoma', 'actinic keratosis'].some(c => risk?.includes(c))
                return (
                  <div key={s.lesion_id} className="flex items-center gap-4 bg-slate-50 rounded-2xl px-4 py-3">
                    <div className={`w-8 h-8 rounded-xl flex items-center justify-center font-bold text-sm ${isHigh ? 'bg-rose-50 text-rose-500' : 'bg-emerald-50 text-emerald-600'}`}>
                      {i + 1}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-semibold text-slate-900 capitalize truncate">{s.prediction || 'Unknown'}</p>
                      <p className="text-xs text-slate-400">{s.created_at ? new Date(s.created_at).toLocaleDateString() : '—'}</p>
                    </div>
                    {isHigh && <Chip color="rose">High Risk</Chip>}
                  </div>
                )
              })}
            </div>
          )}

          {data && tab === 'routine' && (
            <div className="space-y-6">
              {data.ai_routine.length === 0 ? (
                <div className="text-center py-16 text-slate-400">
                  <LuSparkles size={40} className="mx-auto mb-3 opacity-30" />
                  <p>Patient hasn't generated an AI routine yet</p>
                </div>
              ) : (
                <>
                  {amSteps.length > 0 && (
                    <div>
                      <div className="flex items-center gap-2 mb-3">
                        <LuSun size={16} className="text-amber-500" />
                        <p className="text-sm font-bold text-slate-800">Morning ({amSteps.length} steps)</p>
                      </div>
                      <div className="space-y-2">
                        {amSteps.map(s => (
                          <div key={s.step} className="flex gap-3 bg-amber-50/60 rounded-xl px-4 py-3 border border-amber-100">
                            <span className="w-5 h-5 rounded-lg bg-amber-100 text-amber-700 text-xs font-bold flex items-center justify-center flex-shrink-0 mt-0.5">{s.step}</span>
                            <div>
                              <p className="text-sm font-semibold text-slate-900">{s.product}</p>
                              {s.brand && <p className="text-xs text-primary-500 font-medium">{s.brand}</p>}
                              {s.instructions && <p className="text-xs text-slate-500 mt-0.5">{s.instructions}</p>}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  {pmSteps.length > 0 && (
                    <div>
                      <div className="flex items-center gap-2 mb-3">
                        <LuMoon size={16} className="text-indigo-500" />
                        <p className="text-sm font-bold text-slate-800">Evening ({pmSteps.length} steps)</p>
                      </div>
                      <div className="space-y-2">
                        {pmSteps.map(s => (
                          <div key={s.step} className="flex gap-3 bg-indigo-50/60 rounded-xl px-4 py-3 border border-indigo-100">
                            <span className="w-5 h-5 rounded-lg bg-indigo-100 text-indigo-700 text-xs font-bold flex items-center justify-center flex-shrink-0 mt-0.5">{s.step}</span>
                            <div>
                              <p className="text-sm font-semibold text-slate-900">{s.product}</p>
                              {s.brand && <p className="text-xs text-primary-500 font-medium">{s.brand}</p>}
                              {s.instructions && <p className="text-xs text-slate-500 mt-0.5">{s.instructions}</p>}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          {data && tab === 'plans' && (
            <div className="space-y-3">
              {data.active_plans.length === 0 ? (
                <div className="text-center py-16 text-slate-400">
                  <LuPill size={40} className="mx-auto mb-3 opacity-30" />
                  <p>No active treatment plans</p>
                </div>
              ) : data.active_plans.map(p => (
                <div key={p.plan_id} className="bg-emerald-50 border border-emerald-100 rounded-2xl px-4 py-3 flex items-start gap-3">
                  <LuActivity size={18} className="text-emerald-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm font-semibold text-slate-900">{p.diagnosis}</p>
                    <p className="text-xs text-slate-400 mt-0.5">{p.created_at ? `Started ${new Date(p.created_at).toLocaleDateString()}` : ''}</p>
                  </div>
                  <Chip color="emerald">Active</Chip>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer actions */}
        {data && (
          <div className="px-6 py-4 border-t border-slate-100 flex gap-3 flex-shrink-0 bg-white">
            <Link
              to={`/messages?patient_id=${data.patient.patient_id}`}
              className="flex-1 bg-primary-500 hover:bg-primary-600 text-white py-2.5 rounded-xl text-sm font-semibold flex items-center justify-center gap-2 transition-colors"
            >
              <LuMessageSquare size={16} /> Message
            </Link>
            <Link
              to={`/treatment-plans?patient_id=${data.patient.patient_id}`}
              className="flex-1 bg-slate-100 hover:bg-slate-200 text-slate-700 py-2.5 rounded-xl text-sm font-semibold flex items-center justify-center gap-2 transition-colors"
            >
              <LuPill size={16} /> Add Plan
            </Link>
          </div>
        )}
      </motion.div>
    </motion.div>
  )
}

export default function DoctorPatients() {
  const [allPatients, setAllPatients] = useState<PatientOut[]>([])
  const [q, setQ] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedId, setSelectedId] = useState<number | null>(null)

  useEffect(() => {
    ;(async () => {
      try {
        setError(null)
        const pts = await api.getDoctorPatients()
        setAllPatients(pts || [])
      } catch (err: any) {
        setError(err?.message || 'Failed to load patients')
      } finally {
        setLoading(false)
      }
    })()
  }, [])

  const patients = useMemo(() => {
    let list = [...allPatients]
    if (q) {
      const query = q.toLowerCase()
      list = list.filter(p => String(p.patient_id).includes(query) || p.name.toLowerCase().includes(query))
    }
    return list
  }, [allPatients, q])

  const containerVars = { hidden: { opacity: 0 }, show: { opacity: 1, transition: { staggerChildren: 0.05 } } }
  const itemVars = { hidden: { opacity: 0, y: 10 }, show: { opacity: 1, y: 0 } }

  return (
    <div className="space-y-8 pb-12">
      <header className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-slate-900 flex items-center gap-3">
            <LuUsers className="text-primary-500" /> Assigned Patients
          </h1>
          <p className="text-slate-500 mt-1">Derived from your appointment history and active cases.</p>
        </div>
        <div className="relative w-full md:w-72">
          <LuSearch className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
          <input
            className="w-full bg-white border border-slate-200 rounded-xl py-2.5 pl-10 pr-4 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none focus:border-primary-500 focus:ring-4 focus:ring-primary-500/10 transition-all shadow-sm"
            placeholder="Search name or ID..."
            value={q}
            onChange={e => setQ(e.target.value)}
          />
        </div>
      </header>

      {error && (
        // @ts-ignore
        <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="flex items-center gap-3 px-4 py-3 rounded-xl bg-red-50 border border-red-200 text-red-700 text-sm shadow-sm">
          <LuTriangleAlert size={18} />
          <span>{error}</span>
          <button className="ml-auto text-xs font-semibold underline" onClick={() => window.location.reload()}>Retry</button>
        </motion.div>
      )}

      {loading ? (
        <div className="flex items-center justify-center min-h-[40vh]">
          <div className="animate-pulse flex items-center gap-3 text-slate-400 font-medium">
            <div className="w-5 h-5 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
            Loading patients...
          </div>
        </div>
      ) : patients.length === 0 ? (
        // @ts-ignore
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-center py-20 bg-slate-50/50 rounded-3xl border border-slate-200 border-dashed">
          <LuUsers className="mx-auto mb-4 text-slate-300" size={48} />
          <h3 className="text-xl font-bold text-slate-900 mb-2">No Patients Found</h3>
          <p className="text-slate-500">No matching patients in your active directory.</p>
        </motion.div>
      ) : (
        // @ts-ignore
        <motion.div variants={containerVars} initial="hidden" animate="show" className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-5">
          {patients.map(p => (
            // @ts-ignore
            <motion.div key={p.patient_id} variants={itemVars}>
              <Card className="p-6 h-full flex flex-col group" hover={true}>
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center justify-center w-12 h-12 rounded-2xl bg-gradient-to-br from-primary-100 to-primary-50 text-primary-600 font-bold text-lg shadow-sm">
                    {(p.name.split(' ')[0]?.[0] || 'P').toUpperCase()}
                    {(p.name.split(' ')[1]?.[0] || '').toUpperCase()}
                  </div>
                  <div className="bg-slate-100 px-2 py-1 rounded-lg text-xs font-mono text-slate-500 font-medium">
                    ID: {p.patient_id}
                  </div>
                </div>

                <div className="mb-4">
                  <h3 className="text-lg font-bold text-slate-900 truncate" title={p.name}>{p.name}</h3>
                  <div className="mt-2 space-y-1.5">
                    <p className="text-xs text-slate-500 flex items-center gap-2">
                      <LuCalendarHeart size={14} className="text-rose-400" />
                      <span className="font-medium text-slate-700">{p.visits}</span> Total Encounters
                    </p>
                    <p className="text-xs text-slate-500 flex items-center gap-2">
                      <span className="w-3.5 h-3.5 rounded-full bg-slate-100 flex items-center justify-center border border-slate-200 text-[8px]">LV</span>
                      {p.lastVisit ? new Date(p.lastVisit).toLocaleDateString() : 'Never'}
                    </p>
                  </div>
                </div>

                <div className="mt-auto border-t border-slate-100 pt-4 flex gap-2">
                  <button
                    onClick={() => setSelectedId(p.patient_id)}
                    className="flex-1 bg-primary-50 hover:bg-primary-100 text-primary-600 border border-primary-200 py-2.5 rounded-xl text-sm font-semibold transition-all flex items-center justify-center gap-2"
                  >
                    <LuUser size={15} /> View
                  </button>
                  <Link
                    to={`/messages?patient_id=${p.patient_id}`}
                    className="flex-1 bg-slate-50 hover:bg-slate-100 text-slate-600 border border-slate-200 py-2.5 rounded-xl text-sm font-semibold transition-all flex items-center justify-center gap-2"
                  >
                    <LuMessageSquare size={15} /> Chat
                  </Link>
                </div>
              </Card>
            </motion.div>
          ))}
        </motion.div>
      )}

      <AnimatePresence>
        {selectedId !== null && (
          <PatientDrawer patientId={selectedId} onClose={() => setSelectedId(null)} />
        )}
      </AnimatePresence>
    </div>
  )
}

