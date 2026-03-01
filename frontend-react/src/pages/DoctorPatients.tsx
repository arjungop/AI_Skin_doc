import React, { useEffect, useMemo, useState } from 'react'
import { api } from '../services/api.js'

type Appt = { appointment_id:number; patient_id:number; doctor_id:number; appointment_date:string; status:string }

export default function DoctorPatients(){
  const [appointments, setAppointments] = useState<Appt[]>([])
  const [q, setQ] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string|null>(null)

  useEffect(()=>{
    ;(async()=>{
      try{
        setError(null)
        setAppointments(await api.listAppointments())
      }
      catch(err: any){ setError(err?.message || 'Failed to load patients') }
      finally{ setLoading(false) }
    })()
  },[])

  const patients = useMemo(()=>{
    const m = new Map<number, { patient_id:number; lastVisit:string|null; visits:number }>()
    for(const a of appointments){
      const e = m.get(a.patient_id) || { patient_id:a.patient_id, lastVisit:null, visits:0 }
      e.visits += 1
      if(!e.lastVisit || new Date(a.appointment_date) > new Date(e.lastVisit)) e.lastVisit = a.appointment_date
      m.set(a.patient_id, e)
    }
    let list = Array.from(m.values())
    if(q){ list = list.filter(p=> String(p.patient_id).includes(q)) }
    return list.sort((a,b)=> (b.lastVisit||'').localeCompare(a.lastVisit||''))
  },[appointments,q])

  return (
    <div className="space-y-4">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Assigned Patients</h1>
          <p className="text-sm text-slate-500">Derived from your appointment history.</p>
        </div>
        <input
          className="input w-64"
          placeholder="Search patient ID"
          value={q}
          onChange={e=>setQ(e.target.value)}
        />
      </header>

      {error && (
        <div className="flex items-center gap-3 px-4 py-3 rounded-xl bg-red-50 border border-red-200 text-red-700 text-sm">
          <span>⚠</span>
          <span>{error}</span>
          <button className="ml-auto text-xs underline" onClick={()=>window.location.reload()}>Retry</button>
        </div>
      )}

      {loading ? (
        <div className="flex justify-center py-12">
          <div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin" />
        </div>
      ) : (
        <div className="card overflow-x-auto">
          <table className="table-auto">
            <thead>
              <tr>
                <th>Patient</th>
                <th>Visits</th>
                <th>Last Visit</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {patients.map(p=> (
                <tr key={p.patient_id}>
                  <td>
                    <div className="flex items-center gap-2">
                      <div className="w-8 h-8 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-bold">
                        {p.patient_id}
                      </div>
                      <span className="text-sm font-medium">PID {p.patient_id}</span>
                    </div>
                  </td>
                  <td>{p.visits}</td>
                  <td>{p.lastVisit ? new Date(p.lastVisit).toLocaleString() : '-'}</td>
                  <td>
                    <a
                      className="btn btn-ghost btn-sm"
                      href={`/messages?patient_id=${p.patient_id}`}
                    >
                      Open Chat
                    </a>
                  </td>
                </tr>
              ))}
              {!patients.length && (
                <tr>
                  <td colSpan={4} className="text-center text-slate-400 py-6">
                    No assigned patients yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
