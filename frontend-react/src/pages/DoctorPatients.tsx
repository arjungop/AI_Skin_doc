import React, { useEffect, useMemo, useState } from 'react'
import { api } from '../services/api.js'

type Appt = { appointment_id:number; patient_id:number; doctor_id:number; appointment_date:string; status:string }

export default function DoctorPatients(){
  const [appointments, setAppointments] = useState<Appt[]>([])
  const [q, setQ] = useState('')
  useEffect(()=>{(async()=>{ try{ setAppointments(await api.listAppointments()) }catch{} })()},[])

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
          <p className="muted">Derived from your appointment history.</p>
        </div>
        <input className="w-64" placeholder="Search patient ID" value={q} onChange={e=>setQ(e.target.value)} />
      </header>
      <div className="card">
        <table>
          <thead>
            <tr><th>Patient</th><th>Visits</th><th>Last Visit</th><th>Actions</th></tr>
          </thead>
          <tbody>
            {patients.map(p=> (
              <tr key={p.patient_id}>
                <td>PID {p.patient_id}</td>
                <td>{p.visits}</td>
                <td>{p.lastVisit ? new Date(p.lastVisit).toLocaleString() : '-'}</td>
                <td>
                  <a className="btn-ghost btn-sm" href={`/messages?patient_id=${p.patient_id}`}>Open Chat</a>
                </td>
              </tr>
            ))}
            {!patients.length && (
              <tr><td colSpan={4} className="muted">No assigned patients yet.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
