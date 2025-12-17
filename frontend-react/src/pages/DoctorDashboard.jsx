import { useEffect, useMemo, useState } from 'react'
import { api } from '../services/api.js'

export default function DoctorDashboard(){
  const name = localStorage.getItem('username') || 'Doctor'
  const [appts, setAppts] = useState([])
  const [summary, setSummary] = useState({ pending:0, completed:0, failed:0, refunded:0, count:0 })
  useEffect(()=>{(async()=>{ try{ setAppts(await api.listAppointments()); setSummary(await api.transactionsSummary()) }catch{} })()},[])

  const today = new Date()
  const todayApps = useMemo(()=> (appts||[]).filter(a=>{ const d=new Date(a.appointment_date); return d.toDateString()===today.toDateString() }).sort((a,b)=> a.appointment_date.localeCompare(b.appointment_date)),[appts])
  const upcoming = useMemo(()=> (appts||[]).filter(a=> new Date(a.appointment_date) > today).length,[appts])
  const completedToday = useMemo(()=> (appts||[]).filter(a=> a.status==='Completed' && new Date(a.appointment_date).toDateString()===today.toDateString()).length,[appts])
  const activePatients = useMemo(()=> new Set((appts||[]).map(a=>a.patient_id)).size,[appts])

  return (
    <div className="space-y-6">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Welcome, {name}</h1>
          <p className="muted">Manage your practice</p>
        </div>
      </header>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <KPICard title="Upcoming" value={upcoming} desc="appointments" />
        <KPICard title="Active Patients" value={activePatients} desc="this month" />
        <KPICard title="Completed Today" value={completedToday} desc="consultations" />
        <KPICard title="Revenue (week)" value={`₹ ${(summary.completed||0).toFixed(2)}`} desc="from payments" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="card lg:col-span-2">
          <div className="font-semibold mb-2">Today's Schedule</div>
          <div className="space-y-2">
            {todayApps.map(a=> (
              <div key={a.appointment_id} className="flex items-center justify-between">
                <div className="text-sm">{new Date(a.appointment_date).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'})} · PID {a.patient_id} · {a.reason||'Consultation'}</div>
                <a className="text-primaryBlue" href="/appointments">Open</a>
              </div>
            ))}
            {!todayApps.length && <div className="muted">No appointments today</div>}
          </div>
        </div>
        <div className="card">
          <div className="font-semibold mb-2">Notifications</div>
          <ul className="space-y-2 text-sm">
            <li className="flex items-center justify-between"><span className="muted">System ready</span><span className="chip">Info</span></li>
            <li className="flex items-center justify-between"><span className="muted">New messages appear here</span><span className="chip">Chat</span></li>
          </ul>
        </div>
      </div>

      <div className="grid-cards">
        <Card title="Appointments" desc="Review and manage visits." href="/doctor/appointments" />
        <Card title="Availability" desc="Set weekly schedule." href="/doctor/availability" />
        <Card title="Patients" desc="Assigned patient list." href="/doctor/patients" />
        <Card title="Transactions" desc="Your earnings & payouts." href="/doctor/transactions" />
        <Card title="Profile" desc="Manage your profile." href="/doctor/profile" />
        <Card title="Settings" desc="Preferences and security." href="/doctor/settings" />
      </div>
    </div>
  )
}

function Card({ title, desc, href }){
  return (
    <a className="card card-link" href={href}>
      <h3>{title}</h3>
      <p className="muted">{desc}</p>
      <div className="spacer" />
      <span className="button">Open</span>
    </a>
  )
}

function KPICard({ title, value, desc }){
  return (
    <div className="card">
      <div className="muted">{title}</div>
      <div className="text-2xl font-semibold">{value}</div>
      <div className="text-sm text-textLuxuryMuted">{desc}</div>
    </div>
  )
}
