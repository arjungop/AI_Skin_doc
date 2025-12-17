import { LuActivity, LuMessageSquare, LuCalendar, LuCreditCard, LuUpload, LuClock, LuWallet } from 'react-icons/lu'
import { useEffect, useState } from 'react'
import { api } from '../services/api'
import { Navigate } from 'react-router-dom'

export default function Dashboard(){
  const role = (localStorage.getItem('role')||'').toUpperCase()
  if (role==='ADMIN') return <Navigate to="/admin" replace />
  const name = localStorage.getItem('username') || 'Patient'
  const [loading, setLoading] = useState(true)
  const [nextAppt, setNextAppt] = useState(null)
  const [outstanding, setOutstanding] = useState(0)
  const [lastAI, setLastAI] = useState('')

  useEffect(()=>{
    async function load(){
      try{
        const apps = await api.listAppointments()
        const now = new Date()
        const future = apps
          .map(a=> ({...a, dt:new Date(a.appointment_date)}))
          .filter(a=> a.dt > now)
          .sort((a,b)=> a.dt - b.dt)
        setNextAppt(future[0] || null)
      }catch{}
      try{
        const s = await api.transactionsSummary()
        setOutstanding(s && s.pending ? s.pending : 0)
      }catch{}
      try{
        const saved = JSON.parse(localStorage.getItem('chat_history')||'[]')
        const last = [...saved].reverse().find(m=> m.who==='AI')
        setLastAI(last ? (last.text||'') : '')
      }catch{}
      setLoading(false)
    }
    load()
  },[])

  const fmt = (d)=> { try{ return new Date(d).toLocaleString() }catch{ return '' } }
  return (
    <div>
      <section className="mb-4">
        <div className="rounded-2xl p-5 gradient-lux-banner gold-glow border border-borderElegant shadow-sm">
          <div className="flex items-center justify-between gap-3">
            <div>
              <h1 className="text-3xl font-semibold text-textPrimary dark:text-white">Welcome, {name}</h1>
              <p className="text-textSecondary dark:text-slate-300">Your health hub</p>
            </div>
            <div className="hidden md:flex items-center gap-2">
              <a href="/appointments" className="btn-ghost btn-sm"><LuCalendar className="mr-2"/>Book slot</a>
              <a href="/chat" className="btn-ghost btn-sm"><LuMessageSquare className="mr-2"/>Start chat</a>
              <a href="/lesions" className="btn-ghost btn-sm"><LuUpload className="mr-2"/>Upload lesion</a>
            </div>
          </div>
        </div>
      </section>

      {/* Overview metrics */}
      <section className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-6">
        <MetricCard icon={<LuClock />} title="Next appointment" value={loading? '—' : (nextAppt? fmt(nextAppt.appointment_date) : 'No upcoming') } />
        <MetricCard icon={<LuWallet />} title="Outstanding" value={`₹ ${Number(outstanding||0).toFixed(2)}`} />
        <MetricCard icon={<LuMessageSquare />} title="Last AI reply" value={lastAI ? (lastAI.length>42? lastAI.slice(0,42)+'…' : lastAI) : '—'} />
      </section>

      <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card icon={<LuActivity />} title="Lesion Classification" desc="Upload a lesion image and get AI prediction." href="/lesions" />
        <Card icon={<LuMessageSquare />} title="AI Doctor Chat" desc="Chat with the AI Doctor for guidance." href="/chat" />
        <Card icon={<LuCalendar />} title="Appointments" desc="Book and manage visits." href="/appointments" />
        <Card icon={<LuCreditCard />} title="Transactions" desc="View and manage payments." href="/transactions" />
      </section>
    </div>
  )
}

function Card({ title, desc, href, icon }){
  return (
    <a href={href} className="block rounded-2xl bg-cardBackground border border-borderElegant shadow-md p-5 hover:shadow-lg hover:scale-[1.02] transition-all duration-300">
      <div className="h-10 w-10 rounded-full bg-gradient-to-tr from-primaryBlue to-accentPurple2 flex items-center justify-center text-white text-lg mb-3">{icon}</div>
      <h3 className="text-xl font-semibold mb-1 text-textLuxury dark:text-white">{title}</h3>
      <p className="text-sm text-textMuted dark:text-slate-300 mb-6">{desc}</p>
      <span className="btn-primary btn-sm">Open</span>
    </a>
  )
}

function MetricCard({ icon, title, value }){
  return (
    <div className="card flex items-center gap-3">
      <div className="h-8 w-8 rounded-full bg-gradient-to-tr from-primaryBlue to-accentPurple2 text-white flex items-center justify-center">{icon}</div>
      <div>
        <div className="text-sm text-textMuted dark:text-slate-300">{title}</div>
        <div className="text-lg font-semibold text-textLuxury dark:text-white">{value}</div>
      </div>
    </div>
  )
}
