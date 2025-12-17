import React, { useEffect, useMemo, useState } from 'react'
import ConfirmModal from '../components/ConfirmModal.jsx'
import { api } from '../services/api.js'

type Appt = {
  appointment_id: number
  patient_id: number
  doctor_id: number
  appointment_date: string
  reason?: string
  status: 'Scheduled' | 'Confirmed' | 'Completed' | 'Cancelled'
}

function StatusBadge({ status }: { status: Appt['status'] }){
  const cls =
    status === 'Confirmed' ? 'bg-successLux text-white' :
    status === 'Completed' ? 'bg-primaryBlue text-white' :
    status === 'Cancelled' ? 'bg-errorLux text-white' :
    'bg-warmGray text-textLuxury'
  return <span className={`px-2 py-1 rounded-full text-xs ${cls}`}>{status}</span>
}

function useDoctorId(){
  const [doctorId, setDoctorId] = useState<number|null>(null)
  const [loading, setLoading] = useState(true)
  useEffect(()=>{(async()=>{
    try{
      const me = await api.me()
      const docs = await api.listDoctors('')
      const mine = (docs||[]).find((d:any)=> d.user_id === me.user_id)
      setDoctorId(mine?.doctor_id ?? null)
    }catch{ setDoctorId(null) } finally{ setLoading(false) }
  })()},[])
  return { doctorId, loading }
}

export default function DoctorAppointments(){
  const { doctorId, loading } = useDoctorId()
  const [items, setItems] = useState<Appt[]>([])
  const [view, setView] = useState<'day'|'week'|'month'|'table'>('table')
  const [confirm, setConfirm] = useState<{ id:number, action:'Confirm'|'Complete'|'Cancel' }|null>(null)
  const [noteFor, setNoteFor] = useState<Appt|null>(null)
  const [noteText, setNoteText] = useState('')

  async function load(){
    try{
      const data: Appt[] = await api.listAppointments()
      setItems(data||[])
    }catch{}
  }
  useEffect(()=>{ if(!loading) load() },[loading])

  const today = new Date()
  const grouped = useMemo(()=>{
    const byDate: Record<string, Appt[]> = {}
    for(const a of items){
      const d = new Date(a.appointment_date)
      const key = d.toISOString().slice(0,10)
      ;(byDate[key] ||= []).push(a)
    }
    return byDate
  },[items])

  async function doAction(id:number, action:'Confirm'|'Complete'|'Cancel'){
    await api.updateAppointmentStatus(id, action)
    setConfirm(null)
    await load()
  }

  async function addNote(appt: Appt){
    // Post a note into doctor-patient room as a message
    try{
      const rooms = await api.listRooms()
      let room = (rooms||[]).find((r:any)=> r.patient_id===appt.patient_id && r.doctor_id===appt.doctor_id)
      if(!room){
        room = await api.createRoom({ patient_id: appt.patient_id, doctor_id: appt.doctor_id })
      }
      await api.postMessage(room.room_id, `[Note for appointment #${appt.appointment_id}] ${noteText}`)
      setNoteText('')
      setNoteFor(null)
    }catch(e){ console.error(e) }
  }

  return (
    <div className="space-y-6">
      <header className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold">Appointments</h1>
          <p className="muted">Manage daily schedule and actions.</p>
        </div>
        <div className="flex gap-2">
          {(['day','week','month','table'] as const).map(v=> (
            <button key={v} onClick={()=>setView(v)} className={`px-3 py-2 rounded-md border ${view===v?'bg-primaryBlue text-white border-primaryBlue':'border-borderElegant hover:bg-slate-50'}`}>{v.toUpperCase()}</button>
          ))}
        </div>
      </header>

      {view==='table' && (
        <div className="card">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Date/Time</th>
                <th>Patient</th>
                <th>Reason</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {items.map(a=>{
                const d = new Date(a.appointment_date)
                return (
                  <tr key={a.appointment_id}>
                    <td>#{a.appointment_id}</td>
                    <td>{d.toLocaleString()}</td>
                    <td>PID {a.patient_id}</td>
                    <td>{a.reason||'-'}</td>
                    <td><StatusBadge status={a.status} /></td>
                    <td className="space-x-2">
                      {a.status==='Scheduled' && (
                        <button className="btn-primary btn-sm" onClick={()=>setConfirm({ id:a.appointment_id, action:'Confirm' })}>Confirm</button>
                      )}
                      {['Scheduled','Confirmed'].includes(a.status) && (
                        <button className="btn-ghost btn-sm" onClick={()=>setConfirm({ id:a.appointment_id, action:'Cancel' })}>Cancel</button>
                      )}
                      {['Scheduled','Confirmed'].includes(a.status) && (
                        <button className="btn-ghost btn-sm" onClick={()=>setConfirm({ id:a.appointment_id, action:'Complete' })}>Complete</button>
                      )}
                      <button className="btn-ghost btn-sm" onClick={()=>{ setNoteFor(a); setNoteText('') }}>Add Notes</button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {view!=='table' && (
        <div className="card">
          <h3 className="font-semibold mb-3">{view==='day'?'Today':view==='week'?'This Week':'This Month'}</h3>
          <div className="grid gap-3">
            {Object.entries(grouped).map(([day, list])=>{
              const dt = new Date(day+'T00:00:00')
              const show = (
                view==='day' ? dt.toDateString()===today.toDateString() :
                view==='week' ? (():boolean=>{ const t=new Date(today); const s=new Date(t); s.setDate(t.getDate()-t.getDay()); const e=new Date(s); e.setDate(s.getDate()+6); return dt>=s && dt<=e })() :
                true
              )
              if(!show) return null
              return (
                <div key={day} className="p-3 rounded-xl border border-borderElegant bg-white">
                  <div className="font-medium text-textLuxury mb-2">{dt.toDateString()}</div>
                  <div className="space-y-2">
                    {list.sort((a,b)=> a.appointment_date.localeCompare(b.appointment_date)).map(a=>{
                      const d = new Date(a.appointment_date)
                      return (
                        <div key={a.appointment_id} className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <StatusBadge status={a.status} />
                            <div className="text-sm">{d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} · PID {a.patient_id} · {a.reason||'Consultation'}</div>
                          </div>
                          <div className="flex gap-2">
                            {a.status==='Scheduled' && <button className="btn-primary btn-sm" onClick={()=>setConfirm({ id:a.appointment_id, action:'Confirm' })}>Confirm</button>}
                            {['Scheduled','Confirmed'].includes(a.status) && <button className="btn-ghost btn-sm" onClick={()=>setConfirm({ id:a.appointment_id, action:'Cancel' })}>Cancel</button>}
                            {['Scheduled','Confirmed'].includes(a.status) && <button className="btn-ghost btn-sm" onClick={()=>setConfirm({ id:a.appointment_id, action:'Complete' })}>Complete</button>}
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      <ConfirmModal
        open={!!confirm}
        title={`${confirm?.action} appointment?`}
        onClose={()=>setConfirm(null)}
        onConfirm={()=> confirm && doAction(confirm.id, confirm.action)}
      >
        <div className="space-y-2">
          <div>This will update the appointment status. This action is logged.</div>
          <div className="muted">Appointments after today will be cancelled automatically. This action is logged.</div>
        </div>
      </ConfirmModal>

      <ConfirmModal
        open={!!noteFor}
        title={`Add notes`}
        confirmText='Send to patient'
        onClose={()=>{ setNoteFor(null); setNoteText('') }}
        onConfirm={()=> noteFor && addNote(noteFor)}
      >
        <textarea className="w-full" rows={4} placeholder="Notes visible to patient via chat" value={noteText} onChange={e=>setNoteText(e.target.value)} />
      </ConfirmModal>
    </div>
  )
}
