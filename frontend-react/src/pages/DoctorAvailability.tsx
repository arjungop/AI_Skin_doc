import React, { useEffect, useMemo, useState } from 'react'
import { api } from '../services/api.js'

type Slot = { weekday:number; start_time:string; end_time:string; timezone?:string }

const WEEKDAYS = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

function useDoctorId(){
  const [doctorId, setDoctorId] = useState<number|null>(null)
  useEffect(()=>{(async()=>{
    try{
      const me = await api.me(); const docs = await api.listDoctors('')
      const mine = (docs||[]).find((d:any)=> d.user_id===me.user_id)
      setDoctorId(mine?.doctor_id ?? null)
    }catch{ setDoctorId(null) }
  })()},[])
  return doctorId
}

export default function DoctorAvailability(){
  const doctorId = useDoctorId()
  const [slots, setSlots] = useState<Slot[]>([])
  const [tz, setTz] = useState<string>(Intl.DateTimeFormat().resolvedOptions().timeZone || 'local')
  const [saving, setSaving] = useState(false)

  async function load(){
    if(!doctorId) return
    try{
      const rows: any[] = await api.listDoctorAvailability(doctorId)
      setSlots((rows||[]).map(r=>({ weekday:r.weekday, start_time:r.start_time, end_time:r.end_time, timezone:r.timezone })))
    }catch{}
  }
  useEffect(()=>{ load() },[doctorId])

  function addSlot(day:number){ setSlots(s=>[...s, { weekday:day, start_time:'09:00', end_time:'17:00', timezone: tz }]) }
  function removeSlot(i:number){ setSlots(s=> s.filter((_,idx)=> idx!==i)) }

  async function save(){
    if(!doctorId) return
    setSaving(true)
    try{
      await api.setDoctorAvailability(doctorId, slots.map(s=>({ ...s, timezone: tz })))
    }finally{ setSaving(false) }
  }

  const grouped = useMemo(()=>{
    const m: Record<number, Slot[]> = {}
    for(const s of slots){ (m[s.weekday] ||= []).push(s) }
    return m
  },[slots])

  return (
    <div className="space-y-6">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Availability</h1>
          <p className="muted">Weekly recurring slots with timezone awareness.</p>
        </div>
        <div className="flex items-center gap-2">
          <select className="px-3 py-2 rounded-md border border-borderElegant" value={tz} onChange={e=>setTz(e.target.value)}>
            <option value={tz}>{tz}</option>
            <option value="local">local</option>
          </select>
          <button className="btn-primary" onClick={save} disabled={saving}>{saving?'Saving…':'Save'}</button>
        </div>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {WEEKDAYS.map((d,di)=> (
          <div key={di} className="card">
            <div className="flex items-center justify-between mb-3">
              <div className="font-medium">{d}</div>
              <button className="btn-ghost btn-sm" onClick={()=>addSlot(di)}>Add slot</button>
            </div>
            <div className="space-y-2">
              {(grouped[di]||[]).map((s,idx)=> (
                <div key={idx} className="flex items-center gap-2">
                  <input className="w-28" type="time" value={s.start_time} onChange={e=> setSlots(prev=> prev.map((p,i)=> i===slots.indexOf(s)? { ...p, start_time:e.target.value }:p))} />
                  <span className="text-textLuxuryMuted">to</span>
                  <input className="w-28" type="time" value={s.end_time} onChange={e=> setSlots(prev=> prev.map((p,i)=> i===slots.indexOf(s)? { ...p, end_time:e.target.value }:p))} />
                  <button className="btn-ghost btn-sm" onClick={()=> removeSlot(slots.indexOf(s))}>Remove</button>
                </div>
              ))}
              {!(grouped[di]||[]).length && (
                <div className="muted">No slots</div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
