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
  const [tz, setTz] = useState<string>(Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC')
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)

  async function load(){
    if(!doctorId) return
    try{
      const rows: any[] = await api.listDoctorAvailability(doctorId)
      setSlots((rows||[]).map(r=>({ weekday:r.weekday, start_time:r.start_time, end_time:r.end_time, timezone:r.timezone })))
    }catch{}
  }
  useEffect(()=>{ load() },[doctorId])

  function addSlot(day:number){
    setSlots(s=>[...s, { weekday:day, start_time:'09:00', end_time:'17:00', timezone: tz }])
  }

  function removeSlot(globalIdx:number){
    setSlots(s=> s.filter((_,i)=> i!==globalIdx))
  }

  function updateSlot(globalIdx:number, field: 'start_time'|'end_time', value:string){
    setSlots(s=> s.map((p,i)=> i===globalIdx ? { ...p, [field]: value } : p))
  }

  async function save(){
    if(!doctorId) return
    setSaving(true)
    try{
      await api.setDoctorAvailability(doctorId, slots.map(s=>({ ...s, timezone: tz })))
      setSaved(true)
      setTimeout(()=>setSaved(false), 2000)
    }finally{ setSaving(false) }
  }

  // Build a map from weekday → { slot, globalIndex }[]
  const grouped = useMemo(()=>{
    const m: Record<number, { slot:Slot; idx:number }[]> = {}
    slots.forEach((s,i)=>{ (m[s.weekday] ||= []).push({ slot:s, idx:i }) })
    return m
  },[slots])

  const TIMEZONES = [
    'UTC',
    'America/New_York',
    'America/Chicago',
    'America/Denver',
    'America/Los_Angeles',
    'Europe/London',
    'Europe/Paris',
    'Asia/Kolkata',
    'Asia/Tokyo',
    'Australia/Sydney',
  ]

  return (
    <div className="space-y-6">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Availability</h1>
          <p className="text-sm text-slate-500">Weekly recurring slots with timezone awareness.</p>
        </div>
        <div className="flex items-center gap-2">
          <select
            className="input"
            value={tz}
            onChange={e=>setTz(e.target.value)}
          >
            {TIMEZONES.includes(tz) ? null : <option value={tz}>{tz}</option>}
            {TIMEZONES.map(t=> <option key={t} value={t}>{t}</option>)}
          </select>
          <button
            className="btn btn-primary"
            onClick={save}
            disabled={saving}
          >
            {saving ? 'Saving…' : saved ? '✓ Saved' : 'Save'}
          </button>
        </div>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {WEEKDAYS.map((d,di)=> {
          const daySlots = grouped[di] || []
          return (
            <div key={di} className="card">
              <div className="flex items-center justify-between mb-3">
                <div className="font-medium">
                  {d}
                  {daySlots.length > 0 && (
                    <span className="ml-2 text-xs text-slate-400">{daySlots.length} slot{daySlots.length!==1?'s':''}</span>
                  )}
                </div>
                <button className="btn btn-ghost btn-sm" onClick={()=>addSlot(di)}>+ Add slot</button>
              </div>
              <div className="space-y-2">
                {daySlots.map(({ slot:s, idx })=> (
                  <div key={idx} className="flex items-center gap-2">
                    <input
                      className="input w-28"
                      type="time"
                      value={s.start_time}
                      onChange={e=> updateSlot(idx, 'start_time', e.target.value)}
                    />
                    <span className="text-slate-400 text-sm">to</span>
                    <input
                      className="input w-28"
                      type="time"
                      value={s.end_time}
                      onChange={e=> updateSlot(idx, 'end_time', e.target.value)}
                    />
                    <button className="btn btn-ghost btn-sm" onClick={()=> removeSlot(idx)}>✕</button>
                  </div>
                ))}
                {daySlots.length===0 && (
                  <div className="text-sm text-slate-400">No slots — click + Add slot</div>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
