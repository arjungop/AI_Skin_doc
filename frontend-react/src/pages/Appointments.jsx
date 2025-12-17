import { useEffect, useMemo, useState } from 'react'
import { api } from '../services/api'
import { useToast } from '../components/Toast.jsx'

export default function Appointments(){
  const [list, setList] = useState([])
  const [doctors, setDoctors] = useState([])
  const [doctorId, setDoctorId] = useState('')
  const [date, setDate] = useState('')
  const [slots, setSlots] = useState([])
  const [selectedSlot, setSelectedSlot] = useState('')
  const [reason, setReason] = useState('')
  const pid = parseInt(localStorage.getItem('patient_id'))
  const did = parseInt(localStorage.getItem('doctor_id'))
  const role = (localStorage.getItem('role')||'').toUpperCase()
  const { push } = useToast()
  const [availability, setAvailability] = useState([])

  const [loading, setLoading] = useState(false)

  async function load(){
    setLoading(true)
    try{
      let data = await api.listAppointments()
      if (role === 'DOCTOR' && did) data = data.filter(a=> a.doctor_id === did)
      else if (pid) data = data.filter(a=> a.patient_id === pid)
      setList(data)
      const docs = await api.listDoctors()
      setDoctors(docs)
    }catch{}
    setLoading(false)
  }
  useEffect(()=>{ load() },[])
  useEffect(()=>{ 
    if (doctorId) api.listDoctorAvailability(doctorId).then(setAvailability).catch(()=>setAvailability([])) 
  }, [doctorId])

  useEffect(()=>{
    // regenerate slots when date or availability changes
    if (!date || availability.length===0) { setSlots([]); return }
    const d = new Date(date)
    const weekday = d.getDay() // 0 Sun
    const avail = availability.filter(a=> a.weekday === ((weekday+6)%7)) // backend uses 0=Mon
    const dur = 30
    const s = []
    for (const a of avail){
      let [h1,m1] = a.start_time.split(':').map(Number)
      let [h2,m2] = a.end_time.split(':').map(Number)
      let t1 = h1*60+m1, t2 = h2*60+m2
      for (let t=t1; t+dur<=t2; t+=dur){
        const hh = String(Math.floor(t/60)).padStart(2,'0')
        const mm = String(t%60).padStart(2,'0')
        s.push(`${hh}:${mm}`)
      }
    }
    setSlots(s)
    setSelectedSlot('')
  }, [date, availability])

  async function submit(e){
    e.preventDefault()
    if (!pid) return push('Login required','error')
    if (!selectedSlot || !date) return push('Pick a date and slot','error')
    try{
      const iso = new Date(`${date}T${selectedSlot}:00`).toISOString()
      await api.createAppointment({ patient_id: pid, doctor_id: parseInt(doctorId), appointment_date: iso, reason: reason||undefined })
      push('Appointment booked','success')
      setDoctorId(''); setDate(''); setReason(''); load()
    }catch(err){ push(err.message||'Booking failed','error') }
  }

  return (
    <div>
      <h1>Appointments</h1>
      {role !== 'DOCTOR' && (
        <form onSubmit={submit} className="card stack">
          <select value={doctorId} onChange={e=>setDoctorId(e.target.value)} required>
            <option value="" disabled>Select Doctor</option>
          {doctors.map(d=> (
            <option key={d.doctor_id} value={d.doctor_id}>{d.username || `Doctor #${d.doctor_id}`} {d.specialization?`(${d.specialization})`:''}</option>
          ))}
          </select>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <input type="date" value={date} onChange={e=>setDate(e.target.value)} required />
            <input placeholder="Reason" value={reason} onChange={e=>setReason(e.target.value)} />
          </div>
          {date && slots.length>0 ? (
            <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 gap-2">
              {slots.map(s=> (
                <button type="button" key={s} onClick={()=>setSelectedSlot(s)} className={("px-3 py-2 rounded-md border "+(selectedSlot===s?"bg-primary text-white border-primary":"border-borderGray hover:bg-slate-50"))}>{s}</button>
              ))}
            </div>
          ) : (
            <div className="muted">{date? 'No slots for this day' : 'Choose a date to see available slots'}</div>
          )}
          <button className="btn-primary" type="submit">Book</button>
        </form>
      )}
      {doctorId && availability.length===0 && (
        <div className="muted" style={{marginTop:8}}>No availability set for this doctor. Try another doctor or time.</div>
      )}
      <div className="card">
        <table>
          <thead><tr><th>ID</th><th>Doctor</th><th>Time</th><th>Reason</th><th>Status</th><th>Actions</th></tr></thead>
          <tbody>
            {loading && (
              <tr><td colSpan="6">
                <div className="grid grid-cols-3 gap-2">
                  <div className="skeleton h-6"></div>
                  <div className="skeleton h-6"></div>
                  <div className="skeleton h-6"></div>
                </div>
              </td></tr>
            )}
            {!loading && list.map(a=> {
              const d = doctors.find(x=> x.doctor_id === a.doctor_id)
              return (
                <tr key={a.appointment_id}>
                  <td>{a.appointment_id}</td>
                  <td>{d ? `${d.username}${d.specialization?` (${d.specialization})`:''}` : a.doctor_id}</td>
                  <td>{a.appointment_date}</td>
                  <td>{a.reason||'-'}</td>
                  <td><span className="chip">{a.status}</span></td>
                  <td>
                    {role==='PATIENT' && ['Scheduled','Confirmed'].includes(a.status) && (
                      <button className="btn-primary" onClick={async()=>{ await api.updateAppointmentStatus(a.appointment_id,'Cancelled'); load() }}>Cancel</button>
                    )}
                    {role==='DOCTOR' && a.status==='Scheduled' && (
                      <button className="btn-primary" onClick={async()=>{ await api.updateAppointmentStatus(a.appointment_id,'Confirmed'); load() }}>Confirm</button>
                    )}
                    {role==='DOCTOR' && ['Scheduled','Confirmed'].includes(a.status) && (
                      <button className="btn-primary" onClick={async()=>{ await api.updateAppointmentStatus(a.appointment_id,'Cancelled'); load() }}>Cancel</button>
                    )}
                    {role==='DOCTOR' && ['Scheduled','Confirmed'].includes(a.status) && (
                      <button className="btn-primary" onClick={async()=>{ await api.updateAppointmentStatus(a.appointment_id,'Completed'); load() }}>Complete</button>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
