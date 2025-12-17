import { useState } from 'react'
import { api } from '../services/api'

export default function ApplyDoctor(){
  const [form, setForm] = useState({ first_name:'', last_name:'', email:'', password:'', specialization:'', license_no:'', department:'' })
  const [msg, setMsg] = useState('')
  const set = (k,v)=> setForm(prev=>({...prev,[k]:v}))

  const submit = async (e)=>{
    e.preventDefault()
    setMsg('')
    try{
      await api.applyDoctor({ ...form })
      setMsg('Application submitted. An admin will review it.')
    }catch(err){ setMsg(err.message) }
  }

  return (
    <div className="min-h-screen bg-background grid place-items-center px-4">
      <div className="w-full max-w-lg">
        <div className="text-center mb-4">
          <div className="mx-auto h-10 w-10 rounded-md bg-gradient-to-tr from-primary to-accent" />
        </div>
        <h1 className="text-2xl font-semibold text-center">Apply as Doctor</h1>
        <p className="text-center text-textMuted">Your account will be reviewed by an admin.</p>
        <form onSubmit={submit} className="card stack mt-3">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <input className="border border-borderGray rounded-lg px-3 py-2" placeholder="First name" value={form.first_name} onChange={e=>set('first_name', e.target.value)} required />
            <input className="border border-borderGray rounded-lg px-3 py-2" placeholder="Last name" value={form.last_name} onChange={e=>set('last_name', e.target.value)} required />
          </div>
          <input className="border border-borderGray rounded-lg px-3 py-2" type="email" placeholder="Email" value={form.email} onChange={e=>set('email', e.target.value)} required />
          <input className="border border-borderGray rounded-lg px-3 py-2" type="password" placeholder="Password" value={form.password} onChange={e=>set('password', e.target.value)} required />
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <input className="border border-borderGray rounded-lg px-3 py-2" placeholder="Specialization" value={form.specialization} onChange={e=>set('specialization', e.target.value)} />
            <input className="border border-borderGray rounded-lg px-3 py-2" placeholder="License No" value={form.license_no} onChange={e=>set('license_no', e.target.value)} />
          </div>
          <select className="border border-borderGray rounded-lg px-3 py-2" value={form.department} onChange={e=>set('department', e.target.value)} required>
            <option value="">Select Department</option>
            <option>Dermatology</option>
            <option>Cardiology</option>
            <option>Orthopedics</option>
            <option>Pediatrics</option>
            <option>Oncology</option>
            <option>General Medicine</option>
          </select>
          <button className="btn-primary" type="submit">Submit Application</button>
          {msg && <div className="text-success text-sm">{msg}</div>}
        </form>
      </div>
    </div>
  )
}
