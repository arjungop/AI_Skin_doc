import React, { useEffect, useState } from 'react'
import { api } from '../services/api.js'

export default function DoctorProfile(){
  const [me, setMe] = useState<any>(null)
  const [doc, setDoc] = useState<any>(null)
  const [bio, setBio] = useState('')
  const [specialization, setSpecialization] = useState('')
  const [visible, setVisible] = useState(true)

  useEffect(()=>{(async()=>{
    try{
      const m = await api.me(); setMe(m)
      const docs = await api.listDoctors(''); const mine = (docs||[]).find((d:any)=> d.user_id===m.user_id)
      setDoc(mine||null)
      setSpecialization(mine?.specialization||'')
    }catch{}
  })()},[])

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold">Profile</h1>
        <p className="muted">Manage your public profile and preferences.</p>
      </header>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="card space-y-3">
          <div className="font-semibold">About</div>
          <textarea className="w-full" rows={5} placeholder="Short bio (saved locally)" value={bio} onChange={e=>setBio(e.target.value)} />
          <div>
            <label className="block text-sm mb-1">Specializations</label>
            <input value={specialization} onChange={e=>setSpecialization(e.target.value)} placeholder="e.g., Dermatology, Oncology" />
            <div className="muted text-xs mt-1">Note: Specialization field is shown from server; editing here is local-only.</div>
          </div>
          <label className="flex items-center gap-2"><input type="checkbox" checked={visible} onChange={e=>setVisible(e.target.checked)} /> Visible in listings</label>
          <button className="btn-primary">Save (local)</button>
        </div>
        <div className="card space-y-2">
          <div className="font-semibold">Credentials (view-only)</div>
          <div className="grid grid-cols-3 gap-2 items-center"><div className="muted">Username</div><div className="col-span-2">{me?.username||'-'}</div></div>
          <div className="grid grid-cols-3 gap-2 items-center"><div className="muted">Email</div><div className="col-span-2">{me?.email||'-'}</div></div>
          <div className="grid grid-cols-3 gap-2 items-center"><div className="muted">Role</div><div className="col-span-2">{me?.role||'-'}</div></div>
          <div className="grid grid-cols-3 gap-2 items-center"><div className="muted">Doctor ID</div><div className="col-span-2">{doc?.doctor_id||'-'}</div></div>
          <div className="grid grid-cols-3 gap-2 items-center"><div className="muted">Specialization</div><div className="col-span-2">{doc?.specialization||'-'}</div></div>
        </div>
      </div>
    </div>
  )
}
