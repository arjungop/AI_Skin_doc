import React, { useEffect, useState } from 'react'
import { api } from '../services/api.js'
import { Card, CardTitle, CardDescription } from '../components/Card'
import { useToast } from '../components/Toast.jsx'
import { LuUser, LuStethoscope, LuEye, LuEyeOff, LuSave } from 'react-icons/lu'

export default function DoctorProfile(){
  const [me, setMe] = useState<any>(null)
  const [doc, setDoc] = useState<any>(null)
  const [bio, setBio] = useState('')
  const [specialization, setSpecialization] = useState('')
  const [visible, setVisible] = useState(true)
  const [saving, setSaving] = useState(false)
  const { push } = useToast()

  useEffect(()=>{(async()=>{
    try{
      const m = await api.me(); setMe(m)
      const docs = await api.listDoctors('')
      const mine = (docs||[]).find((d:any)=> d.user_id===m.user_id)
      setDoc(mine||null)
      setSpecialization(mine?.specialization||'')
      // Load locally saved bio / visibility
      try{
        const saved = JSON.parse(localStorage.getItem('doctor_profile') || '{}')
        if(saved.bio) setBio(saved.bio)
        if(typeof saved.visible === 'boolean') setVisible(saved.visible)
      }catch{}
    }catch{}
  })()},[])

  function saveLocal(){
    setSaving(true)
    try{
      localStorage.setItem('doctor_profile', JSON.stringify({ bio, visible }))
      push('Profile saved locally', 'success')
    }catch{
      push('Failed to save', 'error')
    }finally{ setSaving(false) }
  }

  function Row({ label, value }: { label: string; value: string }){
    return (
      <div className="flex items-center py-3 border-b border-slate-100 last:border-0">
        <span className="w-36 text-sm text-slate-500 font-medium">{label}</span>
        <span className="text-slate-900 font-medium">{value || '—'}</span>
      </div>
    )
  }

  return (
    <div className="max-w-3xl mx-auto space-y-6 pb-20">
      <h1 className="text-3xl font-bold text-slate-900 mb-6">Profile</h1>

      <Card>
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-xl bg-primary-50 flex items-center justify-center text-primary-500">
            <LuUser size={20} />
          </div>
          <div>
            <CardTitle>About You</CardTitle>
            <CardDescription>Locally-saved bio and listing preferences</CardDescription>
          </div>
        </div>

        <div className="space-y-5">
          <div>
            <label className="block text-sm font-semibold text-slate-700 mb-2">Short Bio</label>
            <textarea
              className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-slate-900 placeholder:text-slate-400 focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20 outline-none transition-all resize-none"
              rows={4}
              placeholder="A brief introduction visible to patients…"
              value={bio}
              onChange={e=>setBio(e.target.value)}
            />
          </div>

          <div>
            <label className="block text-sm font-semibold text-slate-700 mb-2">Specializations</label>
            <input
              className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-slate-900 placeholder:text-slate-400 focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20 outline-none transition-all"
              value={specialization}
              onChange={e=>setSpecialization(e.target.value)}
              placeholder="e.g., Dermatology, Oncology"
            />
            <p className="text-xs text-slate-400 mt-1.5">Specialization is read from the server — edits here are local-only.</p>
          </div>

          <div className="flex items-center justify-between p-4 rounded-xl bg-slate-50 border border-slate-200 hover:border-primary-200 transition-all">
            <div className="flex items-center gap-3">
              {visible ? <LuEye className="text-slate-400" /> : <LuEyeOff className="text-slate-400" />}
              <div>
                <p className="font-medium text-slate-900">Visible in Listings</p>
                <p className="text-xs text-slate-500">Patients can find and book with you</p>
              </div>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" className="sr-only peer" checked={visible} onChange={e=>setVisible(e.target.checked)} />
              <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-500"></div>
            </label>
          </div>

          <button
            onClick={saveLocal}
            disabled={saving}
            className="flex items-center gap-2 px-5 py-2.5 bg-primary-500 hover:bg-primary-600 text-white rounded-xl font-semibold transition-all disabled:opacity-50"
          >
            <LuSave size={16} /> {saving ? 'Saving…' : 'Save'}
          </button>
        </div>
      </Card>

      <Card>
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-xl bg-emerald-50 flex items-center justify-center text-emerald-600">
            <LuStethoscope size={20} />
          </div>
          <div>
            <CardTitle>Credentials</CardTitle>
            <CardDescription>Account details from the server (read-only)</CardDescription>
          </div>
        </div>

        <div className="bg-slate-50 rounded-xl border border-slate-200 px-4">
          <Row label="Username" value={me?.username} />
          <Row label="Email" value={me?.email} />
          <Row label="Role" value={me?.role} />
          <Row label="Doctor ID" value={doc?.doctor_id ? `#${doc.doctor_id}` : ''} />
          <Row label="Specialization" value={doc?.specialization} />
        </div>
      </Card>
    </div>
  )
}
