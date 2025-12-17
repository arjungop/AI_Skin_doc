import React, { useEffect, useState } from 'react'

type Toggles = { notifications:boolean; privacyPublic:boolean; theme:'light'|'dark'|'system'; language:'en'|'es'|'fr' }

const LS_KEY = 'doctor_settings'

export default function DoctorSettings(){
  const [s, setS] = useState<Toggles>({ notifications:true, privacyPublic:true, theme:'light', language:'en' })
  useEffect(()=>{ try{ const v = localStorage.getItem(LS_KEY); if(v) setS(JSON.parse(v)) }catch{} },[])
  useEffect(()=>{ try{ localStorage.setItem(LS_KEY, JSON.stringify(s)) }catch{} },[s])

  function logoutAll(){ try{ localStorage.clear(); window.location.href = '/login' }catch{} }

  return (
    <div className="space-y-6">
      <div className="card">
        <h3 className="font-semibold mb-2">Preferences</h3>
        <div className="space-y-3">
          <label className="flex items-center gap-2"><input type="checkbox" checked={s.notifications} onChange={e=>setS({...s, notifications:e.target.checked})} /> Enable notifications</label>
          <label className="flex items-center gap-2"><input type="checkbox" checked={s.privacyPublic} onChange={e=>setS({...s, privacyPublic:e.target.checked})} /> Public profile visibility</label>
          <div className="flex items-center gap-3">
            <div className="w-40">Theme</div>
            <select value={s.theme} onChange={e=>setS({...s, theme:e.target.value as any})}>
              <option value="light">Light</option>
              <option value="dark">Dark</option>
              <option value="system">System</option>
            </select>
          </div>
          <div className="flex items-center gap-3">
            <div className="w-40">Language</div>
            <select value={s.language} onChange={e=>setS({...s, language:e.target.value as any})}>
              <option value="en">English</option>
              <option value="es">Español</option>
              <option value="fr">Français</option>
            </select>
          </div>
        </div>
      </div>

      <div className="card">
        <h3 className="font-semibold mb-2">Security</h3>
        <div className="space-y-3">
          <button className="btn-ghost">Change password (coming soon)</button>
          <button className="btn-primary" onClick={logoutAll}>Logout all devices</button>
        </div>
      </div>
    </div>
  )
}

