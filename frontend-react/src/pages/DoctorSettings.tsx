import React, { useEffect, useState } from 'react'
import { Card, CardTitle, CardDescription } from '../components/Card'
import { LuBell, LuEye, LuMoon, LuLanguages, LuLogOut, LuLock, LuShield } from 'react-icons/lu'

const LS_KEY = 'doctor_settings'

export default function DoctorSettings() {
  const [s, setS] = useState({ notifications: true, privacyPublic: true, theme: 'light', language: 'en' })

  useEffect(() => {
    try {
      const v = localStorage.getItem(LS_KEY)
      if (v) setS(JSON.parse(v))
    } catch { }
  }, [])

  useEffect(() => {
    try { localStorage.setItem(LS_KEY, JSON.stringify(s)) } catch { }
  }, [s])

  function logoutAll() {
    try { localStorage.clear(); window.location.href = '/login' } catch { }
  }

  return (
    <div className="max-w-3xl mx-auto space-y-6 pb-20">
      <h1 className="text-3xl font-bold text-slate-900 mb-6">Settings</h1>

      <Card>
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-xl bg-primary-50 flex items-center justify-center text-primary-500">
            <LuShield size={20} />
          </div>
          <div>
            <CardTitle>Preferences</CardTitle>
            <CardDescription>Manage how you interact with the platform</CardDescription>
          </div>
        </div>

        <div className="space-y-6">
          <div className="flex items-center justify-between p-4 rounded-xl bg-slate-50 border border-slate-200 hover:border-primary-200 transition-all">
            <div className="flex items-center gap-3">
              <LuBell className="text-slate-400" />
              <div>
                <p className="font-medium text-slate-900">Notifications</p>
                <p className="text-xs text-slate-500">Receive alerts about appointments</p>
              </div>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" className="sr-only peer" checked={s.notifications} onChange={e => setS({ ...s, notifications: e.target.checked })} />
              <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-500"></div>
            </label>
          </div>

          <div className="flex items-center justify-between p-4 rounded-xl bg-slate-50 border border-slate-200 hover:border-primary-200 transition-all">
            <div className="flex items-center gap-3">
              <LuEye className="text-slate-400" />
              <div>
                <p className="font-medium text-slate-900">Public Profile</p>
                <p className="text-xs text-slate-500">Allow patients to find you in search</p>
              </div>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" className="sr-only peer" checked={s.privacyPublic} onChange={e => setS({ ...s, privacyPublic: e.target.checked })} />
              <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-500"></div>
            </label>
          </div>

          <div className="flex items-center justify-between p-4 rounded-xl bg-slate-50 border border-slate-200 hover:border-primary-200 transition-all">
            <div className="flex items-center gap-3">
              <LuMoon className="text-slate-400" />
              <span className="font-medium text-slate-900">Theme</span>
            </div>
            <select
              className="bg-white border border-slate-200 text-slate-900 text-sm rounded-lg focus:ring-primary-500 focus:border-primary-500 block p-2.5 outline-none"
              value={s.theme}
              onChange={e => setS({ ...s, theme: e.target.value })}
            >
              <option value="light">Light</option>
              <option value="dark">Dark</option>
              <option value="system">System</option>
            </select>
          </div>

          <div className="flex items-center justify-between p-4 rounded-xl bg-slate-50 border border-slate-200 hover:border-primary-200 transition-all">
            <div className="flex items-center gap-3">
              <LuLanguages className="text-slate-400" />
              <span className="font-medium text-slate-900">Language</span>
            </div>
            <select
              className="bg-white border border-slate-200 text-slate-900 text-sm rounded-lg focus:ring-primary-500 focus:border-primary-500 block p-2.5 outline-none"
              value={s.language}
              onChange={e => setS({ ...s, language: e.target.value })}
            >
              <option value="en">English</option>
              <option value="es">Español</option>
              <option value="fr">Français</option>
            </select>
          </div>
        </div>
      </Card>

      <Card>
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-xl bg-rose-50 flex items-center justify-center text-rose-600">
            <LuLock size={20} />
          </div>
          <div>
            <CardTitle>Security</CardTitle>
            <CardDescription>Protect your account</CardDescription>
          </div>
        </div>

        <div className="space-y-4">
          <button className="w-full flex items-center justify-between p-4 rounded-xl bg-slate-50 border border-slate-200 hover:bg-slate-100 transition-all text-left">
            <span className="font-medium text-slate-900">Change Password</span>
            <span className="text-xs bg-slate-100 border border-slate-200 px-2 py-1 rounded text-slate-400">Coming Soon</span>
          </button>

          <button
            onClick={logoutAll}
            className="w-full flex items-center justify-center gap-2 p-4 rounded-xl bg-rose-50 text-rose-600 border border-rose-200 hover:bg-rose-100 transition-all font-semibold"
          >
            <LuLogOut size={18} /> Logout All Devices
          </button>
        </div>
      </Card>
    </div>
  )
}
