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
      <h1 className="text-3xl font-bold text-text-primary mb-6">Settings</h1>

      <Card variant="glass">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-xl bg-primary-500/10 flex items-center justify-center text-primary-400">
            <LuShield size={20} />
          </div>
          <div>
            <CardTitle>Preferences</CardTitle>
            <CardDescription>Manage how you interact with the platform</CardDescription>
          </div>
        </div>

        <div className="space-y-6">
          <div className="flex items-center justify-between p-4 rounded-xl bg-white/5 border border-white/10 hover:border-primary-500/30 transition-all">
            <div className="flex items-center gap-3">
              <LuBell className="text-text-tertiary" />
              <div>
                <p className="font-medium text-text-primary">Notifications</p>
                <p className="text-xs text-text-secondary">Receive alerts about appointments</p>
              </div>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" className="sr-only peer" checked={s.notifications} onChange={e => setS({ ...s, notifications: e.target.checked })} />
              <div className="w-11 h-6 bg-white/10 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-500"></div>
            </label>
          </div>

          <div className="flex items-center justify-between p-4 rounded-xl bg-white/5 border border-white/10 hover:border-primary-500/30 transition-all">
            <div className="flex items-center gap-3">
              <LuEye className="text-text-tertiary" />
              <div>
                <p className="font-medium text-text-primary">Public Profile</p>
                <p className="text-xs text-text-secondary">Allow patients to find you in search</p>
              </div>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" className="sr-only peer" checked={s.privacyPublic} onChange={e => setS({ ...s, privacyPublic: e.target.checked })} />
              <div className="w-11 h-6 bg-white/10 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-500"></div>
            </label>
          </div>

          <div className="flex items-center justify-between p-4 rounded-xl bg-white/5 border border-white/10 hover:border-primary-500/30 transition-all">
            <div className="flex items-center gap-3">
              <LuMoon className="text-text-tertiary" />
              <span className="font-medium text-text-primary">Theme</span>
            </div>
            <select
              className="bg-surface-elevated border border-white/10 text-text-primary text-sm rounded-lg focus:ring-primary-500 focus:border-primary-500 block p-2.5 outline-none"
              value={s.theme}
              onChange={e => setS({ ...s, theme: e.target.value })}
            >
              <option value="light">Light</option>
              <option value="dark">Dark</option>
              <option value="system">System</option>
            </select>
          </div>

          <div className="flex items-center justify-between p-4 rounded-xl bg-white/5 border border-white/10 hover:border-primary-500/30 transition-all">
            <div className="flex items-center gap-3">
              <LuLanguages className="text-text-tertiary" />
              <span className="font-medium text-text-primary">Language</span>
            </div>
            <select
              className="bg-surface-elevated border border-white/10 text-text-primary text-sm rounded-lg focus:ring-primary-500 focus:border-primary-500 block p-2.5 outline-none"
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

      <Card variant="glass">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-xl bg-danger/10 flex items-center justify-center text-danger">
            <LuLock size={20} />
          </div>
          <div>
            <CardTitle>Security</CardTitle>
            <CardDescription>Protect your account</CardDescription>
          </div>
        </div>

        <div className="space-y-4">
          <button className="w-full flex items-center justify-between p-4 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10 transition-all text-left">
            <span className="font-medium text-text-primary">Change Password</span>
            <span className="text-xs bg-white/10 px-2 py-1 rounded text-text-muted">Coming Soon</span>
          </button>

          <button
            onClick={logoutAll}
            className="w-full flex items-center justify-center gap-2 p-4 rounded-xl bg-danger/10 text-danger border border-danger/20 hover:bg-danger/20 transition-all font-semibold"
          >
            <LuLogOut size={18} /> Logout All Devices
          </button>
        </div>
      </Card>
    </div>
  )
}
