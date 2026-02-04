import { useEffect, useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { api } from '../services/api'
import { LuMail, LuLock, LuArrowRight, LuCheck } from 'react-icons/lu'
import { motion } from 'framer-motion'

export default function Login() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [show, setShow] = useState(false)
  const [msg, setMsg] = useState('')
  const [loading, setLoading] = useState(false)
  const [remember, setRemember] = useState(false)
  const navigate = useNavigate()

  useEffect(() => {
    const saved = localStorage.getItem('remember_email')
    if (saved) { setEmail(saved); setRemember(true) }
  }, [])

  const onSubmit = async (e) => {
    e.preventDefault()
    setMsg('')
    if (!email || !password) { setMsg('Please enter email and password'); return }
    try {
      setLoading(true)
      const res = await api.login({ email, password })
      if (res.patient_id) localStorage.setItem('patient_id', res.patient_id)
      if (res.user_id) localStorage.setItem('user_id', res.user_id)
      const display = (res.first_name && res.last_name) ? (res.first_name + ' ' + res.last_name) : (res.username || '')
      if (display) localStorage.setItem('username', display)
      if (res.role) localStorage.setItem('role', res.role)
      if (res.doctor_id) localStorage.setItem('doctor_id', res.doctor_id)
      if (res.access_token) localStorage.setItem('access_token', res.access_token)
      if (remember) localStorage.setItem('remember_email', email); else localStorage.removeItem('remember_email')

      const r = (res.role || '').toUpperCase()
      const dest = r === 'DOCTOR' ? '/doctor' : (r === 'ADMIN' ? '/admin' : '/journey')

      setMsg('Welcome back')
      setTimeout(() => navigate(dest), 500)
    } catch (err) {
      setMsg(err.message || 'Login failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#FCFCFC] p-6 relative">
      <div className="w-full max-w-lg">
        {/* Logo Brand */}
        <div className="text-center mb-10">
          <h1 className="font-serif text-5xl font-bold text-[#10201D] mb-2 tracking-tight">Skin.AI</h1>
          <p className="text-slate-400 font-medium uppercase tracking-widest text-xs">Precision Dermatology</p>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="card-ceramic p-10 md:p-12"
        >
          <div className="mb-8">
            <h2 className="font-serif text-3xl font-bold text-[#10201D] mb-1">Welcome Back</h2>
            <p className="text-slate-500">Sign in to your dashboard.</p>
          </div>

          <form onSubmit={onSubmit} className="space-y-6">
            <div className="space-y-2">
              <label className="text-xs uppercase tracking-widest font-bold text-[#10201D]">Email</label>
              <div className="relative">
                <input
                  className="pl-4 pr-4 bg-surface border-2 border-border-light focus:bg-white focus:border-accent-medical transition-colors rounded-xl py-4 font-medium"
                  type="email"
                  placeholder="name@example.com"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  required
                />
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <label className="text-xs uppercase tracking-widest font-bold text-[#10201D]">Password</label>
                <Link className="text-xs font-bold text-slate-400 hover:text-[#10201D] transition-colors" to="/forgot-password">Forgot?</Link>
              </div>

              <div className="relative">
                <input
                  className="pl-4 pr-16 bg-surface border-2 border-border-light focus:bg-white focus:border-accent-medical transition-colors rounded-xl py-4 font-medium"
                  type={show ? 'text' : 'password'}
                  placeholder="••••••••"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  required
                />
                <button
                  type="button"
                  className="absolute right-4 top-1/2 -translate-y-1/2 text-xs font-bold text-slate-400 hover:text-[#10201D] uppercase tracking-wider"
                  onClick={() => setShow(!show)}
                >
                  {show ? 'Hide' : 'Show'}
                </button>
              </div>
            </div>

            <div className="py-2">
              <label className="flex items-center gap-3 cursor-pointer group">
                <div className={`w-6 h-6 rounded border-2 flex items-center justify-center transition-colors ${remember ? 'bg-accent-medical border-accent-medical' : 'bg-white border-border-medium group-hover:border-accent-medical'}`}>
                  {remember && <LuCheck className="text-white text-xs stroke-[4]" />}
                </div>
                <input type="checkbox" checked={remember} onChange={e => setRemember(e.target.checked)} className="hidden" />
                <span className="text-sm text-slate-600 font-medium group-hover:text-[#10201D] transition-colors">Keep me signed in</span>
              </label>
            </div>

            <button
              className="w-full btn-primary py-5 text-lg shadow-none hover:shadow-xl relative overflow-hidden group rounded-xl"
              type="submit"
              disabled={loading}
            >
              <span className="relative z-10 flex items-center justify-center gap-3 font-bold">
                {loading ? 'Please Wait...' : 'Sign In'}
                {!loading && <LuArrowRight className="group-hover:translate-x-1 transition-transform stroke-[3]" />}
              </span>
            </button>

            {msg && (
              <div className={`p-4 rounded-xl text-center text-sm font-bold ${msg.includes('Welcome') ? 'bg-green-100 text-green-800' : 'bg-red-50 text-red-600'}`}>
                {msg}
              </div>
            )}
          </form>

          <div className="mt-10 text-center">
            <p className="text-slate-500 font-medium">
              New to Skin.AI? <Link className="text-[#10201D] font-bold hover:underline" to="/register">Create Account</Link>
            </p>
          </div>
        </motion.div>

        <div className="mt-8 text-center">
          <Link className="text-xs text-slate-400 font-bold uppercase tracking-widest hover:text-[#10201D] transition-colors" to="/apply-doctor">Apply as Specialist</Link>
        </div>
      </div>
    </div>
  )
}
