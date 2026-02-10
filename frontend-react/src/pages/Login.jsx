import { useEffect, useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { api } from '../services/api'
import { LuArrowRight, LuCheck, LuEye, LuEyeOff, LuActivity } from 'react-icons/lu'
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
    <div className="min-h-screen flex bg-white">
      {/* Left Panel - Soft Brand Aesthetic */}
      <div className="hidden lg:flex lg:w-1/2 relative bg-slate-50 overflow-hidden items-center justify-center p-12">
        <div className="absolute inset-0">
          <div className="absolute top-0 left-0 w-full h-full bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-primary-100/40 via-transparent to-transparent" />
          <div className="absolute bottom-0 left-0 w-full h-full bg-[radial-gradient(ellipse_at_bottom_left,_var(--tw-gradient-stops))] from-secondary-100/40 via-transparent to-transparent" />
        </div>

        <div className="relative z-10 max-w-lg">
          <div className="w-16 h-16 rounded-2xl bg-white shadow-soft-xl flex items-center justify-center mb-8 text-primary-500">
            <LuActivity size={32} />
          </div>

          <h1 className="text-5xl font-bold text-slate-900 tracking-tight leading-tight mb-6">
            Your skin health, <br />
            <span className="text-primary-500">demystified.</span>
          </h1>

          <p className="text-xl text-slate-500 leading-relaxed mb-8">
            Join thousands of patients using medical-grade AI to track, analyze, and improve their skin health from home.
          </p>

          <div className="flex gap-4">
            <div className="flex -space-x-3">
              {[1, 2, 3, 4].map(i => (
                <div key={i} className={`w-10 h-10 rounded-full border-2 border-white bg-slate-200`} />
              ))}
            </div>
            <div className="flex flex-col justify-center">
              <span className="text-sm font-bold text-slate-900">10,000+</span>
              <span className="text-xs text-slate-500">Active users</span>
            </div>
          </div>
        </div>
      </div>

      {/* Right Panel - Login Form */}
      <div className="w-full lg:w-1/2 flex items-center justify-center p-6 lg:p-24">
        <div className="w-full max-w-md space-y-8">
          <div className="text-center lg:text-left">
            <h2 className="text-3xl font-bold text-slate-900 tracking-tight">Welcome back</h2>
            <p className="mt-2 text-slate-500">Please enter your details to sign in.</p>
          </div>

          <form onSubmit={onSubmit} className="space-y-6">
            <div className="space-y-5">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Email address</label>
                <input
                  type="email"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  className="input w-full"
                  placeholder="name@example.com"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Password</label>
                <div className="relative">
                  <input
                    type={show ? 'text' : 'password'}
                    value={password}
                    onChange={e => setPassword(e.target.value)}
                    className="input w-full pr-10"
                    placeholder="••••••••"
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setShow(!show)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
                  >
                    {show ? <LuEyeOff size={20} /> : <LuEye size={20} />}
                  </button>
                </div>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={remember}
                  onChange={e => setRemember(e.target.checked)}
                  className="rounded border-slate-300 text-primary-600 focus:ring-primary-500 w-4 h-4"
                />
                <span className="text-sm text-slate-600">Remember me</span>
              </label>
              <Link to="/forgot-password" className="text-sm font-medium text-primary-600 hover:text-primary-700">
                Forgot password?
              </Link>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="btn btn-primary w-full py-3.5 text-base shadow-primary-500/20"
            >
              {loading ? 'Signing in...' : 'Sign in'}
            </button>

            {msg && (
              <div className={`p-4 rounded-xl text-sm font-medium text-center ${msg.includes('Welcome') ? 'bg-emerald-50 text-emerald-700' : 'bg-rose-50 text-rose-700'}`}>
                {msg}
              </div>
            )}
          </form>

          <div className="text-center text-sm text-slate-500">
            Don't have an account?{' '}
            <Link to="/register" className="font-semibold text-slate-900 hover:underline">
              Sign up for free
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}
