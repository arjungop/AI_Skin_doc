import { useEffect, useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { api } from '../services/api'
import { LuArrowRight, LuCheck, LuEye, LuEyeOff, LuSparkles } from 'react-icons/lu'
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
    <div className="min-h-screen flex">
      {/* Left Panel - Animated Gradient Background */}
      <div className="hidden lg:flex lg:w-1/2 relative overflow-hidden">
        {/* Animated mesh gradient */}
        <div className="absolute inset-0 bg-surface">
          <div className="absolute inset-0 opacity-60">
            <div className="absolute top-0 left-0 w-[600px] h-[600px] bg-primary-500/20 rounded-full blur-[120px] animate-float" />
            <div className="absolute bottom-0 right-0 w-[500px] h-[500px] bg-accent-500/20 rounded-full blur-[100px] animate-float animation-delay-200" />
            <div className="absolute top-1/2 left-1/2 w-[400px] h-[400px] bg-ai-500/15 rounded-full blur-[80px] animate-float animation-delay-500" />
          </div>
        </div>

        {/* Content */}
        <div className="relative z-10 flex flex-col justify-between p-12 w-full">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center">
                <LuSparkles className="text-white" size={20} />
              </div>
              <span className="text-2xl font-bold text-gradient-primary">Skin.AI</span>
            </div>
          </motion.div>

          <motion.div
            className="space-y-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <h1 className="text-5xl xl:text-6xl font-bold text-text-primary leading-tight">
              Your Skin's
              <br />
              <span className="text-gradient-primary">Digital Twin</span>
            </h1>
            <p className="text-xl text-text-secondary max-w-md leading-relaxed">
              AI-powered dermatology insights, personalized routines, and expert consultations — all in one place.
            </p>

            {/* Feature pills */}
            <div className="flex flex-wrap gap-3 pt-4">
              {['AI Analysis', 'Expert Doctors', 'Personalized Care'].map((feature, i) => (
                <motion.div
                  key={feature}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.6 + i * 0.1 }}
                  className="chip"
                >
                  {feature}
                </motion.div>
              ))}
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.8 }}
            className="text-sm text-text-tertiary"
          >
            Trusted by 10,000+ patients worldwide
          </motion.div>
        </div>
      </div>

      {/* Right Panel - Login Form */}
      <div className="w-full lg:w-1/2 flex items-center justify-center p-6 lg:p-12">
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
          className="w-full max-w-md"
        >
          {/* Mobile Logo */}
          <div className="lg:hidden flex items-center gap-3 mb-10">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center">
              <LuSparkles className="text-white" size={20} />
            </div>
            <span className="text-2xl font-bold text-gradient-primary">Skin.AI</span>
          </div>

          <div className="mb-10">
            <h2 className="text-3xl font-bold text-text-primary mb-2">Welcome back</h2>
            <p className="text-text-secondary">Sign in to continue to your dashboard</p>
          </div>

          <form onSubmit={onSubmit} className="space-y-6">
            {/* Email */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-text-secondary uppercase tracking-wider">
                Email
              </label>
              <input
                type="email"
                placeholder="name@example.com"
                value={email}
                onChange={e => setEmail(e.target.value)}
                className="input"
                required
              />
            </div>

            {/* Password */}
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <label className="text-sm font-medium text-text-secondary uppercase tracking-wider">
                  Password
                </label>
                <Link
                  to="/forgot-password"
                  className="text-sm text-primary-500 hover:text-primary-400 transition-colors font-medium"
                >
                  Forgot?
                </Link>
              </div>
              <div className="relative">
                <input
                  type={show ? 'text' : 'password'}
                  placeholder="••••••••"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  className="input pr-12"
                  required
                />
                <button
                  type="button"
                  className="absolute right-4 top-1/2 -translate-y-1/2 text-text-tertiary hover:text-text-primary transition-colors"
                  onClick={() => setShow(!show)}
                >
                  {show ? <LuEyeOff size={20} /> : <LuEye size={20} />}
                </button>
              </div>
            </div>

            {/* Remember Me */}
            <label className="flex items-center gap-3 cursor-pointer group">
              <div className={`w-5 h-5 rounded-md border-2 flex items-center justify-center transition-all ${remember
                  ? 'bg-primary-500 border-primary-500'
                  : 'border-white/20 group-hover:border-primary-500/50'
                }`}>
                {remember && <LuCheck className="text-white" size={14} strokeWidth={3} />}
              </div>
              <input
                type="checkbox"
                checked={remember}
                onChange={e => setRemember(e.target.checked)}
                className="hidden"
              />
              <span className="text-sm text-text-secondary group-hover:text-text-primary transition-colors">
                Keep me signed in
              </span>
            </label>

            {/* Submit Button */}
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="btn-primary w-full py-4 text-lg"
              type="submit"
              disabled={loading}
            >
              {loading ? (
                <span className="flex items-center gap-2">
                  <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Signing in...
                </span>
              ) : (
                <span className="flex items-center justify-center gap-2">
                  Sign In
                  <LuArrowRight size={20} />
                </span>
              )}
            </motion.button>

            {/* Message */}
            {msg && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`p-4 rounded-xl text-center text-sm font-medium ${msg.includes('Welcome')
                    ? 'bg-success/10 text-success border border-success/20'
                    : 'bg-danger/10 text-danger border border-danger/20'
                  }`}
              >
                {msg}
              </motion.div>
            )}
          </form>

          {/* Sign up link */}
          <div className="mt-10 text-center">
            <p className="text-text-secondary">
              New to Skin.AI?{' '}
              <Link to="/register" className="text-primary-500 hover:text-primary-400 font-semibold transition-colors">
                Create an account
              </Link>
            </p>
          </div>

          {/* Apply as doctor */}
          <div className="mt-6 text-center">
            <Link
              to="/apply-doctor"
              className="text-sm text-text-tertiary hover:text-text-secondary transition-colors"
            >
              Are you a dermatologist? <span className="text-accent-400">Apply here</span>
            </Link>
          </div>
        </motion.div>
      </div>
    </div>
  )
}
