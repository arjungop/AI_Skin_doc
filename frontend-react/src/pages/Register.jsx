import { useMemo, useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { api } from '../services/api'
import { LuArrowRight, LuSparkles, LuShield, LuCheck } from 'react-icons/lu'
import { motion } from 'framer-motion'

export default function Register() {
  const [form, setForm] = useState({ first_name: '', last_name: '', email: '', password: '', age: '', gender: '' })
  const [msg, setMsg] = useState('')
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()
  const set = (k, v) => setForm(prev => ({ ...prev, [k]: v }))

  const pwdScore = useMemo(() => {
    const p = form.password || ''; let s = 0
    if (p.length >= 8) s++
    if (/[A-Z]/.test(p)) s++
    if (/[0-9]/.test(p)) s++
    if (/[^A-Za-z0-9]/.test(p)) s++
    return s
  }, [form.password])

  const strengthLabels = ['Weak', 'Fair', 'Good', 'Strong']
  const strengthColors = ['bg-danger', 'bg-warning', 'bg-ai-500', 'bg-success']

  const onSubmit = async (e) => {
    e.preventDefault()
    setMsg('')
    // Client-side validation
    if (!form.first_name.trim() || !form.last_name.trim()) {
      setMsg('Please enter your full name')
      return
    }
    if (!form.email.includes('@') || !form.email.includes('.')) {
      setMsg('Please enter a valid email address')
      return
    }
    if (form.password.length < 8) {
      setMsg('Password must be at least 8 characters')
      return
    }
    const age = parseInt(form.age)
    if (isNaN(age) || age < 1 || age > 120) {
      setMsg('Please enter a valid age between 1 and 120')
      return
    }
    if (!form.gender) {
      setMsg('Please select your gender')
      return
    }
    try {
      setLoading(true)
      await api.register({ ...form, age })
      setMsg('Account created! Redirecting to login...')
      setTimeout(() => navigate('/login'), 1000)
    } catch (err) {
      setMsg(err.message || 'Registration failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex">
      {/* Left Panel - Animated Gradient Background */}
      <div className="hidden lg:flex lg:w-1/2 relative overflow-hidden">
        <div className="absolute inset-0 bg-surface">
          <div className="absolute inset-0 opacity-60">
            <div className="absolute top-0 right-0 w-[600px] h-[600px] bg-secondary-500/20 rounded-full blur-[120px] animate-float" />
            <div className="absolute bottom-0 left-0 w-[500px] h-[500px] bg-primary-500/20 rounded-full blur-[100px] animate-float animation-delay-200" />
            <div className="absolute top-1/2 left-1/3 w-[400px] h-[400px] bg-primary-300/15 rounded-full blur-[80px] animate-float animation-delay-500" />
          </div>
        </div>

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
            className="space-y-8"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <h1 className="text-5xl xl:text-6xl font-bold text-text-primary leading-tight">
              Start Your
              <br />
              <span className="text-gradient-accent">Skin Journey</span>
            </h1>

            {/* Features */}
            <div className="space-y-4 pt-4">
              {[
                { icon: LuShield, text: 'HIPAA-compliant & secure' },
                { icon: LuSparkles, text: 'AI-powered skin analysis' },
                { icon: LuCheck, text: 'Access to certified dermatologists' },
              ].map((feature, i) => (
                <motion.div
                  key={feature.text}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.6 + i * 0.1 }}
                  className="flex items-center gap-4"
                >
                  <div className="w-10 h-10 rounded-xl bg-white/5 flex items-center justify-center">
                    <feature.icon className="text-primary-500" size={20} />
                  </div>
                  <span className="text-text-secondary">{feature.text}</span>
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
            Join 10,000+ patients taking control of their skin health
          </motion.div>
        </div>
      </div>

      {/* Right Panel - Registration Form */}
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

          <div className="mb-8">
            <h2 className="text-3xl font-bold text-text-primary mb-2">Create account</h2>
            <p className="text-text-secondary">Get started with AI-powered skincare</p>
          </div>

          <form onSubmit={onSubmit} className="space-y-5">
            {/* Name Fields */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-text-secondary uppercase tracking-wider">First name</label>
                <input
                  type="text"
                  placeholder="John"
                  value={form.first_name}
                  onChange={e => set('first_name', e.target.value)}
                  className="input"
                  required
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-text-secondary uppercase tracking-wider">Last name</label>
                <input
                  type="text"
                  placeholder="Doe"
                  value={form.last_name}
                  onChange={e => set('last_name', e.target.value)}
                  className="input"
                  required
                />
              </div>
            </div>

            {/* Email */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-text-secondary uppercase tracking-wider">Email</label>
              <input
                type="email"
                placeholder="name@example.com"
                value={form.email}
                onChange={e => set('email', e.target.value)}
                className="input"
                required
              />
            </div>

            {/* Password with Strength Indicator */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-text-secondary uppercase tracking-wider">Password</label>
              <input
                type="password"
                placeholder="Create a strong password"
                value={form.password}
                onChange={e => set('password', e.target.value)}
                className="input"
                required
              />
              {form.password && (
                <div className="space-y-2">
                  <div className="flex gap-1">
                    {[0, 1, 2, 3].map(i => (
                      <div
                        key={i}
                        className={`h-1 flex-1 rounded-full transition-colors ${i < pwdScore ? strengthColors[pwdScore - 1] : 'bg-white/10'
                          }`}
                      />
                    ))}
                  </div>
                  <p className="text-xs text-text-tertiary">
                    Strength: <span className={`text-${pwdScore > 2 ? 'success' : 'warning'}`}>
                      {strengthLabels[Math.max(0, pwdScore - 1)] || 'Weak'}
                    </span>
                  </p>
                </div>
              )}
            </div>

            {/* Age & Gender */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-text-secondary uppercase tracking-wider">Age</label>
                <input
                  type="number"
                  placeholder="25"
                  min="1"
                  max="120"
                  value={form.age}
                  onChange={e => set('age', e.target.value)}
                  className="input"
                  required
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-text-secondary uppercase tracking-wider">Gender</label>
                <select
                  value={form.gender}
                  onChange={e => set('gender', e.target.value)}
                  className="input"
                  required
                >
                  <option value="">Select</option>
                  <option>Female</option>
                  <option>Male</option>
                  <option>Other</option>
                  <option>Prefer not to say</option>
                </select>
              </div>
            </div>

            {/* Terms */}
            <p className="text-xs text-text-tertiary leading-relaxed">
              By creating an account, you agree to our{' '}
              <a href="#" className="text-primary-500 hover:underline">Terms of Service</a>
              {' '}and{' '}
              <a href="#" className="text-primary-500 hover:underline">Privacy Policy</a>
            </p>

            {/* Submit */}
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="btn-accent w-full py-4 text-lg"
              type="submit"
              disabled={loading}
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Creating...
                </span>
              ) : (
                <span className="flex items-center justify-center gap-2">
                  Create Account
                  <LuArrowRight size={20} />
                </span>
              )}
            </motion.button>

            {/* Message */}
            {msg && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`p-4 rounded-xl text-center text-sm font-medium ${msg.includes('created')
                  ? 'bg-success/10 text-success border border-success/20'
                  : 'bg-danger/10 text-danger border border-danger/20'
                  }`}
              >
                {msg}
              </motion.div>
            )}
          </form>

          <div className="mt-8 text-center">
            <p className="text-text-secondary">
              Already have an account?{' '}
              <Link to="/login" className="text-primary-500 hover:text-primary-400 font-semibold transition-colors">
                Sign in
              </Link>
            </p>
          </div>
        </motion.div>
      </div>
    </div>
  )
}
