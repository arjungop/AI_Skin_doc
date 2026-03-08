import { useState } from 'react'
import { api } from '../services/api'
import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import { LuStethoscope, LuArrowLeft, LuCheck, LuCircleAlert, LuUser, LuMail, LuLock, LuGraduationCap } from 'react-icons/lu'
import { Card } from '../components/Card'

export default function ApplyDoctor() {
  const [form, setForm] = useState({ first_name: '', last_name: '', email: '', password: '', specialization: '', license_no: '', department: '' })
  const [status, setStatus] = useState({ type: '', text: '' })
  const [isSubmitting, setIsSubmitting] = useState(false)

  const set = (k, v) => setForm(prev => ({ ...prev, [k]: v }))

  const submit = async (e) => {
    e.preventDefault()
    setIsSubmitting(true)
    setStatus({ type: '', text: '' })
    try {
      await api.applyDoctor({ ...form })
      setStatus({ type: 'success', text: 'Application successfully submitted. Our administrative intelligence will review your credentials.' })
      setForm({ first_name: '', last_name: '', email: '', password: '', specialization: '', license_no: '', department: '' })
    } catch (err) {
      setStatus({ type: 'error', text: err.message || 'An error occurred while submitting your application.' })
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col justify-center items-center p-4 relative overflow-hidden">
      {/* Background Decorators */}
      <div className="absolute top-[-15%] left-[-10%] w-[50%] h-[50%] bg-primary-500/10 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-15%] right-[-10%] w-[50%] h-[50%] bg-indigo-500/10 rounded-full blur-[120px] pointer-events-none" />

      <Link to="/login" className="absolute top-6 left-6 flex items-center gap-2 text-sm font-medium text-slate-500 hover:text-slate-900 transition-colors bg-white/50 backdrop-blur px-4 py-2 rounded-full shadow-sm border border-slate-200/60 z-10">
        <LuArrowLeft size={16} /> Returns to Login
      </Link>

      <motion.div
        initial={{ opacity: 0, y: 30, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.5, ease: [0.23, 1, 0.32, 1] }}
        className="w-full max-w-2xl z-10 my-12"
      >
        <div className="text-center mb-10">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: "spring", stiffness: 200, damping: 20, delay: 0.2 }}
            className="mx-auto w-16 h-16 bg-gradient-to-tr from-primary-600 to-indigo-600 rounded-2xl flex items-center justify-center shadow-lg shadow-primary-500/30 mb-6 text-white rotate-3"
          >
            <LuStethoscope size={32} />
          </motion.div>
          <h1 className="text-4xl font-extrabold text-slate-900 tracking-tight mb-3">Join <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary-600 to-indigo-600">SkinDoc Network</span></h1>
          <p className="text-lg text-slate-500 max-w-lg mx-auto leading-relaxed">Elevate your dermatology practice by joining our AI-enhanced medical intelligence platform.</p>
        </div>

        <Card className="p-8 md:p-10 border border-white/40 shadow-xl bg-white/80 backdrop-blur-xl" hover={false}>
          {status.text && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className={`p-4 mb-8 rounded-xl flex gap-3 text-sm font-medium border ${status.type === 'success' ? 'bg-emerald-50 text-emerald-800 border-emerald-200' : 'bg-red-50 text-red-800 border-red-200'}`}
            >
              <div className="mt-0.5">
                {status.type === 'success' ? <LuCheck size={18} className="text-emerald-500" /> : <LuCircleAlert size={18} className="text-red-500" />}
              </div>
              <div>{status.text}</div>
            </motion.div>
          )}

          <form onSubmit={submit} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              <div className="space-y-1.5">
                <label className="text-xs font-bold text-slate-500 uppercase tracking-widest pl-1">Legal First Name</label>
                <div className="relative">
                  <LuUser className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
                  <input className="w-full bg-slate-50/50 border border-slate-200 focus:bg-white focus:border-primary-500 focus:ring-4 focus:ring-primary-500/10 rounded-xl py-3 pl-11 pr-4 text-sm font-medium text-slate-900 transition-all outline-none" placeholder="First Name" value={form.first_name} onChange={e => set('first_name', e.target.value)} required />
                </div>
              </div>
              <div className="space-y-1.5">
                <label className="text-xs font-bold text-slate-500 uppercase tracking-widest pl-1">Legal Last Name</label>
                <div className="relative">
                  <LuUser className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
                  <input className="w-full bg-slate-50/50 border border-slate-200 focus:bg-white focus:border-primary-500 focus:ring-4 focus:ring-primary-500/10 rounded-xl py-3 pl-11 pr-4 text-sm font-medium text-slate-900 transition-all outline-none" placeholder="Last Name" value={form.last_name} onChange={e => set('last_name', e.target.value)} required />
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              <div className="space-y-1.5">
                <label className="text-xs font-bold text-slate-500 uppercase tracking-widest pl-1">Professional Email</label>
                <div className="relative">
                  <LuMail className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
                  <input className="w-full bg-slate-50/50 border border-slate-200 focus:bg-white focus:border-primary-500 focus:ring-4 focus:ring-primary-500/10 rounded-xl py-3 pl-11 pr-4 text-sm font-medium text-slate-900 transition-all outline-none" type="email" placeholder="doctor@hospital.com" value={form.email} onChange={e => set('email', e.target.value)} required />
                </div>
              </div>
              <div className="space-y-1.5">
                <label className="text-xs font-bold text-slate-500 uppercase tracking-widest pl-1">Secure Password</label>
                <div className="relative">
                  <LuLock className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
                  <input className="w-full bg-slate-50/50 border border-slate-200 focus:bg-white focus:border-primary-500 focus:ring-4 focus:ring-primary-500/10 rounded-xl py-3 pl-11 pr-4 text-sm font-medium text-slate-900 transition-all outline-none" type="password" placeholder="••••••••" value={form.password} onChange={e => set('password', e.target.value)} required />
                </div>
              </div>
            </div>

            <div className="pt-4 pb-2">
              <div className="h-px w-full bg-gradient-to-r from-transparent via-slate-200 to-transparent"></div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              <div className="space-y-1.5">
                <label className="text-xs font-bold text-slate-500 uppercase tracking-widest pl-1">Specialization</label>
                <div className="relative">
                  <LuGraduationCap className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
                  <input className="w-full bg-slate-50/50 border border-slate-200 focus:bg-white focus:border-primary-500 focus:ring-4 focus:ring-primary-500/10 rounded-xl py-3 pl-11 pr-4 text-sm font-medium text-slate-900 transition-all outline-none" placeholder="e.g. Cosmetic Dermatology" value={form.specialization} onChange={e => set('specialization', e.target.value)} required />
                </div>
              </div>
              <div className="space-y-1.5">
                <label className="text-xs font-bold text-slate-500 uppercase tracking-widest pl-1">Medical License No.</label>
                <div className="relative">
                  <LuCheck className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
                  <input className="w-full font-mono bg-slate-50/50 border border-slate-200 focus:bg-white focus:border-primary-500 focus:ring-4 focus:ring-primary-500/10 rounded-xl py-3 pl-11 pr-4 text-sm font-bold text-slate-700 transition-all outline-none" placeholder="e.g. MD-1029384" value={form.license_no} onChange={e => set('license_no', e.target.value)} required />
                </div>
              </div>
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-bold text-slate-500 uppercase tracking-widest pl-1">Primary Department</label>
              <select className="w-full bg-slate-50/50 border border-slate-200 focus:bg-white focus:border-primary-500 focus:ring-4 focus:ring-primary-500/10 rounded-xl py-3 px-4 text-sm font-medium text-slate-900 transition-all outline-none cursor-pointer appearance-none" style={{ backgroundImage: 'url("data:image/svg+xml,%3csvg xmlns=\'http://www.w3.org/2000/svg\' fill=\'none\' viewBox=\'0 0 20 20\'%3e%3cpath stroke=\'%236b7280\' stroke-linecap=\'round\' stroke-linejoin=\'round\' stroke-width=\'1.5\' d=\'M6 8l4 4 4-4\'/%3e%3c/svg%3e")', backgroundPosition: 'right 0.5rem center', backgroundRepeat: 'no-repeat', backgroundSize: '1.5em 1.5em' }} value={form.department} onChange={e => set('department', e.target.value)} required>
                <option value="" disabled>Select Medical Department</option>
                <option value="Dermatology">Dermatology</option>
                <option value="Cardiology">Cardiology</option>
                <option value="Orthopedics">Orthopedics</option>
                <option value="Pediatrics">Pediatrics</option>
                <option value="Oncology">Oncology</option>
                <option value="General Medicine">General Medicine</option>
              </select>
            </div>

            <motion.button
              whileHover={{ scale: 1.01 }}
              whileTap={{ scale: 0.98 }}
              className="w-full mt-4 flex items-center justify-center gap-2 bg-gradient-to-r from-primary-600 to-indigo-600 hover:from-primary-500 hover:to-indigo-500 text-white font-bold text-sm py-4 rounded-xl shadow-lg shadow-primary-500/30 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              type="submit"
              disabled={isSubmitting}
            >
              {isSubmitting ? (
                <>
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Submitting Application...
                </>
              ) : (
                'Submit Application Package'
              )}
            </motion.button>
            <p className="text-center text-xs text-slate-400 font-medium">By submitting, you agree to our Terms of Service regarding medical professionals.</p>
          </form>
        </Card>
      </motion.div>
    </div>
  )
}
