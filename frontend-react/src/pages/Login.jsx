import { useEffect, useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { api } from '../services/api'
import AuthLayout from '../components/AuthLayout.jsx'
import { LuMail, LuLock } from 'react-icons/lu'

export default function Login() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [show, setShow] = useState(false)
  const [msg, setMsg] = useState('')
  const [loading, setLoading] = useState(false)
  const [remember, setRemember] = useState(false)
  const navigate = useNavigate()

  useEffect(()=>{
    // Pre-fill remembered email
    const saved = localStorage.getItem('remember_email')
    if (saved) { setEmail(saved); setRemember(true) }
  },[])

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
      const r = (res.role||'').toUpperCase()
      const dest = r === 'DOCTOR' ? '/doctor' : (r === 'ADMIN' ? '/admin' : '/dashboard')
      setMsg('Welcome ' + (res.username || ''))
      setTimeout(() => navigate(dest), 300)
    } catch (err) {
      setMsg(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <AuthLayout title="Welcome back" subtitle="Sign in to your account"
      footer={<>
        No account? <Link className="text-primaryBlue" to="/register">Register</Link>
        <span className="mx-2">•</span>
        <Link className="text-primaryBlue" to="/apply-doctor">Apply as Doctor</Link>
      </>}>
      <form onSubmit={onSubmit} className="stack">
        <div>
          <label className="text-sm">Email</label>
          <div className="flex items-center gap-3">
            <span className="h-12 w-12 rounded-md bg-gradient-to-tr from-primaryBlue to-accentPurple2 text-white grid place-items-center text-lg"><LuMail/></span>
            <input className="flex-1 h-12 text-base" type="email" placeholder="you@example.com" value={email} onChange={e=>setEmail(e.target.value)} required />
          </div>
        </div>
        <div>
          <label className="text-sm">Password</label>
          <div className="flex items-center gap-3">
            <span className="h-12 w-12 rounded-md bg-gradient-to-tr from-primaryBlue to-accentPurple2 text-white grid place-items-center text-lg"><LuLock/></span>
            <input className="flex-1 h-12 text-base" type={show?'text':'password'} placeholder="••••••••" value={password} onChange={e=>setPassword(e.target.value)} required />
            <button type="button" className="btn-ghost btn-sm" onClick={()=> setShow(s=>!s)}>{show?'Hide':'Show'}</button>
          </div>
        </div>
        <div className="flex items-center justify-between mt-2 mb-1">
          <label className="flex items-center gap-2 whitespace-nowrap"><input type="checkbox" checked={remember} onChange={e=>setRemember(e.target.checked)} /> <span>Remember me</span></label>
          <Link className="text-primaryBlue text-sm" to="/forgot-password">Forgot password?</Link>
        </div>
        <button className="btn-primary btn-lg" type="submit" disabled={loading}>{loading?'Signing in…':'Sign in'}</button>
        {msg && <div className="text-error text-sm">{msg}</div>}
      </form>
    </AuthLayout>
  )
}
