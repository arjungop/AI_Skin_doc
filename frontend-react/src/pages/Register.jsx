import { useEffect, useMemo, useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { api } from '../services/api'

export default function Register() {
  const [form, setForm] = useState({ first_name:'', last_name:'', email:'', password:'', age:'', gender:'' })
  const [msg, setMsg] = useState('')
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()
  const set = (k,v)=> setForm(prev=>({...prev,[k]:v}))

  const pwdScore = useMemo(()=>{
    const p = form.password||''; let s=0
    if (p.length>=8) s++
    if (/[A-Z]/.test(p)) s++
    if (/[0-9]/.test(p)) s++
    if (/[^A-Za-z0-9]/.test(p)) s++
    return s
  }, [form.password])

  const onSubmit = async (e) => {
    e.preventDefault()
    setMsg('')
    try {
      setLoading(true)
      await api.register({ ...form, age: parseInt(form.age) })
      setMsg('Registered successfully! Please login.')
      setTimeout(()=> navigate('/login'), 600)
    } catch (err) {
      setMsg(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-background grid place-items-center px-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-4">
          <div className="mx-auto h-10 w-10 rounded-md bg-gradient-to-tr from-primary to-accent" />
        </div>
        <h1 className="text-2xl font-semibold text-center">Create account</h1>
        <form onSubmit={onSubmit} className="card stack mt-3">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <input placeholder="First name" value={form.first_name} onChange={e=>set('first_name', e.target.value)} required />
            <input placeholder="Last name" value={form.last_name} onChange={e=>set('last_name', e.target.value)} required />
          </div>
          <input type="email" placeholder="Email" value={form.email} onChange={e=>set('email', e.target.value)} required />
          <div>
            <input type="password" placeholder="Password" value={form.password} onChange={e=>set('password', e.target.value)} required />
            <div className="mt-1 text-xs muted">Strength: {["Weak","Fair","Good","Strong"][Math.max(0,pwdScore-1)] || 'Weak'}</div>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <input type="number" placeholder="Age" value={form.age} onChange={e=>set('age', e.target.value)} required />
            <select value={form.gender} onChange={e=>set('gender', e.target.value)} required>
              <option value="">Gender</option>
              <option>Female</option>
              <option>Male</option>
              <option>Other</option>
              <option>Prefer not to say</option>
            </select>
          </div>
          <button className="btn-primary" type="submit" disabled={loading}>{loading?'Creating…':'Create account'}</button>
          {msg && <div className="text-success text-sm">{msg}</div>}
        </form>
        <p className="text-center text-textMuted mt-3">Have an account? <Link className="text-primary" to="/login">Login</Link></p>
      </div>
    </div>
  )
}
