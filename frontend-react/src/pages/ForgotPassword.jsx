import { useState } from 'react'
import { Link } from 'react-router-dom'
import AuthLayout from '../components/AuthLayout.jsx'
import { api } from '../services/api'
import { LuMail } from 'react-icons/lu'

export default function ForgotPassword(){
  const [email, setEmail] = useState('')
  const [msg, setMsg] = useState('')
  const [loading, setLoading] = useState(false)

  async function submit(e){
    e.preventDefault(); setMsg(''); setLoading(true)
    try{ await api.forgotPassword(email); setMsg('If an account exists, a reset link has been sent.') }
    catch(err){ setMsg(err.message) } finally { setLoading(false) }
  }

  return (
    <AuthLayout title="Forgot password" subtitle="Enter your email to receive reset instructions"
      footer={<Link className="text-primaryBlue" to="/login">Back to login</Link>}>
      <form onSubmit={submit} className="stack">
        <label className="text-sm">Email</label>
        <div className="flex items-center gap-3">
          <span className="h-12 w-12 rounded-md bg-gradient-to-tr from-primaryBlue to-accentPurple2 text-white grid place-items-center text-lg"><LuMail/></span>
          <input className="flex-1 h-12 text-base" type="email" placeholder="you@example.com" value={email} onChange={e=>setEmail(e.target.value)} required />
        </div>
        <button className="btn-primary btn-lg" type="submit" disabled={loading}>{loading?'Sending…':'Send reset link'}</button>
        {msg && <div className="text-sm">{msg}</div>}
      </form>
    </AuthLayout>
  )
}

