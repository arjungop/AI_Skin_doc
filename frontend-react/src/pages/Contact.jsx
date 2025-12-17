import { useState } from 'react'

export default function Contact(){
  const [form, setForm] = useState({ name:'', email:'', subject:'', message:'' })
  const [msg, setMsg] = useState('')

  async function submit(e){
    e.preventDefault(); setMsg('')
    try{
      const BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
      const res = await fetch(`${BASE}/support/ticket`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(form) })
      if (!res.ok) throw new Error('Failed')
      setMsg('Thanks! We will reach out soon.')
      setForm({ name:'', email:'', subject:'', message:'' })
    }catch(err){ setMsg('Failed to send. Please try again.') }
  }

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-2xl font-semibold mb-2">Contact Support</h1>
      <p className="muted mb-4">Have a question or feedback? Send us a note.</p>
      <form onSubmit={submit} className="card stack">
        <input placeholder="Name" value={form.name} onChange={e=>setForm({ ...form, name:e.target.value })} required />
        <input type="email" placeholder="Email" value={form.email} onChange={e=>setForm({ ...form, email:e.target.value })} required />
        <input placeholder="Subject" value={form.subject} onChange={e=>setForm({ ...form, subject:e.target.value })} required />
        <textarea rows={5} placeholder="Message" value={form.message} onChange={e=>setForm({ ...form, message:e.target.value })} required />
        <button className="btn-primary" type="submit">Send</button>
        {msg && <div className="text-sm">{msg}</div>}
      </form>
    </div>
  )
}

