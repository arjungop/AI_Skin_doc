import { Link, useLocation } from 'react-router-dom'
import { ToastProvider } from './Toast.jsx'
import { useEffect, useState } from 'react'

export default function AppShell({ children }){
  const loc = useLocation()
  const role = (localStorage.getItem('role') || '').toUpperCase()
  const loggedIn = !!role
  const onLogout = (e)=>{ e.preventDefault(); localStorage.clear(); location.href='/login' }

  // Disable dark mode globally: ensure class is removed
  useEffect(()=>{ document.documentElement.classList.remove('dark'); try{ localStorage.removeItem('theme') }catch{} }, [])

  const name = (localStorage.getItem('username')||'')
  const initials = name.split(' ').map(s=>s[0]).join('').slice(0,2).toUpperCase() || 'U'

  return (
    <ToastProvider>
    <div className="min-h-screen bg-backgroundSoft dark:bg-slate-900 text-textLuxury dark:text-slate-100 flex flex-col">
      <header className="bg-backgroundSecondary/90 dark:bg-slate-800/80 backdrop-blur supports-[backdrop-filter]:bg-backgroundSecondary/80 border-b border-borderElegant dark:border-slate-700 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
          <Link to={role==='ADMIN'?'/admin':'/dashboard'} className="flex items-center gap-3 text-inherit no-underline">
            <div className="h-8 w-8 rounded-md bg-gradient-to-tr from-primaryBlue to-accentPurple2 ring-2 ring-accentGold/40" />
            <div className="text-xl font-semibold">AI Skin Doctor</div>
          </Link>
          <nav className="hidden md:flex items-center gap-1">
            {loggedIn ? (
              role==='ADMIN' ? (
                <>
                  <Nav to="/admin" label="Admin" active={loc.pathname==='/admin'} />
                  <Nav to="/admin/transactions" label="Transactions" active={loc.pathname==='/admin/transactions'} />
                </>
              ) : (
                <>
                  {role==='DOCTOR' && <Nav to="/doctor" label="Doctor" active={loc.pathname.startsWith('/doctor')} />}
                  <Nav to="/lesions" label="Lesion Classification" active={loc.pathname==='/lesions'} />
                  <Nav to="/chat" label="AI Chat" active={loc.pathname==='/chat'} />
                  <Nav to="/appointments" label="Appointments" active={loc.pathname==='/appointments'} />
                  <Nav to="/transactions" label="Transactions" active={loc.pathname==='/transactions'} />
                  <Nav to="/messages" label="Messages" active={loc.pathname==='/messages'} />
                </>
              )
            ) : (
              <>
                <Nav to="/login" label="Login" active={loc.pathname==='/login'} />
                <Nav to="/register" label="Register" active={loc.pathname==='/register'} />
              </>
            )}
          </nav>
          <div className="flex items-center gap-2">
            {loggedIn ? (
              <>
                <div className="h-8 w-8 rounded-full bg-slate-200 dark:bg-slate-600 flex items-center justify-center text-sm font-semibold text-textLuxury dark:text-white">{initials}</div>
                <a href="#" onClick={onLogout} className="btn-primary">Logout</a>
              </>
            ) : null}
          </div>
        </div>
      </header>
      <main className="max-w-7xl mx-auto px-6 py-8 flex-1 w-full">
        {children}
      </main>
      <footer className="border-t border-borderElegant dark:border-slate-700 bg-backgroundSecondary dark:bg-slate-800">
        <div className="max-w-7xl mx-auto px-6 py-10 grid grid-cols-1 sm:grid-cols-3 gap-8 text-sm text-textLuxuryMuted dark:text-slate-300">
          <div>
            <div className="text-lg font-semibold text-textLuxury dark:text-white mb-2">AI Skin Doctor</div>
            <p>Your private dermatology + AI concierge.</p>
            <div className="mt-4">© {new Date().getFullYear()} AI Skin Doctor</div>
          </div>
          <div>
            <div className="font-semibold text-textLuxury dark:text-white mb-2">Quick Links</div>
            <div className="grid grid-cols-2 gap-2">
              {role==='ADMIN' ? (
                <>
                  <Link to="/admin" className="text-primaryBlue">Admin</Link>
                  <Link to="/transactions" className="text-primaryBlue">Transactions</Link>
                </>
              ) : (
                <>
                  <Link to="/dashboard" className="text-primaryBlue">Dashboard</Link>
                  <Link to="/lesions" className="text-primaryBlue">Lesion Classification</Link>
                  <Link to="/chat" className="text-primaryBlue">AI Chat</Link>
                  <Link to="/appointments" className="text-primaryBlue">Appointments</Link>
                  <Link to="/transactions" className="text-primaryBlue">Transactions</Link>
                  <Link to="/messages" className="text-primaryBlue">Messages</Link>
                  <Link to="/contact" className="text-primaryBlue">Contact</Link>
                </>
              )}
            </div>
          </div>
          <div>
            <div className="font-semibold text-textLuxury dark:text-white mb-2">Newsletter</div>
            <NewsletterForm />
            <div className="mt-4">
              <div>support@example.com</div>
              <div>+91-00000 00000</div>
            </div>
          </div>
        </div>
      </footer>
    </div>
    </ToastProvider>
  )
}

function Nav({ to, label, active }){
  return (
    <Link className={("px-4 py-2 text-sm font-medium text-textLuxury dark:text-slate-100 hover:text-primaryBlue border-b-2 ")+(active?"border-primaryBlue text-primaryBlue":"border-transparent")} to={to}>{label}</Link>
  )
}

function NewsletterForm(){
  const [email, setEmail] = useState('')
  const [ok, setOk] = useState('')
  async function submit(e){
    e.preventDefault(); setOk('')
    try{
      await fetch((import.meta.env.VITE_API_BASE_URL||'http://127.0.0.1:8000')+`/support/newsletter/subscribe`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ email }) })
      setOk('Subscribed!')
      setEmail('')
    }catch{ setOk('Failed to subscribe') }
  }
  return (
    <form onSubmit={submit} className="flex gap-2">
      <input className="flex-1" type="email" placeholder="Your email" value={email} onChange={e=>setEmail(e.target.value)} required />
      <button className="btn-primary btn-sm" type="submit">Join</button>
      {ok && <div className="text-xs">{ok}</div>}
    </form>
  )
}
