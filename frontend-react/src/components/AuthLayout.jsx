import { Link } from 'react-router-dom'
import { useEffect } from 'react'

export default function AuthLayout({ title, subtitle, children, footer }){
  // Force light mode while on auth screens for readability
  useEffect(()=>{
    const prev = document.documentElement.classList.contains('dark')
    document.documentElement.classList.remove('dark')
    return ()=> { if (prev) document.documentElement.classList.add('dark') }
  },[])
  return (
    <div className="min-h-screen relative overflow-hidden flex items-center justify-center px-4" style={{background:"radial-gradient(1200px 400px at -200px -100px, rgba(106,91,255,0.08), transparent), radial-gradient(800px 300px at 110% 10%, rgba(78,115,223,0.08), transparent)"}}>
      <div className="absolute inset-0 pointer-events-none" />
      <div className="w-full max-w-lg">
        <header className="flex items-center justify-center mb-5">
          <Link to="/" className="flex items-center gap-2 text-textLuxury dark:text-white">
            <div className="h-9 w-9 rounded-md bg-gradient-to-tr from-primaryBlue to-accentPurple2 ring-2 ring-accentGold/40" />
            <div className="text-xl font-semibold">AI Skin Doctor</div>
          </Link>
        </header>
        <div className="rounded-2xl border border-borderElegant bg-cardBackground/90 backdrop-blur shadow-md p-8">
          <h1 className="text-2xl font-semibold mb-1 text-textLuxury dark:text-white">{title}</h1>
          {subtitle && <p className="muted mb-4">{subtitle}</p>}
          {children}
        </div>
        {footer && (
          <div className="text-center text-sm text-textLuxuryMuted dark:text-slate-300 mt-4">{footer}</div>
        )}
      </div>
    </div>
  )
}
