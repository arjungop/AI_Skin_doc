import { Link, useLocation } from 'react-router-dom'
import { ToastProvider } from './Toast.jsx'
import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import {
  FaHome, FaStethoscope, FaComments, FaCalendarAlt,
  FaMoneyBillWave, FaEnvelope, FaRobot, FaUserMd,
  FaCog, FaSignOutAlt, FaMapMarkerAlt
} from 'react-icons/fa'
import { LuScan, LuSparkles, LuUser } from 'react-icons/lu'

export default function AppShell({ children }) {
  const loc = useLocation()
  const role = (localStorage.getItem('role') || '').toUpperCase()
  const loggedIn = !!role
  const onLogout = (e) => { e.preventDefault(); localStorage.clear(); location.href = '/login' }
  const [showProfileMenu, setShowProfileMenu] = useState(false)

  // Ensure dark mode is off for clinical theme
  useEffect(() => { document.documentElement.classList.remove('dark'); }, [])

  const name = (localStorage.getItem('username') || 'User')
  const initials = name.split(' ').map(s => s[0]).join('').slice(0, 2).toUpperCase()

  // Full Screen Layout if not logged in
  if (!loggedIn) {
    return <ToastProvider>{children}</ToastProvider>
  }

  return (
    <ToastProvider>
      <div className="relative min-h-screen bg-surface">
        {/* DYNAMIC DOCK - Desktop (Left Center) */}
        <motion.nav 
          initial={{ x: -100, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          className="hidden lg:flex dock left-6 top-1/2 -translate-y-1/2 flex-col gap-1"
        >
          {role === 'ADMIN' ? (
            <>
              <DockIcon to="/admin" icon={FaHome} active={loc.pathname === '/admin'} label="Overview" />
              <DockIcon to="/admin/transactions" icon={FaMoneyBillWave} active={loc.pathname.includes('transactions')} label="Transactions" />
            </>
          ) : (
            <>
              <DockIcon to="/dashboard" icon={FaHome} active={loc.pathname === '/dashboard'} label="Dashboard" />
              <DockIcon to="/lesions" icon={LuScan} active={loc.pathname === '/lesions'} label="AI Scan" aiFeature />
              <DockIcon to="/coach" icon={LuSparkles} active={loc.pathname === '/coach'} label="Coach" isNew />
              <DockIcon to="/chat" icon={FaRobot} active={loc.pathname === '/chat'} label="AI Chat" aiFeature />
              <DockIcon to="/journey" icon={FaComments} active={loc.pathname === '/journey'} label="Journey" isNew />
              <DockIcon to="/routine" icon={FaStethoscope} active={loc.pathname === '/routine'} label="Routine" />
              <DockIcon to="/find-doctors" icon={FaMapMarkerAlt} active={loc.pathname === '/find-doctors'} label="Find Doctors" isNew />
              <div className="h-px bg-white/20 my-2" />
              <DockIcon to="/appointments" icon={FaCalendarAlt} active={loc.pathname === '/appointments'} label="Appointments" />
              <DockIcon to="/messages" icon={FaEnvelope} active={loc.pathname === '/messages'} label="Messages" />
              <DockIcon to="/transactions" icon={FaMoneyBillWave} active={loc.pathname === '/transactions'} label="Payments" />
            </>
          )}
        </motion.nav>

        {/* DYNAMIC DOCK - Mobile (Bottom Center) */}
        <motion.nav 
          initial={{ y: 100, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          className="lg:hidden dock bottom-6 left-1/2 -translate-x-1/2 gap-1"
        >
          <DockIcon to="/dashboard" icon={FaHome} active={loc.pathname === '/dashboard'} label="Home" />
          <DockIcon to="/lesions" icon={LuScan} active={loc.pathname === '/lesions'} label="Scan" aiFeature />
          <DockIcon to="/chat" icon={FaRobot} active={loc.pathname === '/chat'} label="AI" aiFeature />
          <DockIcon to="/appointments" icon={FaCalendarAlt} active={loc.pathname === '/appointments'} label="Appointments" />
          <button onClick={() => setShowProfileMenu(!showProfileMenu)} className="dock-icon">
            <LuUser size={20} />
          </button>
        </motion.nav>

        {/* Mobile Profile Menu */}
        {showProfileMenu && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="lg:hidden fixed bottom-24 left-1/2 -translate-x-1/2 card-glass p-4 z-50 min-w-[200px]"
          >
            <div className="text-center mb-3">
              <div className="h-12 w-12 mx-auto rounded-full bg-accent-medical/10 flex items-center justify-center font-mono font-semibold text-accent-medical mb-2">
                {initials}
              </div>
              <p className="text-sm font-semibold text-text-primary">{name}</p>
              <p className="text-xs text-text-secondary capitalize">{role.toLowerCase()}</p>
            </div>
            <button
              onClick={onLogout}
              className="w-full btn-ghost text-sm py-2 flex items-center justify-center gap-2"
            >
              <FaSignOutAlt size={14} />
              Sign Out
            </button>
          </motion.div>
        )}

        {/* Floating Header - Minimal Branding */}
        <header className="fixed top-6 right-6 z-40 hidden lg:flex items-center gap-4">
          <div className="card-glass px-4 py-2 flex items-center gap-3">
            <div className="h-8 w-8 rounded-full bg-accent-medical/10 flex items-center justify-center font-mono font-semibold text-accent-medical text-sm">
              {initials}
            </div>
            <div className="text-sm">
              <p className="font-medium text-text-primary tracking-tight">{name}</p>
              <p className="text-xs text-text-secondary capitalize">{role.toLowerCase()}</p>
            </div>
            <button onClick={onLogout} className="ml-2 text-text-secondary hover:text-accent-medical transition-colors">
              <FaSignOutAlt size={16} />
            </button>
          </div>
        </header>

        {/* Main Content Area - Full Width */}
        <main className="min-h-screen pt-6 pb-32 lg:pb-12 px-6 lg:pl-32">
          <div className="max-w-[1600px] mx-auto">
            {children}
          </div>
        </main>
      </div>
    </ToastProvider>
  )
}

function DockIcon({ to, icon: Icon, active, label, isNew, aiFeature }) {
  const [showTooltip, setShowTooltip] = useState(false)
  
  return (
    <div className="relative">
      <Link
        to={to}
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        className={`dock-icon ${active ? 'active' : ''} ${aiFeature && active ? '!text-accent-ai !bg-ai-50' : ''}`}
      >
        <Icon size={20} />
        {isNew && (
          <span className="absolute -top-1 -right-1 h-2 w-2 rounded-full bg-accent-ai animate-pulse" />
        )}
      </Link>
      
      {/* Tooltip */}
      {showTooltip && (
        <motion.div
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          className="absolute left-full ml-4 top-1/2 -translate-y-1/2 card-glass px-3 py-1.5 text-xs font-medium text-text-primary whitespace-nowrap pointer-events-none hidden lg:block"
        >
          {label}
        </motion.div>
      )}
    </div>
  )
}
