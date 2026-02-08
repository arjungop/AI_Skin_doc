import { Link, useLocation } from 'react-router-dom'
import { ToastProvider } from './Toast.jsx'
import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  FaHome, FaStethoscope, FaComments, FaCalendarAlt,
  FaMoneyBillWave, FaEnvelope, FaUserMd, FaSignOutAlt
} from 'react-icons/fa'
import { LuScan, LuSparkles, LuUser, LuMessageCircle, LuMapPin, LuHeart, LuX } from 'react-icons/lu'

export default function AppShell({ children }) {
  const loc = useLocation()
  const role = (localStorage.getItem('role') || '').toUpperCase()
  const loggedIn = !!role
  const onLogout = (e) => { e.preventDefault(); localStorage.clear(); location.href = '/login' }
  const [showProfileMenu, setShowProfileMenu] = useState(false)

  useEffect(() => { document.documentElement.classList.remove('dark'); }, [])

  const name = (localStorage.getItem('username') || 'User')
  const initials = name.split(' ').map(s => s[0]).join('').slice(0, 2).toUpperCase()

  if (!loggedIn) {
    return <ToastProvider>{children}</ToastProvider>
  }

  // Navigation items
  const navItems = role === 'ADMIN' ? [
    { to: '/admin', icon: FaHome, label: 'Overview' },
    { to: '/admin/transactions', icon: FaMoneyBillWave, label: 'Transactions' },
  ] : [
    { to: '/dashboard', icon: FaHome, label: 'Dashboard' },
    { to: '/lesions', icon: LuScan, label: 'AI Scan', isAI: true },
    { to: '/coach', icon: LuSparkles, label: 'Coach', isNew: true },
    { to: '/chat', icon: LuMessageCircle, label: 'AI Chat', isAI: true },
    { to: '/journey', icon: LuHeart, label: 'Journey', isNew: true },
    { to: '/routine', icon: FaStethoscope, label: 'Routine' },
    { to: '/find-doctors', icon: LuMapPin, label: 'Doctors', isNew: true },
    'divider',
    { to: '/appointments', icon: FaCalendarAlt, label: 'Appointments' },
    { to: '/messages', icon: FaEnvelope, label: 'Messages' },
    { to: '/transactions', icon: FaMoneyBillWave, label: 'Payments' },
  ]

  const mobileNavItems = [
    { to: '/dashboard', icon: FaHome, label: 'Home' },
    { to: '/lesions', icon: LuScan, label: 'Scan', isAI: true },
    { to: '/chat', icon: LuMessageCircle, label: 'AI', isAI: true },
    { to: '/appointments', icon: FaCalendarAlt, label: 'Book' },
  ]

  return (
    <ToastProvider>
      <div className="relative min-h-screen">
        {/* Desktop Dock - Left Side */}
        <motion.nav
          initial={{ x: -100, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }}
          className="hidden lg:flex dock left-6 top-1/2 -translate-y-1/2 flex-col gap-1"
        >
          {navItems.map((item, i) =>
            item === 'divider' ? (
              <div key={i} className="h-px bg-white/10 my-2 mx-2" />
            ) : (
              <DockIcon
                key={item.to}
                to={item.to}
                icon={item.icon}
                active={loc.pathname === item.to}
                label={item.label}
                isNew={item.isNew}
                isAI={item.isAI}
              />
            )
          )}
        </motion.nav>

        {/* Mobile Dock - Bottom */}
        <motion.nav
          initial={{ y: 100, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }}
          className="lg:hidden dock bottom-6 left-1/2 -translate-x-1/2 gap-1"
        >
          {mobileNavItems.map(item => (
            <DockIcon
              key={item.to}
              to={item.to}
              icon={item.icon}
              active={loc.pathname === item.to}
              label={item.label}
              isAI={item.isAI}
              mobile
            />
          ))}
          <button
            onClick={() => setShowProfileMenu(!showProfileMenu)}
            className={`dock-icon ${showProfileMenu ? 'active' : ''}`}
          >
            <LuUser size={20} />
          </button>
        </motion.nav>

        {/* Mobile Profile Menu */}
        <AnimatePresence>
          {showProfileMenu && (
            <motion.div
              initial={{ opacity: 0, y: 20, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 20, scale: 0.95 }}
              transition={{ duration: 0.2 }}
              className="lg:hidden fixed bottom-28 left-1/2 -translate-x-1/2 card-glass p-5 z-50 min-w-[220px]"
            >
              <button
                onClick={() => setShowProfileMenu(false)}
                className="absolute top-3 right-3 text-text-tertiary hover:text-text-primary transition-colors"
              >
                <LuX size={18} />
              </button>
              <div className="text-center mb-4 pt-2">
                <div className="h-14 w-14 mx-auto rounded-2xl bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center font-bold text-white text-lg mb-3">
                  {initials}
                </div>
                <p className="text-text-primary font-semibold">{name}</p>
                <p className="text-xs text-text-tertiary capitalize">{role.toLowerCase()}</p>
              </div>
              <button
                onClick={onLogout}
                className="w-full btn-ghost text-sm py-3 flex items-center justify-center gap-2"
              >
                <FaSignOutAlt size={14} />
                Sign Out
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Desktop Header - Top Right */}
        <header className="fixed top-6 right-6 z-40 hidden lg:flex items-center gap-4">
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="card-glass px-5 py-3 flex items-center gap-4"
          >
            <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center font-bold text-white text-sm">
              {initials}
            </div>
            <div>
              <p className="text-sm font-semibold text-text-primary tracking-tight">{name}</p>
              <p className="text-xs text-text-tertiary capitalize">{role.toLowerCase()}</p>
            </div>
            <button
              onClick={onLogout}
              className="ml-2 p-2 rounded-lg text-text-tertiary hover:text-text-primary hover:bg-white/5 transition-all"
              title="Sign Out"
            >
              <FaSignOutAlt size={16} />
            </button>
          </motion.div>
        </header>

        {/* Main Content */}
        <main className="min-h-screen pt-6 pb-32 lg:pb-12 px-6 lg:pl-32">
          <div className="max-w-[1600px] mx-auto">
            {children}
          </div>
        </main>
      </div>
    </ToastProvider>
  )
}

function DockIcon({ to, icon: Icon, active, label, isNew, isAI, mobile }) {
  const [showTooltip, setShowTooltip] = useState(false)

  return (
    <div className="relative">
      <Link
        to={to}
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        className={`dock-icon ${active ? 'active' : ''} ${isAI && active ? '!text-ai-400 !bg-ai-500/10' : ''}`}
      >
        <Icon size={mobile ? 18 : 20} />
        {isNew && (
          <span className="absolute -top-0.5 -right-0.5 h-2.5 w-2.5 rounded-full bg-accent-500 border-2 border-surface-elevated animate-pulse" />
        )}
      </Link>

      {/* Desktop Tooltip */}
      <AnimatePresence>
        {showTooltip && !mobile && (
          <motion.div
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -10 }}
            className="absolute left-full ml-4 top-1/2 -translate-y-1/2 card-glass px-3 py-2 text-xs font-medium text-text-primary whitespace-nowrap pointer-events-none hidden lg:block"
          >
            {label}
            {isNew && (
              <span className="ml-2 text-accent-400 uppercase text-[10px] font-bold">NEW</span>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
