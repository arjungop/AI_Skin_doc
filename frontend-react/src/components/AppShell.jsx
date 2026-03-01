import { Link, useLocation } from 'react-router-dom'
import { ToastProvider } from './Toast.jsx'
import OfflineBanner from './OfflineBanner.jsx'
import { useEffect, useState, useRef, useCallback } from 'react'
import { api } from '../services/api'
import useNotifications from '../hooks/useNotifications'
import { motion, AnimatePresence } from 'framer-motion'
import {
  FaHome, FaStethoscope, FaComments, FaCalendarAlt,
  FaMoneyBillWave, FaEnvelope, FaUserMd, FaSignOutAlt
} from 'react-icons/fa'
import {
  LuScan, LuSparkles, LuUser, LuMessageCircle, LuMapPin,
  LuHeart, LuX, LuUsers, LuCalendarClock, LuPill, LuSettings, LuClipboardList
} from 'react-icons/lu'

export default function AppShell({ children }) {
  const loc = useLocation()
  const role = (localStorage.getItem('role') || '').toUpperCase()
  const loggedIn = !!role
  const onLogout = (e) => { e.preventDefault(); localStorage.clear(); location.href = '/login' }
  const [showProfileMenu, setShowProfileMenu] = useState(false)
  const [totalUnread, setTotalUnread] = useState(0)
  const { subscribe } = useNotifications()

  // Fetch unread counts
  const fetchUnread = useCallback(async () => {
    try {
      const data = await api.getUnreadCounts()
      setTotalUnread(data.total_unread || 0)
    } catch {}
  }, [])

  // Poll unread counts every 30s + refresh on WS events
  useEffect(() => {
    fetchUnread()
    const interval = setInterval(fetchUnread, 30_000)
    return () => clearInterval(interval)
  }, [fetchUnread])

  useEffect(() => {
    const unsubs = [
      subscribe('new_message', fetchUnread),
      subscribe('messages_read', fetchUnread),
    ]
    return () => unsubs.forEach(u => u())
  }, [subscribe, fetchUnread])

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
    { to: '/admin/users', icon: LuUsers, label: 'Users' },
    'divider',
    { to: '/messages', icon: FaEnvelope, label: 'Messages', badge: totalUnread },
  ] : role === 'DOCTOR' ? [
    { to: '/doctor', icon: FaHome, label: 'Dashboard' },
    { to: '/doctor/patients', icon: LuUsers, label: 'Patients' },
    { to: '/doctor/appointments', icon: FaCalendarAlt, label: 'Appointments' },
    { to: '/doctor/treatment-plans', icon: LuPill, label: 'Treatment Plans' },
    { to: '/doctor/availability', icon: LuCalendarClock, label: 'Availability' },
    'divider',
    { to: '/messages', icon: FaEnvelope, label: 'Messages', badge: totalUnread },
    { to: '/doctor/transactions', icon: FaMoneyBillWave, label: 'Billing' },
    { to: '/doctor/settings', icon: LuSettings, label: 'Settings' },
  ] : [
    { to: '/dashboard', icon: FaHome, label: 'Dashboard' },
    { to: '/lesions', icon: LuScan, label: 'AI Scan', isAI: true },
    { to: '/coach', icon: LuSparkles, label: 'Coach', isNew: true },
    { to: '/chat', icon: LuMessageCircle, label: 'AI Chat', isAI: true },
    { to: '/journey', icon: LuHeart, label: 'Journey', isNew: true },
    { to: '/routine', icon: FaStethoscope, label: 'Treatment Plan' },
    { to: '/find-doctors', icon: LuMapPin, label: 'Doctors', isNew: true },
    'divider',
    { to: '/appointments', icon: FaCalendarAlt, label: 'Appointments' },
    { to: '/messages', icon: FaEnvelope, label: 'Messages', badge: totalUnread },
    { to: '/transactions', icon: FaMoneyBillWave, label: 'Payments' },
  ]

  const mobileNavItems = role === 'DOCTOR' ? [
    { to: '/doctor', icon: FaHome, label: 'Home' },
    { to: '/doctor/patients', icon: LuUsers, label: 'Patients' },
    { to: '/doctor/appointments', icon: FaCalendarAlt, label: 'Appts' },
    { to: '/messages', icon: FaEnvelope, label: 'Messages', badge: totalUnread },
  ] : [
    { to: '/dashboard', icon: FaHome, label: 'Home' },
    { to: '/lesions', icon: LuScan, label: 'Scan', isAI: true },
    { to: '/chat', icon: LuMessageCircle, label: 'AI Chat', isAI: true },
    { to: '/messages', icon: FaEnvelope, label: 'Msgs', badge: totalUnread },
    { to: '/appointments', icon: FaCalendarAlt, label: 'Book' },
  ]

  return (
    <ToastProvider>
      <div className="relative min-h-screen">
        {/* Desktop Sidebar - Left Side */}
        <motion.nav
          initial={{ x: -100, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }}
          className="hidden lg:flex fixed left-0 top-0 bottom-0 w-56 bg-white/90 backdrop-blur-xl border-r border-slate-200/60 shadow-soft-md flex-col z-50 py-6"
        >
          {/* App Title */}
          <div className="px-5 mb-6">
            <h1 className="text-lg font-bold text-primary-600 tracking-tight">Skin Doc</h1>
            <p className="text-[11px] text-slate-400 capitalize">{role.toLowerCase()} portal</p>
          </div>

          {/* Nav Items */}
          <div className="flex-1 overflow-y-auto px-3 space-y-1">
            {navItems.map((item, i) =>
              item === 'divider' ? (
                <div key={i} className="h-px bg-slate-200/60 my-3 mx-2" />
              ) : (
                <SidebarLink
                  key={item.to}
                  to={item.to}
                  icon={item.icon}
                  active={loc.pathname === item.to}
                  label={item.label}
                  isNew={item.isNew}
                  isAI={item.isAI}
                  badge={item.badge}
                />
              )
            )}
          </div>

          {/* Profile & Logout at bottom */}
          <div className="px-3 pt-4 border-t border-slate-200/60 mt-2">
            <div className="flex items-center gap-3 px-3 py-2">
              <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center font-bold text-white text-xs flex-shrink-0">
                {initials}
              </div>
              <div className="min-w-0 flex-1">
                <p className="text-sm font-semibold text-slate-800 truncate">{name}</p>
                <p className="text-[11px] text-slate-400 capitalize">{role.toLowerCase()}</p>
              </div>
            </div>
            <button
              onClick={onLogout}
              className="w-full mt-2 flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm text-slate-500 hover:text-red-600 hover:bg-red-50 transition-all"
            >
              <FaSignOutAlt size={14} />
              <span>Sign Out</span>
            </button>
          </div>
        </motion.nav>

        {/* Mobile Bottom Navigation */}
        <motion.nav
          initial={{ y: 100, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }}
          className="lg:hidden fixed bottom-0 left-0 right-0 bg-white/95 backdrop-blur-xl border-t border-slate-200/60 shadow-soft-xl z-50 px-2 pb-[env(safe-area-inset-bottom)]"
        >
          <div className="flex items-center justify-around py-2">
            {mobileNavItems.map(item => (
              <MobileNavLink
                key={item.to}
                to={item.to}
                icon={item.icon}
                active={loc.pathname === item.to}
                label={item.label}
                isAI={item.isAI}
                badge={item.badge}
              />
            ))}
            <button
              onClick={() => setShowProfileMenu(!showProfileMenu)}
              className={`flex flex-col items-center gap-0.5 px-2 py-1 rounded-xl transition-all ${showProfileMenu ? 'text-primary-600' : 'text-slate-400'}`}
              aria-label="Profile menu"
            >
              <LuUser size={20} />
              <span className="text-[10px] font-medium">Profile</span>
            </button>
          </div>
        </motion.nav>

        {/* Mobile Profile Menu */}
        <AnimatePresence>
          {showProfileMenu && (
            <motion.div
              initial={{ opacity: 0, y: 20, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 20, scale: 0.95 }}
              transition={{ duration: 0.2 }}
              className="lg:hidden fixed bottom-20 left-1/2 -translate-x-1/2 card-glass p-5 z-50 min-w-[220px]"
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

        {/* Offline Banner */}
        <OfflineBanner />

        {/* Main Content */}
        <main className="min-h-screen pt-6 pb-24 lg:pb-12 px-6 lg:pl-60">
          <div className="max-w-[1600px] mx-auto">
            {children}
          </div>
        </main>
      </div>
    </ToastProvider>
  )
}

function SidebarLink({ to, icon: Icon, active, label, isNew, isAI, badge }) {
  return (
    <Link
      to={to}
      className={`flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 group relative ${
        active
          ? isAI ? 'bg-ai-50 text-ai-600' : 'bg-primary-50 text-primary-600 shadow-sm'
          : 'text-slate-500 hover:text-slate-800 hover:bg-slate-100'
      }`}
    >
      <div className="relative flex-shrink-0">
        <Icon size={18} />
        {badge > 0 && (
          <span className="absolute -top-1.5 -right-1.5 min-w-[16px] h-4 px-1 flex items-center justify-center rounded-full bg-rose-500 text-white text-[9px] font-bold leading-none">
            {badge > 99 ? '99+' : badge}
          </span>
        )}
      </div>
      <span className="truncate">{label}</span>
      {isNew && (
        <span className="ml-auto text-[10px] font-bold uppercase text-accent-500 bg-accent-50 px-1.5 py-0.5 rounded-md">NEW</span>
      )}
      {isAI && !isNew && (
        <span className="ml-auto text-[10px] font-bold uppercase text-ai-500 bg-ai-50 px-1.5 py-0.5 rounded-md">AI</span>
      )}
    </Link>
  )
}

function MobileNavLink({ to, icon: Icon, active, label, isAI, badge }) {
  return (
    <Link
      to={to}
      className={`flex flex-col items-center gap-0.5 px-2 py-1 rounded-xl transition-all min-w-[48px] ${
        active
          ? isAI ? 'text-ai-600' : 'text-primary-600'
          : 'text-slate-400 hover:text-slate-600'
      }`}
    >
      <div className="relative">
        <Icon size={20} />
        {badge > 0 && (
          <span className="absolute -top-1 -right-2 min-w-[14px] h-3.5 px-1 flex items-center justify-center rounded-full bg-rose-500 text-white text-[8px] font-bold leading-none">
            {badge > 99 ? '99+' : badge}
          </span>
        )}
      </div>
      <span className={`text-[10px] font-medium ${active ? 'font-semibold' : ''}`}>{label}</span>
    </Link>
  )
}
