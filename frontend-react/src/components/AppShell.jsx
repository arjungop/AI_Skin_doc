import { Link, useLocation } from 'react-router-dom'
import { ToastProvider } from './Toast.jsx'
import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  FaHome, FaStethoscope, FaComments, FaCalendarAlt,
  FaMoneyBillWave, FaEnvelope, FaUserMd, FaSignOutAlt, FaCog
} from 'react-icons/fa'
import { LuScan, LuSparkles, LuUser, LuMessageCircle, LuMapPin, LuHeart, LuX, LuMenu, LuActivity } from 'react-icons/lu'

export default function AppShell({ children }) {
  const loc = useLocation()
  const role = (localStorage.getItem('role') || '').toUpperCase()
  const loggedIn = !!role
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  const onLogout = (e) => { e.preventDefault(); localStorage.clear(); location.href = '/login' }
  const name = (localStorage.getItem('username') || 'User')
  const initials = name.split(' ').map(s => s[0]).join('').slice(0, 2).toUpperCase()

  useEffect(() => { document.documentElement.classList.remove('dark'); }, [])
  useEffect(() => { setMobileMenuOpen(false) }, [loc.pathname])

  if (!loggedIn) {
    return <ToastProvider>{children}</ToastProvider>
  }

  const navItems = role === 'ADMIN' ? [
    { to: '/admin', icon: FaHome, label: 'Overview' },
    { to: '/admin/transactions', icon: FaMoneyBillWave, label: 'Transactions' },
  ] : [
    {
      label: 'Overview', items: [
        { to: '/dashboard', icon: FaHome, label: 'Dashboard' },
        { to: '/journey', icon: LuHeart, label: 'Skin Journey', isNew: true },
      ]
    },
    {
      label: 'AI Health', items: [
        { to: '/lesions', icon: LuScan, label: 'Lesion Scan', isAI: true },
        { to: '/chat', icon: LuMessageCircle, label: 'AI Assistant', isAI: true },
        { to: '/coach', icon: LuSparkles, label: 'Skin Coach', isNew: true },
      ]
    },
    {
      label: 'Care', items: [
        { to: '/appointments', icon: FaCalendarAlt, label: 'Appointments' },
        { to: '/routine', icon: FaStethoscope, label: 'Daily Routine' },
        { to: '/find-doctors', icon: LuMapPin, label: 'Find Doctors' },
        { to: '/messages', icon: FaEnvelope, label: 'Messages' },
      ]
    },
  ]

  const SidebarContent = () => (
    <div className="flex flex-col h-full">
      {/* Brand */}
      <div className="p-6 flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-primary-500 flex items-center justify-center text-white">
          <LuActivity size={18} />
        </div>
        <span className="text-lg font-bold text-slate-900 tracking-tight">Skin.AI</span>
      </div>

      {/* Navigation */}
      <div className="flex-1 overflow-y-auto px-4 py-2 space-y-8">
        {navItems.map((group, i) => (
          <div key={i}>
            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3 px-2">
              {group.label}
            </h3>
            <div className="space-y-1">
              {group.items?.map(item => (
                <Link
                  key={item.to}
                  to={item.to}
                  className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${loc.pathname === item.to
                      ? 'bg-primary-50 text-primary-700'
                      : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
                    }`}
                >
                  <item.icon size={18} className={loc.pathname === item.to ? 'text-primary-500' : 'text-slate-400 group-hover:text-slate-600'} />
                  <span>{item.label}</span>
                  {item.isNew && (
                    <span className="ml-auto text-[10px] font-bold text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full">NEW</span>
                  )}
                  {item.isAI && (
                    <span className="ml-auto text-[10px] font-bold text-violet-600 bg-violet-50 px-2 py-0.5 rounded-full">AI</span>
                  )}
                </Link>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* User Profile */}
      <div className="p-4 border-t border-slate-100">
        <div className="flex items-center gap-3 p-2 rounded-xl hover:bg-slate-50 transition-colors cursor-pointer group">
          <div className="w-9 h-9 rounded-full bg-primary-100 text-primary-600 flex items-center justify-center text-sm font-bold">
            {initials}
          </div>
          <div className="flex-1">
            <p className="text-sm font-medium text-slate-900">{name}</p>
            <p className="text-xs text-slate-500 capitalize">{role.toLowerCase()}</p>
          </div>
          <button
            onClick={onLogout}
            className="text-slate-400 hover:text-rose-500 transition-colors p-1"
            title="Sign out"
          >
            <FaSignOutAlt size={16} />
          </button>
        </div>
      </div>
    </div>
  )

  return (
    <ToastProvider>
      <div className="min-h-screen bg-slate-50 flex">
        {/* Desktop Sidebar */}
        <aside className="hidden lg:block w-64 bg-white border-r border-slate-200 fixed inset-y-0 z-50">
          <SidebarContent />
        </aside>

        {/* Mobile Header */}
        <div className="lg:hidden fixed top-0 inset-x-0 z-40 bg-white border-b border-slate-200 px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-primary-500 flex items-center justify-center text-white">
              <LuActivity size={18} />
            </div>
            <span className="font-bold text-slate-900">Skin.AI</span>
          </div>
          <button onClick={() => setMobileMenuOpen(true)} className="p-2 text-slate-600">
            <LuMenu size={24} />
          </button>
        </div>

        {/* Mobile Sidebar Overlay */}
        <AnimatePresence>
          {mobileMenuOpen && (
            <>
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                onClick={() => setMobileMenuOpen(false)}
                className="fixed inset-0 z-40 bg-slate-900/50 backdrop-blur-sm lg:hidden"
              />
              <motion.div
                initial={{ x: '-100%' }}
                animate={{ x: 0 }}
                exit={{ x: '-100%' }}
                transition={{ type: 'spring', damping: 25, stiffness: 200 }}
                className="fixed inset-y-0 left-0 z-50 w-64 bg-white shadow-xl lg:hidden"
              >
                <div className="absolute top-2 right-2">
                  <button onClick={() => setMobileMenuOpen(false)} className="p-2 text-slate-400 hover:text-slate-600">
                    <LuX size={20} />
                  </button>
                </div>
                <SidebarContent />
              </motion.div>
            </>
          )}
        </AnimatePresence>

        {/* Main Content */}
        <main className="flex-1 lg:pl-64 min-h-screen">
          <div className="max-w-6xl mx-auto px-4 py-8 lg:px-8 mt-14 lg:mt-0">
            {children}
          </div>
        </main>
      </div>
    </ToastProvider>
  )
}
