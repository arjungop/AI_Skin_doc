import { Routes, Route, Navigate, useNavigate } from 'react-router-dom'
import { useEffect, useState } from 'react'
import ErrorBoundary from './components/ErrorBoundary.jsx'
import AppShell from './components/AppShell.jsx'
import Login from './pages/Login.jsx'
import Register from './pages/Register.jsx'
import Dashboard from './pages/Dashboard.jsx'
import DoctorDashboard from './pages/DoctorDashboard.jsx'
import DoctorTreatmentPlans from './pages/DoctorTreatmentPlans.jsx'
import DoctorAppointments from './pages/DoctorAppointments.tsx'
import DoctorAvailability from './pages/DoctorAvailability.tsx'
import DoctorPatients from './pages/DoctorPatients.tsx'
import DoctorTransactions from './pages/DoctorTransactions.tsx'
import DoctorSettings from './pages/DoctorSettings.tsx'
import DoctorProfile from './pages/DoctorProfile.tsx'
import LesionUpload from './pages/LesionUpload.jsx'
import Chat from './pages/Chat.jsx'
import Appointments from './pages/Appointments.jsx'
import Transactions from './pages/Transactions.jsx'
import AdminDashboard from './pages/AdminDashboard.jsx'
import AdminTransactions from './pages/AdminTransactions.jsx'
import AdminUsers from './pages/AdminUsers.jsx'
import ApplyDoctor from './pages/ApplyDoctor.jsx'
import Messages from './pages/Messages.jsx'
import Contact from './pages/Contact.jsx'
import ForgotPassword from './pages/ForgotPassword.jsx'
import SkinCoach from './pages/SkinCoach.jsx'
import SkinJourney from './pages/SkinJourney.jsx'
import Routine from './pages/Routine.jsx'
import Onboarding from './pages/Onboarding.jsx'
import FindDermatologists from './pages/FindDermatologists.jsx'
// Landing removed per simplified hospital app scope

function Protected({ children, allowedRoles }) {
  const navigate = useNavigate()
  const [checked, setChecked] = useState(false)

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const t = localStorage.getItem('access_token')
        if (!t) { navigate('/login', { replace: true }); return }
        const me = await (await import('./services/api.js')).api.me()
        if (!mounted) return
        if (me && me.access_token) { try { localStorage.setItem('access_token', me.access_token) } catch { } }
        if (me && me.role) { localStorage.setItem('role', me.role) }
        if (me && me.user_id) { localStorage.setItem('user_id', String(me.user_id)) }
        if (me && me.patient_id) { localStorage.setItem('patient_id', String(me.patient_id)) }
        if (me && me.doctor_id) { localStorage.setItem('doctor_id', String(me.doctor_id)) }

        // Role-based access control
        if (allowedRoles && allowedRoles.length > 0) {
          const userRole = (me.role || '').toUpperCase()
          if (!allowedRoles.includes(userRole)) {
            const dest = userRole === 'DOCTOR' ? '/doctor' : userRole === 'ADMIN' ? '/admin' : '/dashboard'
            navigate(dest, { replace: true })
            return
          }
        }
        setChecked(true)
      } catch (err) {
        if (!mounted) return
        try { localStorage.clear() } catch { }
        navigate('/login', { replace: true })
      }
    })()

    // Listen for auth expiry from api.js 401 interceptor
    const handleAuthExpired = () => {
      try { localStorage.clear() } catch { }
      navigate('/login', { replace: true })
    }
    window.addEventListener('auth:expired', handleAuthExpired)

    // Listen for localStorage changes from other tabs
    const handleStorage = (e) => {
      if (e.key === 'access_token' && !e.newValue) {
        navigate('/login', { replace: true })
      }
    }
    window.addEventListener('storage', handleStorage)

    return () => {
      mounted = false
      window.removeEventListener('auth:expired', handleAuthExpired)
      window.removeEventListener('storage', handleStorage)
    }
  }, [navigate, allowedRoles])

  if (!checked) return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50">
      <div className="w-8 h-8 border-4 border-slate-200 border-t-primary-500 rounded-full animate-spin" />
    </div>
  )
  return children
}

export default function App() {
  return (
    <ErrorBoundary>
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />
      <Route path="/apply-doctor" element={<ApplyDoctor />} />
      <Route path="/forgot-password" element={<ForgotPassword />} />
      <Route path="/onboarding" element={<Protected allowedRoles={['PATIENT']}><Onboarding /></Protected>} />
      <Route path="/coach" element={<Protected allowedRoles={['PATIENT']}><AppShell><SkinCoach /></AppShell></Protected>} />
      <Route path="/journey" element={<Protected allowedRoles={['PATIENT']}><AppShell><SkinJourney /></AppShell></Protected>} />
      <Route path="/routine" element={<Protected allowedRoles={['PATIENT']}><AppShell><Routine /></AppShell></Protected>} />
      <Route path="/find-doctors" element={<Protected allowedRoles={['PATIENT']}><AppShell><FindDermatologists /></AppShell></Protected>} />

      <Route path="/" element={<IndexRoute />} />
      <Route path="/dashboard" element={<Protected allowedRoles={['PATIENT']}><AppShell><Dashboard /></AppShell></Protected>} />
      <Route path="/doctor" element={<Protected allowedRoles={['DOCTOR']}><AppShell><DoctorDashboard /></AppShell></Protected>} />
      <Route path="/doctor/appointments" element={<Protected allowedRoles={['DOCTOR']}><AppShell><DoctorAppointments /></AppShell></Protected>} />
      <Route path="/doctor/availability" element={<Protected allowedRoles={['DOCTOR']}><AppShell><DoctorAvailability /></AppShell></Protected>} />
      <Route path="/doctor/patients" element={<Protected allowedRoles={['DOCTOR']}><AppShell><DoctorPatients /></AppShell></Protected>} />
      <Route path="/doctor/transactions" element={<Protected allowedRoles={['DOCTOR']}><AppShell><DoctorTransactions /></AppShell></Protected>} />
      <Route path="/doctor/settings" element={<Protected allowedRoles={['DOCTOR']}><AppShell><DoctorSettings /></AppShell></Protected>} />
      <Route path="/doctor/profile" element={<Protected allowedRoles={['DOCTOR']}><AppShell><DoctorProfile /></AppShell></Protected>} />
      <Route path="/doctor/treatment-plans" element={<Protected allowedRoles={['DOCTOR']}><AppShell><DoctorTreatmentPlans /></AppShell></Protected>} />
      <Route path="/lesions" element={<Protected><AppShell><LesionUpload /></AppShell></Protected>} />
      <Route path="/chat" element={<Protected><AppShell><Chat /></AppShell></Protected>} />
      <Route path="/messages" element={<Protected allowedRoles={['PATIENT', 'DOCTOR']}><AppShell><Messages /></AppShell></Protected>} />
      <Route path="/appointments" element={<Protected allowedRoles={['PATIENT', 'DOCTOR']}><AppShell><Appointments /></AppShell></Protected>} />
      <Route path="/transactions" element={<Protected allowedRoles={['PATIENT']}><AppShell><Transactions /></AppShell></Protected>} />
      <Route path="/contact" element={<Protected><AppShell><Contact /></AppShell></Protected>} />
      <Route path="/admin" element={<Protected allowedRoles={['ADMIN']}><AppShell><AdminDashboard /></AppShell></Protected>} />
      <Route path="/admin/transactions" element={<Protected allowedRoles={['ADMIN']}><AppShell><AdminTransactions /></AppShell></Protected>} />
      <Route path="/admin/users" element={<Protected allowedRoles={['ADMIN']}><AppShell><AdminUsers /></AppShell></Protected>} />
      <Route path="*" element={<AppShell><Navigate to="/dashboard" replace /></AppShell>} />
    </Routes>
    </ErrorBoundary>
  )
}

function IndexRoute() {
  const role = (localStorage.getItem('role') || '').toUpperCase()
  const hasSession = !!(localStorage.getItem('user_id') || localStorage.getItem('patient_id'))
  const [target, setTarget] = useState('')

  useEffect(() => {
    if (!role || !hasSession) return
    if (role === 'PATIENT') {
      import('./services/api.js').then(({ api }) => {
        api.getProfile().then(() => setTarget('/dashboard'))
          .catch(() => setTarget('/onboarding'))
      })
    } else if (role === 'DOCTOR') {
      setTarget('/doctor')
    } else if (role === 'ADMIN') {
      setTarget('/admin')
    } else {
      setTarget('/dashboard')
    }
  }, [role, hasSession])

  if (!role || !hasSession) return <Navigate to="/login" replace />
  if (!target) return <div className="min-h-screen bg-[#f8fafc] flex items-center justify-center animate-pulse">Loading...</div>
  return <AppShell><Navigate to={target} replace /></AppShell>
}
