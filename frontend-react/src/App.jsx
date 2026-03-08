import { Routes, Route, Navigate, useNavigate } from 'react-router-dom'
import { lazy, Suspense, useEffect, useState } from 'react'
import ErrorBoundary from './components/ErrorBoundary.jsx'

// Eagerly load the shell — everything else is lazy
import AppShell from './components/AppShell.jsx'

// Lazy-loaded pages
const Login = lazy(() => import('./pages/Login.jsx'))
const Register = lazy(() => import('./pages/Register.jsx'))
const Dashboard = lazy(() => import('./pages/Dashboard.jsx'))
const DoctorDashboard = lazy(() => import('./pages/DoctorDashboard.jsx'))
const DoctorTreatmentPlans = lazy(() => import('./pages/DoctorTreatmentPlans.jsx'))
const DoctorAppointments = lazy(() => import('./pages/DoctorAppointments.tsx'))
const DoctorAvailability = lazy(() => import('./pages/DoctorAvailability.tsx'))
const DoctorPatients = lazy(() => import('./pages/DoctorPatients.tsx'))
const DoctorTransactions = lazy(() => import('./pages/DoctorTransactions.tsx'))
const DoctorSettings = lazy(() => import('./pages/DoctorSettings.tsx'))
const DoctorProfile = lazy(() => import('./pages/DoctorProfile.tsx'))
const LesionUpload = lazy(() => import('./pages/LesionUpload.jsx'))
const Chat = lazy(() => import('./pages/Chat.jsx'))
const Appointments = lazy(() => import('./pages/Appointments.jsx'))
const Transactions = lazy(() => import('./pages/Transactions.jsx'))
const AdminDashboard = lazy(() => import('./pages/AdminDashboard.jsx'))
const AdminTransactions = lazy(() => import('./pages/AdminTransactions.jsx'))
const AdminUsers = lazy(() => import('./pages/AdminUsers.jsx'))
const ApplyDoctor = lazy(() => import('./pages/ApplyDoctor.jsx'))
const Messages = lazy(() => import('./pages/Messages.jsx'))
const Contact = lazy(() => import('./pages/Contact.jsx'))
const ForgotPassword = lazy(() => import('./pages/ForgotPassword.jsx'))
const SkinCoach = lazy(() => import('./pages/SkinCoach.jsx'))
const Routine = lazy(() => import('./pages/Routine.jsx'))
const Onboarding = lazy(() => import('./pages/Onboarding.jsx'))
const FindDermatologists = lazy(() => import('./pages/FindDermatologists.jsx'))
const SkinAgent = lazy(() => import('./pages/SkinAgent.jsx'))

function Protected({ children, allowedRoles }) {
  const navigate = useNavigate()
  const [checked, setChecked] = useState(false)

  useEffect(() => {
    let mounted = true;
    (async () => {
      const attemptAuth = async () => {
        const t = localStorage.getItem('access_token')
        if (!t) return null
        const me = await (await import('./services/api.js')).api.me()
        return me
      }

      try {
        let me = null
        try {
          me = await attemptAuth()
        } catch (firstErr) {
          // First attempt failed — wait and retry once before giving up
          await new Promise(r => setTimeout(r, 1500))
          if (!mounted) return
          try {
            me = await attemptAuth()
          } catch {
            // Second attempt also failed — redirect to login
            if (!mounted) return
            try { localStorage.clear() } catch { }
            navigate('/login', { replace: true })
            return
          }
        }

        if (!me) { navigate('/login', { replace: true }); return }
        if (!mounted) return

        if (me && me.access_token) { try { localStorage.setItem('access_token', me.access_token) } catch { } }
        if (me && me.role) { localStorage.setItem('role', me.role) }
        if (me && me.user_id) { localStorage.setItem('user_id', String(me.user_id)) }
        const pid = me?.patient_id || me?.claims?.patient_id
        const did = me?.doctor_id || me?.claims?.doctor_id
        if (pid) { localStorage.setItem('patient_id', String(pid)) }
        if (did) { localStorage.setItem('doctor_id', String(did)) }

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

function PageLoader() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50">
      <div className="w-8 h-8 border-4 border-slate-200 border-t-primary-500 rounded-full animate-spin" />
    </div>
  )
}

function NotFound() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-slate-50 gap-4">
      <h1 className="text-4xl font-bold text-slate-800">404</h1>
      <p className="text-slate-500">Page not found</p>
      <a href="/dashboard" className="text-primary-600 hover:underline text-sm">Go to Dashboard</a>
    </div>
  )
}

export default function App() {
  return (
    <ErrorBoundary>
      <Suspense fallback={<PageLoader />}>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route path="/apply-doctor" element={<ApplyDoctor />} />
          <Route path="/forgot-password" element={<ForgotPassword />} />
          <Route path="/onboarding" element={<Protected allowedRoles={['PATIENT']}><Onboarding /></Protected>} />
          <Route path="/coach" element={<Protected allowedRoles={['PATIENT']}><AppShell><SkinCoach /></AppShell></Protected>} />
          <Route path="/agent" element={<Protected allowedRoles={['PATIENT']}><AppShell><SkinAgent /></AppShell></Protected>} />
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
          <Route path="/lesions" element={<Protected allowedRoles={['PATIENT']}><AppShell><LesionUpload /></AppShell></Protected>} />
          <Route path="/chat" element={<Protected allowedRoles={['PATIENT', 'DOCTOR']}><AppShell><Chat /></AppShell></Protected>} />
          <Route path="/messages" element={<Protected allowedRoles={['PATIENT', 'DOCTOR']}><AppShell><Messages /></AppShell></Protected>} />
          <Route path="/appointments" element={<Protected allowedRoles={['PATIENT']}><AppShell><Appointments /></AppShell></Protected>} />
          <Route path="/transactions" element={<Protected allowedRoles={['PATIENT']}><AppShell><Transactions /></AppShell></Protected>} />
          <Route path="/contact" element={<Protected allowedRoles={['PATIENT', 'DOCTOR']}><AppShell><Contact /></AppShell></Protected>} />
          <Route path="/admin" element={<Protected allowedRoles={['ADMIN']}><AppShell><AdminDashboard /></AppShell></Protected>} />
          <Route path="/admin/transactions" element={<Protected allowedRoles={['ADMIN']}><AppShell><AdminTransactions /></AppShell></Protected>} />
          <Route path="/admin/users" element={<Protected allowedRoles={['ADMIN']}><AppShell><AdminUsers /></AppShell></Protected>} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </Suspense>
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
