import { Routes, Route, Navigate, useNavigate } from 'react-router-dom'
import { useEffect, useState } from 'react'
import AppShell from './components/AppShell.jsx'
import Login from './pages/Login.jsx'
import Register from './pages/Register.jsx'
import Dashboard from './pages/Dashboard.jsx'
import DoctorDashboard from './pages/DoctorDashboard.jsx'
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

function Protected({ children }) {
  const navigate = useNavigate()
  const [checked, setChecked] = useState(false)
  useEffect(() => {
    (async () => {
      try {
        // If no token at all, bail early
        const t = localStorage.getItem('access_token')
        if (!t) { navigate('/login', { replace: true }); return }
        const me = await (await import('./services/api.js')).api.me()
        // If backend refreshed the token, persist it
        if (me && me.access_token) { try { localStorage.setItem('access_token', me.access_token) } catch { } }
        // Ensure role/id present for downstream UX
        if (me && me.role) { localStorage.setItem('role', me.role) }
        if (me && me.user_id) { localStorage.setItem('user_id', String(me.user_id)) }
        setChecked(true)
      } catch (err) {
        try { localStorage.clear() } catch { }
        navigate('/login', { replace: true })
      }
    })()
  }, [])
  if (!checked) return null
  return children
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />
      <Route path="/apply-doctor" element={<ApplyDoctor />} />
      <Route path="/forgot-password" element={<ForgotPassword />} />
      <Route path="/onboarding" element={<Protected><Onboarding /></Protected>} />
      <Route path="/coach" element={<Protected><AppShell><SkinCoach /></AppShell></Protected>} />
      <Route path="/journey" element={<Protected><AppShell><SkinJourney /></AppShell></Protected>} />
      <Route path="/routine" element={<Protected><AppShell><Routine /></AppShell></Protected>} />
      <Route path="/find-doctors" element={<Protected><AppShell><FindDermatologists /></AppShell></Protected>} />

      <Route path="/" element={<IndexRoute />} />
      <Route path="/dashboard" element={<Protected><AppShell><Dashboard /></AppShell></Protected>} />
      <Route path="/doctor" element={<Protected><AppShell><DoctorDashboard /></AppShell></Protected>} />
      <Route path="/doctor/appointments" element={<Protected><AppShell><DoctorAppointments /></AppShell></Protected>} />
      <Route path="/doctor/availability" element={<Protected><AppShell><DoctorAvailability /></AppShell></Protected>} />
      <Route path="/doctor/patients" element={<Protected><AppShell><DoctorPatients /></AppShell></Protected>} />
      <Route path="/doctor/transactions" element={<Protected><AppShell><DoctorTransactions /></AppShell></Protected>} />
      <Route path="/doctor/settings" element={<Protected><AppShell><DoctorSettings /></AppShell></Protected>} />
      <Route path="/doctor/profile" element={<Protected><AppShell><DoctorProfile /></AppShell></Protected>} />
      <Route path="/lesions" element={<Protected><AppShell><LesionUpload /></AppShell></Protected>} />
      <Route path="/chat" element={<Protected><AppShell><Chat /></AppShell></Protected>} />
      <Route path="/messages" element={<Protected><AppShell><Messages /></AppShell></Protected>} />
      <Route path="/appointments" element={<Protected><AppShell><Appointments /></AppShell></Protected>} />
      <Route path="/transactions" element={<Protected><AppShell><Transactions /></AppShell></Protected>} />
      <Route path="/contact" element={<AppShell><Contact /></AppShell>} />
      <Route path="/admin" element={<Protected><AppShell><AdminDashboard /></AppShell></Protected>} />
      <Route path="/admin/transactions" element={<Protected><AppShell><AdminTransactions /></AppShell></Protected>} />
      <Route path="*" element={<AppShell><Navigate to="/dashboard" replace /></AppShell>} />
    </Routes>
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
    } else {
      setTarget('/dashboard')
    }
  }, [role, hasSession])

  if (!role || !hasSession) return <Navigate to="/login" replace />
  if (!target) return <div className="min-h-screen bg-[#f8fafc] flex items-center justify-center animate-pulse">Loading...</div>
  return <AppShell><Navigate to={target} replace /></AppShell>
}
