import { useState, useEffect, useMemo } from 'react'
import { Link } from 'react-router-dom'
import { FaUserPlus, FaClipboardList, FaChartLine } from 'react-icons/fa'
import { LuSearch, LuArrowRight, LuSparkles } from 'react-icons/lu'
import { motion } from 'framer-motion'
import QuickPrescription from '../components/dashboard/QuickPrescription'
import PatientQueue from '../components/dashboard/PatientQueueList'
import { Card, CardTitle, CardDescription, IconWrapper } from '../components/Card'
import { api } from '../services/api'

export default function DoctorDashboard() {
  const name = (localStorage.getItem('username') || 'Doctor').split(' ')[0]

  const [appointments, setAppointments] = useState([])
  const [patients, setPatients] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function load() {
      try {
        const [apptsRes, patsRes] = await Promise.allSettled([
          api.listAppointments(),
          api.listPatients()
        ])
        if (apptsRes.status === 'fulfilled') setAppointments(apptsRes.value || [])
        if (patsRes.status === 'fulfilled') setPatients(Array.isArray(patsRes.value) ? patsRes.value : patsRes.value?.items || [])
      } catch (err) {
        console.error('Doctor dashboard load failed:', err)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  const todayStr = new Date().toISOString().slice(0, 10)

  const todaysAppts = useMemo(() => {
    return appointments.filter(a => {
      if (!['Scheduled', 'Confirmed', 'Arrived', 'Waiting'].includes(a.status)) return false
      return a.appointment_date.startsWith(todayStr)
    })
  }, [appointments, todayStr])

  const pendingReviews = 0 // Not implemented yet

  const container = {
    hidden: { opacity: 0 },
    show: { opacity: 1, transition: { staggerChildren: 0.08 } }
  }

  const item = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.4, 0, 0.2, 1] } }
  }

  return (
    <div className="relative min-h-screen">
      {/* Ambient Background Glow */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute top-0 right-1/4 w-[500px] h-[500px] bg-primary-500/10 rounded-full blur-[120px] opacity-40" />
        <div className="absolute bottom-1/4 left-1/4 w-[400px] h-[400px] bg-secondary-100/40 rounded-full blur-[100px] opacity-30" />
      </div>

      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="relative z-10"
      >
        {/* Header */}
        <header className="mb-10 flex flex-col md:flex-row md:items-end justify-between gap-4">
          <motion.div variants={item}>
            <p className="text-sm uppercase tracking-widest text-slate-400 font-medium mb-2">
              {new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' })}
            </p>
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight">
              <span className="text-slate-900">Dr.</span>{' '}
              <span className="text-gradient-primary">{name}</span>
            </h1>
            <p className="text-slate-400 mt-3 text-lg">Your daily practice overview</p>
          </motion.div>
          <motion.div variants={item} className="flex gap-3">
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="btn-primary flex items-center gap-2"
            >
              <FaUserPlus /> New Patient
            </motion.button>
          </motion.div>
        </header>

        {/* Doctor Bento Grid */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 auto-rows-[160px] md:auto-rows-[180px]">

          {/* 1. Statistics Card (Wide) */}
          <motion.div variants={item} className="md:col-span-2">
            <Card className="h-full p-8 flex items-center justify-between group" hover>
              <div>
                <p className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-2">Today's Appointments</p>
                <div className="flex items-baseline gap-3">
                  <h2 className="text-5xl md:text-6xl font-black text-slate-900 group-hover:text-primary-600 transition-colors tracking-tighter">
                    {loading ? '-' : todaysAppts.length}
                  </h2>
                </div>
                <p className="text-primary-500 font-semibold text-sm mt-3 flex items-center gap-1.5">
                  <span className="w-5 h-5 rounded-full bg-primary-100 flex items-center justify-center shrink-0">
                    <FaChartLine size={10} />
                  </span>
                  Schedule for today
                </p>
              </div>
              <div className="h-32 w-px bg-slate-100 mx-6" />
              <div>
                <p className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-2">Pending Reviews</p>
                <div className="flex items-baseline gap-3">
                  <h2 className="text-5xl md:text-6xl font-black text-slate-900 tracking-tighter">
                    {loading ? '-' : pendingReviews}
                  </h2>
                </div>
                <Link to="/lesions" className="text-slate-500 hover:text-primary-500 font-semibold text-sm mt-3 flex items-center gap-1.5 transition-colors group/link">
                  <span className="w-5 h-5 rounded-full bg-slate-100 group-hover/link:bg-primary-100 flex items-center justify-center shrink-0 transition-colors">
                    <LuArrowRight size={12} />
                  </span>
                  Review Scans
                </Link>
              </div>
            </Card>
          </motion.div>

          {/* 2. Patient Queue (Tall) - Takes up 2 columns vertically */}
          <motion.div variants={item} className="md:col-span-1 row-span-2 md:row-span-2">
            <PatientQueue appointments={todaysAppts} patients={patients} loading={loading} />
          </motion.div>

          {/* 3. Quick Prescriptions (Tall) - Takes up 2 columns vertically */}
          <motion.div variants={item} className="md:col-span-1 row-span-2 md:row-span-2">
            <QuickPrescription />
          </motion.div>

          {/* 4. Treatment Plans Card (Wide on bottom left) */}
          <motion.div variants={item} className="md:col-span-2 row-span-1">
            <Link to="/doctor/treatment-plans" className="block h-full">
              <Card className="h-full p-8 flex items-center justify-between group overflow-hidden relative" hover>
                <div className="absolute -right-8 -top-8 w-40 h-40 bg-gradient-to-br from-primary-50 to-transparent rounded-full opacity-50 blur-2xl group-hover:opacity-100 transition-opacity" />
                <div className="relative z-10 w-full">
                  <div className="flex justify-between items-start mb-4">
                    <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">Treatment Plans</p>
                    <div className="w-12 h-12 rounded-2xl bg-slate-50 flex items-center justify-center text-slate-400 group-hover:bg-primary-500 group-hover:text-white transition-all shadow-sm group-hover:shadow-primary-500/30">
                      <FaClipboardList size={22} />
                    </div>
                  </div>
                  <h2 className="text-2xl font-bold text-slate-900 mb-1 group-hover:text-primary-600 transition-colors">Prescribe & Monitor</h2>
                  <p className="text-slate-500 text-sm leading-relaxed">Create and manage multi-step treatment regimens.</p>
                </div>
              </Card>
            </Link>
          </motion.div>

        </div>
      </motion.div>
    </div>
  )
}
