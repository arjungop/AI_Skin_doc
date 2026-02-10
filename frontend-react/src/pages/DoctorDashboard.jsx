import { Link } from 'react-router-dom'
import { FaUserPlus, FaClipboardList, FaChartLine } from 'react-icons/fa'
import { LuSearch, LuArrowRight, LuSparkles } from 'react-icons/lu'
import { motion } from 'framer-motion'
import QuickPrescription from '../components/dashboard/QuickPrescription'
import PatientQueue from '../components/dashboard/PatientQueueList'
import { Card, CardTitle, CardDescription, IconWrapper } from '../components/Card'

export default function DoctorDashboard() {
  const name = (localStorage.getItem('username') || 'Doctor').split(' ')[0]

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
        className="relative z-10 max-w-[1600px] mx-auto"
      >
        {/* Header */}
        <header className="mb-10 flex flex-col md:flex-row md:items-end justify-between gap-4 lg:pr-72">
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
        <div className="grid grid-cols-1 md:grid-cols-4 gap-5 auto-rows-[180px]">

          {/* 1. Statistics Card (Wide) */}
          <motion.div variants={item} className="md:col-span-2">
            <Card className="h-full p-8 flex items-center justify-between" hover>
              <div>
                <p className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-2">Today's Appointments</p>
                <h2 className="text-5xl font-bold text-gradient-primary">12</h2>
                <p className="text-primary-400 font-semibold text-sm mt-3 flex items-center gap-1.5">
                  <FaChartLine size={14} /> +2 from yesterday
                </p>
              </div>
              <div className="h-20 w-px bg-slate-200 mx-6" />
              <div>
                <p className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-2">Pending Reviews</p>
                <h2 className="text-5xl font-bold text-secondary-500">5</h2>
                <Link to="/lesions" className="text-slate-500 hover:text-primary-500 font-semibold text-sm mt-3 flex items-center gap-1 transition-colors">
                  Review Scans <LuArrowRight size={14} />
                </Link>
              </div>
            </Card>
          </motion.div>

          {/* 2. Patient Queue (Tall) */}
          <motion.div variants={item} className="md:col-span-1 md:row-span-2">
            <PatientQueue />
          </motion.div>

          {/* 3. Quick Prescriptions (Tall) */}
          <motion.div variants={item} className="md:col-span-1 md:row-span-2">
            <QuickPrescription />
          </motion.div>

          {/* 4. Clinical Notes Card */}
          <motion.div variants={item}>
            <Card className="h-full p-6 cursor-pointer group" hover>
              <IconWrapper variant="ai" className="mb-4">
                <LuSparkles size={22} />
              </IconWrapper>
              <CardTitle>Clinical Notes</CardTitle>
              <CardDescription>Generate documentation with AI</CardDescription>
            </Card>
          </motion.div>

          {/* 5. Search Patient Card */}
          <motion.div variants={item}>
            <Card className="h-full p-6 cursor-pointer group" hover>
              <IconWrapper variant="accent" className="mb-4">
                <LuSearch size={22} />
              </IconWrapper>
              <CardTitle>Find Patient</CardTitle>
              <CardDescription>Search by name or ID</CardDescription>
            </Card>
          </motion.div>

        </div>
      </motion.div>
    </div>
  )
}
