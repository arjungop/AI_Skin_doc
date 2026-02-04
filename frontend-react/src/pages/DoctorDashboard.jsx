import { Link } from 'react-router-dom'
import { FaUserPlus, FaCalendarCheck, FaClipboardList, FaSearch, FaChartLine } from 'react-icons/fa'
import { motion } from 'framer-motion'
import QuickPrescription from '../components/dashboard/QuickPrescription'
import PatientQueue from '../components/dashboard/PatientQueueList'

export default function DoctorDashboard() {
  const name = (localStorage.getItem('username') || 'Doctor').split(' ')[0]

  const container = {
    hidden: { opacity: 0 },
    show: { opacity: 1, transition: { staggerChildren: 0.1 } }
  }

  const item = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 }
  }

  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
      className="max-w-[1600px] mx-auto"
    >
      <header className="mb-10 flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h1 className="text-4xl md:text-5xl font-serif text-slate-900 tracking-tight">
            Dr. <span className="text-slate-800">{name}</span>
          </h1>
          <p className="text-slate-500 mt-2 text-lg font-light">Your daily practice overview.</p>
        </div>
        <div className="flex gap-3">
          <button className="btn-primary gap-2 shadow-lg">
            <FaUserPlus /> New Patient
          </button>
        </div>
      </header>

      {/* Doctor Bento Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 lg:grid-cols-4 gap-6 auto-rows-[180px]">

        {/* 1. Statistics (Wide) */}
        <motion.div variants={item} className="md:col-span-2 bg-white rounded-3xl p-8 border border-slate-100 shadow-sm flex items-center justify-between">
          <div>
            <p className="text-sm font-bold text-slate-400 uppercase tracking-widest mb-1">Today's Appointments</p>
            <h2 className="text-5xl font-serif text-slate-900">12</h2>
            <p className="text-emerald-600 font-bold text-sm mt-2 flex items-center gap-1">
              <FaChartLine /> +2 from yesterday
            </p>
          </div>
          <div className="h-full w-px bg-slate-100 mx-6"></div>
          <div>
            <p className="text-sm font-bold text-slate-400 uppercase tracking-widest mb-1">Pending Reviews</p>
            <h2 className="text-5xl font-serif text-slate-900 text-orange-400">5</h2>
            <Link to="/lesions" className="text-slate-500 font-bold text-sm mt-2 hover:text-slate-800 block">
              Review Scans &rarr;
            </Link>
          </div>
        </motion.div>

        {/* 2. Patient Queue (Tall) */}
        <motion.div variants={item} className="md:col-span-1 md:row-span-2">
          <PatientQueue />
        </motion.div>

        {/* 3. Quick Prescriptions (Tall) */}
        <motion.div variants={item} className="md:col-span-1 md:row-span-2">
          <QuickPrescription />
        </motion.div>

        {/* 4. Action Card */}
        <motion.div variants={item} className="bg-slate-900 rounded-3xl p-6 flex flex-col justify-center text-white cursor-pointer hover:bg-slate-800 transition-colors shadow-xl shadow-slate-900/10">
          <FaClipboardList className="text-gold-400 mb-4" size={32} />
          <h3 className="text-xl font-bold">Clinical Notes</h3>
          <p className="text-slate-400 text-sm mt-1">Generate documentation with AI.</p>
        </motion.div>

        {/* 5. Search Patient */}
        <motion.div variants={item} className="bg-blue-50 rounded-3xl p-6 border border-blue-100 flex flex-col justify-center cursor-pointer hover:bg-blue-100 transition-colors">
          <FaSearch className="text-blue-500 mb-4" size={28} />
          <h3 className="text-xl font-bold text-slate-900">Find Patient</h3>
          <p className="text-slate-500 text-sm mt-1">Search by name or ID.</p>
        </motion.div>

      </div>
    </motion.div>
  )
}
