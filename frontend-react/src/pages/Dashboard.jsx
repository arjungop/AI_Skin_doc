import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useGeolocation } from '../hooks/useGeolocation'
import UVIndexWidget from '../components/dashboard/UVIndexWidget'
import WeatherWidget from '../components/dashboard/WeatherWidget'
import { Card, CardTitle, CardDescription, CardData, IconWrapper, CardBadge } from '../components/Card'
import { api } from '../services/api'
import {
  LuMapPin, LuScan, LuSparkles, LuActivity, LuArrowRight,
  LuMessageCircle, LuCalendar, LuHeart, LuTrendingUp, LuZap, LuClipboardCheck,
  LuCamera, LuBookOpen
} from 'react-icons/lu'

export default function Dashboard() {
  const name = (localStorage.getItem('username') || 'Member').split(' ')[0]
  const { location, loading, error, requestLocation } = useGeolocation()

  const [scanCount, setScanCount] = useState(0)
  const [nextAppointment, setNextAppointment] = useState(null)
  const [routineProgress, setRoutineProgress] = useState({ completed: 0, total: 0 })
  const [dataLoading, setDataLoading] = useState(true)

  useEffect(() => {
    fetchDashboardData()
  }, [])

  const fetchDashboardData = async () => {
    setDataLoading(true)
    try {
      // Fetch real data from APIs
      const [appointments, routine, completions] = await Promise.all([
        api.getAppointments().catch(() => []),
        api.getRoutine().catch(() => []),
        api.getCompletions(new Date().toISOString().split('T')[0]).catch(() => [])
      ])

      // Find next upcoming appointment
      const upcoming = appointments
        .filter(a => new Date(a.schedule) > new Date() && a.status !== 'Cancelled')
        .sort((a, b) => new Date(a.schedule) - new Date(b.schedule))[0]
      setNextAppointment(upcoming)

      // Calculate routine progress for today
      const todayItems = Array.isArray(routine) ? routine : []
      const completedIds = Array.isArray(completions) ? completions.map(c => c.routine_item_id) : []
      setRoutineProgress({
        completed: todayItems.filter(i => completedIds.includes(i.item_id)).length,
        total: todayItems.length
      })

      setScanCount(parseInt(localStorage.getItem('scan_count') || '0'))
    } catch (err) {
      console.error('Dashboard data fetch error:', err)
    } finally {
      setDataLoading(false)
    }
  }

  const container = {
    hidden: { opacity: 0 },
    show: { opacity: 1, transition: { staggerChildren: 0.05, delayChildren: 0.1 } }
  }

  const item = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: "easeOut" } }
  }

  const hour = new Date().getHours()
  const greeting = hour < 12 ? 'Good morning' : hour < 18 ? 'Good afternoon' : 'Good evening'

  // Calculate a simple "wellness" indicator based on real actions
  const hasCompletedOnboarding = !!localStorage.getItem('skin_type')
  const hasScannedBefore = scanCount > 0
  const hasRoutine = routineProgress.total > 0

  const getStartedSteps = [
    { done: hasCompletedOnboarding, label: 'Complete profile', link: '/onboarding' },
    { done: hasRoutine, label: 'Set up routine', link: '/routine' },
    { done: hasScannedBefore, label: 'First AI scan', link: '/lesions' },
  ]
  const completedSteps = getStartedSteps.filter(s => s.done).length
  const isFullySetUp = completedSteps === getStartedSteps.length

  return (
    <div className="min-h-screen pb-20">
      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="relative z-10"
      >
        {/* Clean Header */}
        <header className="mb-8 flex flex-col md:flex-row md:items-end justify-between gap-4">
          <div>
            <motion.h1 variants={item} className="text-3xl md:text-4xl font-bold text-slate-900 tracking-tight">
              {greeting}, {name}
            </motion.h1>
            <motion.p variants={item} className="text-slate-500 mt-1">
              {isFullySetUp ? 'Here is your daily skin health overview.' : 'Let\'s get you started with personalized skin care.'}
            </motion.p>
          </div>

          <motion.div variants={item}>
            {!location ? (
              <button
                onClick={requestLocation}
                disabled={loading}
                className="flex items-center gap-2 text-sm font-medium text-primary-600 bg-primary-50 px-4 py-2 rounded-lg hover:bg-primary-100 transition-colors"
              >
                <LuMapPin size={16} />
                {loading ? 'Locating...' : 'Enable Local Insights'}
              </button>
            ) : (
              <div className="flex items-center gap-2 text-sm font-medium text-emerald-700 bg-emerald-50 px-4 py-2 rounded-lg">
                <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                Synced with environment
              </div>
            )}
            {error && <span className="text-xs text-rose-500 block mt-1 text-right">{error}</span>}
          </motion.div>
        </header>

        {/* Bento Grid layout */}
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-6">

          {/* MAIN STATUS CARD - Now shows real progress */}
          <motion.div variants={item} className="md:col-span-2 lg:col-span-2 lg:row-span-2">
            <Card className="h-full flex flex-col justify-between relative overflow-hidden group">
              <div className="absolute top-0 right-0 w-64 h-64 bg-primary-50 rounded-full blur-3xl opacity-50 -mr-16 -mt-16 transition-transform group-hover:scale-110 duration-700" />

              <div className="relative z-10">
                <div className="flex justify-between items-start mb-6">
                  <div className="flex items-center gap-3">
                    <IconWrapper variant="primary">
                      <LuActivity size={24} />
                    </IconWrapper>
                    <span className="text-sm font-bold text-slate-400 uppercase tracking-wider">
                      {isFullySetUp ? 'Your Progress' : 'Get Started'}
                    </span>
                  </div>
                  {isFullySetUp && (
                    <CardBadge variant="success" className="flex gap-1">
                      <LuTrendingUp size={12} /> Active
                    </CardBadge>
                  )}
                </div>

                {!isFullySetUp ? (
                  // Show onboarding steps if not fully set up
                  <div className="space-y-4">
                    <div className="mb-6">
                      <span className="text-5xl font-bold text-slate-900 tracking-tighter">{completedSteps}</span>
                      <span className="text-2xl font-bold text-slate-300">/{getStartedSteps.length}</span>
                      <p className="text-slate-500 mt-2">steps completed</p>
                    </div>
                    <div className="space-y-3">
                      {getStartedSteps.map((step, i) => (
                        <Link
                          key={i}
                          to={step.link}
                          className={`flex items-center gap-3 p-3 rounded-xl transition-all ${step.done
                              ? 'bg-emerald-50 text-emerald-700'
                              : 'bg-slate-50 text-slate-600 hover:bg-primary-50 hover:text-primary-700'
                            }`}
                        >
                          <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${step.done ? 'bg-emerald-500 text-white' : 'bg-slate-200 text-slate-500'
                            }`}>
                            {step.done ? '✓' : i + 1}
                          </div>
                          <span className={`font-medium ${step.done ? 'line-through opacity-60' : ''}`}>
                            {step.label}
                          </span>
                          {!step.done && <LuArrowRight className="ml-auto" size={16} />}
                        </Link>
                      ))}
                    </div>
                  </div>
                ) : (
                  // Show actual stats when fully set up
                  <>
                    <div className="mb-4">
                      <p className="text-slate-600 leading-relaxed max-w-sm">
                        You're on track! Keep up with your routine for the best results.
                      </p>
                    </div>
                    <div className="grid grid-cols-3 gap-4 mt-8 pt-6 border-t border-slate-100">
                      <CardData label="Scans" size="lg">{scanCount}</CardData>
                      <CardData label="Today's Routine" size="lg">
                        {routineProgress.total > 0
                          ? `${routineProgress.completed}/${routineProgress.total}`
                          : '—'}
                      </CardData>
                      <CardData label="Streak" size="lg">
                        {localStorage.getItem('streak_days') || '—'}
                      </CardData>
                    </div>
                  </>
                )}
              </div>
            </Card>
          </motion.div>

          {/* AI SCAN - Action */}
          <motion.div variants={item} className="md:col-span-1 lg:col-span-1 lg:row-span-2">
            <Link to="/lesions" className="block h-full">
              <Card className="h-full flex flex-col justify-between group hover:border-primary-200 hover:ring-2 hover:ring-primary-50 transition-all">
                <div>
                  <IconWrapper variant="primary" className="mb-6 bg-primary-100 text-primary-600">
                    <LuScan size={24} />
                  </IconWrapper>
                  <CardTitle className="mb-2">New Scan</CardTitle>
                  <CardDescription>
                    Analyze a spot or mole instantly with clinical-grade AI.
                  </CardDescription>
                </div>
                <div className="mt-8 flex items-center justify-between text-primary-600 font-semibold group-hover:translate-x-1 transition-transform">
                  Start Analysis <LuArrowRight />
                </div>
              </Card>
            </Link>
          </motion.div>

          {/* AI ASSISTANT */}
          <motion.div variants={item} className="md:col-span-1 lg:col-span-1 lg:row-span-2">
            <Link to="/chat" className="block h-full">
              <Card className="h-full flex flex-col justify-between group hover:border-violet-200 hover:ring-2 hover:ring-violet-50 transition-all">
                <div>
                  <IconWrapper className="mb-6 bg-violet-100 text-violet-600">
                    <LuMessageCircle size={24} />
                  </IconWrapper>
                  <CardTitle className="mb-2">AI Assistant</CardTitle>
                  <CardDescription>
                    Ask about ingredients, symptoms, or routines.
                  </CardDescription>
                </div>
                <div className="mt-8 flex items-center justify-between text-violet-600 font-semibold group-hover:translate-x-1 transition-transform">
                  Chat Now <LuArrowRight />
                </div>
              </Card>
            </Link>
          </motion.div>

          {/* ENVIRONMENT - if available */}
          {location && (
            <>
              <motion.div variants={item} className="md:col-span-1">
                <UVIndexWidget location={location} />
              </motion.div>
              <motion.div variants={item} className="md:col-span-1">
                <WeatherWidget location={location} />
              </motion.div>
            </>
          )}

          {/* APPOINTMENTS - Real data */}
          <motion.div variants={item} className="md:col-span-2">
            <Card className="flex items-center justify-between p-6">
              <div className="flex items-center gap-4">
                <IconWrapper variant="default" size="sm">
                  <LuCalendar size={20} />
                </IconWrapper>
                <div>
                  <CardTitle className="text-base">Next Appointment</CardTitle>
                  <CardDescription>
                    {dataLoading ? 'Loading...' :
                      nextAppointment
                        ? `${new Date(nextAppointment.schedule).toLocaleDateString()} at ${new Date(nextAppointment.schedule).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`
                        : 'No upcoming visits scheduled.'
                    }
                  </CardDescription>
                </div>
              </div>
              <Link to="/appointments" className="btn btn-secondary text-sm py-2 px-4 whitespace-nowrap">
                {nextAppointment ? 'View' : 'Book Visit'}
              </Link>
            </Card>
          </motion.div>

          {/* SKIN JOURNEY - More useful than "coming soon" */}
          <motion.div variants={item} className="md:col-span-1">
            <Link to="/journey" className="block h-full">
              <Card className="flex flex-col justify-center items-center text-center p-6 bg-gradient-to-br from-emerald-500 to-teal-600 text-white border-transparent hover:scale-[1.02] transition-transform">
                <LuCamera size={28} className="mb-3 opacity-90" />
                <div className="font-bold text-lg mb-1">Track Progress</div>
                <div className="text-white/80 text-sm">Skin Journey →</div>
              </Card>
            </Link>
          </motion.div>

        </div>
      </motion.div>
    </div>
  )
}
