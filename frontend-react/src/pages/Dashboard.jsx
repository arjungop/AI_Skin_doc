import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useGeolocation } from '../hooks/useGeolocation'
import { Suspense, lazy, useState, useEffect, Component } from 'react'
import { api } from '../services/api'
import UVIndexWidget from '../components/dashboard/UVIndexWidget'
import WeatherWidget from '../components/dashboard/WeatherWidget'
import { Card, CardTitle, CardDescription, CardData, IconWrapper, CardBadge } from '../components/Card'
import {
  LuMapPin, LuScan, LuSparkles, LuActivity, LuArrowRight,
  LuMessageCircle, LuCalendar, LuHeart, LuTrendingUp, LuZap, LuMail,
  LuDroplets, LuStethoscope
} from 'react-icons/lu'

// Lazy load 3D component for performance
const FaceMap3D = lazy(() => import('../components/FaceMap3D'))

// Silent error boundary for optional 3D background — crashes shouldn't break the dashboard
class Safe3DBoundary extends Component {
  state = { hasError: false }
  static getDerivedStateFromError() { return { hasError: true } }
  render() { return this.state.hasError ? null : this.props.children }
}

export default function Dashboard() {
  const name = (localStorage.getItem('username') || 'Member').split(' ')[0]
  const { location, loading, error, requestLocation } = useGeolocation()

  const [profile, setProfile] = useState(null)
  const [upcomingAppts, setUpcomingAppts] = useState(null)
  const [dataLoading, setDataLoading] = useState(true)

  useEffect(() => {
    let mounted = true
    async function fetchDashboardData() {
      try {
        const [profileRes, apptsRes] = await Promise.allSettled([
          api.getProfile(),
          api.listAppointments(),
        ])
        if (!mounted) return
        if (profileRes.status === 'fulfilled') setProfile(profileRes.value)
        if (apptsRes.status === 'fulfilled') {
          const now = new Date()
          const upcoming = (apptsRes.value || []).filter(a => new Date(a.appointment_date) > now)
          setUpcomingAppts(upcoming)
        }
      } catch {}
      finally { if (mounted) setDataLoading(false) }
    }
    fetchDashboardData()
    return () => { mounted = false }
  }, [])

  const container = {
    hidden: { opacity: 0 },
    show: { opacity: 1, transition: { staggerChildren: 0.08, delayChildren: 0.1 } }
  }

  const item = {
    hidden: { opacity: 0, y: 30 },
    show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: [0.4, 0, 0.2, 1] } }
  }

  const hour = new Date().getHours()
  const greeting = hour < 12 ? 'Good morning' : hour < 18 ? 'Good afternoon' : 'Good evening'

  return (
    <div className="relative min-h-screen pb-20">
      {/* Ambient Background Glow */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute top-0 left-1/4 w-[600px] h-[600px] bg-primary-500/10 rounded-full blur-[150px] opacity-50" />
        <div className="absolute bottom-1/4 right-1/4 w-[500px] h-[500px] bg-accent-500/10 rounded-full blur-[120px] opacity-40" />
      </div>

      {/* 3D Background Layer */}
      <div className="fixed inset-0 pointer-events-none opacity-10 z-0">
        <Safe3DBoundary>
          <Suspense fallback={null}>
            <FaceMap3D />
          </Suspense>
        </Safe3DBoundary>
      </div>

      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="relative z-10"
      >
        {/* Hero Header */}
        <header className="mb-12 pt-4">
          <motion.div variants={item} className="mb-2">
            <span className="text-sm uppercase tracking-widest text-text-tertiary font-medium">
              {new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' })}
            </span>
          </motion.div>

          <motion.h1
            variants={item}
            className="text-5xl md:text-6xl lg:text-7xl font-bold tracking-tight"
          >
            <span className="text-text-primary">{greeting},</span>
            <br />
            <span className="text-gradient-primary">{name}</span>
          </motion.h1>

          {/* Location Sync */}
          <motion.div variants={item} className="flex items-center gap-4 mt-8">
            {!location ? (
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={requestLocation}
                disabled={loading}
                className="btn-ghost btn-sm flex items-center gap-2"
              >
                <LuMapPin size={16} />
                {loading ? 'Syncing...' : 'Enable Environmental Sync'}
              </motion.button>
            ) : (
              <div className="flex items-center gap-2 px-4 py-2 bg-primary-500/10 text-primary-500 rounded-full text-sm font-medium border border-primary-500/20">
                <div className="w-2 h-2 rounded-full bg-primary-500 animate-pulse" />
                <span className="font-mono text-xs uppercase tracking-wider">ENV SYNCED</span>
              </div>
            )}
            {error && <span className="text-xs text-danger font-mono">{error}</span>}
          </motion.div>
        </header>

        {/* Bento Grid */}
        <div className="bento-grid">

          {/* HERO CARD - Skin Health Score */}
          <motion.div variants={item} className="bento-large">
            <Card variant="glow-primary" className="h-full flex flex-col justify-between p-8 group" padding="none">
              {/* Ambient glow effect */}
              <div className="absolute -top-20 -right-20 w-60 h-60 bg-primary-500/20 rounded-full blur-[80px] group-hover:bg-primary-500/30 transition-colors duration-700" />

              <div className="relative z-10">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-12 h-12 rounded-2xl bg-primary-500/10 flex items-center justify-center">
                    <LuActivity className="text-primary-500" size={24} />
                  </div>
                  <div>
                    <span className="text-xs uppercase tracking-widest text-text-tertiary font-medium">
                      Skin Profile
                    </span>
                  </div>
                </div>

                {/* Health Overview */}
                <div className="mb-6">
                  {profile?.skin_type ? (
                    <>
                      <span className="text-5xl md:text-6xl font-bold text-gradient-primary tracking-tighter capitalize">
                        {profile.skin_type}
                      </span>
                      <p className="text-text-secondary text-sm mt-2">Your skin type profile</p>
                    </>
                  ) : (
                    <>
                      <span className="text-4xl md:text-5xl font-bold text-gradient-primary tracking-tighter">
                        {dataLoading ? '...' : 'Set Up Profile'}
                      </span>
                      <p className="text-text-secondary text-sm mt-2">{dataLoading ? 'Loading...' : 'Complete onboarding for personalized insights'}</p>
                    </>
                  )}
                </div>

                <p className="text-text-secondary text-sm max-w-md leading-relaxed">
                  {profile?.skin_type
                    ? 'Track your skin health progress and get personalized recommendations based on your profile.'
                    : 'Complete your skin profile to unlock AI-powered insights and personalized care recommendations.'}
                </p>
              </div>

              <div className="relative z-10 grid grid-cols-3 gap-4 mt-8 pt-6 border-t border-white/5">
                <CardData label="Skin Type" size="sm">{profile?.skin_type || '—'}</CardData>
                <CardData label="Sensitivity" size="sm">{profile?.sensitivity_level || '—'}</CardData>
                <CardData label="Location" size="sm">{profile?.location_city || '—'}</CardData>
              </div>
            </Card>
          </motion.div>

          {/* AI SCAN - Primary CTA */}
          <motion.div variants={item} className="bento-tall">
            <Link to="/lesions" className="block h-full">
              <Card variant="glass" className="h-full flex flex-col justify-between group" hover>
                <div>
                  <IconWrapper variant="ai" size="lg" className="mb-6">
                    <LuScan size={28} />
                  </IconWrapper>
                  <CardTitle className="mb-3 text-xl">AI Lesion Scan</CardTitle>
                  <CardDescription>
                    Clinical-grade analysis powered by deep learning. Get instant insights on any skin concern.
                  </CardDescription>
                </div>

                <motion.div
                  whileHover={{ scale: 1.02 }}
                  className="btn-ai w-full flex items-center justify-center gap-2 mt-6"
                >
                  <LuZap size={18} />
                  Start Scan
                  <LuArrowRight size={18} className="group-hover:translate-x-1 transition-transform" />
                </motion.div>
              </Card>
            </Link>
          </motion.div>

          {/* SKIN COACH */}
          <motion.div variants={item}>
            <Link to="/coach" className="block h-full">
              <Card variant="glass" className="h-full flex flex-col justify-between group" hover>
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <IconWrapper variant="primary">
                      <LuSparkles size={22} />
                    </IconWrapper>
                    <CardBadge variant="accent">NEW</CardBadge>
                  </div>
                  <CardTitle className="mb-2">Skin Coach</CardTitle>
                  <CardDescription>
                    Personalized AI recommendations for your unique skin profile.
                  </CardDescription>
                </div>
                <div className="flex items-center gap-2 text-primary-500 text-sm font-medium mt-4 group-hover:gap-3 transition-all">
                  Get Insights <LuArrowRight size={16} />
                </div>
              </Card>
            </Link>
          </motion.div>

          {/* ENVIRONMENTAL WIDGETS */}
          {location && (
            <>
              <motion.div variants={item}>
                <UVIndexWidget location={location} />
              </motion.div>
              <motion.div variants={item}>
                <WeatherWidget location={location} />
              </motion.div>
            </>
          )}

          {/* APPOINTMENTS */}
          <motion.div variants={item}>
            <Card variant="elevated" className="h-full flex flex-col justify-between" hover>
              <div>
                <div className="flex items-center gap-3 mb-4">
                  <IconWrapper variant="accent" size="sm">
                    <LuCalendar size={18} />
                  </IconWrapper>
                  <CardTitle>Appointments</CardTitle>
                </div>
                <div className="font-mono text-3xl font-bold text-text-primary mb-2">{upcomingAppts ? upcomingAppts.length : '—'}</div>
                <CardDescription>{upcomingAppts?.length ? `${upcomingAppts.length} upcoming` : 'No upcoming appointments'}</CardDescription>
              </div>
              <Link to="/appointments">
                <motion.div whileHover={{ scale: 1.02 }} className="btn-ghost w-full mt-4">
                  Book Consultation
                </motion.div>
              </Link>
            </Card>
          </motion.div>

          {/* MESSAGES */}
          <motion.div variants={item}>
            <Link to="/messages" className="block h-full">
              <Card variant="glass" className="h-full flex flex-col justify-between group" hover>
                <div>
                  <div className="flex items-center gap-3 mb-4">
                    <IconWrapper variant="primary" size="sm">
                      <LuMail size={18} />
                    </IconWrapper>
                    <CardTitle>Messages</CardTitle>
                  </div>
                  <CardDescription>
                    Chat with your dermatologist, ask questions, and get expert advice.
                  </CardDescription>
                </div>
                <div className="flex items-center gap-2 text-primary-500 text-sm font-medium mt-4 group-hover:gap-3 transition-all">
                  Open Messages <LuArrowRight size={16} />
                </div>
              </Card>
            </Link>
          </motion.div>

          {/* AI ASSISTANT - Wide Card */}
          <motion.div variants={item} className="bento-wide">
            <Link to="/chat" className="block">
              <Card variant="glass" className="flex items-center justify-between p-8 group" hover padding="none">
                <div className="flex items-center gap-6">
                  <IconWrapper variant="ai" size="lg">
                    <LuMessageCircle size={28} />
                  </IconWrapper>
                  <div>
                    <CardTitle className="mb-1 text-xl">AI Health Assistant</CardTitle>
                    <CardDescription className="max-w-sm">
                      Ask anything about skincare, ingredients, or routines.
                      <span className="text-ai-400 ml-2 font-mono text-xs">"Is vitamin C good for acne?"</span>
                    </CardDescription>
                  </div>
                </div>
                <div className="w-14 h-14 rounded-2xl bg-ai-500/10 text-ai-400 flex items-center justify-center group-hover:bg-ai-500/20 group-hover:translate-x-2 transition-all">
                  <LuArrowRight size={24} />
                </div>
              </Card>
            </Link>
          </motion.div>

          {/* SKIN JOURNEY */}
          <motion.div variants={item}>
            <Link to="/journey" className="block h-full">
              <Card variant="glass" className="h-full flex flex-col justify-between group" hover>
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <IconWrapper variant="primary">
                      <LuHeart size={22} />
                    </IconWrapper>
                    <CardBadge>NEW</CardBadge>
                  </div>
                  <CardTitle className="mb-2">Skin Journey</CardTitle>
                  <CardDescription>
                    Track your progress and see how far you've come.
                  </CardDescription>
                </div>
                <div className="flex items-center gap-2 text-primary-500 text-sm font-medium mt-4 group-hover:gap-3 transition-all">
                  View Timeline <LuArrowRight size={16} />
                </div>
              </Card>
            </Link>
          </motion.div>

          {/* SKINCARE ROUTINE */}
          <motion.div variants={item}>
            <Link to="/routine" className="block h-full">
              <Card variant="glass" className="h-full flex flex-col justify-between group" hover>
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <IconWrapper variant="accent">
                      <LuDroplets size={22} />
                    </IconWrapper>
                  </div>
                  <CardTitle className="mb-2">My Routine</CardTitle>
                  <CardDescription>
                    Your personalized AM & PM skincare steps tailored to your skin type.
                  </CardDescription>
                </div>
                <div className="flex items-center gap-2 text-primary-500 text-sm font-medium mt-4 group-hover:gap-3 transition-all">
                  View Routine <LuArrowRight size={16} />
                </div>
              </Card>
            </Link>
          </motion.div>

          {/* FIND DERMATOLOGISTS */}
          <motion.div variants={item}>
            <Link to="/find-doctors" className="block h-full">
              <Card variant="glass" className="h-full flex flex-col justify-between group" hover>
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <IconWrapper variant="primary">
                      <LuStethoscope size={22} />
                    </IconWrapper>
                  </div>
                  <CardTitle className="mb-2">Find Doctors</CardTitle>
                  <CardDescription>
                    Discover top-rated dermatologists near you and book in-person visits.
                  </CardDescription>
                </div>
                <div className="flex items-center gap-2 text-primary-500 text-sm font-medium mt-4 group-hover:gap-3 transition-all">
                  Browse Clinics <LuArrowRight size={16} />
                </div>
              </Card>
            </Link>
          </motion.div>

          {/* PROFILE SNAPSHOT */}
          <motion.div variants={item} className="bento-wide">
            <Card variant="elevated" className="p-8" padding="none" hover={false}>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                <div>
                  <div className="text-xs uppercase tracking-widest text-text-tertiary font-medium mb-3">
                    Skin Type
                  </div>
                  <div className="flex gap-2 flex-wrap">
                    <span className="chip">{profile?.skin_type || 'Not set'}</span>
                  </div>
                </div>

                <div>
                  <div className="text-xs uppercase tracking-widest text-text-tertiary font-medium mb-3">
                    Sensitivity
                  </div>
                  <div className="font-mono text-2xl font-bold text-text-primary">{profile?.sensitivity_level || '—'}</div>
                </div>

                <div>
                  <div className="text-xs uppercase tracking-widest text-text-tertiary font-medium mb-3">
                    Primary Goals
                  </div>
                  <div className="text-text-primary font-medium">{(() => { try { const g = JSON.parse(profile?.goals || '[]'); return g.length ? g.join(', ') : 'Not set' } catch { return profile?.goals || 'Not set' } })()}</div>
                </div>
              </div>
            </Card>
          </motion.div>

        </div>
      </motion.div>
    </div>
  )
}
