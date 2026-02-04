import { Link } from 'react-router-dom'
import { FaStethoscope, FaRobot, FaCalendarAlt, FaArrowRight } from 'react-icons/fa'
import { motion } from 'framer-motion'
import { useGeolocation } from '../hooks/useGeolocation'
import { Suspense, lazy } from 'react'
import UVIndexWidget from '../components/dashboard/UVIndexWidget'
import WeatherWidget from '../components/dashboard/WeatherWidget'
import { Card, CardHeader, CardTitle, CardData, CardDescription } from '../components/Card'
import { LuMapPin, LuScan, LuSparkles, LuActivity } from 'react-icons/lu'

// Lazy load 3D component for performance
const FaceMap3D = lazy(() => import('../components/FaceMap3D'))

export default function Dashboard() {
  const name = (localStorage.getItem('username') || 'Member').split(' ')[0]
  const { location, loading, error, requestLocation } = useGeolocation()

  const container = {
    hidden: { opacity: 0 },
    show: { opacity: 1, transition: { staggerChildren: 0.08 } }
  }

  const item = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 }
  }

  return (
    <div className="relative min-h-screen">
      {/* 3D BACKGROUND LAYER - Cockpit Style */}
      <div className="fixed inset-0 pointer-events-none opacity-[0.15] z-0">
        <Suspense fallback={<div className="w-full h-full bg-gradient-to-br from-accent-medical/5 to-accent-ai/5" />}>
          <FaceMap3D />
        </Suspense>
      </div>

      {/* MAIN CONTENT - Floating above 3D */}
      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="relative z-10"
      >
        {/* EDITORIAL HERO */}
        <header className="mb-12">
          <motion.div variants={item} className="mb-6">
            <div className="text-sm uppercase tracking-wider text-text-secondary font-medium mb-3">
              {new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' })}
            </div>
            <h1 className="text-6xl md:text-7xl lg:text-8xl font-sans font-semibold text-text-primary tracking-tighter leading-[0.9]">
              Good morning,<br />
              <span className="bg-gradient-to-r from-accent-medical to-accent-ai bg-clip-text text-transparent">
                {name}
              </span>
            </h1>
          </motion.div>

          {/* Location Control */}
          <motion.div variants={item} className="flex items-center gap-4 mt-6">
            {!location ? (
              <button
                onClick={requestLocation}
                disabled={loading}
                className="btn-primary btn-sm flex items-center gap-2"
              >
                <LuMapPin size={16} />
                {loading ? 'Detecting Environment...' : 'Enable Environmental Sync'}
              </button>
            ) : (
              <div className="flex items-center gap-2 px-4 py-2 bg-medical-50 text-accent-medical rounded-full text-xs font-medium border border-medical-100">
                <div className="w-2 h-2 rounded-full bg-accent-medical animate-pulse" />
                <span className="font-mono">ENV SYNCED</span>
              </div>
            )}
            {error && <span className="text-xs text-red-500 font-mono">{error}</span>}
          </motion.div>
        </header>

        {/* ASYMMETRIC BENTO GRID - Glass Tiles over 3D */}
        <div className="bento-grid">
          
          {/* HERO METRIC - Skin Health Score */}
          <motion.div variants={item} className="bento-large">
            <Card variant="glass" className="h-full flex flex-col justify-between p-8 group relative overflow-hidden">
              {/* Glow Effect */}
              <div className="absolute inset-0 bg-gradient-to-br from-accent-medical/10 to-accent-ai/10 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
              
              <div className="relative z-10">
                <div className="flex items-center gap-2 mb-4">
                  <LuActivity className="text-accent-medical" size={24} />
                  <span className="text-xs uppercase tracking-wider text-text-secondary font-medium">
                    Skin Health Index
                  </span>
                </div>
                
                <div className="font-mono text-[120px] leading-none font-bold text-text-primary tracking-tighter mb-4">
                  98<span className="text-5xl text-accent-medical">.2</span>
                </div>
                
                <p className="text-text-secondary text-sm max-w-md leading-relaxed">
                  Your skin health is in excellent condition. Environmental factors are optimal. 
                  Continue current routine for maintained results.
                </p>
              </div>
              
              <div className="relative z-10 grid grid-cols-3 gap-4 mt-8">
                <CardData label="Hydration">87%</CardData>
                <CardData label="UV Exposure">Low</CardData>
                <CardData label="Progress">+12%</CardData>
              </div>
            </Card>
          </motion.div>

          {/* AI SCAN - Primary Action */}
          <motion.div variants={item} className="bento-tall">
            <Link to="/lesions" className="block h-full">
              <Card variant="ceramic" className="h-full flex flex-col justify-between group hover:border-accent-ai/30">
                <div>
                  <div className="h-14 w-14 rounded-full bg-ai-50 flex items-center justify-center text-accent-ai mb-6 group-hover:scale-110 transition-transform">
                    <LuScan size={28} />
                  </div>
                  <CardTitle className="mb-2">AI Scan</CardTitle>
                  <CardDescription>
                    Launch bio-digital imaging analysis. Real-time lesion detection powered by clinical AI.
                  </CardDescription>
                </div>
                
                <div className="btn-ai w-full flex items-center justify-center gap-2">
                  Start Scan <FaArrowRight size={14} />
                </div>
              </Card>
            </Link>
          </motion.div>

          {/* SKIN COACH */}
          <motion.div variants={item}>
            <Link to="/coach" className="block h-full">
              <Card variant="glass" className="h-full flex flex-col justify-between group">
                <div>
                  <div className="flex items-center gap-2 mb-4">
                    <LuSparkles className="text-accent-medical" size={20} />
                    <span className="text-xs uppercase tracking-wider text-accent-medical font-medium">NEW</span>
                  </div>
                  <CardTitle className="mb-2">Skin Coach</CardTitle>
                  <CardDescription>
                    Personalized insights and routine optimization.
                  </CardDescription>
                </div>
                <div className="font-mono text-xs text-text-secondary group-hover:text-accent-medical transition-colors">
                  OPEN →
                </div>
              </Card>
            </Link>
          </motion.div>

          {/* ENVIRONMENTAL DATA */}
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
            <Card variant="ceramic" className="h-full flex flex-col justify-between">
              <div>
                <div className="flex items-center gap-2 mb-4">
                  <FaCalendarAlt className="text-text-secondary" size={18} />
                  <CardTitle>Next Visit</CardTitle>
                </div>
                <div className="font-mono text-3xl font-bold text-text-primary mb-2">—</div>
                <CardDescription>No upcoming appointments scheduled.</CardDescription>
              </div>
              <Link to="/appointments" className="btn-ghost w-full mt-4">
                Book Consultation
              </Link>
            </Card>
          </motion.div>

          {/* AI ASSISTANT */}
          <motion.div variants={item} className="md:col-span-2">
            <Link to="/chat" className="block h-full">
              <Card variant="glass" className="h-full flex items-center justify-between group p-8">
                <div className="flex items-center gap-6">
                  <div className="h-16 w-16 rounded-full bg-ai-50 flex items-center justify-center text-accent-ai group-hover:scale-110 transition-transform">
                    <FaRobot size={32} />
                  </div>
                  <div>
                    <CardTitle className="mb-2">AI Assistant</CardTitle>
                    <CardDescription className="max-w-md">
                      Ask questions about ingredients, routines, or skin conditions. 
                      <span className="font-mono text-xs text-accent-ai ml-2">"Is niacinamide safe for me?"</span>
                    </CardDescription>
                  </div>
                </div>
                <div className="h-12 w-12 rounded-full bg-ai-50 text-accent-ai flex items-center justify-center group-hover:translate-x-2 transition-transform">
                  <FaArrowRight size={20} />
                </div>
              </Card>
            </Link>
          </motion.div>

          {/* PROFILE SNAPSHOT */}
          <motion.div variants={item} className="md:col-span-2">
            <Card variant="ceramic" className="h-full p-8">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div>
                  <div className="text-xs uppercase tracking-wider text-text-secondary font-medium mb-3">
                    Skin Type
                  </div>
                  <div className="flex gap-2">
                    <span className="px-3 py-1.5 bg-surface rounded-full text-xs font-mono text-text-mono border border-border-subtle">
                      Combination
                    </span>
                  </div>
                </div>
                
                <div>
                  <div className="text-xs uppercase tracking-wider text-text-secondary font-medium mb-3">
                    Fitzpatrick
                  </div>
                  <div className="font-mono text-xl font-semibold text-text-primary">Type III</div>
                </div>
                
                <div>
                  <div className="text-xs uppercase tracking-wider text-text-secondary font-medium mb-3">
                    Primary Goal
                  </div>
                  <div className="text-sm font-medium text-text-primary">Anti-Aging & Hydration</div>
                </div>
              </div>
            </Card>
          </motion.div>

        </div>
      </motion.div>
    </div>
  )
}
