import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { LuPhone, LuArrowRight, LuShield, LuActivity, LuUserCheck, LuSearch, LuCalendar } from 'react-icons/lu'
import { IconWrapper } from '../components/Card'

function TopBar() {
  return (
    <div className="bg-background-darker border-b border-white/5 text-xs font-medium relative z-50">
      <div className="max-w-7xl mx-auto px-6 py-2 flex items-center justify-between text-text-secondary">
        <div className="flex items-center gap-2">
          <LuPhone className="text-primary-400" size={14} />
          <span>24x7 Helpline: <b className="text-text-primary">+91-00000 00000</b></span>
        </div>
        <div className="hidden sm:flex gap-6">
          <a className="hover:text-primary-400 transition-colors" href="#departments">Departments</a>
          <a className="hover:text-primary-400 transition-colors" href="#appointments">Appointments</a>
          <a className="hover:text-primary-400 transition-colors" href="#contact">Contact</a>
        </div>
      </div>
    </div>
  )
}

function Header() {
  const loggedIn = !!localStorage.getItem('role')
  return (
    <header className="fixed top-0 left-0 right-0 bg-background/80 backdrop-blur-md border-b border-white/5 z-40 mt-[33px]">
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center shadow-lg shadow-primary-500/20">
            <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.384-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
            </svg>
          </div>
          <div className="text-xl font-bold text-text-primary tracking-tight">AI Skin Doctor</div>
        </div>
        <nav className="hidden md:flex items-center gap-8 text-sm font-medium">
          {['Departments', 'Why Us', 'Research', 'Contact'].map(item => (
            <a
              key={item}
              href={`#${item.toLowerCase().replace(' ', '')}`}
              className="text-text-secondary hover:text-white transition-colors relative group"
            >
              {item}
              <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-primary-500 transition-all group-hover:w-full" />
            </a>
          ))}
        </nav>
        <div className="flex items-center gap-3">
          {loggedIn ? (
            <Link to="/dashboard" className="btn-primary flex items-center gap-2 shadow-lg shadow-primary-500/20">
              Go to App <LuArrowRight />
            </Link>
          ) : (
            <>
              <Link to="/login" className="px-5 py-2.5 rounded-xl border border-white/10 text-sm font-medium text-text-primary hover:bg-white/5 transition-colors">
                Sign in
              </Link>
              <Link to="/register" className="btn-primary shadow-lg shadow-primary-500/20">
                Get Started
              </Link>
            </>
          )}
        </div>
      </div>
    </header>
  )
}

function Hero() {
  return (
    <section className="relative pt-32 pb-20 overflow-hidden min-h-screen flex items-center">
      {/* Background Ambience */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none">
        <div className="absolute top-0 right-[-10%] w-[800px] h-[800px] bg-primary-500/10 rounded-full blur-[120px]" />
        <div className="absolute bottom-0 left-[-10%] w-[600px] h-[600px] bg-accent-500/10 rounded-full blur-[120px]" />
      </div>

      <div className="relative max-w-7xl mx-auto px-6 grid lg:grid-cols-2 gap-12 items-center">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }}>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 text-xs font-medium text-primary-400 mb-6">
            <span className="w-2 h-2 rounded-full bg-primary-400 animate-pulse" />
            Next-Gen Dermatology
          </div>
          <h1 className="text-5xl md:text-7xl font-bold text-text-primary mb-6 leading-tight">
            World-class care with <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary-400 to-accent-400">AI precision</span>
          </h1>
          <p className="text-lg text-text-secondary mb-8 leading-relaxed max-w-xl">
            Experience the future of skin health. Instant AI analysis, expert dermatologist consultations, and personalized care journeys—all in one secure platform.
          </p>
          <div className="flex flex-wrap gap-4">
            <Link to="/register" className="btn-primary text-lg px-8 py-4 shadow-xl shadow-primary-500/20 hover:scale-105 transition-transform">
              Start Diagnosis
            </Link>
            <a href="#departments" className="px-8 py-4 rounded-xl bg-white/5 border border-white/10 text-text-primary font-medium hover:bg-white/10 transition-colors backdrop-blur-sm">
              Explore Departments
            </a>
          </div>
          <div className="mt-12 pt-8 border-t border-white/5 grid grid-cols-3 gap-8">
            <Stat k="Expert Doctors" v="50+" />
            <Stat k="Active Patients" v="25k+" />
            <Stat k="Partner Clinics" v="12+" />
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="hidden lg:block relative"
        >
          <div className="absolute inset-0 bg-gradient-to-br from-primary-500/20 to-accent-500/20 rounded-[3rem] blur-3xl -z-10" />
          <div className="bg-surface-elevated/80 backdrop-blur-xl border border-white/10 rounded-[2rem] p-8 shadow-2xl relative overflow-hidden">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-primary-500 to-accent-500" />
            <div className="flex items-center gap-4 mb-6">
              <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center shadow-lg">
                <LuSearch className="text-white" size={24} />
              </div>
              <div>
                <h3 className="font-bold text-text-primary">AI Diagnostic Preview</h3>
                <p className="text-xs text-text-secondary">Real-time analysis running...</p>
              </div>
            </div>

            <div className="space-y-4">
              <div className="p-4 rounded-xl bg-white/5 border border-white/5">
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-text-secondary">Analysis</span>
                  <span className="text-primary-400 font-mono">98.5% Confidence</span>
                </div>
                <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                  <div className="h-full w-[98.5%] bg-gradient-to-r from-primary-500 to-accent-500" />
                </div>
              </div>

              <div className="flex gap-4">
                <div className="flex-1 p-4 rounded-xl bg-primary-500/10 border border-primary-500/20">
                  <div className="text-2xl font-bold text-primary-400 mb-1">Low Risk</div>
                  <div className="text-xs text-text-secondary">Immediate attention not required</div>
                </div>
                <div className="flex-1 p-4 rounded-xl bg-surface-elevated border border-white/5">
                  <div className="text-sm font-medium text-text-primary mb-2">Recommendation</div>
                  <div className="text-xs text-text-muted">Routine monitoring and UV protection advised.</div>
                </div>
              </div>
            </div>
          </div>

          {/* Floating Badge */}
          <motion.div
            animate={{ y: [0, -10, 0] }}
            transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
            className="absolute -bottom-6 -right-6 bg-surface-elevated border border-white/10 p-4 rounded-2xl shadow-xl flex items-center gap-3 z-10"
          >
            <div className="w-10 h-10 rounded-full bg-green-500/20 flex items-center justify-center text-green-500">
              <LuShield size={20} />
            </div>
            <div>
              <div className="font-bold text-text-primary">HIPAA Compliant</div>
              <div className="text-xs text-text-secondary">100% Secure Data</div>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </section>
  )
}

function Stat({ k, v }) {
  return (
    <div>
      <div className="text-3xl font-bold text-white mb-1">{v}</div>
      <div className="text-sm text-text-tertiary font-medium uppercase tracking-wider">{k}</div>
    </div>
  )
}

function Departments() {
  const items = [
    { t: 'Dermatology', d: 'Expert care for skin, hair, and nail conditions', i: <LuUserCheck /> },
    { t: 'Cosmetology', d: 'Advanced aesthetic procedures & treatments', i: <LuActivity /> },
    { t: 'Oncology', d: 'Specialized skin cancer screening & care', i: <LuSearch /> },
    { t: 'Pediatrics', d: 'Gentle dermatological care for children', i: <LuUserCheck /> },
  ]
  return (
    <section id="departments" className="max-w-7xl mx-auto px-6 py-20">
      <div className="text-center mb-16">
        <h2 className="text-3xl md:text-4xl font-bold text-text-primary mb-4">Specialized Departments</h2>
        <p className="text-text-secondary max-w-2xl mx-auto">Our range of specialized clinics ensures you get the exact care you need from field experts.</p>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        {items.map((x, i) => (
          <div key={i} className="group p-6 rounded-2xl bg-surface-elevated border border-white/5 hover:border-primary-500/30 hover:bg-white/5 transition-all hover:-translate-y-1">
            <div className="h-12 w-12 rounded-xl bg-gradient-to-br from-primary-500/10 to-accent-500/10 flex items-center justify-center text-primary-400 mb-4 group-hover:scale-110 transition-transform">
              {React.cloneElement(x.i, { size: 24 })}
            </div>
            <h3 className="text-xl font-bold text-text-primary mb-2">{x.t}</h3>
            <p className="text-sm text-text-secondary leading-relaxed">{x.d}</p>
          </div>
        ))}
      </div>
    </section>
  )
}

function Why() {
  const feats = [
    { t: 'AI-Assisted Triage', d: 'Instant, preliminary analysis to prioritize urgent cases and guide care.', i: '🤖' },
    { t: 'Top Specialists', d: 'Access to board-certified dermatologists across all major subspecialties.', i: '👨‍⚕️' },
    { t: 'Secure & Private', d: 'Bank-grade encryption protecting your sensitive medical data at all times.', i: '🔒' },
  ]
  return (
    <section id="why" className="py-20 bg-surface-elevated/30 border-y border-white/5">
      <div className="max-w-7xl mx-auto px-6">
        <div className="grid md:grid-cols-3 gap-12">
          {feats.map((f, i) => (
            <div key={i} className="text-center">
              <div className="text-4xl mb-6 bg-white/5 w-20 h-20 rounded-full flex items-center justify-center mx-auto">{f.i}</div>
              <h3 className="text-xl font-bold text-text-primary mb-3">{f.t}</h3>
              <p className="text-text-secondary leading-relaxed">{f.d}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

function CTA() {
  return (
    <section id="appointments" className="max-w-7xl mx-auto px-6 py-20">
      <div className="relative rounded-3xl overflow-hidden p-12 text-center md:text-left md:flex items-center justify-between gap-8">
        <div className="absolute inset-0 bg-gradient-to-r from-primary-600 to-accent-600 opacity-90" />
        <div className="absolute inset-0 noise-bg opacity-20" /> {/* Assuming noise class exists or is ignored */}

        <div className="relative z-10 max-w-2xl">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">Ready to transform your skin health?</h2>
          <p className="text-primary-100 text-lg">Join thousands of patients using AI Skin Doctor for smarter, faster dermatological care.</p>
        </div>
        <div className="relative z-10 mt-8 md:mt-0">
          <Link to="/appointments" className="inline-flex items-center justify-center px-8 py-4 rounded-xl bg-white text-primary-600 font-bold hover:bg-white/90 transition-colors shadow-xl">
            Book Appointment
          </Link>
        </div>
      </div>
    </section>
  )
}

function Footer() {
  return (
    <footer id="contact" className="border-t border-white/10 bg-surface-elevated/20 pt-16 pb-8">
      <div className="max-w-7xl mx-auto px-6 grid sm:grid-cols-2 lg:grid-cols-4 gap-12 mb-12">
        <div>
          <div className="flex items-center gap-2 mb-4">
            <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-primary-500 to-accent-500" />
            <span className="text-lg font-bold text-text-primary">AI Skin Doctor</span>
          </div>
          <p className="text-text-secondary text-sm leading-relaxed">
            Pioneering the future of dermatology through artificial intelligence and human expertise.
          </p>
        </div>

        <div>
          <h4 className="font-bold text-text-primary mb-4">Contact</h4>
          <ul className="space-y-2 text-sm text-text-secondary">
            <li>+91-00000 00000</li>
            <li>support@aiskindoc.com</li>
            <li>123 Medical Hub, Tech Park</li>
          </ul>
        </div>

        <div>
          <h4 className="font-bold text-text-primary mb-4">Quick Links</h4>
          <ul className="space-y-2 text-sm text-text-secondary">
            <li><Link to="/login" className="hover:text-primary-400">Patient Login</Link></li>
            <li><Link to="/apply-doctor" className="hover:text-primary-400">Doctor Portal</Link></li>
            <li><a href="#" className="hover:text-primary-400">Privacy Policy</a></li>
            <li><a href="#" className="hover:text-primary-400">Terms of Service</a></li>
          </ul>
        </div>

        <div>
          <h4 className="font-bold text-text-primary mb-4">Newsletter</h4>
          <div className="flex gap-2">
            <input placeholder="Email address" className="bg-white/5 border border-white/10 rounded-lg px-4 py-2 text-sm w-full focus:outline-none focus:border-primary-500/50" />
            <button className="p-2 bg-primary-500 rounded-lg text-white hover:bg-primary-600"><LuArrowRight /></button>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 pt-8 border-t border-white/5 text-center text-xs text-text-tertiary">
        &copy; {new Date().getFullYear()} AI Skin Doctor. All rights reserved.
      </div>
    </footer>
  )
}
import React from 'react'

export default function Landing() {
  return (
    <div className="min-h-screen bg-background text-text-primary font-sans selection:bg-primary-500/30">
      <TopBar />
      <Header />
      <Hero />
      <Departments />
      <Why />
      <CTA />
      <Footer />
    </div>
  )
}
