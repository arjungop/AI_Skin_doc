import { Link } from 'react-router-dom'

function TopBar(){
  return (
    <div className="bg-slate-100 border-b border-borderLight text-sm">
      <div className="max-w-7xl mx-auto px-6 py-2 flex items-center justify-between text-textDark">
        <div>24x7 Helpline: <b>+91-00000 00000</b></div>
        <div className="hidden sm:flex gap-4">
          <a className="text-primary" href="#departments">Departments</a>
          <a className="text-primary" href="#appointments">Appointments</a>
          <a className="text-primary" href="#contact">Contact</a>
        </div>
      </div>
    </div>
  )
}

function Header(){
  const loggedIn = !!localStorage.getItem('role')
  return (
    <header className="bg-white/90 backdrop-blur supports-[backdrop-filter]:bg-white/80 border-b border-borderGray sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-md bg-gradient-to-tr from-primary to-accent" />
          <div className="text-xl font-semibold text-textDark">AI Skin Doctor</div>
        </div>
        <nav className="hidden md:flex items-center gap-6 text-sm">
          <a href="#departments" className="text-textDark hover:text-primary">Departments</a>
          <a href="#why" className="text-textDark hover:text-primary">Why Us</a>
          <a href="#research" className="text-textDark hover:text-primary">Research</a>
          <a href="#contact" className="text-textDark hover:text-primary">Contact</a>
        </nav>
        <div className="flex items-center gap-2">
          {loggedIn ? (
            <Link to="/dashboard" className="btn-primary">Go to App</Link>
          ) : (
            <>
              <Link to="/login" className="px-4 py-2 rounded-md border border-borderGray text-sm hover:bg-slate-50">Sign in</Link>
              <Link to="/register" className="btn-primary">Create account</Link>
            </>
          )}
        </div>
      </div>
    </header>
  )
}

function Hero(){
  return (
    <section className="relative overflow-hidden">
      <div className="absolute inset-0 gradient-calm opacity-60" />
      <div className="relative max-w-7xl mx-auto px-6 py-16 md:py-24 grid md:grid-cols-2 gap-8 items-center">
        <div>
          <h1 className="text-4xl md:text-5xl font-semibold text-textDark mb-4">World‑class dermatology care with AI guidance</h1>
          <p className="text-textMuted mb-6">Book appointments, chat with our AI Doctor for safe skin guidance, and manage your reports — all in one secure place.</p>
          <div className="flex gap-3">
            <Link to="/register" className="btn-primary">Get Started</Link>
            <a href="#departments" className="btn-secondary">Explore Departments</a>
          </div>
          <div className="mt-6 grid grid-cols-3 gap-4 text-center">
            <Stat k="Doctors" v="50+"/>
            <Stat k="Patients" v="25k+"/>
            <Stat k="Clinics" v="5"/>
          </div>
        </div>
        <div className="hidden md:block">
          <div className="rounded-2xl border border-borderGray bg-white shadow-sm p-6">
            <div className="text-sm text-textMuted mb-2">AI Preview</div>
            <div className="space-y-2">
              <div className="bubble">What are melanoma warning signs?</div>
              <div className="bubble me">Look for ABCDE: Asymmetry, irregular Borders, varied Color, Diameter >6mm, and Evolution. See a dermatologist for changing or bleeding moles.</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

function Stat({k,v}){ return (
  <div className="card">
    <div className="text-2xl font-semibold">{v}</div>
    <div className="muted text-sm">{k}</div>
  </div>
) }

function Departments(){
  const items = [
    { t:'Dermatology', d:'Skin, hair and nail care' },
    { t:'Cosmetology', d:'Aesthetics & procedures' },
    { t:'Oncology', d:'Skin cancer clinic' },
    { t:'Pediatrics', d:'Child skin conditions' },
  ]
  return (
    <section id="departments" className="max-w-7xl mx-auto px-6 py-12">
      <h2 className="text-2xl font-semibold mb-4">Departments</h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        {items.map((x,i)=> (
          <div key={i} className="card hover:shadow-lg hover:scale-[1.01] transition">
            <div className="h-10 w-10 rounded-md bg-gradient-to-tr from-primary to-accent mb-3" />
            <div className="text-lg font-semibold">{x.t}</div>
            <div className="muted text-sm">{x.d}</div>
          </div>
        ))}
      </div>
    </section>
  )
}

function Why(){
  const feats = [
    { t:'AI‑assisted triage', d:'Safe guidance with clear red‑flag alerts.'},
    { t:'Top specialists', d:'Consult experienced dermatologists across subspecialties.'},
    { t:'Secure records', d:'Privacy‑first platform with encrypted storage.'},
  ]
  return (
    <section id="why" className="max-w-7xl mx-auto px-6 py-12">
      <div className="grid md:grid-cols-3 gap-6">
        {feats.map((f,i)=> (
          <div key={i} className="card">
            <div className="text-lg font-semibold mb-1">{f.t}</div>
            <div className="muted text-sm">{f.d}</div>
          </div>
        ))}
      </div>
    </section>
  )
}

function CTA(){
  return (
    <section id="appointments" className="max-w-7xl mx-auto px-6 py-12">
      <div className="card gradient-innovation text-white p-8 flex items-center justify-between">
        <div>
          <div className="text-2xl font-semibold">Book an appointment in minutes</div>
          <div className="opacity-90">Find a doctor and choose a convenient time slot.</div>
        </div>
        <Link to="/appointments" className="px-4 py-2 rounded-lg bg-white text-textDark hover:opacity-90">Book now</Link>
      </div>
    </section>
  )
}

function Footer(){
  return (
    <footer id="contact" className="bg-slate-50 border-t border-borderLight mt-12">
      <div className="max-w-7xl mx-auto px-6 py-8 grid sm:grid-cols-2 lg:grid-cols-4 gap-6 text-sm">
        <div>
          <div className="text-lg font-semibold mb-2">AI Skin Doctor</div>
          <div className="muted">Modern dermatology with AI assistance.</div>
        </div>
        <div>
          <div className="font-semibold mb-2">Contact</div>
          <div className="muted">+91-00000 00000</div>
          <div className="muted">support@example.com</div>
        </div>
        <div>
          <div className="font-semibold mb-2">Links</div>
          <div><Link to="/login" className="text-primary">Patient Login</Link></div>
          <div><Link to="/apply-doctor" className="text-primary">Apply as Doctor</Link></div>
        </div>
        <div className="muted">© {new Date().getFullYear()} AI Skin Doctor</div>
      </div>
    </footer>
  )
}

export default function Landing(){
  return (
    <div className="bg-background text-textDark">
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

