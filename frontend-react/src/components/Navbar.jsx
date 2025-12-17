import { Link, useLocation } from 'react-router-dom'

export default function Navbar({ onLogout }) {
  const loc = useLocation()
  const role = (localStorage.getItem('role') || '').toUpperCase()
  const isDoctor = role === 'DOCTOR'
  const isAdmin = role === 'ADMIN'
  const loggedIn = !!role
  return (
    <div className="navbar">
      <div className="brand">AI Skin Doctor</div>
      <div className="links">
        {!loggedIn ? (
          <>
            <Nav to="/login" label="Login" active={loc.pathname==='/login'} />
            <Nav to="/register" label="Register" active={loc.pathname==='/register'} />
            <Nav to="/apply-doctor" label="Apply as Doctor" active={loc.pathname==='/apply-doctor'} />
          </>
        ) : isAdmin ? (
          <>
            <Nav to="/admin" label="Admin" active={loc.pathname==='/admin'} />
            <Nav to="/appointments" label="Appointments" active={loc.pathname==='/appointments'} />
            <Nav to="/transactions" label="Transactions" active={loc.pathname==='/transactions'} />
            <Nav to="/chat" label="AI Chat" active={loc.pathname==='/chat'} />
            <Nav to="/messages" label="Messages" active={loc.pathname==='/messages'} />
          </>
        ) : isDoctor ? (
          <>
            <Nav to="/doctor" label="Doctor Portal" active={loc.pathname==='/doctor'} />
            <Nav to="/appointments" label="Appointments" active={loc.pathname==='/appointments'} />
            <Nav to="/transactions" label="Transactions" active={loc.pathname==='/transactions'} />
            <Nav to="/chat" label="AI Chat" active={loc.pathname==='/chat'} />
            <Nav to="/messages" label="Messages" active={loc.pathname==='/messages'} />
          </>
        ) : (
          <>
            <Nav to="/dashboard" label="Dashboard" active={loc.pathname==='/dashboard'} />
            <Nav to="/lesions" label="Lesion Classification" active={loc.pathname==='/lesions'} />
            <Nav to="/chat" label="AI Chat" active={loc.pathname==='/chat'} />
            <Nav to="/appointments" label="Appointments" active={loc.pathname==='/appointments'} />
            <Nav to="/transactions" label="Transactions" active={loc.pathname==='/transactions'} />
            <Nav to="/messages" label="Messages" active={loc.pathname==='/messages'} />
          </>
        )}
      </div>
      {loggedIn ? (
        <button className="button" onClick={onLogout}>Logout</button>
      ) : null}
    </div>
  )
}

function Nav({ to, label, active }) {
  return <Link className={"navlink" + (active ? " active" : "")} to={to}>{label}</Link>
}
