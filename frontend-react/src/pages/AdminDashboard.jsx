import { useEffect, useState, useRef, useCallback } from 'react'
import { api } from '../services/api'
import ConfirmModal from '../components/ConfirmModal.jsx'
import { useToast } from '../components/Toast.jsx'
import { motion, AnimatePresence } from 'framer-motion'
import {
  LuUsers, LuStethoscope,
  LuSettings, LuHistory, LuCheck, LuX, LuBan, LuSearch,
  LuDownload, LuFileText, LuShield, LuRefreshCw, LuChevronLeft, LuChevronRight
} from 'react-icons/lu'
import { Card, CardTitle, CardDescription } from '../components/Card.jsx'

export default function AdminDashboard() {
  const [overview, setOverview] = useState(null)
  const [tab, setTab] = useState('apps')
  // Applications
  const [apps, setApps] = useState([])
  const [appsTotal, setAppsTotal] = useState(0)
  const [appsPage, setAppsPage] = useState(1)
  const [appsStatus, setAppsStatus] = useState('PENDING')
  const [appsQ, setAppsQ] = useState('')
  // Users
  const [users, setUsers] = useState([])
  const [usersTotal, setUsersTotal] = useState(0)
  const [usersPage, setUsersPage] = useState(1)
  const [usersQ, setUsersQ] = useState('')
  const [usersRole, setUsersRole] = useState('')
  const [suspend, setSuspend] = useState({ open: false, user: null })
  const [terminate, setTerminate] = useState({ open: false, user: null, reason_code: 'REQUEST', reason_text: '' })
  // Doctors
  const [docs, setDocs] = useState([])
  const [docsTotal, setDocsTotal] = useState(0)
  const [docsPage, setDocsPage] = useState(1)
  const [docsQ, setDocsQ] = useState('')
  // Audit
  const [logs, setLogs] = useState([])
  const [logsTotal, setLogsTotal] = useState(0)
  const [logsPage, setLogsPage] = useState(1)
  // Settings
  const [settings, setSettings] = useState({})
  // Approve/Reject confirmation
  const [confirmAction, setConfirmAction] = useState({ open: false, id: null, what: '' })
  // Debounce refs
  const appsDebounce = useRef(null)
  const usersDebounce = useRef(null)
  const docsDebounce = useRef(null)

  const { push } = useToast()

  useEffect(() => { (async () => { try { setOverview(await api.adminOverview()) } catch (err) { push(err?.message || 'Failed to load overview', 'error') } })() }, [])
  async function refreshOverview() { try { setOverview(await api.adminOverview()) } catch (err) { push(err?.message || 'Failed to refresh overview', 'error') } }

  async function loadApps(p = appsPage) {
    const res = await api.adminListDoctorAppsPaged({ status: appsStatus || undefined, q: appsQ || undefined, page: p, page_size: 20 })
    setApps(res.items || []); setAppsTotal(res.total || 0); setAppsPage(res.page || p)
  }
  useEffect(() => { clearTimeout(appsDebounce.current); appsDebounce.current = setTimeout(() => loadApps(1), 300) }, [appsStatus, appsQ])

  async function loadUsers(p = usersPage) {
    const res = await api.adminListUsers({ q: usersQ || undefined, role: usersRole || undefined, page: p, page_size: 20 })
    setUsers(res.items || []); setUsersTotal(res.total || 0); setUsersPage(res.page || p)
  }
  useEffect(() => { if (tab === 'users') { clearTimeout(usersDebounce.current); usersDebounce.current = setTimeout(() => loadUsers(1), 300) } }, [tab, usersQ, usersRole])

  async function loadDocs(p = docsPage) {
    const res = await api.adminListDoctors({ q: docsQ || undefined, page: p, page_size: 20 })
    setDocs(res.items || []); setDocsTotal(res.total || 0); setDocsPage(res.page || p)
  }
  useEffect(() => { if (tab === 'doctors') { clearTimeout(docsDebounce.current); docsDebounce.current = setTimeout(() => loadDocs(1), 300) } }, [tab, docsQ])

  async function loadLogs(p = logsPage) {
    const res = await api.adminListAudit({ page: p, page_size: 50 })
    setLogs(res.items || []); setLogsTotal(res.total || 0); setLogsPage(res.page || p)
  }
  useEffect(() => { if (tab === 'audit') loadLogs(1) }, [tab])

  async function loadSettings() { try { setSettings(await api.adminGetSettings()) } catch { } }
  useEffect(() => { if (tab === 'settings') loadSettings() }, [tab])

  const appsMax = Math.max(1, Math.ceil(appsTotal / 20))
  const usersMax = Math.max(1, Math.ceil(usersTotal / 20))
  const docsMax = Math.max(1, Math.ceil(docsTotal / 20))
  const logsMax = Math.max(1, Math.ceil(logsTotal / 50))

  const act = async (id, what) => {
    try {
      if (what === 'approve') await api.adminApproveDoctor(id)
      else await api.adminRejectDoctor(id)
      push(`Application ${what}d successfully`, 'success')
      setConfirmAction({ open: false, id: null, what: '' })
      await loadApps()
    } catch (e) { push(e.message, 'error') }
  }

  async function exportApps() {
    try {
      const BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
      const params = new URLSearchParams()
      if (appsStatus) params.set('status', appsStatus)
      if (appsQ) params.set('q', appsQ)
      const resp = await fetch(`${BASE}/admin/doctor_applications/export.csv?${params}`, {
        headers: { Authorization: `Bearer ${localStorage.getItem('access_token')}` },
      })
      if (!resp.ok) throw new Error('Export failed')
      const blob = await resp.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `doctor_applications_${new Date().toISOString().split('T')[0]}.csv`
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
    } catch (e) { push(e.message, 'error') }
  }

  async function doSuspend() {
    if (!suspend.user) return;
    const status = (suspend.user.status === 'SUSPENDED') ? 'ACTIVE' : 'SUSPENDED';
    try {
      await api.adminUpdateUserStatus(suspend.user.user_id, status);
      push(`User ${status.toLowerCase()}`, 'success');
    } catch (e) {
      push(e?.message, 'error');
    }
    setSuspend({ open: false, user: null });
    loadUsers();
  }

  async function doTerminate() {
    if (!terminate.user) return;
    try {
      await api.adminTerminateUser(terminate.user.user_id, terminate.reason_code, terminate.reason_text);
      push('User terminated successfully', 'success');
    } catch (e) {
      push(e?.message, 'error');
    }
    setTerminate({ open: false, user: null, reason_code: 'REQUEST', reason_text: '' });
    loadUsers();
  }

  async function saveSettings(e) {
    e.preventDefault();
    const form = new FormData(e.target);
    const obj = {};
    for (const [k, v] of form.entries()) { obj[k] = v };
    try {
      await api.adminSetSettings(obj);
      push('Settings saved', 'success');
    } catch (err) {
      push(err?.message, 'error');
    }
    loadSettings();
  }

  const container = { hidden: { opacity: 0 }, show: { opacity: 1, transition: { staggerChildren: 0.05 } } }
  const item = { hidden: { opacity: 0, y: 10 }, show: { opacity: 1, y: 0 } }

  return (
    <div className="pb-12">
      <header className="mb-8 p-1">
        <h1 className="text-3xl font-bold text-slate-900 tracking-tight flex items-center gap-3">
          <LuShield className="text-indigo-500" /> Admin Portal
        </h1>
        <p className="text-slate-500 mt-2 max-w-2xl">Manage users, review doctor applications, audit logs, and configure system settings directly from the administrative dashboard.</p>
      </header>

      {overview ? (
        <motion.div variants={container} initial="hidden" animate="show" className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-4 mb-8">
          {Object.entries(overview).map(([k, v]) => (
            <motion.div key={k} variants={item}>
              <Card className="p-5 flex flex-col justify-center h-full border border-slate-200/60 shadow-soft-sm" hover={true}>
                <div className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">{k.replace('_', ' ')}</div>
                <div className="text-2xl font-bold text-slate-800">{v}</div>
              </Card>
            </motion.div>
          ))}
        </motion.div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-4 mb-8 animate-pulse">
          {[...Array(6)].map((_, i) => <Card key={i} className="p-5 h-[88px] bg-slate-100/50 border-none shadow-none" hover={false} />)}
        </div>
      )}

      {/* Tabs Layout */}
      <div className="flex overflow-x-auto hide-scrollbar gap-2 p-1 bg-slate-100/80 rounded-2xl border border-slate-200/60 shadow-inner w-fit mb-6">
        {[
          { id: 'apps', label: 'Applications', icon: LuFileText },
          { id: 'users', label: 'Users', icon: LuUsers },
          { id: 'doctors', label: 'Doctors', icon: LuStethoscope },
          { id: 'settings', label: 'Settings', icon: LuSettings },
          { id: 'audit', label: 'Audit Log', icon: LuHistory },
        ].map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all whitespace-nowrap focus:outline-none focus:ring-2 focus:ring-primary-500/20 ${tab === t.id
              ? 'bg-white text-primary-600 shadow-sm border border-slate-200/60'
              : 'text-slate-500 hover:text-slate-800 hover:bg-slate-200/50'
              }`}
          >
            <t.icon size={16} /> {t.label}
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={tab}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.98 }}
          transition={{ duration: 0.2 }}
        >
          {/* Applications Tab */}
          {tab === 'apps' && (
            <Card className="overflow-hidden border border-slate-200/60 shadow-soft-md">
              <div className="p-6 border-b border-slate-100 flex flex-col md:flex-row md:items-center justify-between gap-4 bg-white">
                <div>
                  <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2">Doctor Applications <span className="bg-primary-50 text-primary-600 text-xs px-2 py-0.5 rounded-full">{appsTotal}</span></h3>
                  <p className="text-sm text-slate-500">Review pending doctors to grant them platform access.</p>
                </div>
                <div className="flex items-center gap-3">
                  <div className="relative">
                    <LuSearch className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={16} />
                    <input className="input pl-9 w-full md:w-64 text-sm bg-slate-50" placeholder="Search name/email" value={appsQ} onChange={e => setAppsQ(e.target.value)} />
                  </div>
                  <select className="input text-sm bg-slate-50 font-medium cursor-pointer" value={appsStatus} onChange={e => setAppsStatus(e.target.value)}>
                    <option value="">All Statuses</option>
                    <option value="PENDING">Pending</option>
                    <option value="APPROVED">Approved</option>
                    <option value="REJECTED">Rejected</option>
                  </select>
                  <button className="btn-secondary whitespace-nowrap text-sm flex items-center gap-2 bg-slate-50 border border-slate-200" onClick={exportApps}>
                    <LuDownload size={14} /> Export CSV
                  </button>
                </div>
              </div>
              <div className="overflow-x-auto w-full">
                <table className="w-full text-left text-sm text-slate-600 whitespace-nowrap">
                  <thead className="bg-slate-50 text-slate-500 text-xs uppercase tracking-wider font-bold border-b border-slate-100">
                    <tr><th className="p-4 pl-6">ID</th><th className="p-4">Name</th><th className="p-4">Email</th><th className="p-4">Specialization</th><th className="p-4">License</th><th className="p-4">Status</th><th className="p-4 pr-6 text-right">Actions</th></tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {apps.map(a => (
                      <tr key={a.application_id} className="hover:bg-slate-50/50 transition-colors">
                        <td className="p-4 pl-6 font-mono text-xs">{a.application_id}</td>
                        <td className="p-4 font-semibold text-slate-800">{(a.first_name || '') + ' ' + (a.last_name || '')}</td>
                        <td className="p-4">{a.email}</td>
                        <td className="p-4">{a.specialization || '-'}</td>
                        <td className="p-4 font-mono text-xs">{a.license_no || '-'}</td>
                        <td className="p-4">
                          <span className={`px-2 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest ${a.status === 'APPROVED' ? 'bg-emerald-50 text-emerald-600 border border-emerald-200' : a.status === 'REJECTED' ? 'bg-rose-50 text-rose-600 border border-rose-200' : 'bg-amber-50 text-amber-600 border border-amber-200'}`}>{a.status}</span>
                        </td>
                        <td className="p-4 pr-6 text-right">
                          {a.status === 'PENDING' && (
                            <div className="flex items-center justify-end gap-2">
                              <button className="p-1.5 text-emerald-600 bg-emerald-50 hover:bg-emerald-100 rounded-lg transition-colors border border-emerald-100" title="Approve" onClick={() => setConfirmAction({ open: true, id: a.application_id, what: 'approve' })}><LuCheck size={16} /></button>
                              <button className="p-1.5 text-rose-600 bg-rose-50 hover:bg-rose-100 rounded-lg transition-colors border border-rose-100" title="Reject" onClick={() => setConfirmAction({ open: true, id: a.application_id, what: 'reject' })}><LuX size={16} /></button>
                            </div>
                          )}
                        </td>
                      </tr>
                    ))}
                    {apps.length === 0 && <tr><td colSpan="7" className="p-8 text-center text-slate-400">No applications found.</td></tr>}
                  </tbody>
                </table>
              </div>
              <div className="p-4 bg-slate-50/50 border-t border-slate-100 flex items-center justify-between text-sm">
                <div className="text-slate-500 font-medium pl-2">Page <span className="text-slate-900 font-bold">{appsPage}</span> of {appsMax}</div>
                <div className="flex gap-2">
                  <button className="p-2 rounded-lg bg-white border border-slate-200 text-slate-600 hover:bg-slate-50 hover:text-slate-900 disabled:opacity-50 disabled:cursor-not-allowed transition-colors" disabled={appsPage <= 1} onClick={() => loadApps(appsPage - 1)}><LuChevronLeft size={16} /></button>
                  <button className="p-2 rounded-lg bg-white border border-slate-200 text-slate-600 hover:bg-slate-50 hover:text-slate-900 disabled:opacity-50 disabled:cursor-not-allowed transition-colors" disabled={appsPage >= appsMax} onClick={() => loadApps(appsPage + 1)}><LuChevronRight size={16} /></button>
                </div>
              </div>
            </Card>
          )}

          {/* Users Tab */}
          {tab === 'users' && (
            <Card className="overflow-hidden border border-slate-200/60 shadow-soft-md">
              <div className="p-6 border-b border-slate-100 flex flex-col md:flex-row md:items-center justify-between gap-4 bg-white">
                <div>
                  <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2">Registered Users <span className="bg-primary-50 text-primary-600 text-xs px-2 py-0.5 rounded-full">{usersTotal}</span></h3>
                  <p className="text-sm text-slate-500">Manage user accounts, roles, and platform access.</p>
                </div>
                <div className="flex items-center gap-3">
                  <div className="relative">
                    <LuSearch className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={16} />
                    <input className="input pl-9 w-full md:w-64 text-sm bg-slate-50" placeholder="Search username/email" value={usersQ} onChange={e => setUsersQ(e.target.value)} />
                  </div>
                  <select className="input text-sm bg-slate-50 font-medium cursor-pointer" value={usersRole} onChange={e => setUsersRole(e.target.value)}>
                    <option value="">All Roles</option>
                    <option value="ADMIN">Admin</option>
                    <option value="DOCTOR">Doctor</option>
                    <option value="PATIENT">Patient</option>
                    <option value="PENDING_DOCTOR">Pending Doctor</option>
                  </select>
                </div>
              </div>
              <div className="overflow-x-auto w-full">
                <table className="w-full text-left text-sm text-slate-600 whitespace-nowrap">
                  <thead className="bg-slate-50 text-slate-500 text-xs uppercase tracking-wider font-bold border-b border-slate-100">
                    <tr><th className="p-4 pl-6">ID</th><th className="p-4">Username</th><th className="p-4">Email</th><th className="p-4">Role</th><th className="p-4 pr-6 text-right">Actions</th></tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {users.map(u => (
                      <tr key={u.user_id} className="hover:bg-slate-50/50 transition-colors">
                        <td className="p-4 pl-6 font-mono text-xs">{u.user_id}</td>
                        <td className="p-4 font-semibold text-slate-800">{u.username}</td>
                        <td className="p-4">{u.email}</td>
                        <td className="p-4">
                          <span className="px-2 py-1 rounded-md text-[10px] bg-slate-100 text-slate-600 font-bold uppercase tracking-widest">{u.role}</span>
                        </td>
                        <td className="p-4 pr-6 text-right">
                          <div className="flex items-center justify-end gap-2">
                            <button className="px-3 py-1.5 text-xs font-semibold text-slate-600 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors" onClick={() => setSuspend({ open: true, user: u })}>{u.status === 'SUSPENDED' ? 'Unsuspend' : 'Suspend'}</button>
                            <button className="px-3 py-1.5 text-xs font-semibold text-rose-600 bg-rose-50 hover:bg-rose-100 border border-rose-100 rounded-lg transition-colors flex items-center gap-1.5" onClick={() => setTerminate({ open: true, user: u, reason_code: 'REQUEST', reason_text: '' })}><LuBan size={12} />Terminate</button>
                          </div>
                        </td>
                      </tr>
                    ))}
                    {users.length === 0 && <tr><td colSpan="5" className="p-8 text-center text-slate-400">No users found.</td></tr>}
                  </tbody>
                </table>
              </div>
              <div className="p-4 bg-slate-50/50 border-t border-slate-100 flex items-center justify-between text-sm">
                <div className="text-slate-500 font-medium pl-2">Page <span className="text-slate-900 font-bold">{usersPage}</span> of {usersMax}</div>
                <div className="flex gap-2">
                  <button className="p-2 rounded-lg bg-white border border-slate-200 text-slate-600 hover:bg-slate-50 hover:text-slate-900 disabled:opacity-50 disabled:cursor-not-allowed transition-colors" disabled={usersPage <= 1} onClick={() => loadUsers(usersPage - 1)}><LuChevronLeft size={16} /></button>
                  <button className="p-2 rounded-lg bg-white border border-slate-200 text-slate-600 hover:bg-slate-50 hover:text-slate-900 disabled:opacity-50 disabled:cursor-not-allowed transition-colors" disabled={usersPage >= usersMax} onClick={() => loadUsers(usersPage + 1)}><LuChevronRight size={16} /></button>
                </div>
              </div>
            </Card>
          )}

          {/* Doctors Tab */}
          {tab === 'doctors' && (
            <Card className="overflow-hidden border border-slate-200/60 shadow-soft-md">
              <div className="p-6 border-b border-slate-100 flex flex-col md:flex-row md:items-center justify-between gap-4 bg-white">
                <div>
                  <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2">Platform Doctors <span className="bg-primary-50 text-primary-600 text-xs px-2 py-0.5 rounded-full">{docsTotal}</span></h3>
                  <p className="text-sm text-slate-500">View active doctors synchronized with the system.</p>
                </div>
                <div className="flex items-center gap-3">
                  <div className="relative">
                    <LuSearch className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={16} />
                    <input className="input pl-9 w-full md:w-64 text-sm bg-slate-50" placeholder="Search name/specialization" value={docsQ} onChange={e => setDocsQ(e.target.value)} />
                  </div>
                  <button className="btn-primary text-sm whitespace-nowrap px-4 py-2" onClick={async () => { await api.adminSyncDoctors(); await loadDocs(1); await refreshOverview(); push('Doctors synced successfully', 'success') }}>
                    <LuRefreshCw size={14} /> Sync Doctors
                  </button>
                </div>
              </div>
              <div className="overflow-x-auto w-full">
                <table className="w-full text-left text-sm text-slate-600 whitespace-nowrap">
                  <thead className="bg-slate-50 text-slate-500 text-xs uppercase tracking-wider font-bold border-b border-slate-100">
                    <tr><th className="p-4 pl-6">Doc ID</th><th className="p-4">Username</th><th className="p-4">Email</th><th className="p-4 pr-6 text-right">Specialization</th></tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {docs.map(d => (
                      <tr key={d.doctor_id} className="hover:bg-slate-50/50 transition-colors">
                        <td className="p-4 pl-6 font-mono text-xs">{d.doctor_id}</td>
                        <td className="p-4 font-semibold text-slate-800">{d.username}</td>
                        <td className="p-4">{d.email}</td>
                        <td className="p-4 pr-6 text-right font-medium">{d.specialization || '-'}</td>
                      </tr>
                    ))}
                    {docs.length === 0 && <tr><td colSpan="4" className="p-8 text-center text-slate-400">No doctors found.</td></tr>}
                  </tbody>
                </table>
              </div>
              <div className="p-4 bg-slate-50/50 border-t border-slate-100 flex items-center justify-between text-sm">
                <div className="text-slate-500 font-medium pl-2">Page <span className="text-slate-900 font-bold">{docsPage}</span> of {docsMax}</div>
                <div className="flex gap-2">
                  <button className="p-2 rounded-lg bg-white border border-slate-200 text-slate-600 hover:bg-slate-50 hover:text-slate-900 disabled:opacity-50 disabled:cursor-not-allowed transition-colors" disabled={docsPage <= 1} onClick={() => loadDocs(docsPage - 1)}><LuChevronLeft size={16} /></button>
                  <button className="p-2 rounded-lg bg-white border border-slate-200 text-slate-600 hover:bg-slate-50 hover:text-slate-900 disabled:opacity-50 disabled:cursor-not-allowed transition-colors" disabled={docsPage >= docsMax} onClick={() => loadDocs(docsPage + 1)}><LuChevronRight size={16} /></button>
                </div>
              </div>
            </Card>
          )}

          {/* Settings Tab */}
          {tab === 'settings' && (
            <Card className="max-w-3xl overflow-hidden border border-slate-200/60 shadow-soft-md" hover={false}>
              <div className="p-6 border-b border-slate-100 bg-white">
                <h3 className="text-lg font-bold text-slate-900">System Parameters</h3>
                <p className="text-sm text-slate-500 mt-1">Configure global application variables stored securely in the database.</p>
              </div>
              <div className="p-6 bg-slate-50/30">
                <form onSubmit={saveSettings} className="space-y-6">
                  <div className="grid md:grid-cols-2 gap-6">
                    <div>
                      <label className="text-xs font-bold text-slate-700 uppercase tracking-widest mb-1.5 block">Malignant Threshold</label>
                      <input className="input w-full bg-white border-slate-200 focus:border-primary-500" name="LESION_MALIGNANT_THRESHOLD" defaultValue={settings['LESION_MALIGNANT_THRESHOLD'] || ''} placeholder="0.5" />
                      <p className="text-[10px] text-slate-500 mt-1.5">Float value for model malignancy cutoff.</p>
                    </div>
                    <div>
                      <label className="text-xs font-bold text-slate-700 uppercase tracking-widest mb-1.5 block">JWT Leeway</label>
                      <input className="input w-full bg-white border-slate-200 focus:border-primary-500" name="JWT_LEEWAY_SECONDS" defaultValue={settings['JWT_LEEWAY_SECONDS'] || ''} placeholder="300" />
                      <p className="text-[10px] text-slate-500 mt-1.5">Seconds allowed for clock drift on tokens.</p>
                    </div>
                    <div className="md:col-span-2">
                      <label className="text-xs font-bold text-slate-700 uppercase tracking-widest mb-1.5 block">Frontend Origins</label>
                      <input className="input w-full bg-white border-slate-200 focus:border-primary-500" name="FRONTEND_ORIGINS" defaultValue={settings['FRONTEND_ORIGINS'] || ''} placeholder="http://localhost:3000,https://app.com" />
                      <p className="text-[10px] text-slate-500 mt-1.5">Comma-separated CORS origins.</p>
                    </div>
                  </div>
                  <div className="pt-4 border-t border-slate-200/60 pb-2">
                    <button className="btn-primary w-full md:w-auto px-8" type="submit">Save Configurations</button>
                  </div>
                </form>
              </div>
            </Card>
          )}

          {/* Audit Tab */}
          {tab === 'audit' && (
            <Card className="overflow-hidden border border-slate-200/60 shadow-soft-md">
              <div className="p-6 border-b border-slate-100 flex flex-col md:flex-row md:items-center justify-between gap-4 bg-white">
                <div>
                  <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2">Audit Logs <span className="bg-primary-50 text-primary-600 text-xs px-2 py-0.5 rounded-full">{logsTotal}</span></h3>
                  <p className="text-sm text-slate-500">Immutable trail of administrative and elevated actions.</p>
                </div>
              </div>
              <div className="overflow-x-auto w-full max-h-[600px] overflow-y-auto">
                <table className="w-full text-left text-sm text-slate-600 whitespace-nowrap">
                  <thead className="sticky top-0 bg-slate-50 text-slate-400 text-xs uppercase tracking-wider font-bold border-b border-slate-200 z-10">
                    <tr><th className="p-4 pl-6">Log ID</th><th className="p-4">User</th><th className="p-4">Action</th><th className="p-4">Meta</th><th className="p-4 pr-6 text-right">Timestamp</th></tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {logs.map(r => (
                      <tr key={r.id} className="hover:bg-slate-50/50 transition-colors">
                        <td className="p-4 pl-6 font-mono text-[10px] text-slate-400">{r.id}</td>
                        <td className="p-4 font-mono text-xs text-primary-600">{r.user_id || '-'}</td>
                        <td className="p-4 font-semibold text-slate-800">
                          <span className="px-2.5 py-1 rounded bg-slate-100 text-[10px] uppercase font-bold tracking-widest">{r.action}</span>
                        </td>
                        <td className="p-4 text-xs font-mono text-slate-500 truncate max-w-[200px]" title={r.meta}>{r.meta || '-'}</td>
                        <td className="p-4 pr-6 text-right text-xs text-slate-500">{new Date(r.created_at).toLocaleString([], { hour12: true, month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })}</td>
                      </tr>
                    ))}
                    {logs.length === 0 && <tr><td colSpan="5" className="p-8 text-center text-slate-400">No logs found.</td></tr>}
                  </tbody>
                </table>
              </div>
              <div className="p-4 bg-slate-50 border-t border-slate-100 flex items-center justify-between text-sm">
                <div className="text-slate-500 font-medium pl-2">Page <span className="text-slate-900 font-bold">{logsPage}</span> of {logsMax}</div>
                <div className="flex gap-2">
                  <button className="p-2 rounded-lg bg-white border border-slate-200 text-slate-600 hover:bg-slate-50 hover:text-slate-900 disabled:opacity-50 disabled:cursor-not-allowed transition-colors" disabled={logsPage <= 1} onClick={() => loadLogs(logsPage - 1)}><LuChevronLeft size={16} /></button>
                  <button className="p-2 rounded-lg bg-white border border-slate-200 text-slate-600 hover:bg-slate-50 hover:text-slate-900 disabled:opacity-50 disabled:cursor-not-allowed transition-colors" disabled={logsPage >= logsMax} onClick={() => loadLogs(logsPage + 1)}><LuChevronRight size={16} /></button>
                </div>
              </div>
            </Card>
          )}
        </motion.div>
      </AnimatePresence>

      <ConfirmModal
        open={confirmAction.open}
        title={`${confirmAction.what === 'approve' ? 'Approve' : 'Reject'} Application?`}
        onClose={() => setConfirmAction({ open: false, id: null, what: '' })}
        onConfirm={() => act(confirmAction.id, confirmAction.what)}
        confirmText={confirmAction.what === 'approve' ? 'Approve Doctor' : 'Reject Doctor'}
      >
        <div className="text-sm text-slate-600 leading-relaxed pt-2">Are you sure you want to <strong>{confirmAction.what}</strong> this doctor application? This action will be logged in the system audit trail and notify the user.</div>
      </ConfirmModal>

      <ConfirmModal open={suspend.open} title={suspend.user ? `${suspend?.user?.status === 'SUSPENDED' ? 'Unsuspend' : 'Suspend'} Account?` : ''} onClose={() => setSuspend({ open: false, user: null })} onConfirm={doSuspend} confirmText="Confirm">
        <div className="text-sm text-slate-600 leading-relaxed pt-2">This will <strong>{suspend?.user?.status === 'SUSPENDED' ? 'reactivate' : 'temporarily block'}</strong> the account {suspend.user?.username}. Are you sure?</div>
      </ConfirmModal>

      <ConfirmModal open={terminate.open} title={terminate.user ? `Terminate ${terminate.user.username}?` : ''} onClose={() => setTerminate({ open: false, user: null, reason_code: 'REQUEST', reason_text: '' })} onConfirm={doTerminate} confirmText="Terminate Account">
        <div className="grid gap-4 pt-2">
          <div>
            <label className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-1 block">Reason code</label>
            <select className="input text-sm w-full bg-slate-50 border-slate-200" value={terminate.reason_code} onChange={e => setTerminate({ ...terminate, reason_code: e.target.value })}>
              <option value="REQUEST">User Request</option>
              <option value="ABUSE">Abuse/Violation</option>
              <option value="FRAUD">Fraud</option>
              <option value="OTHER">Other</option>
            </select>
          </div>
          <div>
            <label className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-1 block">Details (Optional)</label>
            <input className="input text-sm w-full bg-slate-50 border-slate-200" value={terminate.reason_text} onChange={e => setTerminate({ ...terminate, reason_text: e.target.value })} placeholder="Add termination context" />
          </div>
          <div className="p-3 bg-amber-50 border border-amber-200 text-amber-800 text-xs rounded-xl flex items-start gap-2">
            <LuShield size={14} className="mt-0.5" />
            <p>Appointments after today will be cancelled automatically. This destructive action is logged.</p>
          </div>
        </div>
      </ConfirmModal>
    </div>
  )
}
