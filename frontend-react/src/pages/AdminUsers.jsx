import { useEffect, useState } from 'react'
import { api } from '../services/api'
import { Card } from '../components/Card'
import { LuSearch, LuChevronLeft, LuChevronRight, LuShieldAlert, LuUserCheck } from 'react-icons/lu'
import { useToast } from '../components/Toast'

export default function AdminUsers() {
  const [list, setList] = useState([])
  const [q, setQ] = useState('')
  const [role, setRole] = useState('')
  const [page, setPage] = useState(1)
  const [total, setTotal] = useState(0)
  const pageSize = 20
  const { push } = useToast()

  async function load(p = page) {
    try {
      const res = await api.adminListUsers({ q, role, page: p, page_size: pageSize })
      setList(res.items || []); setTotal(res.total || 0); setPage(res.page || p)
    } catch (e) {
      push('Failed to load users: ' + (e.message || 'Error'), 'error')
    }
  }
  useEffect(() => { load(1) }, [q, role])

  async function promote(uid, newRole) {
    try {
      await api.adminUpdateUserRole(uid, newRole);
      push(`User ID ${uid} promoted to ${newRole}`, 'success')
      load()
    } catch (e) {
      push(`Failed to promote: ${e.message}`, 'error')
    }
  }

  const maxPage = Math.max(1, Math.ceil(total / pageSize))

  return (
    <div className="pb-12">
      <header className="mb-8 p-1">
        <h1 className="text-3xl font-bold text-slate-900 tracking-tight flex items-center gap-3">
          <LuShieldAlert className="text-indigo-500" /> Advanced User Management
        </h1>
        <p className="text-slate-500 mt-2 max-w-2xl">Directly query and escalate user roles (promote patients to doctors, grant admin rights). Warning: These actions bypass standard application flows.</p>
      </header>

      <Card className="overflow-hidden border border-slate-200/60 shadow-soft-md">
        <div className="p-6 border-b border-slate-100 flex flex-col md:flex-row md:items-center justify-between gap-4 bg-white">
          <div>
            <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2">Global Users <span className="bg-primary-50 text-primary-600 text-xs px-2 py-0.5 rounded-full">{total}</span></h3>
          </div>
          <div className="flex items-center gap-3">
            <div className="relative">
              <LuSearch className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={16} />
              <input className="input pl-9 w-full md:w-64 text-sm bg-slate-50" placeholder="Search username/email" value={q} onChange={e => setQ(e.target.value)} />
            </div>
            <select className="input text-sm bg-slate-50 font-medium cursor-pointer" value={role} onChange={e => setRole(e.target.value)}>
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
              <tr><th className="p-4 pl-6">ID</th><th className="p-4">Username</th><th className="p-4">Email</th><th className="p-4">Role</th><th className="p-4 pr-6 text-right">Escalation Actions</th></tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {list.map(u => (
                <tr key={u.user_id} className="hover:bg-slate-50/50 transition-colors">
                  <td className="p-4 pl-6 font-mono text-xs">{u.user_id}</td>
                  <td className="p-4 font-semibold text-slate-800">{u.username}</td>
                  <td className="p-4">{u.email}</td>
                  <td className="p-4">
                    <span className="px-2 py-1 rounded-md text-[10px] bg-indigo-50 text-indigo-700 border border-indigo-100 font-bold uppercase tracking-widest">{u.role}</span>
                  </td>
                  <td className="p-4 pr-6 text-right">
                    <div className="flex items-center justify-end gap-2">
                      {u.role !== 'ADMIN' && <button className="px-3 py-1.5 text-xs font-semibold text-purple-700 bg-purple-50 hover:bg-purple-100 border border-purple-100 rounded-lg transition-colors flex items-center gap-1.5" onClick={() => promote(u.user_id, 'ADMIN')}><LuShieldAlert size={12} /> Make Admin</button>}
                      {u.role !== 'DOCTOR' && <button className="px-3 py-1.5 text-xs font-semibold text-emerald-700 bg-emerald-50 hover:bg-emerald-100 border border-emerald-100 rounded-lg transition-colors flex items-center gap-1.5" onClick={() => promote(u.user_id, 'DOCTOR')}><LuUserCheck size={12} /> Make Doctor</button>}
                      {u.role !== 'PATIENT' && <button className="px-3 py-1.5 text-xs font-semibold text-sky-700 bg-sky-50 hover:bg-sky-100 border border-sky-100 rounded-lg transition-colors flex items-center gap-1.5" onClick={() => promote(u.user_id, 'PATIENT')}><LuUserCheck size={12} /> Make Patient</button>}
                    </div>
                  </td>
                </tr>
              ))}
              {list.length === 0 && <tr><td colSpan="5" className="p-8 text-center text-slate-400">No users found.</td></tr>}
            </tbody>
          </table>
        </div>
        <div className="p-4 bg-slate-50/50 border-t border-slate-100 flex items-center justify-between text-sm">
          <div className="text-slate-500 font-medium pl-2">Page <span className="text-slate-900 font-bold">{page}</span> of {maxPage}</div>
          <div className="flex gap-2">
            <button className="p-2 rounded-lg bg-white border border-slate-200 text-slate-600 hover:bg-slate-50 hover:text-slate-900 disabled:opacity-50 disabled:cursor-not-allowed transition-colors" disabled={page <= 1} onClick={() => load(page - 1)}><LuChevronLeft size={16} /></button>
            <button className="p-2 rounded-lg bg-white border border-slate-200 text-slate-600 hover:bg-slate-50 hover:text-slate-900 disabled:opacity-50 disabled:cursor-not-allowed transition-colors" disabled={page >= maxPage} onClick={() => load(page + 1)}><LuChevronRight size={16} /></button>
          </div>
        </div>
      </Card>
    </div>
  )
}
