import { useEffect, useState } from 'react'
import { api } from '../services/api'

export default function AdminUsers(){
  const [list, setList] = useState([])
  const [q, setQ] = useState('')
  const [role, setRole] = useState('')
  const [page, setPage] = useState(1)
  const [total, setTotal] = useState(0)
  const pageSize = 20

  async function load(p=page){
    const res = await api.adminListUsers({ q, role, page:p, page_size: pageSize })
    setList(res.items||[]); setTotal(res.total||0); setPage(res.page||p)
  }
  useEffect(()=>{ load(1) }, [q, role])

  async function promote(uid, newRole){
    await api.adminUpdateUserRole(uid, newRole); load()
  }

  const maxPage = Math.max(1, Math.ceil(total / pageSize))
  return (
    <div className="card">
      <div className="row" style={{justifyContent:'space-between'}}>
        <h3 className="text-lg font-semibold">Users</h3>
        <div className="row">
          <input placeholder="Search username/email" value={q} onChange={e=>setQ(e.target.value)} />
          <select value={role} onChange={e=>setRole(e.target.value)}>
            <option value="">All roles</option>
            <option value="ADMIN">ADMIN</option>
            <option value="DOCTOR">DOCTOR</option>
            <option value="PATIENT">PATIENT</option>
            <option value="PENDING_DOCTOR">PENDING_DOCTOR</option>
          </select>
        </div>
      </div>
      <div style={{overflowX:'auto'}}>
        <table>
          <thead><tr><th>ID</th><th>Username</th><th>Email</th><th>Role</th><th>Actions</th></tr></thead>
          <tbody>
            {list.map(u=> (
              <tr key={u.user_id}>
                <td>{u.user_id}</td>
                <td>{u.username}</td>
                <td>{u.email}</td>
                <td><span className="chip">{u.role}</span></td>
                <td>
                  <div className="row">
                    {u.role!=='ADMIN' && <button className="button" onClick={()=>promote(u.user_id,'ADMIN')}>Make Admin</button>}
                    {u.role!=='DOCTOR' && <button className="button" onClick={()=>promote(u.user_id,'DOCTOR')}>Make Doctor</button>}
                    {u.role!=='PATIENT' && <button className="button" onClick={()=>promote(u.user_id,'PATIENT')}>Make Patient</button>}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="row mt-2" style={{justifyContent:'space-between'}}>
        <div className="muted">Page {page} of {maxPage} ({total} total)</div>
        <div className="row">
          <button className="button" disabled={page<=1} onClick={()=>load(page-1)}>Prev</button>
          <button className="button" disabled={page>=maxPage} onClick={()=>load(page+1)}>Next</button>
        </div>
      </div>
    </div>
  )
}

