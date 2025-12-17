import { useEffect, useState } from 'react'
import { api } from '../services/api'
import ConfirmModal from '../components/ConfirmModal.jsx'

export default function AdminDashboard(){
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
  const [suspend, setSuspend] = useState({ open:false, user:null })
  const [terminate, setTerminate] = useState({ open:false, user:null, reason_code:'REQUEST', reason_text:'' })
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

  useEffect(()=>{ (async()=>{ try{ setOverview(await api.adminOverview()) }catch{} })() },[])
  async function refreshOverview(){ try{ setOverview(await api.adminOverview()) }catch{} }

  async function loadApps(p=appsPage){
    const res = await api.adminListDoctorAppsPaged({ status: appsStatus||undefined, q: appsQ||undefined, page:p, page_size: 20 })
    setApps(res.items||[]); setAppsTotal(res.total||0); setAppsPage(res.page||p)
  }
  useEffect(()=>{ loadApps(1) }, [appsStatus, appsQ])

  async function loadUsers(p=usersPage){
    const res = await api.adminListUsers({ q: usersQ||undefined, role: usersRole||undefined, page:p, page_size:20 })
    setUsers(res.items||[]); setUsersTotal(res.total||0); setUsersPage(res.page||p)
  }
  useEffect(()=>{ if(tab==='users') loadUsers(1) }, [tab, usersQ, usersRole])

  async function loadDocs(p=docsPage){
    const res = await api.adminListDoctors({ q: docsQ||undefined, page:p, page_size:20 })
    setDocs(res.items||[]); setDocsTotal(res.total||0); setDocsPage(res.page||p)
  }
  useEffect(()=>{ if(tab==='doctors') loadDocs(1) }, [tab, docsQ])

  async function loadLogs(p=logsPage){
    const res = await api.adminListAudit({ page:p, page_size:50 })
    setLogs(res.items||[]); setLogsTotal(res.total||0); setLogsPage(res.page||p)
  }
  useEffect(()=>{ if(tab==='audit') loadLogs(1) }, [tab])

  async function loadSettings(){ try{ setSettings(await api.adminGetSettings()) }catch{} }
  useEffect(()=>{ if(tab==='settings') loadSettings() }, [tab])

  const appsMax = Math.max(1, Math.ceil(appsTotal/20))
  const usersMax = Math.max(1, Math.ceil(usersTotal/20))
  const docsMax = Math.max(1, Math.ceil(docsTotal/20))
  const logsMax = Math.max(1, Math.ceil(logsTotal/50))

  const act = async (id, what)=>{
    try{
      if (what==='approve') await api.adminApproveDoctor(id)
      else await api.adminRejectDoctor(id)
      await loadApps()
    }catch(e){ alert(e.message) }
  }

  async function exportApps(){ try{ await api.adminExportDoctorApps({ status: appsStatus||undefined, q: appsQ||undefined }) }catch(e){ alert(e.message) } }

  // Role changes are handled via Applications approval; no direct role changes here per requirements.
  async function doSuspend(){ if(!suspend.user) return; const status = (suspend.user.status==='SUSPENDED')?'ACTIVE':'SUSPENDED'; await api.adminUpdateUserStatus(suspend.user.user_id, status); setSuspend({open:false,user:null}); loadUsers() }
  async function doTerminate(){ if(!terminate.user) return; await api.adminTerminateUser(terminate.user.user_id, terminate.reason_code, terminate.reason_text); setTerminate({open:false,user:null,reason_code:'REQUEST',reason_text:''}); loadUsers() }

  async function saveSettings(e){ e.preventDefault(); const form = new FormData(e.target); const obj={}; for(const [k,v] of form.entries()){ obj[k]=v }; await api.adminSetSettings(obj); loadSettings() }

  return (
    <div>
      <div className="row mb-4" style={{justifyContent:'space-between'}}>
        <h1 className="text-2xl font-semibold">Admin</h1>
        <div className="row">
          <button className={"btn-ghost "+(tab==='apps'?'underline':'')} onClick={()=>setTab('apps')}>Applications</button>
          <button className={"btn-ghost "+(tab==='users'?'underline':'')} onClick={()=>setTab('users')}>Users</button>
          <button className={"btn-ghost "+(tab==='doctors'?'underline':'')} onClick={()=>setTab('doctors')}>Doctors</button>
          <button className={"btn-ghost "+(tab==='settings'?'underline':'')} onClick={()=>setTab('settings')}>Settings</button>
          <button className={"btn-ghost "+(tab==='audit'?'underline':'')} onClick={()=>setTab('audit')}>Audit Log</button>
        </div>
      </div>

      {overview ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-4 mb-6">
          {Object.entries(overview).map(([k,v])=> (
            <div key={k} className="card">
              <div className="muted text-sm capitalize">{k.replace('_',' ')}</div>
              <div className="text-2xl font-semibold">{v}</div>
            </div>
          ))}
        </div>
      ) : (
        <div className="card">Loading metrics…</div>
      )}

      {tab==='apps' && (
        <div className="card">
          <div className="row" style={{justifyContent:'space-between'}}>
            <div className="row">
              <h3 className="text-lg font-semibold">Doctor Applications</h3>
              <span className="chip">{appsTotal} total</span>
            </div>
            <div className="row">
              <input placeholder="Search name/email" value={appsQ} onChange={e=>setAppsQ(e.target.value)} />
              <select value={appsStatus} onChange={e=>setAppsStatus(e.target.value)}>
                <option value="">All</option>
                <option value="PENDING">Pending</option>
                <option value="APPROVED">Approved</option>
                <option value="REJECTED">Rejected</option>
              </select>
              <a className="button" onClick={exportApps}>Export CSV</a>
            </div>
          </div>
          <div style={{overflowX:'auto'}}>
            <table>
              <thead className="sticky top-0 bg-white"><tr><th>ID</th><th>Name</th><th>Email</th><th>Specialization</th><th>License</th><th>Department</th><th>Status</th><th>Actions</th></tr></thead>
              <tbody>
                {apps.map(a=> (
                  <tr key={a.application_id}>
                    <td>{a.application_id}</td>
                    <td>{(a.first_name||'') + ' ' + (a.last_name||'')}</td>
                    <td>{a.email}</td>
                    <td>{a.specialization||'-'}</td>
                    <td>{a.license_no||'-'}</td>
                    <td>{a.department||'-'}</td>
                    <td><span className="chip">{a.status}</span></td>
                    <td>
                      {a.status==='PENDING' && (
                        <div className="row">
                          <button className="button" onClick={()=>act(a.application_id,'approve')}>Approve</button>
                          <button className="button" onClick={()=>act(a.application_id,'reject')}>Reject</button>
                        </div>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="row mt-2" style={{justifyContent:'space-between'}}>
            <div className="muted">Page {appsPage} of {appsMax}</div>
            <div className="row">
              <button className="button" disabled={appsPage<=1} onClick={()=>loadApps(appsPage-1)}>Prev</button>
              <button className="button" disabled={appsPage>=appsMax} onClick={()=>loadApps(appsPage+1)}>Next</button>
            </div>
          </div>
        </div>
      )}

      {tab==='users' && (
        <div className="card">
          <div className="row" style={{justifyContent:'space-between'}}>
            <h3 className="text-lg font-semibold">Users</h3>
            <div className="row">
              <input placeholder="Search username/email" value={usersQ} onChange={e=>setUsersQ(e.target.value)} />
              <select value={usersRole} onChange={e=>setUsersRole(e.target.value)}>
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
                {users.map(u=> (
                  <tr key={u.user_id}>
                    <td>{u.user_id}</td>
                    <td>{u.username}</td>
                    <td>{u.email}</td>
                    <td><span className="chip">{u.role}</span></td>
                    <td>
                      <div className="row">
                        <button className="btn-ghost" onClick={()=>setSuspend({ open:true, user:u })}>Suspend/Unsuspend</button>
                        <button className="button" onClick={()=>setTerminate({ open:true, user:u, reason_code:'REQUEST', reason_text:'' })}>Terminate</button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="row mt-2" style={{justifyContent:'space-between'}}>
            <div className="muted">Page {usersPage} of {usersMax}</div>
            <div className="row">
              <button className="button" disabled={usersPage<=1} onClick={()=>loadUsers(usersPage-1)}>Prev</button>
              <button className="button" disabled={usersPage>=usersMax} onClick={()=>loadUsers(usersPage+1)}>Next</button>
            </div>
          </div>
          <ConfirmModal open={suspend.open} title={suspend.user?`Confirm ${suspend?.user?.status==='SUSPENDED'?'Unsuspend':'Suspend'} ${suspend.user.username}`:''} onClose={()=>setSuspend({open:false,user:null})} onConfirm={doSuspend} confirmText="Confirm">
            <div className="text-sm">This will {suspend?.user?.status==='SUSPENDED'?'reactivate':'temporarily block'} the account. Are you sure?</div>
          </ConfirmModal>
          <ConfirmModal open={terminate.open} title={terminate.user?`Terminate ${terminate.user.username}`:''} onClose={()=>setTerminate({open:false,user:null,reason_code:'REQUEST',reason_text:''})} onConfirm={doTerminate} confirmText="Terminate">
            <div className="grid gap-2">
              <label className="text-sm">Reason code</label>
              <select value={terminate.reason_code} onChange={e=>setTerminate({...terminate, reason_code:e.target.value})}>
                <option value="REQUEST">REQUEST</option>
                <option value="ABUSE">ABUSE</option>
                <option value="FRAUD">FRAUD</option>
                <option value="OTHER">OTHER</option>
              </select>
              <label className="text-sm">Details (optional)</label>
              <input value={terminate.reason_text} onChange={e=>setTerminate({...terminate, reason_text:e.target.value})} placeholder="Add context" />
              <div className="muted">Appointments after today will be cancelled automatically. This action is logged.</div>
            </div>
          </ConfirmModal>
        </div>
      )}

      {tab==='doctors' && (
        <div className="card">
          <div className="row" style={{justifyContent:'space-between'}}>
            <h3 className="text-lg font-semibold">Doctors</h3>
            <div className="row" style={{gap:8}}>
              <input placeholder="Search name/email/specialization" value={docsQ} onChange={e=>setDocsQ(e.target.value)} />
              <button className="button" onClick={async()=>{ await api.adminSyncDoctors(); await loadDocs(1); await refreshOverview() }}>Sync Doctors</button>
            </div>
          </div>
          <div style={{overflowX:'auto'}}>
            <table>
              <thead><tr><th>ID</th><th>User</th><th>Email</th><th>Specialization</th></tr></thead>
              <tbody>
                {docs.map(d=> (
                  <tr key={d.doctor_id}>
                    <td>{d.doctor_id}</td>
                    <td>{d.username}</td>
                    <td>{d.email}</td>
                    <td>{d.specialization||'-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="row mt-2" style={{justifyContent:'space-between'}}>
            <div className="muted">Page {docsPage} of {docsMax}</div>
            <div className="row">
              <button className="button" disabled={docsPage<=1} onClick={()=>loadDocs(docsPage-1)}>Prev</button>
              <button className="button" disabled={docsPage>=docsMax} onClick={()=>loadDocs(docsPage+1)}>Next</button>
            </div>
          </div>
        </div>
      )}

      {tab==='settings' && (
        <div className="card">
          <h3 className="text-lg font-semibold mb-2">System Settings</h3>
          <form onSubmit={saveSettings} className="grid gap-3" style={{gridTemplateColumns:'1fr 1fr'}}>
            <div>
              <label className="text-sm">LESION_MALIGNANT_THRESHOLD</label>
              <input name="LESION_MALIGNANT_THRESHOLD" defaultValue={settings['LESION_MALIGNANT_THRESHOLD']||''} />
            </div>
            <div>
              <label className="text-sm">JWT_LEEWAY_SECONDS</label>
              <input name="JWT_LEEWAY_SECONDS" defaultValue={settings['JWT_LEEWAY_SECONDS']||''} />
            </div>
            <div>
              <label className="text-sm">FRONTEND_ORIGINS</label>
              <input name="FRONTEND_ORIGINS" defaultValue={settings['FRONTEND_ORIGINS']||''} />
            </div>
            <div className="col-span-2">
              <button className="button" type="submit">Save</button>
            </div>
          </form>
        </div>
      )}

      {tab==='audit' && (
        <div className="card">
          <div className="row" style={{justifyContent:'space-between'}}>
            <h3 className="text-lg font-semibold">Audit Log</h3>
            <div className="muted">Page {logsPage} of {logsMax} ({logsTotal} total)</div>
          </div>
          <div style={{overflowX:'auto'}}>
            <table>
              <thead><tr><th>ID</th><th>User</th><th>Action</th><th>Meta</th><th>When</th></tr></thead>
              <tbody>
                {logs.map(r=> (
                  <tr key={r.id}>
                    <td>{r.id}</td>
                    <td>{r.user_id||'-'}</td>
                    <td>{r.action}</td>
                    <td>{r.meta||''}</td>
                    <td>{new Date(r.created_at).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="row mt-2" style={{justifyContent:'space-between'}}>
            <div />
            <div className="row">
              <button className="button" disabled={logsPage<=1} onClick={()=>loadLogs(logsPage-1)}>Prev</button>
              <button className="button" disabled={logsPage>=logsMax} onClick={()=>loadLogs(logsPage+1)}>Next</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
