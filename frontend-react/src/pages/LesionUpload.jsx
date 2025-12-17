import { useState, useEffect } from 'react'
import { api } from '../services/api'

export default function LesionUpload(){
  const [file, setFile] = useState(null)
  const [res, setRes] = useState(null)
  const [diag, setDiag] = useState('')
  const [report, setReport] = useState(null)
  const patientId = parseInt(localStorage.getItem('patient_id'))
  const role = (localStorage.getItem('role')||'').toUpperCase()
  const [overridePatientId, setOverridePatientId] = useState('')
  const [error, setError] = useState('')
  const [patientSearch, setPatientSearch] = useState('')
  const [patientOptions, setPatientOptions] = useState([])
  const [doctorSearch, setDoctorSearch] = useState('')
  const [doctorOptions, setDoctorOptions] = useState([])
  const [selectedDoctor, setSelectedDoctor] = useState('')
  const [tab, setTab] = useState('upload')
  const [highSensitivity, setHighSensitivity] = useState(true)

  const onSubmit = async (e) => {
    e.preventDefault()
    setRes(null); setDiag(''); setError(''); setReport(null)
    const pid = role==='ADMIN' ? parseInt(overridePatientId||'0') : patientId
    if (!pid) { setError('Select a patient first'); return }
    if (!file) { setError('Select an image'); return }
    try {
      const result = await api.predictLesion(pid, file, { sensitivity: highSensitivity ? 'high' : undefined })
      setRes(result)
    } catch (err) { setError(err.message||'Upload failed') }
  }

  const runDiagnosis = async () => {
    if (!res?.lesion_id) return
    try{
      const pid = role==='ADMIN' ? parseInt(overridePatientId||'0') : patientId
      // Create structured report (compassionate details)
      const rep = await api.createDiagnosisReport(res.lesion_id, pid)
      setReport(rep)
      setDiag(rep.details || '')
    }catch(err){ setDiag(String(err.message||err)) }
  }

  // Admin: search patients for selection
  async function fetchPatients(q){
    try{ setPatientOptions(await api.listPatients(q||'')) }catch{}
  }
  async function fetchDoctors(q){
    try{ setDoctorOptions(await api.listDoctors(q||'')) }catch{}
  }

  useEffect(()=>{
    if (role!=='PATIENT'){
      const h = setTimeout(()=> fetchPatients(patientSearch), 250)
      return ()=> clearTimeout(h)
    }
  }, [patientSearch])
  useEffect(()=>{
    const h = setTimeout(()=> fetchDoctors(doctorSearch), 250)
    return ()=> clearTimeout(h)
  }, [doctorSearch])

  return (
    <div>
      <div className="row" style={{justifyContent:'space-between', alignItems:'center'}}>
        <h1>Lesion Classification</h1>
        <div className="row">
          <button className={"btn-ghost "+(tab==='upload'?'underline':'')} onClick={()=>setTab('upload')}>Upload</button>
          <button className={"btn-ghost "+(tab==='history'?'underline':'')} onClick={()=>setTab('history')}>History</button>
        </div>
      </div>
      {tab==='upload' && (
      <form onSubmit={onSubmit} className="card stack">
        {role!=='PATIENT' && (
          <div className="grid gap-2" style={{gridTemplateColumns:'1fr 260px'}}>
            <div>
              <label className="text-sm">Search patient</label>
              <input placeholder="Name, email or ID" value={patientSearch} onChange={e=>setPatientSearch(e.target.value)} />
            </div>
            <div>
              <label className="text-sm">Select patient</label>
              <select value={overridePatientId} onChange={e=>setOverridePatientId(e.target.value)}>
                <option value="">-- choose --</option>
                {patientOptions.map(p=> (
                  <option key={p.patient_id} value={p.patient_id}>
                    {p.first_name} {p.last_name} (P{p.patient_id}) — {p.email}
                  </option>
                ))}
              </select>
            </div>
          </div>
        )}
        <input type="file" accept="image/*" onChange={e=>setFile(e.target.files?.[0])} />
        <label className="flex items-center gap-2"><input type="checkbox" checked={highSensitivity} onChange={e=>setHighSensitivity(e.target.checked)} /> High sensitivity (more cautious)</label>
        <button className="button" type="submit">Upload & Predict</button>
        {error && <div className="text-error text-sm">{error}</div>}
      </form>
      )}

      {res && (
        <div className="card stack">
          <div><b>Prediction:</b> {res.prediction}</div>
          {typeof res.risk_score === 'number' && (
            <div><b>Risk score:</b> {(res.risk_score*100).toFixed(1)}%</div>
          )}
          <div><b>Lesion ID:</b> {res.lesion_id}</div>
          {res.explain_url && (
            <div>
              <div className="muted">Model explanation (Grad‑CAM)</div>
              <img src={res.explain_url} alt="Explanation heatmap" style={{maxWidth:'100%', borderRadius:8, border:'1px solid #E2E8F0'}} />
            </div>
          )}
          <button className="button" onClick={runDiagnosis}>Get AI Diagnosis</button>
          {diag && (
            <div className="stack">
              <h3>AI Guidance</h3>
              <pre className="pre">{diag}</pre>
              <div className="grid gap-2" style={{gridTemplateColumns:'1fr 260px'}}>
                <div>
                  <label className="text-sm">Search doctor</label>
                  <input placeholder="Name or specialization" value={doctorSearch} onChange={e=>setDoctorSearch(e.target.value)} />
                </div>
                <div>
                  <label className="text-sm">Send to doctor</label>
                  <div className="row">
                    <select value={selectedDoctor} onChange={e=>setSelectedDoctor(e.target.value)}>
                      <option value="">-- choose --</option>
                      {doctorOptions.map(d=> (
                        <option key={d.doctor_id} value={d.doctor_id}>{d.username} — {d.specialization||'General'}</option>
                      ))}
                    </select>
                    <button className="button" disabled={!report||!selectedDoctor} onClick={async()=>{ if(!report||!selectedDoctor) return; await api.sendDiagnosisReport(report.report_id, parseInt(selectedDoctor)); alert('Sent to doctor'); }}>Send</button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {tab==='history' && (
        <History patientId={role==='ADMIN'? (overridePatientId||patientId) : patientId} />
      )}
    </div>
  )
}

function History({ patientId }){
  const [items, setItems] = useState([])
  useEffect(()=>{ (async()=>{ try{ setItems(await api.listDiagnosisReports(patientId)) }catch{} })() },[patientId])
  return (
    <div className="card" style={{overflowX:'auto'}}>
      <table>
        <thead><tr><th>When</th><th>Prediction</th><th>Summary</th><th>Details</th></tr></thead>
        <tbody>
          {items.map(r=> (
            <tr key={r.report_id}>
              <td>{new Date(r.created_at).toLocaleString()}</td>
              <td>{r.prediction||'-'}</td>
              <td>{r.summary||'-'}</td>
              <td><details><summary>View</summary><pre className="pre" style={{whiteSpace:'pre-wrap'}}>{r.details}</pre></details></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
