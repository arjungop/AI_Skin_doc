import { useEffect, useState } from 'react'
import { api } from '../services/api'
import ConfirmModal from '../components/ConfirmModal.jsx'
import DetailsDrawer from '../components/DetailsDrawer.jsx'

export default function AdminTransactions(){
  const [items, setItems] = useState([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [q, setQ] = useState('')
  const [status, setStatus] = useState('')
  const [method, setMethod] = useState('')
  const [category, setCategory] = useState('')
  const [start, setStart] = useState('')
  const [end, setEnd] = useState('')
  const [amountMin, setAmountMin] = useState('')
  const [amountMax, setAmountMax] = useState('')
  const [sel, setSel] = useState(null)
  const [confirm, setConfirm] = useState({ open:false, id:null, newStatus:'', reason:'' })

  async function load(p=page){
    const res = await api.adminListTransactions({ q, status, method, category, start, end, amount_min: amountMin||undefined, amount_max: amountMax||undefined, page:p, page_size: 30 })
    setItems(res.items||[]); setTotal(res.total||0); setPage(res.page||p)
  }
  useEffect(()=>{ load(1) }, [q, status, method, category, start, end, amountMin, amountMax])

  const maxPage = Math.max(1, Math.ceil(total/30))

  function openConfirm(id, newStatus){ setConfirm({ open:true, id, newStatus, reason:'' }) }
  async function doConfirm(){
    await api.updateTransactionStatus(confirm.id, confirm.newStatus, confirm.reason||'')
    setConfirm({ open:false, id:null, newStatus:'', reason:'' }); load()
  }

  function openDetails(t){ setSel(t) }

  return (
    <div>
      <div className="row mb-3" style={{justifyContent:'space-between', flexWrap:'wrap', gap:8}}>
        <h1 className="text-2xl font-semibold">Transactions (Admin)</h1>
        <div className="row" style={{gap:8}}>
          <input placeholder="Search (email/id/reference)" value={q} onChange={e=>setQ(e.target.value)} />
          <input type="date" value={start} onChange={e=>setStart(e.target.value)} />
          <input type="date" value={end} onChange={e=>setEnd(e.target.value)} />
          <select value={status} onChange={e=>setStatus(e.target.value)}>
            <option value="">All</option>
            <option value="pending">Pending</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
            <option value="refunded">Refunded</option>
          </select>
          <input placeholder="Method" value={method} onChange={e=>setMethod(e.target.value)} style={{width:120}} />
          <select value={category} onChange={e=>setCategory(e.target.value)}>
            <option value="">All categories</option>
            <option value="consultation">Consultation</option>
            <option value="procedure">Procedure</option>
            <option value="pharmacy">Pharmacy</option>
            <option value="general">General</option>
          </select>
          <input placeholder="Min ₹" value={amountMin} onChange={e=>setAmountMin(e.target.value)} style={{width:90}} />
          <input placeholder="Max ₹" value={amountMax} onChange={e=>setAmountMax(e.target.value)} style={{width:90}} />
          <a className="button" href={`${api.base}/transactions/export.csv`} target="_blank" rel="noreferrer">Export CSV</a>
        </div>
      </div>

      <div className="card" style={{overflowX:'auto'}}>
        <table>
          <thead className="sticky top-0"><tr><th>ID</th><th>User</th><th>Email</th><th>Amount</th><th>Status</th><th>Method</th><th>Reference</th><th>Created</th><th>Actions</th></tr></thead>
          <tbody>
            {items.map(t=> (
              <tr key={t.transaction_id}>
                <td>{t.transaction_id}</td>
                <td>{t.username||'-'}</td>
                <td>{t.email||'-'}</td>
                <td>₹ {t.amount}</td>
                <td><span className="chip">{t.status}</span></td>
                <td>{t.method||'-'}</td>
                <td>{t.reference||'-'}</td>
                <td>{new Date(t.created_at).toLocaleString()}</td>
                <td>
                  <div className="row">
                    <button className="btn-ghost" onClick={()=>openDetails(t)}>Details</button>
                    <button className="button" onClick={()=>openConfirm(t.transaction_id,'completed')}>Complete</button>
                    <button className="button" onClick={()=>openConfirm(t.transaction_id,'failed')}>Fail</button>
                    <button className="button" onClick={()=>openConfirm(t.transaction_id,'refunded')}>Refund</button>
                    <a className="button" href={`${api.base}/transactions/${t.transaction_id}/receipt.pdf`} target="_blank" rel="noreferrer">PDF</a>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="row mt-2" style={{justifyContent:'space-between'}}>
        <div className="muted">Page {page} of {Math.max(1, Math.ceil(total/30))} ({total} total)</div>
        <div className="row">
          <button className="button" disabled={page<=1} onClick={()=>load(page-1)}>Prev</button>
          <button className="button" disabled={page>=maxPage} onClick={()=>load(page+1)}>Next</button>
        </div>
      </div>

      <ConfirmModal open={confirm.open} title={`Confirm ${confirm.newStatus}`} onClose={()=>setConfirm({open:false,id:null,newStatus:'',reason:''})} onConfirm={doConfirm} confirmText="Confirm">
        <div className="grid gap-2">
          {(confirm.newStatus==='failed' || confirm.newStatus==='refunded') && (
            <>
              <label className="text-sm">Reason (required)</label>
              <input value={confirm.reason} onChange={e=>setConfirm({...confirm, reason:e.target.value})} placeholder="Type reason" />
            </>
          )}
        </div>
      </ConfirmModal>

      <DetailsDrawer open={!!sel} title={sel?`Transaction #${sel.transaction_id}`:''} onClose={()=>setSel(null)}>
        {sel && (
          <div className="stack">
            <div><b>User:</b> {sel.username} ({sel.email})</div>
            <div><b>Amount:</b> ₹ {sel.amount}</div>
            <div><b>Status:</b> {sel.status}</div>
            <div><b>Method:</b> {sel.method||'-'}</div>
            <div><b>Reference:</b> {sel.reference||'-'}</div>
            <div><b>Created:</b> {new Date(sel.created_at).toLocaleString()}</div>
            <div className="mt-2"><a className="button" href={`${api.base}/transactions/${sel.transaction_id}/receipt.pdf`} target="_blank" rel="noreferrer">Receipt PDF</a></div>
            <div className="muted">Audit trail (most recent): open Admin → Audit Log</div>
          </div>
        )}
      </DetailsDrawer>
    </div>
  )
}

