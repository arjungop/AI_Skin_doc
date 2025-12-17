import { useEffect, useState } from 'react'
import { api } from '../services/api'
import { useToast } from '../components/Toast.jsx'

export default function Transactions(){
  const [list, setList] = useState([])
  const [amount, setAmount] = useState('')
  const [method, setMethod] = useState('UPI')
  const [reference, setReference] = useState('')
  const [note, setNote] = useState('')
  const [filter, setFilter] = useState('')
  const [search, setSearch] = useState('')
  const [refFilter, setRefFilter] = useState('')
  const [start, setStart] = useState('')
  const [end, setEnd] = useState('')
  const [category, setCategory] = useState('')
  const [methodFilter, setMethodFilter] = useState('')
  const [amountMin, setAmountMin] = useState('')
  const [amountMax, setAmountMax] = useState('')
  const [monthly, setMonthly] = useState([])
  const [summary, setSummary] = useState(null)
  const uid = parseInt(localStorage.getItem('user_id'))
  const role = (localStorage.getItem('role')||'').toUpperCase()
  const { push } = useToast()

  async function load(){
    try{
      const isTxnId = /^\d+$/.test(refFilter.trim())
      const q = [search, isTxnId ? '' : refFilter].filter(Boolean).join(' ').trim() || undefined
      const transaction_id = isTxnId ? parseInt(refFilter.trim()) : undefined
      const common = { status: filter||undefined, q, start: start||undefined, end: end||undefined, category: category||undefined, method: methodFilter||undefined, amount_min: amountMin||undefined, amount_max: amountMax||undefined }
      if (role==='ADMIN'){
        const res = await api.adminListTransactions({ ...common, page:1, page_size:200 })
        setList(res.items||[])
      } else {
        setList(await api.listTransactions({ ...common, transaction_id }))
      }
      setSummary(await api.transactionsSummary(role==='ADMIN'?undefined:uid))
      setMonthly(await api.monthly(12))
    }catch{}
  }
  useEffect(()=>{ load() },[filter, search, start, end, category, methodFilter, amountMin, amountMax])
  useEffect(()=>{ load() },[refFilter])

  async function submit(e){
    e.preventDefault()
    if (!uid) return push('Login required','error')
    const val = parseFloat(amount)
    if (isNaN(val) || val <= 0) return push('Enter a valid amount','error')
    try{
      const t = await api.createTransaction({ user_id: uid, amount: val, status: 'pending', category: category||'general' })
      await api.setTransactionMeta(t.transaction_id, { method, reference, note })
      setAmount(''); push('Payment recorded','success'); load()
    }catch(err){ push(err.message,'error') }
  }

  return (
    <div>
      <div className="row" style={{justifyContent:'space-between'}}>
        <h1>Transactions</h1>
        <div className="row" style={{gap:'8px', alignItems:'flex-end', flexWrap:'wrap'}}>
          <div style={{display:'flex', flexDirection:'column', flex:'1 1 320px', minWidth:280}}>
            <label className="muted" style={{fontSize:12, marginBottom:4}}>Search reference or note</label>
            <input placeholder="Search reference or note" value={search} onChange={e=>setSearch(e.target.value)} style={{width:'100%'}} />
          </div>
          <div style={{display:'flex', flexDirection:'column', flex:'0 1 200px', minWidth:180}}>
            <label className="muted" style={{fontSize:12, marginBottom:4}}>Ref / Txn ID</label>
            <input
              placeholder="e.g. TXN-123 or 456"
              value={refFilter}
              onChange={e=>setRefFilter(e.target.value)}
              style={{width:'100%'}}
              title="Enter exact transaction ID (numbers) or reference text"
            />
          </div>
          <div style={{display:'flex', flexDirection:'column', flex:'0 1 180px', minWidth:160}}>
            <label className="muted" style={{fontSize:12, marginBottom:4}}>From (dd/mm/yyyy)</label>
            <input
              type="date"
              value={start}
              onChange={e=>setStart(e.target.value)}
              placeholder="dd/mm/yyyy"
              title="Use dd/mm/yyyy or yyyy-mm-dd"
            />
          </div>
          <div style={{display:'flex', flexDirection:'column', flex:'0 1 180px', minWidth:160}}>
            <label className="muted" style={{fontSize:12, marginBottom:4}}>To (dd/mm/yyyy)</label>
            <input
              type="date"
              value={end}
              onChange={e=>setEnd(e.target.value)}
              placeholder="dd/mm/yyyy"
              title="Use dd/mm/yyyy or yyyy-mm-dd"
            />
          </div>
          <select value={category} onChange={e=>setCategory(e.target.value)} style={{flex:'0 1 180px'}}>
            <option value="">All categories</option>
            <option value="consultation">Consultation</option>
            <option value="procedure">Procedure</option>
            <option value="pharmacy">Pharmacy</option>
            <option value="general">General</option>
          </select>
          <input placeholder="Method (UPI/Card/Cash)" value={methodFilter} onChange={e=>setMethodFilter(e.target.value)} style={{flex:'0 1 160px'}} />
          <input placeholder="Min ₹" value={amountMin} onChange={e=>setAmountMin(e.target.value)} style={{width:100}} />
          <input placeholder="Max ₹" value={amountMax} onChange={e=>setAmountMax(e.target.value)} style={{width:100}} />
          <select value={filter} onChange={e=>setFilter(e.target.value)} style={{flex:'0 1 160px'}}>
            <option value="">All</option>
            <option value="pending">Pending</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
            <option value="refunded">Refunded</option>
          </select>
        </div>
      </div>

      {/* Summary cards */}
      {summary && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
          <div className="card"><div className="muted">Pending</div><div className="text-2xl font-semibold">₹ {summary.pending.toFixed(2)}</div></div>
          <div className="card"><div className="muted">Completed</div><div className="text-2xl font-semibold">₹ {summary.completed.toFixed(2)}</div></div>
          <div className="card"><div className="muted">Failed</div><div className="text-2xl font-semibold">₹ {summary.failed.toFixed(2)}</div></div>
          <div className="card"><div className="muted">Count</div><div className="text-2xl font-semibold">{summary.count}</div></div>
        </div>
      )}

      {/* Create payment (admin only) */}
      {role==='ADMIN' ? (
        <form onSubmit={submit} className="card grid grid-cols-1 md:grid-cols-6 gap-3 mb-4">
          <input type="number" step="0.01" placeholder="Amount (₹)" value={amount} onChange={e=>setAmount(e.target.value)} required />
          <select value={category} onChange={e=>setCategory(e.target.value)}>
            <option value="consultation">Consultation</option>
            <option value="procedure">Procedure</option>
            <option value="pharmacy">Pharmacy</option>
            <option value="general">General</option>
          </select>
          <select value={method} onChange={e=>setMethod(e.target.value)}>
            <option>UPI</option>
            <option>Card</option>
            <option>Cash</option>
            <option>Bank Transfer</option>
            <option>Other</option>
          </select>
          <input placeholder="Reference (Txn ID / UPI)" value={reference} onChange={e=>setReference(e.target.value)} />
          <input placeholder="Note (optional)" value={note} onChange={e=>setNote(e.target.value)} />
          <button className="button" type="submit">Record Payment</button>
        </form>
      ) : (
        <div className="card mb-4"><div className="muted">Only admins can record payments.</div></div>
      )}

      {/* Table */}
      <div className="card" style={{overflowX:'auto'}}>
        <table>
          <thead className="sticky top-0"><tr><th>ID</th><th>Amount</th><th>Category</th><th>Method</th><th>Reference</th><th>Status</th><th>Created</th><th>Receipt</th>{role==='ADMIN' && <th>Actions</th>}</tr></thead>
          <tbody>
            {list.map(t=> (
              <tr key={t.transaction_id}>
                <td>{t.transaction_id}</td>
                <td>₹ {t.amount}</td>
                <td>{t.category||'general'}</td>
                <td>{t.method||'-'}</td>
                <td>{t.reference||'-'}</td>
                <td><span className="chip">{t.status}</span></td>
                <td>{new Date(t.created_at).toLocaleString()}</td>
                <td>
                  <div className="row">
                    <a className="button" href={`${api.base}/transactions/${t.transaction_id}/receipt.pdf`} target="_blank" rel="noreferrer">PDF</a>
                  </div>
                </td>
                {role==='ADMIN' && (
                  <td>
                    <div className="row">
                      <button className="button" onClick={async()=>{ await api.updateTransactionStatus(t.transaction_id,'completed',''); load() }}>Mark Completed</button>
                      <button className="button" onClick={async()=>{ const r=prompt('Reason for failure?')||''; await api.updateTransactionStatus(t.transaction_id,'failed', r); load() }}>Mark Failed</button>
                      <button className="button" onClick={async()=>{ const r=prompt('Reason for refund?')||''; await api.updateTransactionStatus(t.transaction_id,'refunded', r); load() }}>Refund</button>
                    </div>
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Export buttons */}
      <div className="row mt-3">
        <a className="button" href={`${api.base}/transactions/export.csv`} target="_blank" rel="noreferrer">Export CSV</a>
        <a className="button" href={`${api.base}/transactions/export.pdf`} target="_blank" rel="noreferrer">Export PDF</a>
      </div>

      {/* Monthly sparkline */}
      {monthly.length>0 && (
        <div className="card mt-4">
          <div className="muted">Last 12 months</div>
          <Sparkline data={monthly.map(m=>m.total)} labels={monthly.map(m=>m.month)} />
        </div>
      )}
    </div>
  )
}

function Sparkline({ data, labels }){
  const w = 400, h = 80, pad = 6
  const max = Math.max(...data, 1)
  const step = (w - pad*2) / Math.max(data.length-1, 1)
  const points = data.map((v,i)=> [pad + i*step, h - pad - (v/max)*(h-pad*2)])
  const d = points.map((p,i)=> (i? 'L':'M') + p[0] + ' ' + p[1]).join(' ')
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <path d={d} fill="none" stroke="#0078D4" strokeWidth="2" />
      {points.map((p,i)=> <circle key={i} cx={p[0]} cy={p[1]} r="2" fill="#6B5BFF" />)}
    </svg>
  )
}
