import React, { useEffect, useMemo, useState } from 'react'
import { api } from '../services/api.js'

type Txn = { transaction_id:number; amount:number; status:string; created_at:string; method?:string; reference?:string; note?:string }

export default function DoctorTransactions(){
  const [rows, setRows] = useState<Txn[]>([])
  const [q, setQ] = useState('')
  const [status, setStatus] = useState('')

  async function load(){
    const data = await api.listTransactions({ q, status })
    setRows(data||[])
  }
  useEffect(()=>{ load() },[])

  const totalCompleted = useMemo(()=> rows.filter(r=>r.status==='completed').reduce((s,r)=> s + (r.amount||0), 0),[rows])

  async function exportCSV(){
    const res = await fetch(`${api.base}/transactions/export.csv`, { headers: { ...((window as any).HEADERS||{}), ...(localStorage.getItem('access_token')? { 'Authorization': `Bearer ${String(localStorage.getItem('access_token')).replace(/^\"|\"$/g,'')}` } : {}) } })
    const blob = await res.blob(); const url = URL.createObjectURL(blob)
    const a = document.createElement('a'); a.href = url; a.download = 'transactions.csv'; a.click(); URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
        <div className="card"><div className="muted">Revenue (completed)</div><div className="text-2xl font-semibold">₹ {totalCompleted.toFixed(2)}</div></div>
        <div className="card"><div className="muted">Payments</div><div className="text-2xl font-semibold">{rows.length}</div></div>
        <div className="card"><div className="muted">Pending</div><div className="text-2xl font-semibold">{rows.filter(r=>r.status==='pending').length}</div></div>
        <div className="card"><div className="muted">Refunded</div><div className="text-2xl font-semibold">{rows.filter(r=>r.status==='refunded').length}</div></div>
      </div>

      <div className="flex items-center gap-2">
        <input className="w-64" placeholder="Search reference/note/method" value={q} onChange={e=>setQ(e.target.value)} />
        <select className="w-40" value={status} onChange={e=>setStatus(e.target.value)}>
          <option value="">All</option>
          <option value="pending">Pending</option>
          <option value="completed">Completed</option>
          <option value="failed">Failed</option>
          <option value="refunded">Refunded</option>
        </select>
        <button className="btn-primary" onClick={load}>Filter</button>
        <button className="btn-ghost" onClick={exportCSV}>Export CSV</button>
      </div>

      <div className="card">
        <table>
          <thead>
            <tr><th>ID</th><th>Amount</th><th>Status</th><th>Created</th><th>Method</th><th>Reference</th></tr>
          </thead>
          <tbody>
            {rows.map(r=> (
              <tr key={r.transaction_id}>
                <td>#{r.transaction_id}</td>
                <td>₹ {r.amount.toFixed(2)}</td>
                <td>{r.status}</td>
                <td>{new Date(r.created_at).toLocaleString()}</td>
                <td>{r.method||'-'}</td>
                <td>{r.reference||'-'}</td>
              </tr>
            ))}
            {!rows.length && <tr><td className="muted" colSpan={6}>No transactions</td></tr>}
          </tbody>
        </table>
      </div>
    </div>
  )
}
