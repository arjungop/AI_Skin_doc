import React, { useEffect, useMemo, useState } from 'react'
import { api } from '../services/api.js'

type Txn = { transaction_id:number; amount:number; status:string; created_at:string; method?:string; reference?:string; note?:string }

function StatusBadge({ status }: { status: string }){
  const cls =
    status === 'completed' ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' :
    status === 'pending'   ? 'bg-amber-50 text-amber-700 border border-amber-200' :
    status === 'failed'    ? 'bg-red-50 text-red-600 border border-red-200' :
    status === 'refunded'  ? 'bg-purple-50 text-purple-700 border border-purple-200' :
    'bg-slate-100 text-slate-600 border border-slate-200'
  return <span className={`px-2 py-1 rounded-full text-xs font-medium capitalize ${cls}`}>{status}</span>
}

export default function DoctorTransactions(){
  const [rows, setRows] = useState<Txn[]>([])
  const [q, setQ] = useState('')
  const [status, setStatus] = useState('')
  const [loading, setLoading] = useState(true)

  async function load(){
    setLoading(true)
    try{
      const data = await api.listTransactions({ q, status })
      setRows(data||[])
    }catch{ setRows([]) }
    finally{ setLoading(false) }
  }
  useEffect(()=>{ load() },[])

  const totalCompleted = useMemo(()=>
    rows.filter(r=>r.status==='completed').reduce((s,r)=> s + (r.amount||0), 0)
  ,[rows])

  async function exportCSV(){
    const token = localStorage.getItem('access_token')
    const headers: Record<string,string> = token
      ? { 'Authorization': `Bearer ${String(token).replace(/^\"|\"$/g,'')}` }
      : {}
    const res = await fetch(`${(api as any).base}/transactions/export.csv`, { headers })
    const blob = await res.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a'); a.href = url; a.download = 'transactions.csv'; a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-4">
      <header>
        <h1 className="text-2xl font-semibold">Transactions</h1>
        <p className="text-sm text-slate-500">Billing history and payment records.</p>
      </header>

      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
        <div className="card">
          <div className="text-xs text-slate-500 uppercase tracking-wide mb-1">Revenue</div>
          <div className="text-2xl font-semibold text-emerald-600">₹ {totalCompleted.toFixed(2)}</div>
        </div>
        <div className="card">
          <div className="text-xs text-slate-500 uppercase tracking-wide mb-1">Total Payments</div>
          <div className="text-2xl font-semibold">{rows.length}</div>
        </div>
        <div className="card">
          <div className="text-xs text-slate-500 uppercase tracking-wide mb-1">Pending</div>
          <div className="text-2xl font-semibold text-amber-600">{rows.filter(r=>r.status==='pending').length}</div>
        </div>
        <div className="card">
          <div className="text-xs text-slate-500 uppercase tracking-wide mb-1">Refunded</div>
          <div className="text-2xl font-semibold text-purple-600">{rows.filter(r=>r.status==='refunded').length}</div>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <input
          className="input w-64"
          placeholder="Search reference / note / method"
          value={q}
          onChange={e=>setQ(e.target.value)}
          onKeyDown={e=>{ if(e.key==='Enter') load() }}
        />
        <select
          className="input w-40"
          value={status}
          onChange={e=>setStatus(e.target.value)}
        >
          <option value="">All statuses</option>
          <option value="pending">Pending</option>
          <option value="completed">Completed</option>
          <option value="failed">Failed</option>
          <option value="refunded">Refunded</option>
        </select>
        <button className="btn btn-primary" onClick={load}>Filter</button>
        <button className="btn btn-ghost" onClick={exportCSV}>Export CSV</button>
      </div>

      {loading ? (
        <div className="flex justify-center py-12">
          <div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin" />
        </div>
      ) : (
        <div className="card overflow-x-auto">
          <table className="table-auto">
            <thead>
              <tr>
                <th>ID</th>
                <th>Amount</th>
                <th>Status</th>
                <th>Created</th>
                <th>Method</th>
                <th>Reference</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(r=> (
                <tr key={r.transaction_id}>
                  <td className="font-mono text-sm">#{r.transaction_id}</td>
                  <td className="font-semibold">₹ {r.amount.toFixed(2)}</td>
                  <td><StatusBadge status={r.status} /></td>
                  <td className="text-sm">{new Date(r.created_at).toLocaleString()}</td>
                  <td>{r.method||'-'}</td>
                  <td className="text-sm text-slate-500">{r.reference||'-'}</td>
                </tr>
              ))}
              {!rows.length && (
                <tr>
                  <td colSpan={6} className="text-center text-slate-400 py-6">No transactions found.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
