import { useEffect, useState } from 'react'
import { api } from '../services/api'
import { motion } from 'framer-motion'
import {
  LuSearch, LuCalendar, LuWallet
} from 'react-icons/lu'
import { Card, CardBadge } from '../components/Card'

export default function AdminTransactions() {
  const [items, setItems] = useState([])
  const [status, setStatus] = useState('')
  const [category, setCategory] = useState('')
  const [start, setStart] = useState('')
  const [end, setEnd] = useState('')
  const [loading, setLoading] = useState(false)

  async function load() {
    setLoading(true)
    try {
      const params = {}
      if (status) params.status = status
      if (category) params.category = category
      if (start) params.start = start
      if (end) params.end = end

      const res = await api.listTransactions(params)
      setItems(res || [])
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [status, category, start, end])

  const getStatusColor = (s) => {
    switch (s) {
      case 'completed': return 'success'
      case 'pending': return 'warning'
      case 'failed': return 'danger'
      default: return 'default'
    }
  }

  return (
    <div className="min-h-screen pb-20 space-y-6">
      <h1 className="text-3xl font-bold text-slate-900">All Transactions</h1>

      {/* Filters */}
      <Card className="p-4" hover={false}>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="relative">
            <LuCalendar className="absolute left-3 top-3 text-slate-400" />
            <input type="date"
              className="w-full bg-white border border-slate-200 rounded-xl py-2.5 pl-10 pr-4 text-sm text-slate-900 focus:outline-none focus:border-primary-500 focus:ring-4 focus:ring-primary-500/10"
              value={start} onChange={e => setStart(e.target.value)}
            />
          </div>
          <div className="relative">
            <LuCalendar className="absolute left-3 top-3 text-slate-400" />
            <input type="date"
              className="w-full bg-white border border-slate-200 rounded-xl py-2.5 pl-10 pr-4 text-sm text-slate-900 focus:outline-none focus:border-primary-500 focus:ring-4 focus:ring-primary-500/10"
              value={end} onChange={e => setEnd(e.target.value)}
            />
          </div>
          <select
            className="w-full bg-white border border-slate-200 rounded-xl py-2.5 px-4 text-sm text-slate-900 focus:outline-none focus:border-primary-500 focus:ring-4 focus:ring-primary-500/10"
            value={status} onChange={e => setStatus(e.target.value)}
          >
            <option value="">All Status</option>
            <option value="pending">Pending</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
          </select>
          <select
            className="w-full bg-white border border-slate-200 rounded-xl py-2.5 px-4 text-sm text-slate-900 focus:outline-none focus:border-primary-500 focus:ring-4 focus:ring-primary-500/10"
            value={category} onChange={e => setCategory(e.target.value)}
          >
            <option value="">All Categories</option>
            <option value="consultation">Consultation</option>
            <option value="procedure">Procedure</option>
            <option value="pharmacy">Pharmacy</option>
          </select>
        </div>
      </Card>

      {/* Table */}
      <Card className="overflow-hidden p-0" hover={false}>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-slate-600">
            <thead className="bg-slate-50 text-xs uppercase font-semibold text-slate-400">
              <tr>
                <th className="p-4">ID</th>
                <th className="p-4">User</th>
                <th className="p-4">Amount</th>
                <th className="p-4">Category</th>
                <th className="p-4">Status</th>
                <th className="p-4">Date</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {items.map(t => (
                <tr key={t.transaction_id} className="hover:bg-slate-50 transition-colors">
                  <td className="p-4 font-mono text-slate-400">#{t.transaction_id}</td>
                  <td className="p-4 text-slate-700">{t.user_id}</td>
                  <td className="p-4 font-semibold text-slate-900">₹ {t.amount}</td>
                  <td className="p-4 capitalize text-slate-500">{t.category || 'consultation'}</td>
                  <td className="p-4">
                    <CardBadge variant={getStatusColor(t.status)}>{t.status}</CardBadge>
                  </td>
                  <td className="p-4 text-slate-400">{new Date(t.created_at).toLocaleDateString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {loading && <div className="text-center py-10 text-slate-400">Loading...</div>}
      {!loading && items.length === 0 && (
        <div className="text-center py-20 bg-slate-50 rounded-3xl border border-slate-200 border-dashed">
          <LuWallet className="mx-auto mb-4 text-slate-300" size={48} />
          <h3 className="text-xl font-bold text-slate-900">No transactions</h3>
        </div>
      )}
    </div>
  )
}
