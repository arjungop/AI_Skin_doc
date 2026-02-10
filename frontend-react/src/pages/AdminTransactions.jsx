import { useEffect, useState } from 'react'
import { api } from '../services/api'
import ConfirmModal from '../components/ConfirmModal.jsx'
import DetailsDrawer from '../components/DetailsDrawer.jsx'
import { motion, AnimatePresence } from 'framer-motion'
import {
  LuSearch, LuFilter, LuDownload, LuFileText, LuCheck,
  LuX, LuRefreshCcw, LuArrowUpRight, LuWallet, LuCalendar, LuEye
} from 'react-icons/lu'
import { Card, CardTitle, CardBadge } from '../components/Card'

export default function AdminTransactions() {
  const [items, setItems] = useState([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [q, setQ] = useState('')
  const [status, setStatus] = useState('')
  const [method, setMethod] = useState('')
  const [category, setCategory] = useState('')
  const [start, setStart] = useState('')
  const [end, setEnd] = useState('')
  const [sel, setSel] = useState(null)
  const [confirm, setConfirm] = useState({ open: false, id: null, newStatus: '', reason: '' })
  const [loading, setLoading] = useState(false)

  async function load(p = page) {
    setLoading(true)
    try {
      const res = await api.adminListTransactions({ q, status, method, category, start, end, page: p, page_size: 30 })
      setItems(res.items || [])
      setTotal(res.total || 0)
      setPage(res.page || p)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load(1) }, [q, status, method, category, start, end])

  const maxPage = Math.max(1, Math.ceil(total / 30))

  function openConfirm(id, newStatus) { setConfirm({ open: true, id, newStatus, reason: '' }) }
  async function doConfirm() {
    await api.updateTransactionStatus(confirm.id, confirm.newStatus, confirm.reason || '')
    setConfirm({ open: false, id: null, newStatus: '', reason: '' })
    load()
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return 'success'
      case 'pending': return 'warning'
      case 'failed': return 'danger'
      case 'refunded': return 'primary'
      default: return 'default'
    }
  }

  return (
    <div className="min-h-screen pb-20 space-y-6">
      <div className="flex flex-col md:flex-row justify-between gap-4">
        <h1 className="text-3xl font-bold text-slate-900">Admin Transactions</h1>
        <div className="flex bg-white rounded-xl p-1 border border-slate-200 shadow-soft-sm w-fit">
          <button onClick={() => load(page > 1 ? page - 1 : 1)} disabled={page <= 1} className="px-4 py-2 hover:bg-slate-50 rounded-lg disabled:opacity-50 text-slate-500">Prev</button>
          <span className="px-4 py-2 flex items-center text-sm font-mono text-slate-900 border-x border-slate-200">
            Page {page} / {maxPage}
          </span>
          <button onClick={() => load(page < maxPage ? page + 1 : maxPage)} disabled={page >= maxPage} className="px-4 py-2 hover:bg-slate-50 rounded-lg disabled:opacity-50 text-slate-500">Next</button>
        </div>
      </div>

      {/* Filters */}
      <Card className="p-4" hover={false}>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4">
          <div className="lg:col-span-2 relative">
            <LuSearch className="absolute left-3 top-3 text-slate-400" />
            <input
              className="w-full bg-white border border-slate-200 rounded-xl py-2.5 pl-10 pr-4 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none focus:border-primary-500 focus:ring-4 focus:ring-primary-500/10"
              placeholder="Search email, ID, ref..."
              value={q}
              onChange={e => setQ(e.target.value)}
            />
          </div>
          <input
            type="date"
            className="w-full bg-white border border-slate-200 rounded-xl py-2.5 px-4 text-sm text-slate-900 focus:outline-none focus:border-primary-500 focus:ring-4 focus:ring-primary-500/10"
            value={start}
            onChange={e => setStart(e.target.value)}
          />
          <input
            type="date"
            className="w-full bg-white border border-slate-200 rounded-xl py-2.5 px-4 text-sm text-slate-900 focus:outline-none focus:border-primary-500 focus:ring-4 focus:ring-primary-500/10"
            value={end}
            onChange={e => setEnd(e.target.value)}
          />
          <select
            className="w-full bg-white border border-slate-200 rounded-xl py-2.5 px-4 text-sm text-slate-900 focus:outline-none focus:border-primary-500 focus:ring-4 focus:ring-primary-500/10"
            value={status}
            onChange={e => setStatus(e.target.value)}
          >
            <option value="">All Status</option>
            <option value="pending">Pending</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
            <option value="refunded">Refunded</option>
          </select>
          <select
            className="w-full bg-white border border-slate-200 rounded-xl py-2.5 px-4 text-sm text-slate-900 focus:outline-none focus:border-primary-500 focus:ring-4 focus:ring-primary-500/10"
            value={category}
            onChange={e => setCategory(e.target.value)}
          >
            <option value="">All Categories</option>
            <option value="consultation">Consultation</option>
            <option value="procedure">Procedure</option>
            <option value="pharmacy">Pharmacy</option>
            <option value="general">General</option>
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
                <th className="p-4">Status</th>
                <th className="p-4">Date</th>
                <th className="p-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {items.map(t => (
                <tr key={t.transaction_id} className="hover:bg-slate-50 transition-colors">
                  <td className="p-4 font-mono text-slate-400">#{t.transaction_id}</td>
                  <td className="p-4">
                    <div className="font-medium text-slate-900">{t.username || 'Unknown'}</div>
                    <div className="text-xs text-slate-400">{t.email}</div>
                  </td>
                  <td className="p-4 font-semibold text-slate-900">₹ {t.amount}</td>
                  <td className="p-4">
                    <CardBadge variant={getStatusColor(t.status)}>{t.status}</CardBadge>
                  </td>
                  <td className="p-4 text-slate-400">{new Date(t.created_at).toLocaleDateString()}</td>
                  <td className="p-4">
                    <div className="flex justify-end gap-2">
                      <button onClick={() => setSel(t)} className="p-2 hover:bg-slate-100 rounded-lg text-slate-600" title="Details">
                        <LuEye size={18} />
                      </button>
                      <button onClick={() => openConfirm(t.transaction_id, 'completed')} className="p-2 hover:bg-emerald-50 text-emerald-600 rounded-lg" title="Complete">
                        <LuCheck size={18} />
                      </button>
                      <button onClick={() => openConfirm(t.transaction_id, 'failed')} className="p-2 hover:bg-rose-50 text-rose-600 rounded-lg" title="Fail">
                        <LuX size={18} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <ConfirmModal
        open={confirm.open}
        title={`Confirm ${confirm.newStatus}`}
        onClose={() => setConfirm({ open: false, id: null, newStatus: '', reason: '' })}
        onConfirm={doConfirm}
        confirmText="Confirm"
      >
        <div className="space-y-4">
          <p className="text-slate-500">Are you sure you want to change the status of this transaction?</p>
          {(confirm.newStatus === 'failed' || confirm.newStatus === 'refunded') && (
            <div>
              <label className="block text-sm font-medium text-slate-600 mb-2">Reason</label>
              <input
                value={confirm.reason}
                onChange={e => setConfirm({ ...confirm, reason: e.target.value })}
                placeholder="Enter reason..."
                className="w-full bg-white border border-slate-200 rounded-xl px-4 py-3 text-slate-900 focus:outline-none focus:border-primary-500 focus:ring-4 focus:ring-primary-500/10"
              />
            </div>
          )}
        </div>
      </ConfirmModal>

      <DetailsDrawer open={!!sel} title={sel ? `Transaction #${sel.transaction_id}` : ''} onClose={() => setSel(null)}>
        {sel && (
          <div className="space-y-4 text-slate-900">
            <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2">
              <div className="flex justify-between"><span className="text-slate-500">User:</span> <span>{sel.username}</span></div>
              <div className="flex justify-between"><span className="text-slate-500">Email:</span> <span>{sel.email}</span></div>
              <div className="flex justify-between"><span className="text-slate-500">Amount:</span> <span className="font-bold">₹ {sel.amount}</span></div>
              <div className="flex justify-between"><span className="text-slate-500">Method:</span> <span>{sel.method || '-'}</span></div>
              <div className="flex justify-between"><span className="text-slate-500">Ref:</span> <span className="font-mono">{sel.reference || '-'}</span></div>
            </div>
          </div>
        )}
      </DetailsDrawer>
    </div>
  )
}
