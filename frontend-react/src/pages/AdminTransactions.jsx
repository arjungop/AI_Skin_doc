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
      default: return 'secondary'
    }
  }

  return (
    <div className="min-h-screen pb-20 space-y-6">
      <div className="flex flex-col md:flex-row justify-between gap-4">
        <h1 className="text-3xl font-bold text-text-primary">Admin Transactions</h1>
        <div className="flex bg-surface-elevated rounded-xl p-1 border border-white/10 w-fit">
          <button onClick={() => load(page > 1 ? page - 1 : 1)} disabled={page <= 1} className="px-4 py-2 hover:bg-white/5 rounded-lg disabled:opacity-50 text-text-secondary">Prev</button>
          <span className="px-4 py-2 flex items-center text-sm font-mono text-text-primary border-x border-white/10">
            Page {page} / {maxPage}
          </span>
          <button onClick={() => load(page < maxPage ? page + 1 : maxPage)} disabled={page >= maxPage} className="px-4 py-2 hover:bg-white/5 rounded-lg disabled:opacity-50 text-text-secondary">Next</button>
        </div>
      </div>

      {/* Filters */}
      <Card variant="glass" className="p-4" hover={false}>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4">
          <div className="lg:col-span-2 relative">
            <LuSearch className="absolute left-3 top-3 text-text-muted" />
            <input
              className="w-full bg-white/5 border border-white/10 rounded-xl py-2.5 pl-10 pr-4 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-primary-500/50"
              placeholder="Search email, ID, ref..."
              value={q}
              onChange={e => setQ(e.target.value)}
            />
          </div>
          <input
            type="date"
            className="w-full bg-white/5 border border-white/10 rounded-xl py-2.5 px-4 text-sm text-text-primary focus:outline-none focus:border-primary-500/50"
            value={start}
            onChange={e => setStart(e.target.value)}
          />
          <input
            type="date"
            className="w-full bg-white/5 border border-white/10 rounded-xl py-2.5 px-4 text-sm text-text-primary focus:outline-none focus:border-primary-500/50"
            value={end}
            onChange={e => setEnd(e.target.value)}
          />
          <select
            className="w-full bg-white/5 border border-white/10 rounded-xl py-2.5 px-4 text-sm text-text-primary focus:outline-none focus:border-primary-500/50"
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
            className="w-full bg-white/5 border border-white/10 rounded-xl py-2.5 px-4 text-sm text-text-primary focus:outline-none focus:border-primary-500/50"
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
      <Card variant="glass" className="overflow-hidden p-0" hover={false}>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-text-secondary">
            <thead className="bg-surface-elevated/50 text-xs uppercase font-semibold text-text-tertiary">
              <tr>
                <th className="p-4">ID</th>
                <th className="p-4">User</th>
                <th className="p-4">Amount</th>
                <th className="p-4">Status</th>
                <th className="p-4">Date</th>
                <th className="p-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {items.map(t => (
                <tr key={t.transaction_id} className="hover:bg-white/5 transition-colors">
                  <td className="p-4 font-mono text-text-muted">#{t.transaction_id}</td>
                  <td className="p-4">
                    <div className="font-medium text-text-primary">{t.username || 'Unknown'}</div>
                    <div className="text-xs text-text-muted">{t.email}</div>
                  </td>
                  <td className="p-4 font-semibold text-text-primary">₹ {t.amount}</td>
                  <td className="p-4">
                    <CardBadge variant={getStatusColor(t.status)}>{t.status}</CardBadge>
                  </td>
                  <td className="p-4 text-text-muted">{new Date(t.created_at).toLocaleDateString()}</td>
                  <td className="p-4">
                    <div className="flex justify-end gap-2">
                      <button onClick={() => setSel(t)} className="p-2 hover:bg-white/10 rounded-lg text-text-primary" title="Details">
                        <LuEye size={18} />
                      </button>
                      <button onClick={() => openConfirm(t.transaction_id, 'completed')} className="p-2 hover:bg-success/20 text-success rounded-lg" title="Complete">
                        <LuCheck size={18} />
                      </button>
                      <button onClick={() => openConfirm(t.transaction_id, 'failed')} className="p-2 hover:bg-danger/20 text-danger rounded-lg" title="Fail">
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
          <p className="text-text-secondary">Are you sure you want to change the status of this transaction?</p>
          {(confirm.newStatus === 'failed' || confirm.newStatus === 'refunded') && (
            <div>
              <label className="block text-sm font-medium text-text-secondary mb-2">Reason</label>
              <input
                value={confirm.reason}
                onChange={e => setConfirm({ ...confirm, reason: e.target.value })}
                placeholder="Enter reason..."
                className="w-full bg-surface-elevated border border-white/10 rounded-xl px-4 py-3 text-text-primary focus:outline-none focus:border-danger/50 is-danger"
              />
            </div>
          )}
        </div>
      </ConfirmModal>

      <DetailsDrawer open={!!sel} title={sel ? `Transaction #${sel.transaction_id}` : ''} onClose={() => setSel(null)}>
        {sel && (
          <div className="space-y-4 text-text-primary">
            <div className="p-4 rounded-xl bg-white/5 border border-white/10 space-y-2">
              <div className="flex justify-between"><span className="text-text-secondary">User:</span> <span>{sel.username}</span></div>
              <div className="flex justify-between"><span className="text-text-secondary">Email:</span> <span>{sel.email}</span></div>
              <div className="flex justify-between"><span className="text-text-secondary">Amount:</span> <span className="font-bold">₹ {sel.amount}</span></div>
              <div className="flex justify-between"><span className="text-text-secondary">Method:</span> <span>{sel.method || '-'}</span></div>
              <div className="flex justify-between"><span className="text-text-secondary">Ref:</span> <span className="font-mono">{sel.reference || '-'}</span></div>
            </div>
          </div>
        )}
      </DetailsDrawer>
    </div>
  )
}
