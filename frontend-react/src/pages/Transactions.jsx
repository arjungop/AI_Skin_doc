import { useEffect, useState } from 'react'
import { api } from '../services/api'
import { useToast } from '../components/Toast.jsx'
import { motion, AnimatePresence } from 'framer-motion'
import {
  LuSearch, LuFilter, LuDownload, LuFileText, LuCheck,
  LuX, LuRefreshCcw, LuArrowUpRight, LuWallet, LuCalendar
} from 'react-icons/lu'
import { Card, CardTitle, CardDescription, CardBadge, IconWrapper } from '../components/Card'

export default function Transactions() {
  const [list, setList] = useState([])
  const [amount, setAmount] = useState('')
  const [method, setMethod] = useState('UPI')
  const [reference, setReference] = useState('')
  const [note, setNote] = useState('')
  const [filter, setFilter] = useState('')
  const [search, setSearch] = useState('')
  const [start, setStart] = useState('')
  const [end, setEnd] = useState('')
  const [category, setCategory] = useState('')
  const [loading, setLoading] = useState(false)
  const [summary, setSummary] = useState(null)

  const uid = parseInt(localStorage.getItem('user_id'))
  const role = (localStorage.getItem('role') || '').toUpperCase()
  const { push } = useToast()

  async function load() {
    setLoading(true)
    try {
      const common = {
        status: filter || undefined,
        q: search || undefined,
        start: start || undefined,
        end: end || undefined,
        category: category || undefined
      }

      const res = await api.listTransactions(common)
      setList(res || [])

      const sum = await api.transactionsSummary(role === 'ADMIN' ? undefined : uid)
      setSummary(sum)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [filter, search, start, end, category])

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
    <div className="min-h-screen pb-20 space-y-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">Transactions</h1>
          <p className="text-slate-500 mt-1">Manage your payments and billing history</p>
        </div>
        <div className="flex gap-2">
          <a href={`${api.base}/transactions/export.csv`} target="_blank" rel="noreferrer" className="btn-secondary flex items-center gap-2">
            <LuDownload size={18} /> CSV
          </a>
          <a href={`${api.base}/transactions/export.pdf`} target="_blank" rel="noreferrer" className="btn-secondary flex items-center gap-2">
            <LuFileText size={18} /> PDF
          </a>
        </div>
      </div>

      {/* Summary Cards */}
      {summary && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card className="p-5" hover={false}>
            <div className="flex items-center gap-3 mb-2">
              <div className="w-10 h-10 rounded-2xl bg-amber-50 text-amber-600 flex items-center justify-center"><LuRefreshCcw size={16} /></div>
              <h3 className="text-sm font-medium text-slate-400">Pending</h3>
            </div>
            <p className="text-2xl font-bold text-slate-900">₹ {summary.pending.toFixed(2)}</p>
          </Card>
          <Card className="p-5" hover={false}>
            <div className="flex items-center gap-3 mb-2">
              <div className="w-10 h-10 rounded-2xl bg-emerald-50 text-emerald-600 flex items-center justify-center"><LuCheck size={16} /></div>
              <h3 className="text-sm font-medium text-slate-400">Completed</h3>
            </div>
            <p className="text-2xl font-bold text-slate-900">₹ {summary.completed.toFixed(2)}</p>
          </Card>
          <Card className="p-5" hover={false}>
            <div className="flex items-center gap-3 mb-2">
              <div className="w-10 h-10 rounded-2xl bg-rose-50 text-rose-600 flex items-center justify-center"><LuX size={16} /></div>
              <h3 className="text-sm font-medium text-slate-400">Failed</h3>
            </div>
            <p className="text-2xl font-bold text-slate-900">₹ {summary.failed.toFixed(2)}</p>
          </Card>
          <Card className="p-5" hover={false}>
            <div className="flex items-center gap-3 mb-2">
              <IconWrapper variant="primary" size="sm"><LuWallet size={16} /></IconWrapper>
              <h3 className="text-sm font-medium text-slate-400">Total Count</h3>
            </div>
            <p className="text-2xl font-bold text-slate-900">{summary.count}</p>
          </Card>
        </div>
      )}

      {/* Filters */}
      <Card className="p-4" hover={false}>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
          <div className="lg:col-span-2 relative">
            <LuSearch className="absolute left-3 top-3 text-slate-400" />
            <input
              className="w-full bg-white border border-slate-200 rounded-xl py-2.5 pl-10 pr-4 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none focus:border-primary-500 focus:ring-4 focus:ring-primary-500/10"
              placeholder="Search by ID or reference..."
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
          </div>
          <div className="relative">
            <LuCalendar className="absolute left-3 top-3 text-slate-400" />
            <input
              type="date"
              className="w-full bg-white border border-slate-200 rounded-xl py-2.5 pl-10 pr-4 text-sm text-slate-900 focus:outline-none focus:border-primary-500 focus:ring-4 focus:ring-primary-500/10"
              value={start}
              onChange={e => setStart(e.target.value)}
            />
          </div>
          <div>
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
          <div>
            <select
              className="w-full bg-white border border-slate-200 rounded-xl py-2.5 px-4 text-sm text-slate-900 focus:outline-none focus:border-primary-500 focus:ring-4 focus:ring-primary-500/10"
              value={filter}
              onChange={e => setFilter(e.target.value)}
            >
              <option value="">All Status</option>
              <option value="pending">Pending</option>
              <option value="completed">Completed</option>
              <option value="failed">Failed</option>
              <option value="refunded">Refunded</option>
            </select>
          </div>
        </div>
      </Card>

      {/* List */}
      <div className="space-y-4">
        <AnimatePresence>
          {list.map((t, i) => (
            <motion.div
              key={t.transaction_id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05 }}
            >
              <Card className="flex flex-col md:flex-row items-center justify-between gap-4 p-4 group">
                <div className="flex items-center gap-4 w-full md:w-auto">
                  <div className={`w-12 h-12 rounded-2xl flex items-center justify-center text-xl font-bold ${t.status === 'completed' ? 'bg-emerald-50 text-emerald-600' :
                    t.status === 'failed' ? 'bg-rose-50 text-rose-600' :
                      'bg-amber-50 text-amber-600'
                    }`}>
                    {t.status === 'completed' ? '✓' : t.status === 'failed' ? '!' : '⟳'}
                  </div>
                  <div>
                    <h3 className="font-semibold text-slate-900">₹ {t.amount}</h3>
                    <p className="text-xs text-slate-500 flex items-center gap-1.5">
                      <span className="capitalize">{t.category}</span> • <span>{new Date(t.created_at).toLocaleDateString()}</span>
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-6 w-full md:w-auto justify-between md:justify-end">
                  <div className="text-right hidden md:block">
                    <p className="text-sm text-slate-700 font-medium">{t.method || 'N/A'}</p>
                    <p className="text-xs text-slate-400 font-mono">{t.transaction_id}</p>
                  </div>

                  <div className="flex items-center gap-3">
                    <CardBadge variant={getStatusColor(t.status)}>{t.status}</CardBadge>
                    <a
                      href={`${api.base}/transactions/${t.transaction_id}/receipt.pdf`}
                      target="_blank"
                      rel="noreferrer"
                      className="p-2 text-slate-400 hover:text-primary-500 hover:bg-slate-50 rounded-lg transition-colors"
                      title="Download Receipt"
                    >
                      <LuFileText size={18} />
                    </a>
                  </div>
                </div>
              </Card>
            </motion.div>
          ))}
        </AnimatePresence>

        {loading && <div className="text-center py-10 text-slate-400">Loading transactions...</div>}
        {!loading && list.length === 0 && (
          <div className="text-center py-20 bg-slate-50 rounded-3xl border border-slate-200 border-dashed">
            <LuWallet className="mx-auto mb-4 text-slate-300" size={48} />
            <h3 className="text-xl font-bold text-slate-900">No transactions found</h3>
            <p className="text-slate-400">Adjust filters or create a new transaction.</p>
          </div>
        )}
      </div>
    </div>
  )
}
