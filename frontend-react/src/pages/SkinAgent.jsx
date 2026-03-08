import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import ReactMarkdown from 'react-markdown'
import { api } from '../services/api'
import {
  LuBrain, LuSearch, LuActivity, LuClipboardList,
  LuSparkles, LuChevronRight, LuHistory, LuPlay, LuLoader,
  LuCircleCheck, LuTriangleAlert, LuEye, LuArrowLeft,
  LuCalendar, LuShieldCheck, LuX, LuZap, LuBookOpen, LuBell,
  LuSend, LuMessageCircle
} from 'react-icons/lu'

const STEP_ICONS = {
  thought: LuBrain,
  tool_call: LuSearch,
  observation: LuEye,
  answer: LuSparkles,
}

const STEP_COLORS = {
  thought: 'text-primary-500 bg-primary-50 border-primary-200',
  tool_call: 'text-blue-500 bg-blue-50 border-blue-200',
  observation: 'text-amber-500 bg-amber-50 border-amber-200',
  answer: 'text-emerald-500 bg-emerald-50 border-emerald-200',
}

const STEP_LABELS = {
  thought: 'Thinking',
  tool_call: 'Action',
  observation: 'Observation',
  answer: 'Final Analysis',
}

function StepCard({ step, isLatest }) {
  const Icon = STEP_ICONS[step.step_type] || LuBrain
  const color = STEP_COLORS[step.step_type] || STEP_COLORS.thought
  const label = STEP_LABELS[step.step_type] || step.step_type

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`rounded-2xl border p-4 ${color} ${isLatest ? 'ring-2 ring-offset-2 ring-primary-300' : ''}`}
    >
      <div className="flex items-center gap-2 mb-2">
        <Icon className="w-4 h-4" />
        <span className="text-xs font-semibold uppercase tracking-wide">{label}</span>
        {step.tool_name && (
          <span className="text-xs font-mono bg-white/60 px-2 py-0.5 rounded-full">{step.tool_name}</span>
        )}
        <span className="ml-auto text-xs opacity-50">#{step.step_order}</span>
        {step.iteration > 0 && (
          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-white/40 text-slate-500">
            {step.iteration}/{step.max_iterations || 15}
          </span>
        )}
      </div>
      <div className="text-sm leading-relaxed">
        {step.step_type === 'answer' ? (
          <div className="prose prose-sm prose-slate max-w-none">
            <ReactMarkdown>{step.content}</ReactMarkdown>
          </div>
        ) : step.step_type === 'observation' ? (
          <details className="cursor-pointer">
            <summary className="font-medium text-amber-700">View data</summary>
            <pre className="mt-2 text-xs bg-white/50 rounded-lg p-3 overflow-x-auto whitespace-pre-wrap max-h-48 overflow-y-auto">
              {(() => {
                try { return JSON.stringify(JSON.parse(step.content), null, 2) } catch { return step.content }
              })()}
            </pre>
          </details>
        ) : (
          <p className="whitespace-pre-wrap">{step.content}</p>
        )}
      </div>
    </motion.div>
  )
}

function SessionCard({ session, onClick }) {
  const statusColor = session.status === 'completed'
    ? 'text-emerald-600 bg-emerald-50'
    : session.status === 'failed'
      ? 'text-red-600 bg-red-50'
      : 'text-blue-600 bg-blue-50'

  return (
    <motion.button
      whileHover={{ scale: 1.01 }}
      whileTap={{ scale: 0.99 }}
      onClick={onClick}
      className="w-full text-left bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl border border-white/50 dark:border-slate-700 rounded-2xl p-5 shadow-sm hover:shadow-md transition-all"
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <LuBrain className="w-4 h-4 text-primary-500" />
          <span className="font-semibold text-slate-800 dark:text-slate-100 text-sm">
            Analysis #{session.session_id}
          </span>
        </div>
        <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${statusColor}`}>
          {session.status}
        </span>
      </div>
      <p className="text-xs text-slate-500 dark:text-slate-400 mb-1">
        {session.trigger === 'scan' ? 'Triggered by scan' : 'Manual analysis'} •{' '}
        {new Date(session.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' })}
      </p>
      {session.summary && (
        <p className="text-sm text-slate-600 dark:text-slate-300 line-clamp-2 mt-1">{session.summary.slice(0, 150)}...</p>
      )}
      <div className="flex items-center gap-1 text-primary-500 text-xs font-medium mt-3">
        View details <LuChevronRight className="w-3 h-3" />
      </div>
    </motion.button>
  )
}

const ACTION_ICONS = {
  schedule_appointment: LuCalendar,
  create_routine: LuClipboardList,
  add_skin_log: LuBookOpen,
  set_reminder: LuBell,
}

const ACTION_COLORS = {
  proposed: 'border-blue-200 bg-blue-50/50 dark:border-blue-800 dark:bg-blue-900/20',
  approved: 'border-emerald-200 bg-emerald-50/50 dark:border-emerald-800 dark:bg-emerald-900/20',
  rejected: 'border-slate-200 bg-slate-50/50 dark:border-slate-700 dark:bg-slate-800/30 opacity-60',
  executed: 'border-emerald-300 bg-emerald-50 dark:border-emerald-700 dark:bg-emerald-900/30',
  failed: 'border-red-200 bg-red-50/50 dark:border-red-800 dark:bg-red-900/20',
}

const STATUS_BADGES = {
  proposed: { text: 'Pending', cls: 'text-blue-600 bg-blue-100 dark:text-blue-300 dark:bg-blue-900/50' },
  approved: { text: 'Approved', cls: 'text-emerald-600 bg-emerald-100 dark:text-emerald-300 dark:bg-emerald-900/50' },
  rejected: { text: 'Rejected', cls: 'text-slate-500 bg-slate-100 dark:text-slate-400 dark:bg-slate-700' },
  executed: { text: 'Executed', cls: 'text-emerald-700 bg-emerald-200 dark:text-emerald-200 dark:bg-emerald-800' },
  failed: { text: 'Failed', cls: 'text-red-600 bg-red-100 dark:text-red-300 dark:bg-red-900/50' },
}

function ActionCard({ action, onApprove, onReject, locked }) {
  const Icon = ACTION_ICONS[action.action_type] || LuZap
  const colorCls = ACTION_COLORS[action.status] || ACTION_COLORS.proposed
  const badge = STATUS_BADGES[action.status] || STATUS_BADGES.proposed
  const canToggle = !locked && (action.status === 'proposed' || action.status === 'approved' || action.status === 'rejected')

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className={`rounded-2xl border p-4 transition-all ${colorCls}`}
    >
      <div className="flex items-start gap-3">
        <div className="flex-shrink-0 w-9 h-9 rounded-xl bg-white/80 dark:bg-slate-800/80 border border-slate-200 dark:border-slate-600 flex items-center justify-center">
          <Icon className="w-4.5 h-4.5 text-primary-500" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <h4 className="text-sm font-semibold text-slate-800 dark:text-slate-100 truncate">{action.title}</h4>
            <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full whitespace-nowrap ${badge.cls}`}>
              {badge.text}
            </span>
          </div>
          {action.description && (
            <p className="text-xs text-slate-500 dark:text-slate-400 line-clamp-2 mb-2">{action.description}</p>
          )}
          {action.result && (
            <p className={`text-xs mt-1 ${action.status === 'failed' ? 'text-red-600' : 'text-emerald-600 dark:text-emerald-400'}`}>
              {action.result}
            </p>
          )}
        </div>
        {canToggle && (
          <div className="flex gap-1.5 flex-shrink-0">
            {action.status !== 'approved' && (
              <button
                onClick={() => onApprove(action.action_id)}
                className="p-1.5 rounded-lg bg-emerald-100 hover:bg-emerald-200 dark:bg-emerald-900/50 dark:hover:bg-emerald-800/70 text-emerald-600 dark:text-emerald-400 transition-colors"
                title="Approve"
              >
                <LuShieldCheck className="w-4 h-4" />
              </button>
            )}
            {action.status !== 'rejected' && (
              <button
                onClick={() => onReject(action.action_id)}
                className="p-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 dark:bg-slate-700 dark:hover:bg-slate-600 text-slate-500 dark:text-slate-400 transition-colors"
                title="Reject"
              >
                <LuX className="w-4 h-4" />
              </button>
            )}
          </div>
        )}
      </div>
    </motion.div>
  )
}

function ActionsPanel({ actions, onApprove, onReject, onExecute, executing }) {
  if (!actions || actions.length === 0) return null

  const approvedCount = actions.filter(a => a.status === 'approved').length
  const hasExecutable = approvedCount > 0
  const allDone = actions.every(a => a.status === 'executed' || a.status === 'failed' || a.status === 'rejected')
  const executedCount = actions.filter(a => a.status === 'executed').length
  const failedCount = actions.filter(a => a.status === 'failed').length

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-200 flex items-center gap-2">
          <LuZap className="w-4 h-4 text-amber-500" />
          Proposed Actions
          <span className="text-xs font-normal text-slate-400">
            ({actions.length} action{actions.length !== 1 ? 's' : ''})
          </span>
        </h3>
        {hasExecutable && !allDone && (
          <button
            onClick={onExecute}
            disabled={executing}
            className="flex items-center gap-1.5 px-4 py-2 bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-600 hover:to-teal-600 text-white text-xs font-semibold rounded-xl shadow-sm hover:shadow-md transition-all disabled:opacity-50"
          >
            {executing ? <LuLoader className="w-3.5 h-3.5 animate-spin" /> : <LuZap className="w-3.5 h-3.5" />}
            Execute {approvedCount} Approved
          </button>
        )}
      </div>
      {!allDone && (
        <p className="text-xs text-slate-400 dark:text-slate-500">
          Review each action below. Approve the ones you want, then click Execute.
        </p>
      )}
      {/* Execution result summary banner */}
      {allDone && (executedCount > 0 || failedCount > 0) && (
        <div className={`rounded-xl p-3 text-sm font-medium flex items-center gap-2 ${failedCount > 0
          ? 'bg-amber-50 border border-amber-200 text-amber-800 dark:bg-amber-900/20 dark:border-amber-700 dark:text-amber-300'
          : 'bg-emerald-50 border border-emerald-200 text-emerald-800 dark:bg-emerald-900/20 dark:border-emerald-700 dark:text-emerald-300'
          }`}>
          {failedCount > 0 ? <LuTriangleAlert className="w-4 h-4 flex-shrink-0" /> : <LuCircleCheck className="w-4 h-4 flex-shrink-0" />}
          {failedCount > 0
            ? `${executedCount} of ${executedCount + failedCount} action${executedCount + failedCount !== 1 ? 's' : ''} executed successfully. ${failedCount} failed.`
            : `All ${executedCount} action${executedCount !== 1 ? 's' : ''} executed successfully!`}
        </div>
      )}
      <div className="space-y-2">
        {actions.map(action => (
          <ActionCard
            key={action.action_id}
            action={action}
            onApprove={onApprove}
            onReject={onReject}
            locked={allDone || executing}
          />
        ))}
      </div>
    </div>
  )
}

export default function SkinAgent() {
  const patientId = parseInt(localStorage.getItem('patient_id')) || null
  const [view, setView] = useState('home') // home | running | history | detail
  const [steps, setSteps] = useState([])
  const [running, setRunning] = useState(false)
  const [error, setError] = useState(null)
  const [sessions, setSessions] = useState([])
  const [selectedSession, setSelectedSession] = useState(null)
  const [loadingSessions, setLoadingSessions] = useState(false)
  const [actions, setActions] = useState([])
  const [executing, setExecuting] = useState(false)
  const [hintText, setHintText] = useState('')
  const [hintSending, setHintSending] = useState(false)
  const [iteration, setIteration] = useState(0)
  const [maxIterations, setMaxIterations] = useState(15)
  const [activeSessionId, setActiveSessionId] = useState(null)
  const scrollRef = useRef(null)

  // Auto-scroll to bottom on new steps
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [steps])

  const loadSessions = async () => {
    setLoadingSessions(true)
    try {
      const data = await api.agentListSessions()
      setSessions(Array.isArray(data) ? data : [])
    } catch { setSessions([]) }
    setLoadingSessions(false)
  }

  const viewSession = async (session) => {
    try {
      const data = await api.agentGetSession(session.session_id)
      setSelectedSession(data)
      setActions(data.actions || [])
      setView('detail')
    } catch {
      setError('Failed to load session details')
    }
  }

  const loadActions = async (sessionId) => {
    try {
      const data = await api.agentGetActions(sessionId)
      setActions(Array.isArray(data) ? data : [])
    } catch { /* ignore */ }
  }

  const handleApprove = async (actionId) => {
    try {
      const updated = await api.agentApproveAction(actionId)
      setActions(prev => prev.map(a => a.action_id === actionId ? updated : a))
    } catch { /* ignore */ }
  }

  const handleReject = async (actionId) => {
    try {
      const updated = await api.agentRejectAction(actionId)
      setActions(prev => prev.map(a => a.action_id === actionId ? updated : a))
    } catch { /* ignore */ }
  }

  const handleExecute = async (sessionId) => {
    setExecuting(true)
    try {
      const updated = await api.agentExecuteActions(sessionId)
      setActions(Array.isArray(updated) ? updated : [])
    } catch (err) {
      setError(err.message || 'Failed to execute actions')
    }
    setExecuting(false)
  }

  const startAnalysis = async () => {
    if (!patientId) {
      setError('No patient profile found. Please complete onboarding first.')
      return
    }
    setSteps([])
    setActions([])
    setError(null)
    setRunning(true)
    setView('running')

    let lastSessionId = null
    try {
      await api.agentAnalyze(patientId, null, (eventType, data) => {
        if (eventType === 'step') {
          setSteps(prev => [...prev, data])
          if (data.iteration) setIteration(data.iteration)
          if (data.max_iterations) setMaxIterations(data.max_iterations)
          // Track session_id from the first step
          if (data.step_id && !lastSessionId) {
            // Session ID will be fetched after stream ends
          }
        } else if (eventType === 'done') {
          // Agent completed — we'll load actions below
        } else if (eventType === 'error') {
          setError(data.message || 'Agent encountered an error')
        }
      })
      // After stream ends, fetch the latest session to get actions
      const sessions = await api.agentListSessions()
      if (Array.isArray(sessions) && sessions.length > 0) {
        lastSessionId = sessions[0].session_id
        await loadActions(lastSessionId)
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setRunning(false)
      if (lastSessionId) {
        setSelectedSession({ session_id: lastSessionId })
        setActiveSessionId(lastSessionId)
      }
    }
  }

  const sendHint = async () => {
    const sid = activeSessionId || selectedSession?.session_id
    if (!sid || !hintText.trim() || hintSending) return
    setHintSending(true)
    try {
      await api.agentSendHint(sid, hintText.trim())
      setHintText('')
    } catch { /* ignore — session may have just completed */ }
    setHintSending(false)
  }

  const finalAnswer = steps.find(s => s.step_type === 'answer')

  return (
    <div className="relative min-h-screen pb-20 pt-8 md:pt-12 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
      {/* Ambient Background Glow */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute top-0 left-1/4 w-[600px] h-[600px] bg-primary-500/10 rounded-full blur-[150px] opacity-50" />
        <div className="absolute bottom-1/4 right-1/4 w-[500px] h-[500px] bg-accent-500/10 rounded-full blur-[120px] opacity-40" />
      </div>

      <div className="relative z-10 max-w-4xl mx-auto space-y-8 animate-fade-in">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 mb-4">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary-50/50 dark:bg-primary-900/20 text-primary-600 dark:text-primary-400 text-sm font-medium mb-4 border border-primary-100 dark:border-primary-800">
              <LuBrain className="w-4 h-4" />
              <span>Agentic AI</span>
            </div>
            <h1 className="text-4xl md:text-5xl font-display font-bold text-slate-800 dark:text-slate-100 tracking-tight">
              Skin Health <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary-600 to-accent-600 dark:from-primary-400 dark:to-accent-400">Agent</span>
            </h1>
            <p className="text-slate-500 dark:text-slate-400 mt-3 text-lg font-light max-w-2xl">
              An autonomous AI agent that analyses your scan history, tracks progression, and creates personalised recommendations.
            </p>
          </div>
        </div>

        {/* Navigation */}
        {view !== 'home' && (
          <button
            onClick={() => { setView('home'); setSelectedSession(null); setError(null) }}
            className="flex items-center gap-1.5 text-sm text-slate-500 hover:text-primary-600 transition-colors"
          >
            <LuArrowLeft className="w-4 h-4" /> Back
          </button>
        )}

        {/* Home View */}
        {view === 'home' && (
          <div className="space-y-6">
            {/* Launch Card */}
            <div className="relative group">
              <div className="absolute inset-0 bg-gradient-to-r from-primary-500/20 to-accent-500/20 rounded-3xl blur-xl transition-all duration-500 group-hover:blur-2xl opacity-70" />
              <div className="relative bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl border border-white/50 dark:border-slate-700 rounded-3xl p-8 shadow-xl">
                <div className="flex flex-col md:flex-row items-center gap-8">
                  <div className="flex-1 space-y-4">
                    <h2 className="text-2xl font-display font-bold text-slate-800 dark:text-slate-100">
                      Comprehensive Skin Analysis
                    </h2>
                    <p className="text-slate-500 dark:text-slate-400 text-sm leading-relaxed">
                      The agent autonomously retrieves your scan history, skin profile, and treatment plans, then reasons step-by-step to produce a complete health assessment with actionable recommendations.
                    </p>
                    <div className="flex flex-wrap gap-2 text-xs">
                      {['Scan History', 'Profile Analysis', 'Progression Tracking', 'Skincare Plan', 'Treatment Review'].map(t => (
                        <span key={t} className="px-3 py-1 rounded-full bg-primary-50 dark:bg-primary-900/30 text-primary-600 dark:text-primary-300 border border-primary-200 dark:border-primary-700">
                          {t}
                        </span>
                      ))}
                    </div>
                  </div>
                  <button
                    onClick={startAnalysis}
                    disabled={running}
                    className="flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-primary-600 to-accent-600 hover:from-primary-700 hover:to-accent-700 text-white font-semibold rounded-2xl shadow-lg hover:shadow-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap"
                  >
                    {running ? <LuLoader className="w-5 h-5 animate-spin" /> : <LuPlay className="w-5 h-5" />}
                    {running ? 'Running...' : 'Start Analysis'}
                  </button>
                </div>
              </div>
            </div>

            {/* How it works */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              {[
                { icon: LuBrain, title: 'Profile', desc: 'Loads your skin type & goals' },
                { icon: LuSearch, title: 'Scan History', desc: 'Analyses all past scans' },
                { icon: LuActivity, title: 'Progression', desc: 'Tracks changes over time' },
                { icon: LuClipboardList, title: 'Plan', desc: 'Generates recommendations' },
              ].map((item, i) => (
                <div key={i} className="bg-white/60 dark:bg-slate-800/60 backdrop-blur-sm border border-slate-200 dark:border-slate-700 rounded-2xl p-4 text-center space-y-2">
                  <div className="inline-flex items-center justify-center w-10 h-10 rounded-xl bg-primary-50 dark:bg-primary-900/30 text-primary-500">
                    <item.icon className="w-5 h-5" />
                  </div>
                  <h3 className="font-semibold text-sm text-slate-700 dark:text-slate-200">{item.title}</h3>
                  <p className="text-xs text-slate-400">{item.desc}</p>
                </div>
              ))}
            </div>

            {/* Past Sessions */}
            <div className="space-y-3">
              <button
                onClick={() => { loadSessions(); setView('history') }}
                className="flex items-center gap-2 text-sm font-medium text-slate-600 dark:text-slate-300 hover:text-primary-600 transition-colors"
              >
                <LuHistory className="w-4 h-4" /> View Past Analyses
                <LuChevronRight className="w-3 h-3" />
              </button>
            </div>
          </div>
        )}

        {/* Running View */}
        {view === 'running' && (
          <div className="space-y-4">
            {/* Progress indicator */}
            <div className="flex items-center gap-3 bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl border border-slate-200 dark:border-slate-700 rounded-2xl p-4">
              {running ? (
                <>
                  <LuLoader className="w-5 h-5 text-primary-500 animate-spin" />
                  <span className="text-sm font-medium text-slate-700 dark:text-slate-200">Agent is reasoning...</span>
                  <span className="ml-auto text-xs text-slate-400">{steps.length} steps</span>
                </>
              ) : error ? (
                <>
                  <LuTriangleAlert className="w-5 h-5 text-red-500" />
                  <span className="text-sm font-medium text-red-600">{error}</span>
                </>
              ) : (
                <>
                  <LuCircleCheck className="w-5 h-5 text-emerald-500" />
                  <span className="text-sm font-medium text-emerald-700 dark:text-emerald-300">Analysis complete</span>
                  <span className="ml-auto text-xs text-slate-400">{steps.length} steps</span>
                </>
              )}
            </div>

            {/* Progress bar */}
            {running && iteration > 0 && (
              <div className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl border border-slate-200 dark:border-slate-700 rounded-2xl p-4 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-500 font-medium">Progress</span>
                  <span className="text-slate-400">Step {iteration} of {maxIterations}</span>
                </div>
                <div className="h-2 rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-primary-500 to-accent-500 transition-all duration-500"
                    style={{ width: `${Math.min(100, (iteration / maxIterations) * 100)}%` }}
                  />
                </div>
              </div>
            )}

            {/* Hint input — visible while agent is running */}
            {running && (
              <div className="flex items-center gap-2 bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl border border-slate-200 dark:border-slate-700 rounded-2xl p-3">
                <LuMessageCircle className="w-4 h-4 text-slate-400 flex-shrink-0" />
                <input
                  type="text"
                  value={hintText}
                  onChange={e => setHintText(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter') sendHint() }}
                  placeholder="Send a hint to guide the agent..."
                  className="flex-1 bg-transparent text-sm text-slate-700 dark:text-slate-200 placeholder-slate-400 outline-none"
                />
                <button
                  onClick={sendHint}
                  disabled={!hintText.trim() || hintSending}
                  className="p-1.5 rounded-lg bg-primary-100 hover:bg-primary-200 dark:bg-primary-900/50 dark:hover:bg-primary-800/70 text-primary-600 dark:text-primary-400 transition-colors disabled:opacity-30"
                >
                  {hintSending ? <LuLoader className="w-4 h-4 animate-spin" /> : <LuSend className="w-4 h-4" />}
                </button>
              </div>
            )}

            {/* Steps */}
            <div ref={scrollRef} className="space-y-3 max-h-[60vh] overflow-y-auto pr-1">
              <AnimatePresence>
                {steps.map((step, i) => (
                  <StepCard key={step.step_id || i} step={step} isLatest={i === steps.length - 1 && running} />
                ))}
              </AnimatePresence>
            </div>

            {/* Done actions */}
            {!running && finalAnswer && (
              <div className="space-y-4 pt-2">
                {/* Proposed Actions */}
                <ActionsPanel
                  actions={actions}
                  onApprove={handleApprove}
                  onReject={handleReject}
                  onExecute={() => handleExecute(selectedSession?.session_id)}
                  executing={executing}
                />
                <div className="flex gap-3">
                  <button
                    onClick={startAnalysis}
                    className="flex items-center gap-2 px-5 py-2.5 bg-primary-600 hover:bg-primary-700 text-white font-medium rounded-xl transition-colors text-sm"
                  >
                    <LuPlay className="w-4 h-4" /> Run Again
                  </button>
                  <button
                    onClick={() => { loadSessions(); setView('history') }}
                    className="flex items-center gap-2 px-5 py-2.5 bg-slate-100 hover:bg-slate-200 dark:bg-slate-700 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-200 font-medium rounded-xl transition-colors text-sm"
                  >
                    <LuHistory className="w-4 h-4" /> View History
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* History View */}
        {view === 'history' && (
          <div className="space-y-4">
            <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">Past Analyses</h2>
            {loadingSessions ? (
              <div className="flex items-center justify-center py-12">
                <LuLoader className="w-6 h-6 text-primary-500 animate-spin" />
              </div>
            ) : sessions.length === 0 ? (
              <div className="text-center py-12 text-slate-400">
                <LuBrain className="w-12 h-12 mx-auto mb-3 opacity-30" />
                <p className="text-sm">No analyses yet. Start your first one!</p>
              </div>
            ) : (
              <div className="space-y-3">
                {sessions.map(s => (
                  <SessionCard key={s.session_id} session={s} onClick={() => viewSession(s)} />
                ))}
              </div>
            )}
          </div>
        )}

        {/* Detail View */}
        {view === 'detail' && selectedSession && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">
                Analysis #{selectedSession.session_id}
              </h2>
              <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${selectedSession.status === 'completed' ? 'text-emerald-600 bg-emerald-50' : 'text-red-600 bg-red-50'
                }`}>
                {selectedSession.status}
              </span>
            </div>
            <p className="text-xs text-slate-400">
              {new Date(selectedSession.created_at).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' })}
            </p>
            <div className="space-y-3">
              {(selectedSession.steps || []).map((step, i) => (
                <StepCard key={step.step_id} step={step} isLatest={false} />
              ))}
            </div>
            {/* Actions for this session */}
            <ActionsPanel
              actions={actions}
              onApprove={handleApprove}
              onReject={handleReject}
              onExecute={() => handleExecute(selectedSession.session_id)}
              executing={executing}
            />
          </div>
        )}

        {/* Error Banner */}
        {error && view === 'home' && (
          <div className="bg-red-50 border border-red-200 rounded-2xl p-4 flex items-center gap-3">
            <LuTriangleAlert className="w-5 h-5 text-red-500 flex-shrink-0" />
            <p className="text-sm text-red-700">{error}</p>
          </div>
        )}
      </div>
    </div >
  )
}
