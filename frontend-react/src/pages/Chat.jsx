import { useEffect, useState, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import { motion, AnimatePresence } from 'framer-motion'
import { api } from '../services/api'
import { LuSend, LuSparkles, LuUser, LuTrash2, LuMessageCircle, LuPlus, LuHistory, LuBot } from 'react-icons/lu'
import { Card, CardTitle, IconWrapper } from '../components/Card'

export default function Chat() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [status, setStatus] = useState('')
  const [pending, setPending] = useState(false)
  const [sessions, setSessions] = useState([])
  const [selectedSession, setSelectedSession] = useState(null)
  const patientId = parseInt(localStorage.getItem('patient_id'))
  const scrollRef = useRef(null)

  useEffect(() => {
    api.llmStatus().then(s => setStatus(s.provider)).catch(() => { })
    try {
      const saved = JSON.parse(localStorage.getItem('chat_history') || '[]')
      if (Array.isArray(saved)) setMessages(saved)
    } catch { }
    const hasSession = !!(localStorage.getItem('user_id') || localStorage.getItem('patient_id'))
    if (hasSession) { api.aiListSessions().then(setSessions).catch(() => { }) }
  }, [])

  useEffect(() => {
    try { localStorage.setItem('chat_history', JSON.stringify(messages.slice(-50))) } catch { }
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages])

  async function send(e) {
    e.preventDefault()
    if (!input.trim() || pending) return
    const msg = { role: 'user', content: input }
    setMessages(prev => [...prev, { who: 'You', text: input }])
    setInput('')
    setPending(true)
    try {
      const history = messages.map(m => ({ role: m.who === 'You' ? 'user' : 'assistant', content: m.text }))
      if ((status || '').toLowerCase() === 'ollama') {
        const res = await api.chat({ patient_id: patientId, prompt: msg.content, history })
        setMessages(prev => [...prev, { who: 'AI', text: res.reply || '(no response)' }])
      } else {
        setMessages(prev => [...prev, { who: 'AI', text: '' }])
        const idx = messages.length + 1
        await api.chatStream({ patient_id: patientId, prompt: msg.content, history }, (chunk) => {
          setMessages(prev => {
            const copy = [...prev]
            copy[idx] = { who: 'AI', text: (copy[idx]?.text || '') + chunk }
            return copy
          })
        })
      }
    } catch (err) {
      setMessages(prev => [...prev, { who: 'AI', text: '[Error] ' + (err.message || 'Request failed') }])
    } finally {
      setPending(false)
    }
  }

  const suggestions = [
    'What are melanoma warning signs?',
    'How do I protect my skin from high UV?',
    'Is Retinol safe for sensitive skin?',
    'Explain the ABCDE rule for moles',
  ]

  async function saveCurrentToHistory() {
    if (!messages || messages.length === 0) return
    let title = messages.find(m => m.who === 'You')?.text || 'New Session'
    if (title.length > 30) title = title.slice(0, 30) + '…'
    try {
      const sess = await api.aiCreateSession(title)
      for (const m of messages) {
        const role = m.who === 'You' ? 'user' : 'assistant'
        await api.aiAddSessionMessage(sess.session_id, role, m.text || '')
      }
      const list = await api.aiListSessions(); setSessions(list); setSelectedSession(sess.session_id)
    } catch { }
  }

  async function loadSession(id) {
    setSelectedSession(id)
    try {
      const msgs = await api.aiListSessionMessages(id)
      const ui = msgs.map(m => ({ who: m.role === 'user' ? 'You' : 'AI', text: m.content }))
      setMessages(ui)
    } catch { }
  }

  async function deleteSession(id) {
    try { await api.aiDeleteSession(id) } catch { }
    const list = await api.aiListSessions().catch(() => [])
    setSessions(list)
    if (selectedSession === id) { setSelectedSession(null); setMessages([]) }
  }

  return (
    <div className="h-[calc(100vh-140px)] flex gap-6">
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <Card variant="glass" className="flex-1 flex flex-col p-0 overflow-hidden" hover={false}>
          {/* Header */}
          <div className="bg-surface-elevated/50 border-b border-white/10 p-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <IconWrapper variant="ai">
                <LuBot size={20} />
              </IconWrapper>
              <div>
                <h1 className="font-bold text-text-primary">AI Medical Assistant</h1>
                <p className="text-xs text-primary-400 font-medium flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-primary-400 animate-pulse" />
                  Online • {status || 'Gemini'}
                </p>
              </div>
            </div>
            <div className="flex gap-2">
              <button onClick={() => { setMessages([]); setSelectedSession(null) }} className="btn-ghost text-xs">New Chat</button>
              <button onClick={saveCurrentToHistory} className="btn-ghost text-xs">Save</button>
            </div>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-6 space-y-6" ref={scrollRef}>
            {messages.length === 0 && (
              <div className="h-full flex flex-col items-center justify-center text-center">
                <div className="w-20 h-20 rounded-2xl bg-ai-500/10 flex items-center justify-center mb-6">
                  <LuSparkles className="text-ai-400" size={36} />
                </div>
                <h3 className="text-2xl font-bold text-text-primary mb-2">How can I help your skin today?</h3>
                <p className="text-text-tertiary mb-8">Ask me about symptoms, treatments, or ingredients</p>
                <div className="grid grid-cols-2 gap-3 w-full max-w-lg">
                  {suggestions.map((s, i) => (
                    <motion.button
                      key={i}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={() => setInput(s)}
                      className="p-4 text-sm text-left bg-white/5 border border-white/10 rounded-xl hover:border-ai-500/30 hover:bg-ai-500/5 transition-all text-text-secondary"
                    >
                      {s}
                    </motion.button>
                  ))}
                </div>
              </div>
            )}

            <AnimatePresence>
              {messages.map((m, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`flex gap-4 ${m.who === 'You' ? 'flex-row-reverse' : ''}`}
                >
                  <div className={`w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0 mt-1 ${m.who === 'You' ? 'bg-accent-500/10 text-accent-400' : 'bg-ai-500/10 text-ai-400'}`}>
                    {m.who === 'You' ? <LuUser size={16} /> : <LuBot size={18} />}
                  </div>
                  <div className={`max-w-[80%] p-4 rounded-2xl text-sm leading-relaxed ${m.who === 'You' ? 'bg-accent-500/10 text-text-primary rounded-tr-none border border-accent-500/20' : 'bg-white/5 border border-white/10 text-text-secondary rounded-tl-none'}`}>
                    {m.who === 'AI' ? <ReactMarkdown className="prose prose-sm prose-invert max-w-none">{m.text}</ReactMarkdown> : m.text}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>

            {pending && (
              <div className="flex gap-4">
                <div className="w-9 h-9 rounded-xl bg-ai-500/10 text-ai-400 flex items-center justify-center flex-shrink-0 animate-pulse">
                  <LuBot size={18} />
                </div>
                <div className="bg-white/5 border border-white/10 p-4 rounded-2xl rounded-tl-none text-sm text-text-muted">
                  <span className="inline-flex gap-1">
                    <span className="w-2 h-2 rounded-full bg-ai-400 animate-bounce" style={{ animationDelay: '0ms' }} />
                    <span className="w-2 h-2 rounded-full bg-ai-400 animate-bounce" style={{ animationDelay: '150ms' }} />
                    <span className="w-2 h-2 rounded-full bg-ai-400 animate-bounce" style={{ animationDelay: '300ms' }} />
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Input */}
          <div className="p-4 bg-surface-elevated/50 border-t border-white/10">
            <form onSubmit={send} className="relative">
              <input
                className="w-full pl-5 pr-14 py-4 bg-white/5 border border-white/10 rounded-2xl focus:outline-none focus:ring-2 focus:ring-ai-500/30 focus:border-ai-500/50 focus:bg-white/10 transition-all text-text-primary placeholder:text-text-muted"
                placeholder="Ask about symptoms, treatments, or ingredients..."
                value={input}
                onChange={e => setInput(e.target.value)}
                disabled={pending}
              />
              <button
                type="submit"
                disabled={pending || !input.trim()}
                className="absolute right-2 top-2 bottom-2 aspect-square bg-ai-500 hover:bg-ai-600 disabled:bg-white/10 disabled:text-text-muted text-white rounded-xl flex items-center justify-center transition-all shadow-lg shadow-ai-500/20 disabled:shadow-none"
              >
                <LuSend size={18} />
              </button>
            </form>
          </div>
        </Card>
      </div>

      {/* Sidebar History (Desktop) */}
      <aside className="w-80 hidden xl:flex flex-col gap-4">
        <Card variant="glass" className="h-full p-6 flex flex-col" hover={false}>
          <div className="flex items-center gap-2 mb-4">
            <LuHistory className="text-text-tertiary" size={18} />
            <CardTitle>History</CardTitle>
          </div>
          <div className="flex-1 overflow-y-auto space-y-2 pr-1">
            {sessions.map(s => (
              <div
                key={s.session_id}
                className={`group p-3 rounded-xl border transition-all cursor-pointer relative ${selectedSession === s.session_id ? 'bg-ai-500/10 border-ai-500/30' : 'bg-white/5 border-transparent hover:bg-white/10 hover:border-white/10'}`}
                onClick={() => loadSession(s.session_id)}
              >
                <div className="text-sm font-medium text-text-primary truncate pr-6">{s.title}</div>
                <div className="text-xs text-text-muted mt-1">{new Date(s.created_at).toLocaleDateString()}</div>
                <button
                  onClick={(e) => { e.stopPropagation(); deleteSession(s.session_id) }}
                  className="absolute right-2 top-3 text-text-muted hover:text-danger opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <LuTrash2 size={14} />
                </button>
              </div>
            ))}
            {sessions.length === 0 && <div className="text-text-muted text-sm text-center py-10">No saved chats.</div>}
          </div>
        </Card>
      </aside>
    </div>
  )
}
