import { useEffect, useState, useRef, useCallback, useMemo } from 'react'
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

  // Handle different user roles - doctors/admins may not have patient_id
  const role = localStorage.getItem('role') || 'PATIENT'
  const patientId = parseInt(localStorage.getItem('patient_id')) || null
  const userId = parseInt(localStorage.getItem('user_id')) || null
  const chatUserId = patientId || userId // Fallback to user_id for non-patients

  const scrollRef = useRef(null)

  const chunkBuffer = useRef('')
  const updateTimeout = useRef(null)

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
    if (scrollRef.current) {
      requestAnimationFrame(() => {
        scrollRef.current.scrollTop = scrollRef.current.scrollHeight
      })
    }
  }, [messages])

  const flushBuffer = useCallback(() => {
    if (chunkBuffer.current) {
      const chunk = chunkBuffer.current
      chunkBuffer.current = ''
      setMessages(prev => {
        const copy = [...prev]
        const lastIdx = copy.length - 1
        if (lastIdx >= 0 && copy[lastIdx].who === 'AI') {
          copy[lastIdx] = { who: 'AI', text: (copy[lastIdx].text || '') + chunk }
        }
        return copy
      })
    }
  }, [])

  async function send(e) {
    e.preventDefault()
    if (!input.trim() || pending) return
    const msg = { role: 'user', content: input }
    const userMessage = { who: 'You', text: input }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setPending(true)

    try {
      const history = messages.map(m => ({ role: m.who === 'You' ? 'user' : 'assistant', content: m.text }))
      if ((status || '').toLowerCase() === 'ollama') {
        const res = await api.chat({ patient_id: chatUserId, prompt: msg.content, history })
        setMessages(prev => [...prev, { who: 'AI', text: res.reply || '(no response)' }])
      } else {
        setMessages(prev => [...prev, { who: 'AI', text: '' }])
        await api.chatStream({ patient_id: chatUserId, prompt: msg.content, history }, (chunk) => {
          chunkBuffer.current += chunk
          if (!updateTimeout.current) {
            updateTimeout.current = setTimeout(() => {
              flushBuffer()
              updateTimeout.current = null
            }, 50)
          }
        })
        if (updateTimeout.current) {
          clearTimeout(updateTimeout.current)
          updateTimeout.current = null
        }
        flushBuffer()
      }
    } catch (err) {
      setMessages(prev => [...prev, { who: 'AI', text: '[Error] ' + (err.message || 'Request failed') }])
    } finally {
      setPending(false)
    }
  }

  const suggestions = useMemo(() => [
    'What are melanoma warning signs?',
    'How do I protect my skin from high UV?',
    'Is Retinol safe for sensitive skin?',
    'Explain the ABCDE rule for moles',
  ], [])

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
        <Card className="flex-1 flex flex-col p-0 overflow-hidden shadow-soft-xl border-slate-200" hover={false} padding="none">
          {/* Header */}
          <div className="bg-white border-b border-slate-100 p-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <IconWrapper variant="ai" size="sm" className="bg-violet-50 text-violet-600">
                <LuBot size={20} />
              </IconWrapper>
              <div>
                <h1 className="font-bold text-slate-900">Medical Assistant</h1>
                <p className="text-xs text-slate-500 font-medium flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                  Online • {status || 'Gemini'}
                </p>
              </div>
            </div>
            <div className="flex gap-2">
              <button onClick={() => { setMessages([]); setSelectedSession(null) }} className="btn btn-ghost text-xs px-3 py-2">New Chat</button>
              <button onClick={saveCurrentToHistory} className="btn btn-ghost text-xs px-3 py-2">Save</button>
            </div>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-slate-50/50" ref={scrollRef}>
            {messages.length === 0 && (
              <div className="h-full flex flex-col items-center justify-center text-center">
                <div className="w-16 h-16 rounded-2xl bg-white shadow-soft-md flex items-center justify-center mb-6 text-violet-500">
                  <LuSparkles size={32} />
                </div>
                <h3 className="text-xl font-bold text-slate-900 mb-2">How can I help you?</h3>
                <p className="text-slate-500 mb-8 max-w-xs mx-auto">I can answer questions about skin conditions, ingredients, and routine safety.</p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full max-w-lg">
                  {suggestions.map((s, i) => (
                    <motion.button
                      key={i}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={() => setInput(s)}
                      className="p-4 text-sm text-left bg-white border border-slate-200 rounded-xl hover:border-violet-300 hover:shadow-soft-md transition-all text-slate-600"
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
                  transition={{ duration: 0.15 }}
                  className={`flex gap-4 ${m.who === 'You' ? 'flex-row-reverse' : ''}`}
                >
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 mt-1 shadow-sm ${m.who === 'You' ? 'bg-white text-slate-900 border border-slate-200' : 'bg-violet-600 text-white'}`}>
                    {m.who === 'You' ? <LuUser size={14} /> : <LuBot size={16} />}
                  </div>
                  <div className={`max-w-[85%] px-5 py-3.5 rounded-2xl text-sm leading-relaxed shadow-sm ${m.who === 'You' ? 'bg-white text-slate-800 rounded-tr-none border border-slate-200' : 'bg-white text-slate-700 rounded-tl-none border border-slate-200'}`}>
                    {m.who === 'AI' ? <ReactMarkdown className="prose prose-sm prose-slate max-w-none">{m.text}</ReactMarkdown> : m.text}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>

            {pending && messages[messages.length - 1]?.text === '' && (
              <div className="flex gap-4">
                <div className="w-8 h-8 rounded-full bg-violet-600 text-white flex items-center justify-center flex-shrink-0 animate-pulse">
                  <LuBot size={16} />
                </div>
                <div className="bg-white border border-slate-200 px-5 py-4 rounded-2xl rounded-tl-none shadow-sm">
                  <div className="flex gap-1">
                    <div className="w-2 h-2 rounded-full bg-violet-400 animate-bounce" style={{ animationDelay: '0ms' }} />
                    <div className="w-2 h-2 rounded-full bg-violet-400 animate-bounce" style={{ animationDelay: '150ms' }} />
                    <div className="w-2 h-2 rounded-full bg-violet-400 animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Input */}
          <div className="p-4 bg-white border-t border-slate-100">
            <form onSubmit={send} className="relative">
              <input
                className="w-full pl-5 pr-14 py-4 bg-slate-50 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-violet-500/10 focus:border-violet-500 transition-all text-slate-900 placeholder:text-slate-400"
                placeholder="Type your health question..."
                value={input}
                onChange={e => setInput(e.target.value)}
                disabled={pending}
              />
              <button
                type="submit"
                disabled={pending || !input.trim()}
                className="absolute right-2 top-2 bottom-2 aspect-square bg-violet-600 hover:bg-violet-700 disabled:bg-slate-200 disabled:text-slate-400 text-white rounded-lg flex items-center justify-center transition-all shadow-lg shadow-violet-500/20 disabled:shadow-none"
              >
                <LuSend size={18} />
              </button>
            </form>
          </div>
        </Card>
      </div>

      {/* Sidebar History (Desktop) */}
      <aside className="w-72 hidden xl:flex flex-col gap-4">
        <Card className="h-full flex flex-col" hover={false} padding="lg">
          <div className="flex items-center gap-2 mb-4 text-slate-900 font-bold">
            <LuHistory className="text-slate-400" size={18} />
            <h2>History</h2>
          </div>
          <div className="flex-1 overflow-y-auto space-y-2 pr-1">
            {sessions.map(s => (
              <div
                key={s.session_id}
                className={`group p-3 rounded-xl border transition-all cursor-pointer relative ${selectedSession === s.session_id ? 'bg-violet-50 border-violet-100 text-violet-700' : 'bg-white border-transparent hover:bg-slate-50 hover:border-slate-200'}`}
                onClick={() => loadSession(s.session_id)}
              >
                <div className="text-sm font-medium truncate pr-6">{s.title}</div>
                <div className="text-xs text-slate-400 mt-1">{new Date(s.created_at).toLocaleDateString()}</div>
                <button
                  onClick={(e) => { e.stopPropagation(); deleteSession(s.session_id) }}
                  className="absolute right-2 top-3 text-slate-300 hover:text-rose-500 opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <LuTrash2 size={14} />
                </button>
              </div>
            ))}
            {sessions.length === 0 && <div className="text-slate-400 text-sm text-center py-10 italic">No saved conversations</div>}
          </div>
        </Card>
      </aside>
    </div>
  )
}
