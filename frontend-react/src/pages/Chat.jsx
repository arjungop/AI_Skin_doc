import { useEffect, useState, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import { api } from '../services/api'
import { FaPaperPlane, FaRobot, FaUser, FaNotesMedical, FaHistory, FaTrashAlt, FaPen } from 'react-icons/fa'

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
    <div className="h-[calc(100vh-140px)] animate-fade-in flex gap-6">
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col bg-white dark:bg-slate-800 rounded-3xl shadow-xl border border-slate-200 dark:border-slate-700 overflow-hidden relative">
        {/* Header */}
        <div className="bg-slate-50 dark:bg-slate-900 border-b border-slate-100 p-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-indigo-100 text-indigo-600 flex items-center justify-center">
              <FaRobot size={20} />
            </div>
            <div>
              <h1 className="font-bold text-slate-800 dark:text-white">AI Medical Assistant</h1>
              <p className="text-xs text-green-600 font-medium flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
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
        <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-slate-50/30" ref={scrollRef}>
          {messages.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center text-center opacity-60">
              <FaNotesMedical className="text-6xl text-slate-300 mb-4" />
              <h3 className="text-xl font-serif text-slate-700">How can I help your skin today?</h3>
              <div className="mt-8 grid grid-cols-2 gap-3 w-full max-w-lg">
                {suggestions.map((s, i) => (
                  <button key={i} onClick={() => setInput(s)} className="p-3 text-sm text-left bg-white border border-slate-200 rounded-xl hover:border-indigo-300 hover:shadow-md transition-all text-slate-600">
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((m, i) => (
            <div key={i} className={`flex gap-4 ${m.who === 'You' ? 'flex-row-reverse' : ''}`}>
              <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 mt-1 shadow-sm ${m.who === 'You' ? 'bg-slate-200 text-slate-600' : 'bg-indigo-600 text-white'}`}>
                {m.who === 'You' ? <FaUser size={12} /> : <FaRobot size={14} />}
              </div>
              <div className={`max-w-[80%] p-4 rounded-2xl shadow-sm text-sm leading-relaxed ${m.who === 'You' ? 'bg-indigo-50 text-indigo-900 rounded-tr-none' : 'bg-white border border-slate-100 text-slate-700 rounded-tl-none'}`}>
                {m.who === 'AI' ? <ReactMarkdown className="prose prose-sm">{m.text}</ReactMarkdown> : m.text}
              </div>
            </div>
          ))}
          {pending && (
            <div className="flex gap-4">
              <div className="w-8 h-8 rounded-full bg-indigo-600 text-white flex items-center justify-center flex-shrink-0 animate-bounce"><FaRobot size={14} /></div>
              <div className="bg-white border border-slate-100 p-4 rounded-2xl rounded-tl-none shadow-sm text-sm text-slate-400">Thinking...</div>
            </div>
          )}
        </div>

        {/* Input */}
        <div className="p-4 bg-white dark:bg-slate-800 border-t border-slate-100">
          <form onSubmit={send} className="relative">
            <input
              className="w-full pl-6 pr-14 py-4 bg-slate-50 border border-slate-200 rounded-2xl focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:bg-white transition-all shadow-inner"
              placeholder="Ask about symptoms, treatments, or ingredients..."
              value={input}
              onChange={e => setInput(e.target.value)}
              disabled={pending}
            />
            <button
              type="submit"
              disabled={pending || !input.trim()}
              className="absolute right-2 top-2 bottom-2 aspect-square bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-300 text-white rounded-xl flex items-center justify-center transition-colors shadow-lg shadow-indigo-200"
            >
              <FaPaperPlane />
            </button>
          </form>
        </div>
      </div>

      {/* Sidebar History (Desktop) */}
      <aside className="w-80 hidden xl:flex flex-col gap-4">
        <div className="bg-white p-6 rounded-3xl shadow-lg border border-slate-100 h-full flex flex-col">
          <h3 className="font-serif text-lg text-slate-800 mb-4 flex items-center gap-2">
            <FaHistory className="text-slate-400" /> History
          </h3>
          <div className="flex-1 overflow-y-auto space-y-2 pr-1 custom-scrollbar">
            {sessions.map(s => (
              <div key={s.session_id} className={`group p-3 rounded-xl border transition-all cursor-pointer relative ${selectedSession === s.session_id ? 'bg-indigo-50 border-indigo-200 shadow-sm' : 'bg-slate-50 border-transparent hover:bg-white hover:border-slate-200 hover:shadow-sm'}`} onClick={() => loadSession(s.session_id)}>
                <div className="text-sm font-medium text-slate-700 truncate pr-6">{s.title}</div>
                <div className="text-xs text-slate-400 mt-1">{new Date(s.created_at).toLocaleDateString()}</div>
                <button
                  onClick={(e) => { e.stopPropagation(); deleteSession(s.session_id) }}
                  className="absolute right-2 top-3 text-slate-300 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <FaTrashAlt size={12} />
                </button>
              </div>
            ))}
            {sessions.length === 0 && <div className="text-slate-400 text-sm text-center py-10">No saved chats.</div>}
          </div>
        </div>
      </aside>
    </div>
  )
}
