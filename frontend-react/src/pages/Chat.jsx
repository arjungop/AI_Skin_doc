import { useEffect, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { api } from '../services/api'

export default function Chat(){
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [status, setStatus] = useState('')
  const [pending, setPending] = useState(false)
  const [sessions, setSessions] = useState([])
  const [selectedSession, setSelectedSession] = useState(null)
  const patientId = parseInt(localStorage.getItem('patient_id'))
  const username = localStorage.getItem('username')

  useEffect(()=>{
    api.llmStatus().then(s=> setStatus(s.provider)).catch(()=>{})
    // Load chat history
    try{
      const saved = JSON.parse(localStorage.getItem('chat_history')||'[]')
      if (Array.isArray(saved)) setMessages(saved)
    }catch{}
    const hasSession = !!(localStorage.getItem('user_id') || localStorage.getItem('patient_id'))
    if (hasSession) { api.aiListSessions().then(setSessions).catch(()=>{}) }
  },[])

  useEffect(()=>{
    try{ localStorage.setItem('chat_history', JSON.stringify(messages.slice(-50))) }catch{}
  }, [messages])

  async function send(e){
    e.preventDefault()
    if (!input.trim() || pending) return
    const msg = { role:'user', content: input }
    setMessages(prev=>[...prev, { who:'You', text:input }])
    setInput('')
    setPending(true)
    try {
      const history = messages.map(m=>({ role:m.who==='You'?'user':'assistant', content:m.text }))
      if ((status||'').toLowerCase() === 'ollama'){
        // Use non-streaming for Ollama to avoid garbled tokens on some versions
        const res = await api.chat({ patient_id: patientId, prompt: msg.content, history })
        setMessages(prev=>[...prev, { who:'AI', text: res.reply || '(no response)' }])
      } else {
        // Streaming for Gemini/Azure/OpenAI
        setMessages(prev=>[...prev, { who:'AI', text: '' }])
        const idx = messages.length + 1
        await api.chatStream({ patient_id: patientId, prompt: msg.content, history }, (chunk)=>{
          setMessages(prev=>{
            const copy = [...prev]
            copy[idx] = { who:'AI', text: (copy[idx]?.text || '') + chunk }
            return copy
          })
        })
      }
    } catch(err){
      setMessages(prev=>[...prev, { who:'AI', text: '[Error] '+(err.message||'Request failed') }])
    } finally {
      setPending(false)
    }
  }

  const suggestions = [
    'What are melanoma warning signs?',
    'How to protect skin from sun?',
    'When should I see a dermatologist?',
    'Explain ABCDE rule in simple terms',
  ]

  async function saveCurrentToHistory(){
    if (!messages || messages.length===0) return
    let title = messages.find(m=> m.who==='You')?.text || 'Chat'
    if (title.length>50) title = title.slice(0,50)+'…'
    try{
      const sess = await api.aiCreateSession(title)
      for (const m of messages){
        const role = m.who==='You'?'user':'assistant'
        await api.aiAddSessionMessage(sess.session_id, role, m.text||'')
      }
      const list = await api.aiListSessions(); setSessions(list); setSelectedSession(sess.session_id)
    }catch{}
  }

  async function loadSession(id){
    setSelectedSession(id)
    try{
      const msgs = await api.aiListSessionMessages(id)
      const ui = msgs.map(m=> ({ who: m.role==='user'?'You':'AI', text: m.content }))
      setMessages(ui)
      try{ localStorage.setItem('chat_history', JSON.stringify(ui)) }catch{}
    }catch{}
  }

  async function deleteSession(id){
    try{ await api.aiDeleteSession(id) }catch{}
    const list = await api.aiListSessions().catch(()=>[])
    setSessions(list)
    if (selectedSession===id){ setSelectedSession(null); setMessages([]) }
  }

  const getProviderDisplayName = (provider) => {
    switch(provider?.toLowerCase()) {
      case 'gemini': return 'Gemini AI (Primary)'
      case 'azure': return 'Azure OpenAI'
      case 'openai': return 'OpenAI'
      case 'ollama': return 'Ollama'
      default: return provider || 'Unknown'
    }
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
      <div className="lg:col-span-3 flex flex-col min-h-[60vh]">
        <div className="flex items-center gap-2 mb-2">
          <h1 className="text-2xl font-semibold">AI Doctor Chat</h1>
          <span className="chip">{getProviderDisplayName(status)}</span>
          {pending && <span className="chip">AI is typing…</span>}
        </div>
        <div className="chatbox flex-1 overflow-y-auto">
          {messages.map((m,i)=> (
            <div key={i} className={"bubble "+(m.who==='You'?'me':'ai')}>
              {m.who==='AI' ? (
                <div className="pre"><ReactMarkdown>{m.text||''}</ReactMarkdown></div>
              ) : (
                <>
                  <b>You:</b> {m.text}
                </>
              )}
            </div>
          ))}
        </div>
        <div className="row sticky bottom-0 mt-3">
          <button className="btn-ghost" onClick={()=> { setMessages([]); setSelectedSession(null) }}>New chat</button>
          <button className="btn-ghost" onClick={saveCurrentToHistory}>Save to history</button>
          <button className="btn-ghost" onClick={()=> { const last=messages.filter(m=>m.who==='AI').slice(-1)[0]; if(last) navigator.clipboard.writeText(last.text||'') }}>Copy last</button>
        </div>
        <form onSubmit={send} className="row sticky bottom-0">
          <input className="flex-1" value={input} onChange={e=>setInput(e.target.value)} placeholder={"Message as "+(username||'you')} disabled={pending} />
          <button className="btn-primary" type="submit" disabled={pending}>{pending ? 'Sending…' : 'Send'}</button>
        </form>
      </div>
      <aside className="lg:col-span-1 space-y-4">
        <div className="card">
          <h3 className="text-lg font-semibold mb-2">Suggestions</h3>
          <div className="grid gap-2">
            {suggestions.map((s,i)=> (
              <button key={i} className="px-3 py-2 rounded-lg border border-borderGray text-left hover:bg-slate-50" onClick={()=> setInput(s)}>{s}</button>
            ))}
          </div>
        </div>
        <div className="card">
          <h3 className="text-lg font-semibold mb-2">Chat history</h3>
          {sessions.length===0 ? (
            <div className="muted">No saved sessions.
            </div>
          ) : (
            <ul className="space-y-2">
              {sessions.map(s=> (
                <li key={s.session_id} className={"p-2 rounded-md border "+(selectedSession===s.session_id?"border-primaryBlue bg-slate-50":"border-borderGray hover:bg-slate-50") }>
                  <div className="text-sm font-medium truncate">{s.title}</div>
                  <div className="text-xs muted">{new Date(s.created_at).toLocaleString()}</div>
                  <div className="row mt-1">
                    <button className="btn-ghost btn-sm" onClick={()=> loadSession(s.session_id)}>Open</button>
                    <button className="btn-ghost btn-sm" onClick={()=> deleteSession(s.session_id)}>Delete</button>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </aside>
    </div>
  )
}
