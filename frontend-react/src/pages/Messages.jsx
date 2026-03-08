import { useEffect, useRef, useState, useCallback } from 'react'
import { api } from '../services/api'
import useNotifications from '../hooks/useNotifications'
import { motion, AnimatePresence } from 'framer-motion'
import {
  LuSend, LuPlus, LuVideo,
  LuSearch, LuLoader, LuCheck,
  LuCheckCheck, LuCornerUpLeft,
  LuTriangleAlert, LuExternalLink, LuMessageCircle, LuX, LuUserPlus,
  LuCircle
} from 'react-icons/lu'
import { Card } from '../components/Card'

const POLL_INTERVAL = 10_000 // 10 seconds (fallback when WS connected)
const HEARTBEAT_INTERVAL = 30_000 // 30 seconds
const TYPING_DEBOUNCE = 2500

export default function Messages() {
  const [rooms, setRooms] = useState([])
  const [activeRoom, setActiveRoom] = useState(null)
  const [messages, setMessages] = useState([])
  const [messageText, setMessageText] = useState('')
  const [isUrgent, setIsUrgent] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [sendError, setSendError] = useState('')
  const [loadError, setLoadError] = useState('')

  // New conversation modal state
  const [showNewChat, setShowNewChat] = useState(false)
  const [contacts, setContacts] = useState([])
  const [contactSearch, setContactSearch] = useState('')
  const [loadingContacts, setLoadingContacts] = useState(false)

  // Real-time sync state
  const [onlineUsers, setOnlineUsers] = useState({}) // userId → {status, last_seen}
  const [typingUsers, setTypingUsers] = useState({})  // roomId → {username, ts}
  const { subscribe, connected: wsConnected } = useNotifications()

  const messagesEndRef = useRef(null)
  const pollIntervalRef = useRef(null)
  const heartbeatRef = useRef(null)
  const typingTimeoutRef = useRef(null)
  const lastTypingSentRef = useRef(0)
  const activeRoomRef = useRef(null)  // keep current room id accessible in callbacks
  const mountedRef = useRef(true)

  const currentUserId = parseInt(localStorage.getItem('user_id') || '0')
  const role = (localStorage.getItem('role') || '').toUpperCase()

  // Reset mountedRef on every mount (survives React 18 StrictMode unmount/remount)
  useEffect(() => {
    mountedRef.current = true
    return () => { mountedRef.current = false }
  }, [])

  // Keep activeRoomRef in sync
  useEffect(() => { activeRoomRef.current = activeRoom }, [activeRoom])

  // ── Load rooms ──────────────────────────────────────────────────────
  const loadRooms = useCallback(async () => {
    try {
      const data = await api.listRooms()
      if (mountedRef.current) {
        setRooms(Array.isArray(data) ? data : [])
        setLoadError('')
      }
    } catch (err) {
      console.error('loadRooms error:', err)
      if (mountedRef.current) setLoadError(err.message || 'Failed to load conversations')
    }
  }, [])

  useEffect(() => {
    setIsLoading(true)
    loadRooms().finally(() => setIsLoading(false))
  }, [loadRooms])

  // ── Heartbeat for online status ─────────────────────────────────────
  useEffect(() => {
    const beat = () => { try { api.sendHeartbeat() } catch { } }
    beat() // initial
    heartbeatRef.current = setInterval(beat, HEARTBEAT_INTERVAL)
    // Mark offline on page unload using synchronous XHR (sendBeacon can't carry auth headers)
    const handleUnload = () => {
      try {
        let token = localStorage.getItem('access_token')
        if (token) {
          token = String(token).replace(/^\"|\"$/g, '').trim()
          const xhr = new XMLHttpRequest()
          xhr.open('POST', `${api.base}/chat/offline`, false) // synchronous
          xhr.setRequestHeader('Authorization', `Bearer ${token}`)
          xhr.send()
        }
      } catch { }
    }
    window.addEventListener('beforeunload', handleUnload)
    return () => {
      clearInterval(heartbeatRef.current)
      window.removeEventListener('beforeunload', handleUnload)
      try { api.goOffline() } catch { }
    }
  }, [])

  // ── Fetch online status for participants in current rooms ──────────
  const refreshOnlineStatuses = useCallback(async () => {
    if (!rooms.length) return
    const userIds = new Set()
    rooms.forEach(r => {
      if (r.patient?.user_id) userIds.add(r.patient.user_id)
      if (r.doctor?.user_id) userIds.add(r.doctor.user_id)
    })
    const statuses = {}
    await Promise.allSettled(
      [...userIds].map(async uid => {
        try {
          const s = await api.getOnlineStatus(uid)
          statuses[uid] = s
        } catch { }
      })
    )
    if (mountedRef.current) setOnlineUsers(prev => ({ ...prev, ...statuses }))
  }, [rooms])

  useEffect(() => { refreshOnlineStatuses() }, [refreshOnlineStatuses])

  // ── WebSocket event listeners ───────────────────────────────────────
  useEffect(() => {
    const unsubs = []

    // New message notification
    unsubs.push(subscribe('new_message', (data) => {
      const roomId = data.room_id
      // If we're in this room, reload messages instantly
      if (activeRoomRef.current?.room_id === roomId) {
        loadMessages(roomId)
      }
      // Always refresh rooms for unread counts
      loadRooms()
    }))

    // Read receipt notification
    unsubs.push(subscribe('messages_read', (data) => {
      if (activeRoomRef.current?.room_id === data.room_id) {
        setMessages(prev =>
          prev.map(m =>
            m.sender_user_id === currentUserId
              ? { ...m, status: 'read' }
              : m
          )
        )
      }
      loadRooms()
    }))

    // Typing indicator
    unsubs.push(subscribe('typing', (data) => {
      setTypingUsers(prev => ({
        ...prev,
        [data.room_id]: { username: data.username, ts: Date.now() }
      }))
      // Auto-clear after 3s
      setTimeout(() => {
        setTypingUsers(prev => {
          const entry = prev[data.room_id]
          if (entry && Date.now() - entry.ts >= 2800) {
            const next = { ...prev }
            delete next[data.room_id]
            return next
          }
          return prev
        })
      }, 3000)
    }))

    return () => { unsubs.forEach(u => u()) }
  }, [subscribe, loadRooms, currentUserId]) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Polling (fallback & supplementary) ──────────────────────────────
  const loadMessages = useCallback(async (roomId) => {
    try {
      const res = await api.listMessages(roomId, 1, 100)
      if (mountedRef.current) setMessages(res.messages || [])
    } catch (err) { console.error('loadMessages error:', err) }
  }, [])

  const startPolling = useCallback((roomId) => {
    if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)
    pollIntervalRef.current = setInterval(() => {
      loadMessages(roomId)
      loadRooms()
    }, POLL_INTERVAL)
  }, [loadMessages, loadRooms])

  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)
      clearTimeout(typingTimeoutRef.current)
    }
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // ── Open a room ─────────────────────────────────────────────────────
  const openRoom = async (room) => {
    if (activeRoom?.room_id === room.room_id) return
    setActiveRoom(room)
    await loadMessages(room.room_id)
    // Mark messages as read
    try { await api.markRoomRead(room.room_id) } catch { }
    // Refresh room list to clear unread badge
    loadRooms()
    startPolling(room.room_id)
    if (window.innerWidth < 768) setSidebarOpen(false)
  }

  // ── Send message ────────────────────────────────────────────────────
  const sendMessage = async (e) => {
    e.preventDefault()
    if (!messageText.trim() || !activeRoom) return

    setSendError('')
    try {
      const newMsg = await api.postMessage(activeRoom.room_id, {
        content: messageText.trim(),
        is_urgent: isUrgent,
      })
      setMessages(prev => [...prev, newMsg])
      setMessageText('')
      setIsUrgent(false)
      loadRooms()
      // Re-fetch from DB to ensure consistent state
      loadMessages(activeRoom.room_id)
    } catch (err) {
      console.error(err)
      setSendError(err.message || 'Failed to send message. Please try again.')
    }
  }

  // ── Typing indicator sender ─────────────────────────────────────────
  const handleInputChange = (e) => {
    setMessageText(e.target.value)
    if (sendError) setSendError('')
    if (!activeRoom) return
    const now = Date.now()
    if (now - lastTypingSentRef.current > TYPING_DEBOUNCE) {
      lastTypingSentRef.current = now
      try { api.sendTyping(activeRoom.room_id) } catch { }
    }
  }

  // ── Toggle urgent on existing message ───────────────────────────────
  const toggleUrgent = async (messageId) => {
    try {
      const result = await api.markUrgent(messageId)
      setMessages(prev =>
        prev.map(m => m.message_id === messageId ? { ...m, is_urgent: result.is_urgent } : m)
      )
    } catch (err) { console.error(err) }
  }

  // ── Participant helper (fixed for all roles) ───────────────────────
  const getParticipantName = (room) => {
    if (!room) return 'Unknown'
    if (role === 'PATIENT' && room.doctor) {
      return `Dr. ${room.doctor.first_name || room.doctor.username}`
    }
    if (role === 'DOCTOR' && room.patient) {
      const name = `${room.patient.first_name || ''} ${room.patient.last_name || ''}`.trim()
      return name || room.patient.username || 'Patient'
    }
    if (role === 'ADMIN') {
      const patientName = room.patient ? `${room.patient.first_name} ${room.patient.last_name}`.trim() : 'Patient'
      const doctorName = room.doctor ? `Dr. ${room.doctor.first_name || room.doctor.username}` : 'Doctor'
      return `${patientName} ↔ ${doctorName}`
    }
    return 'Chat'
  }

  // ── Get the other participant's user_id for online status ──────────
  const getOtherUserId = (room) => {
    if (!room) return null
    if (role === 'PATIENT') return room.doctor?.user_id
    if (role === 'DOCTOR') return room.patient?.user_id
    return null // Admin doesn't have a single "other"
  }

  // ── Online status helper ────────────────────────────────────────────
  const isUserOnline = (userId) => {
    if (!userId) return false
    const s = onlineUsers[userId]
    return s?.status === 'online'
  }

  // ── Unread count per room ───────────────────────────────────────────
  const getUnreadCount = (room) => {
    if (role === 'PATIENT') return room.unread_count_patient || 0
    if (role === 'DOCTOR') return room.unread_count_doctor || 0
    if (role === 'ADMIN') return Math.max(room.unread_count_patient || 0, room.unread_count_doctor || 0)
    return 0
  }

  // ── Message status icon ─────────────────────────────────────────────
  const MessageStatusIcon = ({ status }) => {
    const s = (status || '').toLowerCase()
    if (s === 'read') return <LuCheckCheck size={12} className="text-blue-400" />
    if (s === 'delivered') return <LuCheckCheck size={12} />
    return <LuCheck size={12} />  // sent
  }

  // ── New conversation helpers ───────────────────────────────────────
  const openNewChatModal = async () => {
    setShowNewChat(true)
    setContactSearch('')
    setLoadingContacts(true)
    try {
      // Patients pick doctors, doctors pick patients
      const data = role === 'DOCTOR'
        ? await api.listPatients()
        : await api.listDoctors()
      setContacts(Array.isArray(data) ? data : data.items || [])
    } catch (err) { console.error(err); setContacts([]) }
    finally { setLoadingContacts(false) }
  }

  const startConversation = async (contact) => {
    try {
      let payload
      if (role === 'DOCTOR') {
        const doctorId = parseInt(localStorage.getItem('doctor_id') || '0')
        payload = { patient_id: contact.patient_id, doctor_id: doctorId }
      } else {
        const patientId = parseInt(localStorage.getItem('patient_id') || '0')
        payload = { patient_id: patientId, doctor_id: contact.doctor_id }
      }
      const newRoom = await api.createRoom(payload)
      setShowNewChat(false)
      await loadRooms()
      // Open the new/existing room
      const roomToOpen = { ...newRoom, room_id: newRoom.room_id }
      await openRoom(roomToOpen)
    } catch (err) { console.error(err) }
  }

  const getContactName = (c) => {
    if (role === 'DOCTOR') return `${c.first_name || ''} ${c.last_name || ''}`.trim() || c.username || c.email || 'Patient'
    return `Dr. ${c.user?.username || c.username || 'Unknown'}`
  }

  const filteredContacts = contacts.filter(c => {
    if (!contactSearch.trim()) return true
    const term = contactSearch.toLowerCase()
    const nameMatch = getContactName(c).toLowerCase().includes(term)
    const usernameMatch = (c.username || '').toLowerCase().includes(term)
    const emailMatch = (c.email || '').toLowerCase().includes(term)
    return nameMatch || usernameMatch || emailMatch
  })

  // ── Filtered rooms ─────────────────────────────────────────────────
  const filteredRooms = rooms.filter(r => {
    if (!searchQuery.trim()) return true
    return getParticipantName(r).toLowerCase().includes(searchQuery.toLowerCase())
  })

  return (
    <div className="h-[calc(100vh-100px)] flex gap-6 overflow-hidden relative">
      {/* Ambient blobs */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none z-0">
        <div className="absolute top-10 left-10 w-64 h-64 bg-primary-100/40 rounded-full blur-[80px]" />
        <div className="absolute bottom-10 right-10 w-64 h-64 bg-secondary-100/40 rounded-full blur-[80px]" />
      </div>

      {/* ─── Sidebar ───────────────────────────────────────────────────── */}
      <Card
        className={`flex flex-col w-full md:w-80 h-full p-0 transition-all absolute md:relative z-20 ${sidebarOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}`}
        hover={false}
      >
        <div className="p-4 border-b border-slate-100 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <h2 className="font-bold text-lg text-slate-900">Messages</h2>
            <span className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-emerald-500' : 'bg-slate-300'}`} title={wsConnected ? 'Real-time sync active' : 'Polling mode'} />
          </div>
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={openNewChatModal}
            className="w-8 h-8 rounded-lg bg-primary-50 text-primary-500 flex items-center justify-center hover:bg-primary-100 transition-colors"
            title="New conversation"
          >
            <LuPlus size={18} />
          </motion.button>
        </div>

        <div className="p-3">
          <div className="relative">
            <LuSearch className="absolute left-3 top-2.5 text-slate-400" size={16} />
            <input
              className="w-full bg-slate-50 border border-slate-200 rounded-xl py-2 pl-9 pr-4 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none focus:border-primary-500 focus:ring-4 focus:ring-primary-500/10 transition-all"
              placeholder="Search conversations..."
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
            />
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {isLoading ? (
            <div className="flex justify-center p-4"><LuLoader className="animate-spin text-slate-400" /></div>
          ) : loadError ? (
            <div className="text-center p-4 text-sm">
              <p className="text-rose-500 mb-2">{loadError}</p>
              <button onClick={() => { setLoadError(''); setIsLoading(true); loadRooms().finally(() => setIsLoading(false)) }} className="text-primary-500 underline text-xs">Retry</button>
            </div>
          ) : filteredRooms.length === 0 ? (
            <div className="text-center text-slate-400 p-4 text-sm">No conversations yet</div>
          ) : (
            filteredRooms.map(room => {
              const hasUrgent = room.last_message?.is_urgent
              const unread = getUnreadCount(room)
              const otherUid = getOtherUserId(room)
              const online = isUserOnline(otherUid)
              const typing = typingUsers[room.room_id]
              return (
                <motion.button
                  key={room.room_id}
                  onClick={() => openRoom(room)}
                  whileHover={{ backgroundColor: 'rgba(241,245,249,1)' }}
                  className={`w-full text-left p-3 rounded-xl flex items-center gap-3 transition-colors ${activeRoom?.room_id === room.room_id
                    ? 'bg-primary-50 border border-primary-200'
                    : 'border border-transparent hover:bg-slate-50'
                    }`}
                >
                  <div className="relative">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary-400 to-primary-600 flex items-center justify-center flex-shrink-0 text-white font-bold text-sm">
                      {getParticipantName(room).charAt(0)}
                    </div>
                    {/* Online indicator */}
                    {online && (
                      <div className="absolute -bottom-0.5 -right-0.5 w-3.5 h-3.5 rounded-full bg-emerald-500 border-2 border-white" />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <span className={`font-semibold text-sm truncate ${activeRoom?.room_id === room.room_id ? 'text-primary-600' : 'text-slate-900'}`}>
                        {getParticipantName(room)}
                      </span>
                      <div className="flex items-center gap-1.5">
                        {unread > 0 && (
                          <span className="min-w-[20px] h-5 px-1.5 flex items-center justify-center rounded-full bg-primary-500 text-white text-[10px] font-bold">
                            {unread > 99 ? '99+' : unread}
                          </span>
                        )}
                        <span className="text-xs text-slate-400">
                          {new Date(room.last_message_at || room.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center gap-1 mt-0.5">
                      {hasUrgent && <LuTriangleAlert className="text-rose-500 flex-shrink-0" size={12} />}
                      <p className="text-xs text-slate-500 truncate">
                        {typing ? (
                          <span className="text-primary-500 italic">{typing.username} is typing...</span>
                        ) : (
                          room.last_message?.content || 'No messages yet'
                        )}
                      </p>
                    </div>
                  </div>
                </motion.button>
              )
            })
          )}
        </div>
      </Card>

      {/* ─── Chat Area ─────────────────────────────────────────────────── */}
      <div className={`flex-1 flex flex-col h-full relative z-10 transition-all ${!sidebarOpen ? 'w-full' : 'hidden md:flex'}`}>
        {activeRoom ? (
          <Card className="h-full flex flex-col p-0 overflow-hidden" hover={false}>
            {/* Header */}
            <div className="p-4 border-b border-slate-100 flex items-center justify-between bg-white/80 backdrop-blur-md">
              <div className="flex items-center gap-3">
                <button onClick={() => setSidebarOpen(true)} className="md:hidden p-2 -ml-2 text-slate-500">
                  <LuCornerUpLeft size={20} />
                </button>
                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center text-white font-bold shadow-lg shadow-primary-500/20">
                  {getParticipantName(activeRoom).charAt(0)}
                </div>
                <div>
                  <h3 className="font-bold text-slate-900">{getParticipantName(activeRoom)}</h3>
                  <span className="text-xs text-slate-400">
                    {(() => {
                      const uid = getOtherUserId(activeRoom)
                      const typing = typingUsers[activeRoom.room_id]
                      if (typing) return <span className="text-primary-500 italic">{typing.username} is typing...</span>
                      if (isUserOnline(uid)) return <span className="text-emerald-500 flex items-center gap-1"><LuCircle size={8} className="fill-emerald-500" /> Online</span>
                      return 'Secure medical messaging'
                    })()}
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-2">
                {/* Video link removed from here and moved to Appointments page */}
              </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-50/50">
              {messages.map((msg, i) => {
                const isMe = msg.sender_user_id === currentUserId
                const isSystem = msg.message_type === 'system'

                if (isSystem) {
                  return (
                    <div key={msg.message_id || i} className="flex justify-center my-6">
                      <div className="bg-slate-100/80 backdrop-blur-sm text-slate-600 border border-slate-200 text-xs px-4 py-2.5 rounded-2xl flex items-start sm:items-center gap-3 max-w-[85%] text-left sm:text-center shadow-sm">
                        {msg.is_urgent && <LuTriangleAlert className="text-rose-500 shrink-0 mt-0.5 sm:mt-0" size={16} />}
                        <span className={`whitespace-pre-wrap leading-relaxed ${msg.is_urgent ? "text-rose-700 font-medium" : ""}`}>
                          {msg.content}
                        </span>
                      </div>
                    </div>
                  )
                }

                return (
                  <motion.div
                    key={msg.message_id || i}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={`flex ${isMe ? 'justify-end' : 'justify-start'} group`}
                  >
                    <div className="relative max-w-[70%]">
                      {msg.is_urgent && (
                        <div className="absolute -top-2 -right-2 w-5 h-5 rounded-full bg-rose-500 text-white flex items-center justify-center z-10 shadow-sm">
                          <LuTriangleAlert size={10} />
                        </div>
                      )}
                      <div
                        className={`p-3.5 rounded-2xl text-sm leading-relaxed ${msg.is_urgent
                          ? isMe
                            ? 'bg-rose-500 text-white rounded-tr-none shadow-lg shadow-rose-500/20 ring-2 ring-rose-300'
                            : 'bg-rose-50 border-2 border-rose-200 text-slate-900 rounded-tl-none'
                          : isMe
                            ? 'bg-primary-500 text-white rounded-tr-none shadow-lg shadow-primary-500/20'
                            : 'bg-white border border-slate-100 text-slate-900 rounded-tl-none shadow-sm'
                          }`}
                      >
                        <p>{msg.content}</p>
                        <div className={`text-[10px] mt-1 flex items-center justify-end gap-1 ${msg.is_urgent
                          ? isMe ? 'text-white/70' : 'text-rose-400'
                          : isMe ? 'text-white/70' : 'text-slate-400'
                          }`}>
                          {msg.is_urgent && <span className="font-bold uppercase mr-1">URGENT</span>}
                          {new Date(msg.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                          {isMe && <MessageStatusIcon status={msg.status} />}
                        </div>
                      </div>

                      {/* Urgent toggle on hover */}
                      <button
                        onClick={() => toggleUrgent(msg.message_id)}
                        className={`absolute -left-8 top-1/2 -translate-y-1/2 p-1 rounded-md opacity-0 group-hover:opacity-100 transition-opacity ${msg.is_urgent ? 'text-rose-500 hover:bg-rose-50' : 'text-slate-400 hover:bg-slate-100'
                          }`}
                        title={msg.is_urgent ? 'Remove urgent' : 'Mark as urgent'}
                      >
                        <LuTriangleAlert size={14} />
                      </button>
                    </div>
                  </motion.div>
                )
              })}
              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="p-4 bg-white border-t border-slate-100">
              {sendError && (
                <div className="mb-2 px-3 py-2 bg-rose-50 border border-rose-200 text-rose-700 text-xs rounded-xl flex items-center gap-2">
                  <LuTriangleAlert size={12} className="shrink-0" />
                  {sendError}
                </div>
              )}
              <form onSubmit={sendMessage} className="flex gap-3 items-end">
                {/* Urgent toggle */}
                <button
                  type="button"
                  onClick={() => setIsUrgent(!isUrgent)}
                  className={`p-3 rounded-xl transition-all ${isUrgent
                    ? 'bg-rose-500 text-white shadow-lg shadow-rose-500/20'
                    : 'text-slate-400 hover:text-slate-700 hover:bg-slate-100'
                    }`}
                  title={isUrgent ? 'Sending as URGENT' : 'Mark as urgent'}
                >
                  <LuTriangleAlert size={20} />
                </button>

                <div className={`flex-1 border rounded-2xl flex items-center px-4 py-2 transition-all ${isUrgent
                  ? 'bg-rose-50 border-rose-300 focus-within:ring-2 focus-within:ring-rose-500/20'
                  : 'bg-slate-50 border-slate-200 focus-within:ring-2 focus-within:ring-primary-500/20 focus-within:border-primary-500'
                  }`}>
                  <input
                    className="flex-1 bg-transparent border-none focus:outline-none py-2 text-sm text-slate-900 placeholder:text-slate-400"
                    placeholder={isUrgent ? '🚨 Type urgent message...' : 'Type a message...'}
                    value={messageText}
                    onChange={handleInputChange}
                  />
                </div>

                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  type="submit"
                  disabled={!messageText.trim()}
                  className={`p-3 text-white rounded-xl shadow-lg disabled:scale-100 disabled:opacity-50 disabled:shadow-none transition-all ${isUrgent ? 'bg-rose-500 shadow-rose-500/20' : 'bg-primary-500 shadow-primary-500/20'
                    }`}
                >
                  <LuSend size={20} />
                </motion.button>
              </form>
            </div>
          </Card>
        ) : (
          <div className="h-full flex flex-col items-center justify-center text-center p-6 opacity-60">
            <div className="w-20 h-20 rounded-3xl bg-slate-100 border border-slate-200 flex items-center justify-center mb-6">
              <LuMessageCircle className="text-slate-400" size={40} />
            </div>
            <h3 className="text-xl font-bold text-slate-900 mb-2">Select a Conversation</h3>
            <p className="text-slate-400 max-w-sm">Choose a chat from the sidebar to start messaging.</p>
          </div>
        )}
      </div>

      {/* ─── New Conversation Modal ────────────────────────────────────── */}
      <AnimatePresence>
        {showNewChat && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-sm p-4"
            onClick={() => setShowNewChat(false)}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              onClick={e => e.stopPropagation()}
              className="bg-white rounded-2xl shadow-2xl w-full max-w-md max-h-[70vh] flex flex-col overflow-hidden border border-slate-200"
            >
              {/* Header */}
              <div className="p-5 border-b border-slate-100 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-primary-50 text-primary-500 flex items-center justify-center">
                    <LuUserPlus size={20} />
                  </div>
                  <div>
                    <h3 className="font-bold text-slate-900">New Conversation</h3>
                    <p className="text-xs text-slate-400">
                      {role === 'DOCTOR' ? 'Select a patient to message' : 'Select a doctor to message'}
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => setShowNewChat(false)}
                  className="p-2 rounded-lg text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors"
                >
                  <LuX size={18} />
                </button>
              </div>

              {/* Search */}
              <div className="p-4 border-b border-slate-50">
                <div className="relative">
                  <LuSearch className="absolute left-3 top-2.5 text-slate-400" size={16} />
                  <input
                    className="w-full bg-slate-50 border border-slate-200 rounded-xl py-2.5 pl-9 pr-4 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none focus:border-primary-500 focus:ring-4 focus:ring-primary-500/10 transition-all"
                    placeholder={role === 'DOCTOR' ? 'Search patients...' : 'Search doctors...'}
                    value={contactSearch}
                    onChange={e => setContactSearch(e.target.value)}
                    autoFocus
                  />
                </div>
              </div>

              {/* Contact list */}
              <div className="flex-1 overflow-y-auto p-2">
                {loadingContacts ? (
                  <div className="flex justify-center p-8">
                    <LuLoader className="animate-spin text-slate-400" size={24} />
                  </div>
                ) : filteredContacts.length === 0 ? (
                  <div className="text-center text-slate-400 p-8 text-sm">
                    {contactSearch ? 'No matches found' : 'No contacts available'}
                  </div>
                ) : (
                  filteredContacts.map((contact, idx) => {
                    const name = getContactName(contact)
                    const initials = name.replace('Dr. ', '').split(' ').map(s => s[0] || '').join('').slice(0, 2).toUpperCase()
                    return (
                      <motion.button
                        key={contact.patient_id || contact.doctor_id || idx}
                        whileHover={{ backgroundColor: 'rgba(241,245,249,1)' }}
                        onClick={() => startConversation(contact)}
                        className="w-full text-left p-3 rounded-xl flex items-center gap-3 transition-colors border border-transparent hover:border-slate-200"
                      >
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-400 to-primary-600 text-white flex items-center justify-center text-sm font-bold flex-shrink-0">
                          {initials}
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="font-semibold text-sm text-slate-900 truncate">{name}</p>
                          <p className="text-xs text-slate-400 truncate">
                            {contact.email || contact.specialization || 'Tap to start conversation'}
                          </p>
                        </div>
                        <LuMessageCircle className="text-slate-300 flex-shrink-0" size={16} />
                      </motion.button>
                    )
                  })
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
