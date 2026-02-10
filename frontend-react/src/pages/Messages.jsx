import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../services/api'
import { motion, AnimatePresence } from 'framer-motion'
import {
  LuSend, LuPlus, LuImage, LuFile, LuSmile,
  LuPhone, LuVideo, LuInfo, LuSearch, LuX, LuLoader, LuCheck,
  LuCheckCheck, LuTrash2, LuCornerUpLeft
} from 'react-icons/lu'
import { Card, CardTitle, IconWrapper } from '../components/Card'

export default function Messages() {
  const [rooms, setRooms] = useState([])
  const [activeRoom, setActiveRoom] = useState(null)
  const [messages, setMessages] = useState([])
  const [messageText, setMessageText] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [showNewConversation, setShowNewConversation] = useState(false)
  const [availableUsers, setAvailableUsers] = useState([])
  const [searchQuery, setSearchQuery] = useState('')
  const [isGeneratingNotes, setIsGeneratingNotes] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)

  const wsRef = useRef(null)
  const messagesEndRef = useRef(null)
  const currentUserId = parseInt(localStorage.getItem('user_id') || '0')

  useEffect(() => {
    loadRooms()
    return () => wsRef.current?.close()
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const loadRooms = async () => {
    setIsLoading(true)
    try {
      const data = await api.listRooms()
      setRooms(data)
    } catch (err) {
      console.error(err)
    } finally {
      setIsLoading(false)
    }
  }

  const loadMessages = async (roomId) => {
    try {
      const res = await api.listMessages(roomId, 1, 50)
      setMessages(res.messages?.reverse() || [])
    } catch (err) {
      console.error(err)
    }
  }

  const openRoom = async (room) => {
    if (activeRoom?.room_id === room.room_id) return
    setActiveRoom(room)
    await loadMessages(room.room_id)
    connectWebSocket(room.room_id)
    // Mobile handling
    if (window.innerWidth < 768) setSidebarOpen(false)
  }

  const connectWebSocket = (roomId) => {
    wsRef.current?.close()
    const token = localStorage.getItem('access_token')
    const wsUrl = `${api.base.replace('http', 'ws')}/chat/ws?room_id=${roomId}&token=${encodeURIComponent(token || '')}`
    const ws = new WebSocket(wsUrl)

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (data.type === 'new_message') {
          setMessages(prev => [...prev, data.data])
        }
      } catch (err) { console.error(err) }
    }
    wsRef.current = ws
  }

  const sendMessage = async (e) => {
    e.preventDefault()
    if (!messageText.trim() || !activeRoom) return

    const payload = { content: messageText.trim(), message_type: 'text' }
    try {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify(payload))
      } else {
        await api.postMessage(activeRoom.room_id, payload)
      }
      setMessageText('')
    } catch (err) { console.error(err) }
  }

  const getParticipantName = (room) => {
    if (!room) return 'Unknown'
    const role = localStorage.getItem('role')?.toLowerCase()
    if (role === 'patient' && room.doctor) return `Dr. ${room.doctor.username}`
    if (role === 'doctor' && room.patient) return `${room.patient.first_name} ${room.patient.last_name}`
    return 'Chat'
  }

  return (
    <div className="h-[calc(100vh-100px)] flex gap-6 overflow-hidden relative">
      {/* Sidebar background blobs */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none z-0">
        <div className="absolute top-10 left-10 w-64 h-64 bg-primary-100/40 rounded-full blur-[80px]" />
        <div className="absolute bottom-10 right-10 w-64 h-64 bg-secondary-100/40 rounded-full blur-[80px]" />
      </div>

      {/* Sidebar */}
      <Card
        className={`flex flex-col w-full md:w-80 h-full p-0 transition-all absolute md:relative z-20 ${sidebarOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}`}
        hover={false}
      >
        <div className="p-4 border-b border-slate-100 flex items-center justify-between">
          <h2 className="font-bold text-lg text-slate-900">Messages</h2>
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={() => setShowNewConversation(true)}
            className="w-8 h-8 rounded-lg bg-primary-50 text-primary-500 flex items-center justify-center hover:bg-primary-100 transition-colors"
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
          ) : rooms.length === 0 ? (
            <div className="text-center text-slate-400 p-4 text-sm">No conversations yet</div>
          ) : (
            rooms.map(room => (
              <motion.button
                key={room.room_id}
                onClick={() => openRoom(room)}
                whileHover={{ backgroundColor: 'rgba(241,245,249,1)' }}
                className={`w-full text-left p-3 rounded-xl flex items-center gap-3 transition-colors ${activeRoom?.room_id === room.room_id ? 'bg-primary-50 border border-primary-200' : 'border border-transparent hover:bg-slate-50'}`}
              >
                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary-400 to-primary-600 flex items-center justify-center flex-shrink-0 text-white font-bold text-sm">
                  {getParticipantName(room).charAt(0)}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between">
                    <span className={`font-semibold text-sm truncate ${activeRoom?.room_id === room.room_id ? 'text-primary-600' : 'text-slate-900'}`}>
                      {getParticipantName(room)}
                    </span>
                    <span className="text-xs text-slate-400">
                      {new Date(room.last_message_at || room.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                    </span>
                  </div>
                  <p className="text-xs text-slate-500 truncate mt-0.5">
                    {room.last_message?.content || 'No messages yet'}
                  </p>
                </div>
              </motion.button>
            ))
          )}
        </div>
      </Card>

      {/* Chat Area */}
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
                  <div className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                    <span className="text-xs text-slate-400">Online</span>
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button className="p-2 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition-colors">
                  <LuPhone size={20} />
                </button>
                <button className="p-2 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition-colors">
                  <LuVideo size={20} />
                </button>
                <button className="p-2 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition-colors">
                  <LuInfo size={20} />
                </button>
              </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-50/50">
              {messages.map((msg, i) => {
                const isMe = msg.sender_user_id === currentUserId
                return (
                  <motion.div
                    key={msg.message_id || i}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={`flex ${isMe ? 'justify-end' : 'justify-start'}`}
                  >
                    <div className={`max-w-[70%] p-3.5 rounded-2xl text-sm leading-relaxed ${isMe
                      ? 'bg-primary-500 text-white rounded-tr-none shadow-lg shadow-primary-500/20'
                      : 'bg-white border border-slate-100 text-slate-900 rounded-tl-none shadow-sm'
                      }`}>
                      <p>{msg.content}</p>
                      <div className={`text-[10px] mt-1 flex items-center justify-end gap-1 ${isMe ? 'text-white/70' : 'text-slate-400'}`}>
                        {new Date(msg.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        {isMe && <LuCheckCheck size={12} />}
                      </div>
                    </div>
                  </motion.div>
                )
              })}
              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="p-4 bg-white border-t border-slate-100">
              <form onSubmit={sendMessage} className="flex gap-3 items-end">
                <button type="button" className="p-3 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-xl transition-colors">
                  <LuPlus size={20} />
                </button>
                <div className="flex-1 bg-slate-50 border border-slate-200 rounded-2xl flex items-center px-4 py-2 focus-within:ring-2 focus-within:ring-primary-500/20 focus-within:border-primary-500 transition-all">
                  <input
                    className="flex-1 bg-transparent border-none focus:outline-none py-2 text-sm text-slate-900 placeholder:text-slate-400"
                    placeholder="Type a message..."
                    value={messageText}
                    onChange={e => setMessageText(e.target.value)}
                  />
                  <button type="button" className="p-2 text-slate-400 hover:text-slate-700 transition-colors">
                    <LuSmile size={20} />
                  </button>
                </div>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  type="submit"
                  disabled={!messageText.trim()}
                  className="p-3 bg-primary-500 text-white rounded-xl shadow-lg shadow-primary-500/20 disabled:scale-100 disabled:opacity-50 disabled:shadow-none transition-all"
                >
                  <LuSend size={20} />
                </motion.button>
              </form>
            </div>
          </Card>
        ) : (
          <div className="h-full flex flex-col items-center justify-center text-center p-6 opacity-60">
            <div className="w-20 h-20 rounded-3xl bg-slate-100 border border-slate-200 flex items-center justify-center mb-6">
              <LuImage className="text-slate-400" size={40} />
            </div>
            <h3 className="text-xl font-bold text-slate-900 mb-2">Select a Conversation</h3>
            <p className="text-slate-400 max-w-sm">Choose a chat from the sidebar to start messaging your doctor or patient.</p>
          </div>
        )}
      </div>

      {/* New Conversation Modal can be added here */}
    </div>
  )
}
