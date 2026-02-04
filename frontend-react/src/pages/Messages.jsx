import { useEffect, useMemo, useRef, useState, useCallback } from 'react'
import { api } from '../services/api'

const MessageStatus = {
  SENT: 'sent',
  DELIVERED: 'delivered',
  READ: 'read'
}

const UserStatus = {
  ONLINE: 'online',
  OFFLINE: 'offline',
  AWAY: 'away'
}

export default function Messages() {
  const [rooms, setRooms] = useState([])
  const [activeRoom, setActiveRoom] = useState(null)
  const [messages, setMessages] = useState([])
  const [messageText, setMessageText] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const [onlineUsers, setOnlineUsers] = useState({})
  const [typingUsers, setTypingUsers] = useState({})
  const [replyToMessage, setReplyToMessage] = useState(null)
  const [editingMessage, setEditingMessage] = useState(null)
  const [showEmojiPicker, setShowEmojiPicker] = useState(null)
  const [page, setPage] = useState(1)
  const [hasMoreMessages, setHasMoreMessages] = useState(true)
  const [showNewConversation, setShowNewConversation] = useState(false)
  const [availableUsers, setAvailableUsers] = useState([])
  const [searchQuery, setSearchQuery] = useState('')
  const [isLoadingUsers, setIsLoadingUsers] = useState(false)
  const [showNotesModal, setShowNotesModal] = useState(false)
  const [generatedNotes, setGeneratedNotes] = useState('')
  const [isGeneratingNotes, setIsGeneratingNotes] = useState(false)

  const wsRef = useRef(null)
  const messagesEndRef = useRef(null)
  const fileInputRef = useRef(null)
  const typingTimeoutRef = useRef(null)
  const messagesContainerRef = useRef(null)

  const currentUserId = parseInt(localStorage.getItem('user_id') || '0')
  const currentUsername = localStorage.getItem('username') || 'You'

  // Load rooms on component mount
  useEffect(() => {
    loadRooms()
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages])

  // Update user online status
  useEffect(() => {
    const updateStatus = (status) => {
      if (currentUserId) {
        api.updateUserStatus(currentUserId, status).catch(console.error)
      }
    }

    updateStatus(UserStatus.ONLINE)

    const handleVisibilityChange = () => {
      updateStatus(document.hidden ? UserStatus.AWAY : UserStatus.ONLINE)
    }

    const handleBeforeUnload = () => {
      updateStatus(UserStatus.OFFLINE)
    }

    document.addEventListener('visibilitychange', handleVisibilityChange)
    window.addEventListener('beforeunload', handleBeforeUnload)

    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange)
      window.removeEventListener('beforeunload', handleBeforeUnload)
      updateStatus(UserStatus.OFFLINE)
    }
  }, [currentUserId])

  const loadRooms = async () => {
    try {
      setIsLoading(true)
      const roomData = await api.listRooms()
      setRooms(roomData)
    } catch (err) {
      setError(err.message || 'Failed to load rooms')
    } finally {
      setIsLoading(false)
    }
  }

  const loadAvailableUsers = async () => {
    try {
      setIsLoadingUsers(true)
      const currentRole = localStorage.getItem('role')?.toUpperCase()

      if (currentRole === 'PATIENT') {
        // Patients can message doctors
        const doctors = await api.listDoctors()
        setAvailableUsers(doctors.map(doc => ({
          ...doc,
          type: 'doctor',
          displayName: `Dr. ${doc.username}`,
          subtitle: doc.specialization || 'Doctor'
        })))
      } else if (currentRole === 'DOCTOR') {
        // Doctors can message patients
        const patients = await api.listPatients()
        setAvailableUsers(patients.map(patient => ({
          ...patient,
          type: 'patient',
          displayName: `${patient.first_name} ${patient.last_name}`,
          subtitle: `Patient • Age ${patient.age}`
        })))
      }
    } catch (err) {
      setError(err.message || 'Failed to load users')
    } finally {
      setIsLoadingUsers(false)
    }
  }

  const createNewConversation = async (targetUser) => {
    try {
      const currentRole = localStorage.getItem('role')?.toUpperCase()
      const currentPatientId = localStorage.getItem('patient_id')
      const currentDoctorId = localStorage.getItem('doctor_id')

      let roomData = {}

      if (currentRole === 'PATIENT' && targetUser.type === 'doctor') {
        if (!currentPatientId) {
          throw new Error('Patient ID not found. Please log in again.')
        }
        roomData = {
          patient_id: parseInt(currentPatientId),
          doctor_id: targetUser.doctor_id
        }
      } else if (currentRole === 'DOCTOR' && targetUser.type === 'patient') {
        if (!currentDoctorId) {
          throw new Error('Doctor ID not found. Please log in again.')
        }
        roomData = {
          patient_id: targetUser.patient_id,
          doctor_id: parseInt(currentDoctorId)
        }
      } else {
        throw new Error('Invalid conversation participants')
      }

      console.log('Creating room with data:', roomData) // Debug log
      const newRoom = await api.createRoom(roomData)
      await loadRooms() // Refresh the rooms list
      openRoom(newRoom) // Select the new room
      setShowNewConversation(false)
    } catch (err) {
      console.error('Failed to create conversation:', err) // Debug log
      setError(err.message || 'Failed to create conversation')
    }
  }

  const openRoom = async (room) => {
    if (activeRoom?.room_id === room.room_id) return

    setActiveRoom(room)
    setMessages([])
    setPage(1)
    setHasMoreMessages(true)
    setError('')

    try {
      await loadMessages(room.room_id, 1)
      connectWebSocket(room.room_id)
    } catch (err) {
      setError(err.message || 'Failed to load messages')
    }
  }

  const loadMessages = async (roomId, pageNum = 1, append = false) => {
    try {
      const response = await api.listMessages(roomId, pageNum, 50)
      const newMessages = response.messages || []

      if (append) {
        setMessages(prev => [...newMessages, ...prev])
      } else {
        setMessages(newMessages)
      }

      setHasMoreMessages(newMessages.length === 50)
      setPage(pageNum)
    } catch (err) {
      console.error('Failed to load messages:', err)
    }
  }

  const loadMoreMessages = () => {
    if (hasMoreMessages && activeRoom) {
      loadMessages(activeRoom.room_id, page + 1, true)
    }
  }

  const connectWebSocket = (roomId) => {
    if (wsRef.current) {
      wsRef.current.close()
    }

    const token = localStorage.getItem('access_token')
    const wsUrl = `${api.base.replace('http', 'ws')}/chat/ws?room_id=${roomId}&token=${encodeURIComponent(token || '')}`

    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      console.log('WebSocket connected')
      setError('')
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        handleWebSocketMessage(data)
      } catch (err) {
        console.error('Failed to parse WebSocket message:', err)
      }
    }

    ws.onclose = (event) => {
      console.log('WebSocket disconnected:', event.code, event.reason)
      // Attempt to reconnect after a delay
      setTimeout(() => {
        if (activeRoom) {
          connectWebSocket(activeRoom.room_id)
        }
      }, 3000)
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      setError('Connection error. Retrying...')
    }

    wsRef.current = ws
  }

  const handleWebSocketMessage = (data) => {
    switch (data.type) {
      case 'new_message':
        setMessages(prev => [...prev, data.data])
        break
      case 'message_updated':
        setMessages(prev => prev.map(msg =>
          msg.message_id === data.data.message_id ? data.data : msg
        ))
        break
      case 'reaction_updated':
        setMessages(prev => prev.map(msg =>
          msg.message_id === data.data.message_id ? data.data : msg
        ))
        break
      case 'typing':
        setTypingUsers(prev => ({
          ...prev,
          [data.data.user_id]: data.data.typing ? data.data.username : null
        }))
        // Clear typing after 3 seconds
        setTimeout(() => {
          setTypingUsers(prev => ({ ...prev, [data.data.user_id]: null }))
        }, 3000)
        break
      case 'user_status_updated':
        setOnlineUsers(prev => ({
          ...prev,
          [data.data.user_id]: data.data.status
        }))
        break
      case 'pong':
        // Handle ping-pong for connection health
        break
    }
  }

  const sendMessage = async (e) => {
    e.preventDefault()

    const content = messageText.trim()
    if (!content || !activeRoom) return

    const messageData = {
      content,
      message_type: 'text',
      reply_to_message_id: replyToMessage?.message_id
    }

    try {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify(messageData))
      } else {
        await api.postMessage(activeRoom.room_id, messageData)
      }

      setMessageText('')
      setReplyToMessage(null)
    } catch (err) {
      setError(err.message || 'Failed to send message')
    }
  }

  const sendTypingIndicator = (typing) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'typing', typing }))
    }
  }

  const handleTextChange = (e) => {
    setMessageText(e.target.value)

    // Send typing indicator
    sendTypingIndicator(true)

    // Clear previous timeout
    if (typingTimeoutRef.current) {
      clearTimeout(typingTimeoutRef.current)
    }

    // Set new timeout to stop typing indicator
    typingTimeoutRef.current = setTimeout(() => {
      sendTypingIndicator(false)
    }, 1000)
  }

  const handleFileUpload = async (file) => {
    if (!activeRoom) return

    try {
      const uploadResult = await api.uploadChatFile(activeRoom.room_id, file)

      const messageData = {
        content: file.name,
        message_type: uploadResult.message_type,
        file_url: uploadResult.file_url,
        file_name: uploadResult.file_name,
        file_size: uploadResult.file_size
      }

      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify(messageData))
      } else {
        await api.postMessage(activeRoom.room_id, messageData)
      }
    } catch (err) {
      setError(err.message || 'Failed to upload file')
    }
  }

  const handleReaction = async (messageId, emoji) => {
    try {
      await api.addReaction(messageId, emoji)
      setShowEmojiPicker(null)
    } catch (err) {
      setError(err.message || 'Failed to add reaction')
    }
  }

  const editMessage = async (messageId, newContent) => {
    try {
      await api.updateMessage(messageId, { content: newContent })
      setEditingMessage(null)
    } catch (err) {
      setError(err.message || 'Failed to edit message')
    }
  }

  const deleteMessage = async (messageId) => {
    try {
      await api.updateMessage(messageId, { is_deleted: true })
    } catch (err) {
      setError(err.message || 'Failed to delete message')
    }
  }

  const formatTime = (dateString) => {
    const date = new Date(dateString)
    const now = new Date()
    const diffMs = now - date
    const diffMins = Math.floor(diffMs / (1000 * 60))
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60))
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24))

    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins}m ago`
    if (diffHours < 24) return `${diffHours}h ago`
    if (diffDays < 7) return `${diffDays}d ago`
    return date.toLocaleDateString()
  }

  const getParticipantName = (room) => {
    if (!room) return 'Unknown'

    // If current user is patient, show doctor name
    // If current user is doctor, show patient name
    const currentUserRole = localStorage.getItem('role')?.toLowerCase()

    if (currentUserRole === 'patient' && room.doctor) {
      return `Dr. ${room.doctor.username}`
    } else if (currentUserRole === 'doctor' && room.patient) {
      return `${room.patient.first_name} ${room.patient.last_name}`
    }

    return `Room #${room.room_id}`
  }

  const getUnreadCount = (room) => {
    const currentUserRole = localStorage.getItem('role')?.toLowerCase()
    if (currentUserRole === 'patient') {
      return room.unread_count_patient || 0
    } else if (currentUserRole === 'doctor') {
      return room.unread_count_doctor || 0
    }
    return 0
  }

  const typingUsersList = Object.values(typingUsers).filter(Boolean)



  const commonEmojis = ['👍', '❤️', '😂', '😮', '😢', '😡', '👏', '🔥']

  const handleGenerateNotes = async () => {
    if (!activeRoom) return
    setIsGeneratingNotes(true)
    try {
      const res = await api.generateClinicalNotes(activeRoom.room_id)
      setGeneratedNotes(res.notes)
      setShowNotesModal(true)
    } catch (err) {
      alert('Failed to generate notes: ' + err.message)
    } finally {
      setIsGeneratingNotes(false)
    }
  }

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar - Conversations */}
      <div className="w-80 bg-white border-r border-gray-200 flex flex-col">
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <h1 className="text-xl font-semibold text-gray-900">Messages</h1>
            <div className="flex items-center space-x-2">
              <button
                onClick={() => {
                  setShowNewConversation(true)
                  loadAvailableUsers()
                }}
                className="p-2 text-blue-600 hover:text-blue-700 rounded-md hover:bg-blue-50"
                title="Start new conversation"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                </svg>
              </button>
              <button
                onClick={loadRooms}
                className="p-2 text-gray-500 hover:text-gray-700 rounded-md hover:bg-gray-100"
                title="Refresh conversations"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
              </button>
            </div>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto">
          {isLoading ? (
            <div className="p-4 text-center text-gray-500">Loading...</div>
          ) : rooms.length === 0 ? (
            <div className="p-4 text-center text-gray-500">No conversations yet</div>
          ) : (
            <div className="space-y-1 p-2">
              {rooms.map(room => {
                const unreadCount = getUnreadCount(room)
                const isActive = activeRoom?.room_id === room.room_id

                return (
                  <div
                    key={room.room_id}
                    onClick={() => openRoom(room)}
                    className={`p-3 rounded-lg cursor-pointer transition-colors ${isActive
                      ? 'bg-blue-50 border border-blue-200'
                      : 'hover:bg-gray-50 border border-transparent'
                      }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-2">
                          <h3 className={`font-medium truncate ${unreadCount > 0 ? 'text-gray-900' : 'text-gray-700'
                            }`}>
                            {getParticipantName(room)}
                          </h3>
                          {unreadCount > 0 && (
                            <span className="bg-blue-500 text-white text-xs rounded-full px-2 py-1 min-w-[20px] text-center">
                              {unreadCount}
                            </span>
                          )}
                        </div>
                        {room.last_message && (
                          <p className="text-sm text-gray-500 truncate mt-1">
                            {room.last_message.content || 'File attachment'}
                          </p>
                        )}
                      </div>
                      <div className="text-xs text-gray-400 ml-2">
                        {room.last_message_at && formatTime(room.last_message_at)}
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {activeRoom ? (
          <>
            {/* Chat Header */}
            <div className="bg-white border-b border-gray-200 p-4">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-semibold text-gray-900">
                    {getParticipantName(activeRoom)}
                  </h2>
                  <div className="flex items-center space-x-4 mt-1">
                    <p className="text-sm text-gray-500">
                      {activeRoom.patient && activeRoom.doctor &&
                        `Patient: ${activeRoom.patient.first_name} ${activeRoom.patient.last_name} • Doctor: Dr. ${activeRoom.doctor.username}`
                      }
                    </p>
                    {typingUsersList.length > 0 && (
                      <p className="text-sm text-blue-500 animate-pulse">
                        {typingUsersList.join(', ')} {typingUsersList.length === 1 ? 'is' : 'are'} typing...
                      </p>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  {localStorage.getItem('role') === 'DOCTOR' && (
                    <button
                      onClick={handleGenerateNotes}
                      disabled={isGeneratingNotes}
                      className="flex items-center gap-2 px-3 py-1.5 bg-indigo-50 text-indigo-600 rounded-lg hover:bg-indigo-100 transition-colors text-sm font-medium disabled:opacity-50"
                    >
                      {isGeneratingNotes ? (
                        <span className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin"></span>
                      ) : (
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
                      )}
                      <span>{isGeneratingNotes ? 'Generating...' : 'Clinical Notes'}</span>
                    </button>
                  )}
                </div>
              </div>
            </div>

            {/* Messages Area */}
            <div
              ref={messagesContainerRef}
              className="flex-1 overflow-y-auto p-4 space-y-4"
              onScroll={(e) => {
                // Load more messages when scrolled to top
                if (e.target.scrollTop === 0 && hasMoreMessages) {
                  loadMoreMessages()
                }
              }}
            >
              {hasMoreMessages && (
                <div className="text-center">
                  <button
                    onClick={loadMoreMessages}
                    className="text-blue-500 hover:text-blue-600 text-sm font-medium"
                  >
                    Load older messages
                  </button>
                </div>
              )}

              {messages.map((message, index) => {
                const isOwnMessage = message.sender_user_id === currentUserId
                const showSender = index === 0 || messages[index - 1].sender_user_id !== message.sender_user_id

                return (
                  <div
                    key={message.message_id}
                    className={`flex ${isOwnMessage ? 'justify-end' : 'justify-start'}`}
                  >
                    <div className={`max-w-xs lg:max-w-md ${isOwnMessage ? 'order-2' : 'order-1'}`}>
                      {showSender && !isOwnMessage && (
                        <p className="text-sm text-gray-500 mb-1 px-3">
                          {message.sender?.username || 'Unknown'}
                        </p>
                      )}

                      <div
                        className={`relative group rounded-lg px-3 py-2 ${isOwnMessage
                          ? 'bg-blue-500 text-white'
                          : 'bg-white border border-gray-200 text-gray-900'
                          } ${message.is_deleted ? 'opacity-50 italic' : ''}`}
                      >
                        {message.reply_to && (
                          <div className={`text-xs p-2 rounded mb-2 ${isOwnMessage ? 'bg-blue-400' : 'bg-gray-100'
                            }`}>
                            <p className="font-medium">Replying to {message.reply_to.sender?.username}</p>
                            <p className="truncate">{message.reply_to.content}</p>
                          </div>
                        )}

                        {message.message_type === 'image' && message.file_url && (
                          <img
                            src={message.file_url}
                            alt={message.file_name}
                            className="max-w-full h-auto rounded mb-2"
                          />
                        )}

                        {message.message_type === 'file' && message.file_url && (
                          <div className="flex items-center space-x-2 p-2 bg-gray-100 rounded mb-2">
                            <svg className="w-5 h-5 text-gray-500" fill="currentColor" viewBox="0 0 20 20">
                              <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clipRule="evenodd" />
                            </svg>
                            <div className="flex-1">
                              <p className="text-sm font-medium text-gray-900">{message.file_name}</p>
                              {message.file_size && (
                                <p className="text-xs text-gray-500">
                                  {(message.file_size / 1024 / 1024).toFixed(2)} MB
                                </p>
                              )}
                            </div>
                            <a
                              href={message.file_url}
                              download={message.file_name}
                              className="text-blue-500 hover:text-blue-600"
                            >
                              Download
                            </a>
                          </div>
                        )}

                        {message.content && (
                          <p className={message.is_deleted ? 'italic' : ''}>
                            {message.is_deleted ? 'This message was deleted' : message.content}
                          </p>
                        )}

                        {message.reactions && message.reactions.length > 0 && (
                          <div className="flex flex-wrap gap-1 mt-2">
                            {message.reactions.map(reaction => (
                              <span
                                key={reaction.reaction_id}
                                className="text-sm bg-gray-200 rounded-full px-2 py-1"
                                title={`${reaction.emoji} by ${reaction.user?.username}`}
                              >
                                {reaction.emoji}
                              </span>
                            ))}
                          </div>
                        )}

                        <div className="flex items-center justify-between mt-1">
                          <p className={`text-xs ${isOwnMessage ? 'text-blue-100' : 'text-gray-400'}`}>
                            {formatTime(message.created_at)}
                            {message.is_edited && ' (edited)'}
                          </p>

                          {isOwnMessage && (
                            <span className={`text-xs ${message.status === 'read' ? 'text-green-200' :
                              message.status === 'delivered' ? 'text-blue-200' : 'text-gray-300'
                              }`}>
                              {message.status === 'read' ? '✓✓' :
                                message.status === 'delivered' ? '✓' : '○'}
                            </span>
                          )}
                        </div>

                        {/* Message Actions */}
                        <div className="absolute top-0 right-0 hidden group-hover:flex space-x-1 -mt-2 -mr-2">
                          <button
                            onClick={() => setReplyToMessage(message)}
                            className="p-1 bg-gray-500 text-white rounded-full hover:bg-gray-600"
                            title="Reply"
                          >
                            <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                              <path fillRule="evenodd" d="M7.707 3.293a1 1 0 010 1.414L5.414 7H11a7 7 0 017 7v2a1 1 0 11-2 0v-2a5 5 0 00-5-5H5.414l2.293 2.293a1 1 0 11-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z" clipRule="evenodd" />
                            </svg>
                          </button>

                          <button
                            onClick={() => setShowEmojiPicker(message.message_id)}
                            className="p-1 bg-gray-500 text-white rounded-full hover:bg-gray-600"
                            title="Add reaction"
                          >
                            <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM7 9a1 1 0 100-2 1 1 0 000 2zm7-1a1 1 0 11-2 0 1 1 0 012 0zm-.464 5.535a1 1 0 10-1.415-1.414 3 3 0 01-4.242 0 1 1 0 00-1.415 1.414 5 5 0 007.072 0z" clipRule="evenodd" />
                            </svg>
                          </button>

                          {isOwnMessage && !message.is_deleted && (
                            <>
                              <button
                                onClick={() => setEditingMessage(message)}
                                className="p-1 bg-gray-500 text-white rounded-full hover:bg-gray-600"
                                title="Edit"
                              >
                                <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                                  <path d="M13.586 3.586a2 2 0 112.828 2.828l-.793.793-2.828-2.828.793-.793zM11.379 5.793L3 14.172V17h2.828l8.38-8.379-2.83-2.828z" />
                                </svg>
                              </button>

                              <button
                                onClick={() => deleteMessage(message.message_id)}
                                className="p-1 bg-red-500 text-white rounded-full hover:bg-red-600"
                                title="Delete"
                              >
                                <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                                  <path fillRule="evenodd" d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" clipRule="evenodd" />
                                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                                </svg>
                              </button>
                            </>
                          )}
                        </div>

                        {/* Emoji Picker */}
                        {showEmojiPicker === message.message_id && (
                          <div className="absolute bottom-full left-0 mb-2 bg-white border border-gray-200 rounded-lg shadow-lg p-2 z-10">
                            <div className="flex space-x-1">
                              {commonEmojis.map(emoji => (
                                <button
                                  key={emoji}
                                  onClick={() => handleReaction(message.message_id, emoji)}
                                  className="text-lg hover:bg-gray-100 rounded p-1"
                                >
                                  {emoji}
                                </button>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )
              })}
              <div ref={messagesEndRef} />
            </div>

            {/* Message Input */}
            <div className="bg-white border-t border-gray-200 p-4">
              {error && (
                <div className="mb-3 p-2 bg-red-50 border border-red-200 rounded text-red-700 text-sm">
                  {error}
                  <button
                    onClick={() => setError('')}
                    className="float-right text-red-500 hover:text-red-700"
                  >
                    ×
                  </button>
                </div>
              )}

              {replyToMessage && (
                <div className="mb-3 p-3 bg-gray-50 border-l-4 border-blue-500 rounded">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-700">
                        Replying to {replyToMessage.sender?.username}
                      </p>
                      <p className="text-sm text-gray-600 truncate">
                        {replyToMessage.content}
                      </p>
                    </div>
                    <button
                      onClick={() => setReplyToMessage(null)}
                      className="text-gray-400 hover:text-gray-600"
                    >
                      ×
                    </button>
                  </div>
                </div>
              )}

              <form onSubmit={sendMessage} className="flex items-end space-x-3">
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={(e) => e.target.files[0] && handleFileUpload(e.target.files[0])}
                  className="hidden"
                  accept="image/*,.pdf,.doc,.docx,.txt"
                />

                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="p-2 text-gray-500 hover:text-gray-700 rounded-md hover:bg-gray-100"
                  title="Attach file"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                  </svg>
                </button>

                <div className="flex-1">
                  <textarea
                    value={messageText}
                    onChange={handleTextChange}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault()
                        sendMessage(e)
                      }
                    }}
                    placeholder="Type a message... (Enter to send, Shift+Enter for new line)"
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    rows={1}
                    style={{ minHeight: '44px', maxHeight: '120px' }}
                  />
                </div>

                <button
                  type="submit"
                  disabled={!messageText.trim()}
                  className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                  </svg>
                </button>
              </form>
            </div>
          </>
        ) : (
          /* No Room Selected */
          <div className="flex-1 flex items-center justify-center bg-gray-50">
            <div className="text-center">
              <svg className="mx-auto h-16 w-16 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
              </svg>
              <h3 className="mt-4 text-lg font-medium text-gray-900">No conversation selected</h3>
              <p className="mt-2 text-gray-500">Choose a conversation from the sidebar to start messaging</p>
            </div>
          </div>
        )}
      </div>

      {/* Click outside to close emoji picker */}
      {showEmojiPicker && (
        <div
          className="fixed inset-0 z-0"
          onClick={() => setShowEmojiPicker(null)}
        />
      )}

      {/* New Conversation Modal */}
      {showNewConversation && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-md m-4">
            <div className="flex items-center justify-between p-4 border-b border-gray-200">
              <h2 className="text-lg font-semibold text-gray-900">Start New Conversation</h2>
              <button
                onClick={() => setShowNewConversation(false)}
                className="p-2 text-gray-400 hover:text-gray-600 rounded-md hover:bg-gray-100"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="p-4">
              <div className="mb-4">
                <input
                  type="text"
                  placeholder="Search users..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>

              <div className="max-h-64 overflow-y-auto">
                {isLoadingUsers ? (
                  <div className="p-4 text-center text-gray-500">Loading users...</div>
                ) : (
                  availableUsers
                    .filter(user =>
                      searchQuery === '' ||
                      user.displayName.toLowerCase().includes(searchQuery.toLowerCase()) ||
                      user.subtitle.toLowerCase().includes(searchQuery.toLowerCase())
                    )
                    .map(user => (
                      <div
                        key={`${user.type}-${user.type === 'doctor' ? user.doctor_id : user.patient_id}`}
                        onClick={() => createNewConversation(user)}
                        className="p-3 rounded-lg border border-gray-200 mb-2 cursor-pointer hover:bg-blue-50 hover:border-blue-300 transition-colors"
                      >
                        <div className="flex items-center space-x-3">
                          <div className={`w-10 h-10 rounded-full flex items-center justify-center text-white ${user.type === 'doctor' ? 'bg-blue-500' : 'bg-green-500'
                            }`}>
                            {user.type === 'doctor' ? '👨‍⚕️' : '👤'}
                          </div>
                          <div className="flex-1">
                            <h3 className="font-medium text-gray-900">{user.displayName}</h3>
                            <p className="text-sm text-gray-500">{user.subtitle}</p>
                          </div>
                        </div>
                      </div>
                    ))
                )}

                {!isLoadingUsers && availableUsers.length === 0 && (
                  <div className="p-4 text-center text-gray-500">
                    {localStorage.getItem('role')?.toUpperCase() === 'PATIENT'
                      ? 'No doctors available'
                      : 'No patients available'
                    }
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {showNotesModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
          <div className="bg-white rounded-2xl w-full max-w-2xl max-h-[80vh] flex flex-col shadow-2xl">
            <div className="p-6 border-b flex items-center justify-between">
              <h3 className="text-xl font-serif font-bold text-slate-900">Clinical SOAP Notes</h3>
              <button onClick={() => setShowNotesModal(false)} className="text-slate-400 hover:text-slate-600">
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" /></svg>
              </button>
            </div>
            <div className="p-6 overflow-y-auto bg-slate-50 font-mono text-sm leading-relaxed whitespace-pre-wrap">
              {generatedNotes}
            </div>
            <div className="p-4 border-t bg-white rounded-b-2xl flex justify-end gap-3">
              <button
                onClick={() => { navigator.clipboard.writeText(generatedNotes); alert('Copied!') }}
                className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 font-medium transition-all"
              >
                Copy to Clipboard
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

