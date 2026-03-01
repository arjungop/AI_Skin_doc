const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

function authHeaders() {
  const h = {}
  let t = null
  try { t = localStorage.getItem('access_token') } catch { }
  if (t) {
    t = String(t).replace(/^\"|\"$/g, '').trim()
    if (t) h['Authorization'] = `Bearer ${t}`
  }
  return h
}

// Auth expiry event for cross-component session handling
const AUTH_EXPIRED_EVENT = 'auth:expired'

export function onAuthExpired(callback) {
  window.addEventListener(AUTH_EXPIRED_EVENT, callback)
  return () => window.removeEventListener(AUTH_EXPIRED_EVENT, callback)
}

async function request(path, opts = {}) {
  if (!navigator.onLine) {
    throw new Error('You are offline. Please check your internet connection.')
  }
  const headers = { ...(opts.headers || {}), ...authHeaders() }
  const res = await fetch(`${BASE_URL}${path}`, { ...opts, headers })
  let data = null
  try { data = await res.json() } catch { }
  if (!res.ok) {
    const detail = (data && (data.detail || data.message)) || res.statusText
    if (res.status === 401) {
      window.dispatchEvent(new CustomEvent(AUTH_EXPIRED_EVENT))
      throw new Error('Session expired. Please log in again.')
    }
    throw new Error(detail)
  }
  return data
}

export const api = {
  base: BASE_URL,
  request,
  // Auth
  login: (payload) => request('/auth/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
  me: () => request('/auth/me'),
  forgotPassword: (email) => request('/auth/forgot', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ email }) }),
  register: (payload) => request('/patients/register', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
  listPatients: (q) => request(`/patients${q ? `?q=${encodeURIComponent(q)}` : ''}`),
  // Appointments
  createAppointment: (payload) => request('/appointments/', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
  listAppointments: () => request('/appointments/'),
  listDoctorAvailability: (doctorId) => request(`/doctors/${doctorId}/availability`),
  setDoctorAvailability: (doctorId, items) => request(`/doctors/${doctorId}/availability`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(items) }),
  listBookedSlots: (doctorId, date) => request(`/appointments/booked-slots?doctor_id=${doctorId}&date=${date}`).catch(() => []),
  updateAppointmentStatus: (id, status) => request(`/appointments/${id}/status`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ status }) }),
  // Doctors
  listDoctors: (q) => request(`/doctors${q ? `?q=${encodeURIComponent(q)}` : ''}`),
  applyDoctor: (payload) => request('/doctors/apply', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
  // Transactions (simplified billing log)
  createTransaction: (payload) => request('/transactions/', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
  listTransactions: (params) => {
    const q = new URLSearchParams()
    if (!params) params = {}
    if (params.status) q.set('status', params.status)
    if (params.start) q.set('start', params.start)
    if (params.end) q.set('end', params.end)
    if (params.category) q.set('category', params.category)
    const qs = q.toString();
    return request(`/transactions${qs ? `?${qs}` : ''}`)
  },
  transactionsSummary: () => request('/transactions/summary'),
  exportTransactionsCSV: async (status) => {
    const q = new URLSearchParams()
    if (status) q.set('status', status)
    const res = await fetch(`${BASE_URL}/transactions/export.csv${q.toString() ? `?${q}` : ''}`, { headers: authHeaders() })
    if (!res.ok) throw new Error(`Export failed: ${res.status}`)
    const blob = await res.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a'); a.href = url; a.download = 'transactions.csv'; a.click()
    URL.revokeObjectURL(url)
  },
  // Lesions
  predictLesion: async (patientId, file, opts = {}) => {
    const form = new FormData()
    form.append('file', file)
    const q = new URLSearchParams({ patient_id: String(patientId) })
    if (opts && opts.threshold != null) q.set('threshold', String(opts.threshold))
    if (opts && opts.sensitivity) q.set('sensitivity', String(opts.sensitivity))
    // Use request() so Authorization header is attached
    return request(`/lesions/predict?${q.toString()}`, { method: 'POST', body: form })
  },
  diagnoseLesion: (patientId, lesionId) => request('/llm/diagnose', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ patient_id: patientId, lesion_id: lesionId }) }),
  // Diagnosis reports
  createDiagnosisReport: (lesionId, patientId) => request(`/lesions/${lesionId}/report?patient_id=${encodeURIComponent(patientId)}`, { method: 'POST' }),
  listDiagnosisReports: (patientId) => request(`/lesions/reports${patientId ? `?patient_id=${encodeURIComponent(patientId)}` : ''}`),
  sendDiagnosisReport: (reportId, doctorId) => request(`/lesions/reports/${reportId}/send?doctor_id=${encodeURIComponent(doctorId)}`, { method: 'POST' }),
  // LLM
  chat: (payload) => request('/llm/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
  llmStatus: () => request('/llm/status'),
  chatStream: async (payload, onChunk, signal) => {
    const controller = signal ? null : new AbortController()
    const abortSignal = signal || controller?.signal
    const timeoutId = setTimeout(() => controller?.abort(), 60_000)
    try {
      const res = await fetch(`${BASE_URL}/llm/chat_stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders() },
        body: JSON.stringify(payload),
        signal: abortSignal,
      })
      if (!res.ok) throw new Error('Stream error')
      const reader = res.body.getReader()
      const decoder = new TextDecoder('utf-8')
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        const text = decoder.decode(value, { stream: true })
        if (text) onChunk(text)
      }
    } finally {
      clearTimeout(timeoutId)
    }
  },
  // Admin
  adminOverview: () => request('/admin/overview'),
  adminListDoctorApps: () => request('/admin/doctor_applications'),
  adminApproveDoctor: (id) => request(`/admin/doctor_applications/${id}/approve`, { method: 'POST' }),
  adminRejectDoctor: (id) => request(`/admin/doctor_applications/${id}/reject`, { method: 'POST' }),
  adminListDoctorAppsPaged: ({ status, q, page = 1, page_size = 20 } = {}) => {
    const p = new URLSearchParams()
    if (status) p.set('status', status)
    if (q) p.set('q', q)
    p.set('page', page); p.set('page_size', page_size)
    return request(`/admin/doctor_applications?${p.toString()}`)
  },
  adminExportDoctorApps: ({ status, q } = {}) => {
    const p = new URLSearchParams(); if (status) p.set('status', status); if (q) p.set('q', q)
    return fetch(`${BASE_URL}/admin/doctor_applications/export.csv?${p.toString()}`, { headers: authHeaders() })
  },
  adminListUsers: ({ q, role, page = 1, page_size = 20 } = {}) => {
    const p = new URLSearchParams(); if (q) p.set('q', q); if (role) p.set('role', role); p.set('page', page); p.set('page_size', page_size)
    return request(`/admin/users?${p.toString()}`)
  },
  adminUpdateUserRole: (userId, role) => request(`/admin/users/${userId}/role`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ role }) }),
  adminListDoctors: ({ q, page = 1, page_size = 20 } = {}) => {
    const p = new URLSearchParams(); if (q) p.set('q', q); p.set('page', page); p.set('page_size', page_size)
    return request(`/admin/doctors?${p.toString()}`)
  },
  adminListAudit: ({ page = 1, page_size = 50 } = {}) => request(`/admin/audit_logs?page=${page}&page_size=${page_size}`),
  adminGetSettings: () => request('/admin/settings'),
  adminSetSettings: (obj) => request('/admin/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(obj) }),
  adminUpdateUserStatus: (userId, status) => request(`/admin/users/${userId}/status`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ status }) }),
  adminTerminateUser: (userId, reason_code, reason_text) => request(`/admin/users/${userId}/terminate`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ reason_code, reason_text }) }),
  adminSyncDoctors: () => request('/admin/sync_doctors', { method: 'POST' }),
  // Messaging (real-time sync + polling fallback)
  listRooms: () => request('/chat/rooms'),
  createRoom: (payload) => request('/chat/rooms', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
  listMessages: (roomId, page = 1, limit = 50) => request(`/chat/rooms/${roomId}/messages?page=${page}&limit=${limit}`),
  postMessage: (roomId, payload) => request(`/chat/rooms/${roomId}/messages`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
  updateMessage: (messageId, payload) => request(`/chat/messages/${messageId}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
  markUrgent: (messageId) => request(`/chat/messages/${messageId}/urgent`, { method: 'PUT' }),
  setVideoLink: (roomId, link) => request(`/chat/rooms/${roomId}/video-link`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ video_link: link }) }),
  // Read receipts & unread
  markRoomRead: (roomId) => request(`/chat/rooms/${roomId}/read`, { method: 'POST' }),
  getUnreadCounts: () => request('/chat/unread'),
  // Online status
  sendHeartbeat: () => request('/chat/online', { method: 'POST' }),
  goOffline: () => request('/chat/offline', { method: 'POST' }),
  getOnlineStatus: (userId) => request(`/chat/online/${userId}`),
  // Typing indicator
  sendTyping: (roomId) => request(`/chat/rooms/${roomId}/typing`, { method: 'POST' }),
  // Profile
  getProfile: () => request('/profile/me'),
  updateProfile: (data) => request('/profile/me', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }),
  // AI Chat sessions
  aiListSessions: () => request('/ai_chat/sessions'),
  aiCreateSession: (title) => request('/ai_chat/sessions', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ title }) }),
  aiDeleteSession: (id) => request(`/ai_chat/sessions/${id}`, { method: 'DELETE' }),
  aiListSessionMessages: (id) => request(`/ai_chat/sessions/${id}/messages`),
  aiAddSessionMessage: (id, role, content) => request(`/ai_chat/sessions/${id}/messages`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ role, content }) }),
  // Skin Journey
  getJourney: () => request('/journey/'),
  addJourneyLog: (data) => request('/journey/', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }),
  deleteJourneyLog: (logId) => request(`/journey/${logId}`, { method: 'DELETE' }),
  // Treatment Plans (replaces Routine)
  getTreatmentPlans: () => request('/routine/'),
  getTreatmentSteps: (planId) => request(`/routine/plans/${planId}/steps`),
  getTreatmentAdherence: (planId, date) => request(`/routine/plans/${planId}/adherence${date ? `?date=${date}` : ''}`),
  recordAdherence: (planId, data) => request(`/routine/plans/${planId}/adherence`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }),
  reportSideEffect: (planId, data) => request(`/routine/plans/${planId}/report-side-effect`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }),
  // Doctor: create plans + steps
  createTreatmentPlan: (data) => request('/routine/plans', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }),
  addTreatmentStep: (planId, data) => request(`/routine/plans/${planId}/steps`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }),
  // UV Risk (uses backend with Fitzpatrick-aware assessment)
  getUVRisk: (city) => request(`/recommendations/uv-risk${city ? `?city=${encodeURIComponent(city)}` : ''}`),
  // Doctor Suggestions
  addSuggestion: (data) => request('/recommendations/suggest', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }),
  getSuggestions: (patientId) => request(`/recommendations/suggestions/${patientId}`),
  // Doctor Copilot
  generateClinicalNotes: (roomId) => request('/llm/generate_notes', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ room_id: roomId }) }),
}
