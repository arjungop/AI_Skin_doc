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

async function request(path, opts = {}) {
  const headers = { ...(opts.headers || {}), ...authHeaders() }
  const res = await fetch(`${BASE_URL}${path}`, { ...opts, headers })
  let data = null
  try { data = await res.json() } catch { }
  if (!res.ok) {
    const detail = (data && (data.detail || data.message)) || res.statusText
    // Do not auto-clear auth on 401; surface error and let the UI handle.
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
  updateAppointmentStatus: (id, status) => request(`/appointments/${id}/status`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ status }) }),
  // Doctors
  listDoctors: (q) => request(`/doctors${q ? `?q=${encodeURIComponent(q)}` : ''}`),
  applyDoctor: (payload) => request('/doctors/apply', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
  // Transactions
  createTransaction: (payload) => request('/transactions/', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
  listTransactions: (params) => {
    const q = new URLSearchParams()
    if (!params) params = {}
    if (params.status) q.set('status', params.status)
    if (params.q) q.set('search', params.q)
    if (params.start) q.set('start', params.start)
    if (params.end) q.set('end', params.end)
    if (params.category) q.set('category', params.category)
    if (params.method) q.set('method', params.method)
    if (params.amount_min != null) q.set('amount_min', params.amount_min)
    if (params.amount_max != null) q.set('amount_max', params.amount_max)
    if (params.transaction_id) q.set('transaction_id', params.transaction_id)
    const qs = q.toString();
    return request(`/transactions${qs ? `?${qs}` : ''}`)
  },
  adminListTransactions: (params = {}) => {
    const q = new URLSearchParams(); Object.entries(params || {}).forEach(([k, v]) => { if (v != null && v !== '') q.set(k, v) })
    return request(`/transactions/admin_list?${q.toString()}`)
  },
  transactionsSummary: (userId) => request(`/transactions/summary${userId ? `?user_id=${encodeURIComponent(userId)}` : ''}`),
  updateTransactionStatus: (id, status, reason) => request(`/transactions/${id}/status`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ status, reason }) }),
  setTransactionMeta: (id, payload) => request(`/transactions/${id}/meta`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
  refundTransaction: (id) => request(`/transactions/${id}/refund`, { method: 'POST' }),
  monthly: (months = 12) => request(`/transactions/monthly?months=${months}`),
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
  chatStream: async (payload, onChunk) => {
    const res = await fetch(`${BASE_URL}/llm/chat_stream`, { method: 'POST', headers: { 'Content-Type': 'application/json', ...authHeaders() }, body: JSON.stringify(payload) })
    if (!res.ok) throw new Error('Stream error')
    const reader = res.body.getReader()
    const decoder = new TextDecoder('utf-8')
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      const text = decoder.decode(value, { stream: true })
      if (text) onChunk(text)
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
  // Messaging
  listRooms: () => request('/chat/rooms'),
  createRoom: (payload) => request('/chat/rooms', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
  listMessages: (roomId, page = 1, limit = 50) => request(`/chat/rooms/${roomId}/messages?page=${page}&limit=${limit}`),
  postMessage: (roomId, payload) => request(`/chat/rooms/${roomId}/messages`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
  updateMessage: (messageId, payload) => request(`/chat/messages/${messageId}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
  addReaction: (messageId, emoji) => request(`/chat/messages/${messageId}/reactions`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ emoji }) }),
  uploadChatFile: async (roomId, file) => {
    const form = new FormData()
    form.append('file', file)
    return request(`/chat/rooms/${roomId}/upload`, { method: 'POST', body: form })
  },
  updateUserStatus: (userId, status) => request(`/chat/users/${userId}/status`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ status }) }),
  // User lists for messaging
  listDoctors: () => request('/doctors/'),
  listPatients: () => request('/patients/'),
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
  // Routine
  getRoutine: () => request('/routine/'),
  addRoutineItem: (data) => request('/routine/', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }),
  deleteRoutineItem: (id) => request(`/routine/${id}`, { method: 'DELETE' }),
  getCompletions: (date) => request(`/routine/completions?date=${date}`), // date: YYYY-MM-DD
  checkRoutineItem: (data) => request('/routine/check', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }),
  // Doctor Copilot
  generateClinicalNotes: (roomId) => request('/llm/generate_notes', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ room_id: roomId }) }),
}
