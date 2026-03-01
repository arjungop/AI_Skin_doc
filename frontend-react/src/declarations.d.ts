/// <reference types="vite/client" />

// ---------------------------------------------------------------------------
// services/api.js
// ---------------------------------------------------------------------------
declare module '*/services/api.js' {
  export function onAuthExpired(callback: EventListener): () => void

  export const api: {
    base: string
    request: (path: string, opts?: RequestInit) => Promise<unknown>
    // Auth
    login: (payload: Record<string, unknown>) => Promise<unknown>
    me: () => Promise<unknown>
    forgotPassword: (email: string) => Promise<unknown>
    register: (payload: Record<string, unknown>) => Promise<unknown>
    listPatients: (q?: string) => Promise<unknown>
    // Appointments
    createAppointment: (payload: Record<string, unknown>) => Promise<unknown>
    listAppointments: () => Promise<unknown>
    listDoctorAvailability: (doctorId: number | string) => Promise<unknown>
    setDoctorAvailability: (doctorId: number | string, items: unknown[]) => Promise<unknown>
    listBookedSlots: (doctorId: number | string, date: string) => Promise<unknown>
    updateAppointmentStatus: (id: number | string, status: string) => Promise<unknown>
    // Doctors
    listDoctors: (q?: string) => Promise<unknown>
    applyDoctor: (payload: Record<string, unknown>) => Promise<unknown>
    // Transactions
    createTransaction: (payload: Record<string, unknown>) => Promise<unknown>
    listTransactions: (params?: Record<string, string | undefined>) => Promise<unknown>
    transactionsSummary: () => Promise<unknown>
    exportTransactionsCSV: (status?: string) => Promise<void>
    // Lesions
    predictLesion: (patientId: number | string, file: File, opts?: Record<string, unknown>) => Promise<unknown>
    diagnoseLesion: (patientId: number | string, lesionId: number | string) => Promise<unknown>
    createDiagnosisReport: (lesionId: number | string, patientId: number | string) => Promise<unknown>
    listDiagnosisReports: (patientId?: number | string) => Promise<unknown>
    sendDiagnosisReport: (reportId: number | string, doctorId: number | string) => Promise<unknown>
    // LLM
    chat: (payload: Record<string, unknown>) => Promise<unknown>
    llmStatus: () => Promise<unknown>
    chatStream: (payload: Record<string, unknown>, onChunk: (text: string) => void, signal?: AbortSignal) => Promise<void>
    // Admin
    adminOverview: () => Promise<unknown>
    adminListDoctorApps: () => Promise<unknown>
    adminApproveDoctor: (id: number | string) => Promise<unknown>
    adminRejectDoctor: (id: number | string) => Promise<unknown>
    adminListDoctorAppsPaged: (opts?: Record<string, unknown>) => Promise<unknown>
    adminExportDoctorApps: (opts?: Record<string, unknown>) => Promise<Response>
    adminListUsers: (opts?: Record<string, unknown>) => Promise<unknown>
    adminUpdateUserRole: (userId: number | string, role: string) => Promise<unknown>
    adminListDoctors: (opts?: Record<string, unknown>) => Promise<unknown>
    adminListAudit: (opts?: Record<string, unknown>) => Promise<unknown>
    adminGetSettings: () => Promise<unknown>
    adminSetSettings: (obj: Record<string, unknown>) => Promise<unknown>
    adminUpdateUserStatus: (userId: number | string, status: string) => Promise<unknown>
    adminTerminateUser: (userId: number | string, reason_code: string, reason_text: string) => Promise<unknown>
    adminSyncDoctors: () => Promise<unknown>
    // Messaging
    listRooms: () => Promise<unknown>
    createRoom: (payload: Record<string, unknown>) => Promise<unknown>
    listMessages: (roomId: number | string, page?: number, limit?: number) => Promise<unknown>
    postMessage: (roomId: number | string, payload: Record<string, unknown>) => Promise<unknown>
    [key: string]: unknown
  }
}

// ---------------------------------------------------------------------------
// components/Toast.jsx
// ---------------------------------------------------------------------------
declare module '*/components/Toast.jsx' {
  import type { ReactNode } from 'react'
  export function ToastProvider(props: { children: ReactNode }): JSX.Element
  export function useToast(): {
    push: (message: string, kind?: 'success' | 'error' | 'info' | 'warning') => void
  }
}

// ---------------------------------------------------------------------------
// components/ConfirmModal.jsx
// ---------------------------------------------------------------------------
declare module '*/components/ConfirmModal.jsx' {
  import type { ReactNode } from 'react'
  export default function ConfirmModal(props: {
    open: boolean
    title?: string
    children?: ReactNode
    confirmText?: string
    cancelText?: string
    onConfirm: () => void
    onClose: () => void
  }): JSX.Element | null
}

// ---------------------------------------------------------------------------
// components/Card.jsx  (also resolvable without extension)
// ---------------------------------------------------------------------------
declare module '*/components/Card.jsx' {
  export * from '*/components/Card'
}
declare module '*/components/Card' {
  import type { ReactNode, CSSProperties } from 'react'
  interface BaseProps { children?: ReactNode; className?: string }
  export function Card(props: BaseProps & { style?: CSSProperties; onClick?: () => void }): JSX.Element
  export function CardHeader(props: BaseProps): JSX.Element
  export function CardTitle(props: BaseProps & { gradient?: boolean }): JSX.Element
  export function CardDescription(props: BaseProps): JSX.Element
  export function CardData(props: BaseProps & { label?: string; glow?: boolean; size?: string }): JSX.Element
  export function CardBadge(props: BaseProps & { variant?: string }): JSX.Element
  export function IconWrapper(props: BaseProps & { variant?: string; size?: string }): JSX.Element
}

// ---------------------------------------------------------------------------
// Remaining .jsx components — typed as any to suppress implicit-any errors
// ---------------------------------------------------------------------------
declare module '*/components/AppShell.jsx'      { const C: React.ComponentType<any>; export default C }
declare module '*/components/AuthLayout.jsx'    { const C: React.ComponentType<any>; export default C }
declare module '*/components/DetailsDrawer.jsx' { const C: React.ComponentType<any>; export default C }
declare module '*/components/ErrorBoundary.jsx' { const C: React.ComponentType<any>; export default C }
declare module '*/components/FaceMap.jsx'       { const C: React.ComponentType<any>; export default C }
declare module '*/components/FaceMap3D.jsx'     { const C: React.ComponentType<any>; export default C }
declare module '*/components/Navbar.jsx'        { const C: React.ComponentType<any>; export default C }
declare module '*/components/OfflineBanner.jsx' { const C: React.ComponentType<any>; export default C }
declare module '*/components/ProductSearch.jsx' { const C: React.ComponentType<any>; export default C }
declare module '*/components/Skeleton.jsx'      { const C: React.ComponentType<any>; export default C }
declare module '*/components/WeatherWidget.jsx' { const C: React.ComponentType<any>; export default C }
