import type { ReactNode } from 'react'

export function ToastProvider(props: { children: ReactNode }): JSX.Element

export function useToast(): {
  push: (message: string, kind?: 'success' | 'error' | 'info' | 'warning') => void
}
