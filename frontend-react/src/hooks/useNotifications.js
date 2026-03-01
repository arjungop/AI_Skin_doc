import { useEffect, useRef, useCallback, useState } from 'react'

const WS_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000')
  .replace(/^http/, 'ws')

/**
 * useNotifications — WebSocket hook for real-time sync.
 *
 * Connects to the backend `/notifications/ws` endpoint using the
 * JWT token from localStorage.  Provides:
 *  • `lastEvent`  — latest event object { event, ...payload }
 *  • `connected`  — whether the WebSocket is open
 *  • `subscribe(eventName, callback)` — register a listener
 *
 * Auto-reconnects on disconnect with exponential back-off (max 30s).
 */
export default function useNotifications() {
  const wsRef = useRef(null)
  const reconnectTimer = useRef(null)
  const pingTimer = useRef(null)
  const retryDelay = useRef(2000)
  const listenersRef = useRef({})   // eventName → Set<callback>
  const [connected, setConnected] = useState(false)
  const [lastEvent, setLastEvent] = useState(null)
  const mountedRef = useRef(true)

  const connect = useCallback(() => {
    let token = null
    try { token = localStorage.getItem('access_token') } catch { }
    if (!token) return
    token = String(token).replace(/^\"|\"$/g, '').trim()
    if (!token) return

    // Close existing
    try { wsRef.current?.close() } catch { }

    const ws = new WebSocket(`${WS_BASE}/notifications/ws?token=${encodeURIComponent(token)}`)
    wsRef.current = ws

    ws.onopen = () => {
      if (!mountedRef.current) return
      setConnected(true)
      retryDelay.current = 2000 // reset back-off
      // Keep WS alive with periodic pings (every 25s)
      clearInterval(pingTimer.current)
      pingTimer.current = setInterval(() => {
        try { if (ws.readyState === WebSocket.OPEN) ws.send('ping') } catch {}
      }, 25000)
    }

    ws.onmessage = (evt) => {
      if (!mountedRef.current) return
      try {
        const data = JSON.parse(evt.data)
        setLastEvent(data)
        // Dispatch to registered listeners
        const eventName = data.event || data.type
        if (eventName && listenersRef.current[eventName]) {
          for (const cb of listenersRef.current[eventName]) {
            try { cb(data) } catch { }
          }
        }
        // Also dispatch to '*' wildcard listeners
        if (listenersRef.current['*']) {
          for (const cb of listenersRef.current['*']) {
            try { cb(data) } catch { }
          }
        }
      } catch { }
    }

    ws.onclose = () => {
      if (!mountedRef.current) return
      setConnected(false)
      clearInterval(pingTimer.current)
      // Reconnect with exponential back-off
      reconnectTimer.current = setTimeout(() => {
        retryDelay.current = Math.min(retryDelay.current * 1.5, 30000)
        connect()
      }, retryDelay.current)
    }

    ws.onerror = () => {
      try { ws.close() } catch { }
    }
  }, [])

  // Subscribe to a specific event type.  Returns an unsubscribe fn.
  const subscribe = useCallback((eventName, callback) => {
    if (!listenersRef.current[eventName]) {
      listenersRef.current[eventName] = new Set()
    }
    listenersRef.current[eventName].add(callback)
    return () => {
      listenersRef.current[eventName]?.delete(callback)
    }
  }, [])

  useEffect(() => {
    mountedRef.current = true
    connect()
    return () => {
      mountedRef.current = false
      clearTimeout(reconnectTimer.current)
      clearInterval(pingTimer.current)
      try { wsRef.current?.close() } catch { }
    }
  }, [connect])

  return { lastEvent, connected, subscribe }
}
