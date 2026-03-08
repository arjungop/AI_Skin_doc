import { useState, useRef, useCallback, useEffect, useMemo } from 'react'
import ReactMarkdown from 'react-markdown'
import { motion, AnimatePresence } from 'framer-motion'
import { api } from '../services/api'
import {
    LuSend, LuSparkles, LuUser, LuBot, LuSquare,
    LuChevronDown, LuMessageCircle, LuX, LuMaximize2, LuMinimize2
} from 'react-icons/lu'

/**
 * Inline streaming AI chat widget designed for embedding in SkinCoach.
 * Features: streaming responses, typing indicator, avatars, auto-scroll,
 * suggested prompts, abort control, expandable/collapsible panel.
 */

const SUGGESTIONS = [
    '💊 Best ingredients for acne-prone skin?',
    '☀️ How to choose the right SPF?',
    '🧴 Is retinol safe with niacinamide?',
    '🌿 Natural remedies for eczema?',
    '💧 How do I fix a damaged skin barrier?',
    '🔬 Explain AHA vs BHA exfoliants',
]

export default function InlineDermChat({ className = '' }) {
    const [messages, setMessages] = useState([])
    const [input, setInput] = useState('')
    const [pending, setPending] = useState(false)
    const [expanded, setExpanded] = useState(false)

    const scrollRef = useRef(null)
    const chunkBuffer = useRef('')
    const updateTimeout = useRef(null)
    const abortRef = useRef(null)
    const mountedRef = useRef(true)
    const inputRef = useRef(null)

    const chatUserId = parseInt(localStorage.getItem('patient_id')) || parseInt(localStorage.getItem('user_id')) || null

    useEffect(() => {
        return () => {
            mountedRef.current = false
            if (updateTimeout.current) clearTimeout(updateTimeout.current)
            abortRef.current?.abort()
        }
    }, [])

    // Auto-scroll on new messages
    useEffect(() => {
        if (scrollRef.current) {
            requestAnimationFrame(() => {
                if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight
            })
        }
    }, [messages])

    const flushBuffer = useCallback(() => {
        if (chunkBuffer.current) {
            const chunk = chunkBuffer.current
            chunkBuffer.current = ''
            setMessages(prev => {
                const copy = [...prev]
                const lastIdx = copy.length - 1
                if (lastIdx >= 0 && copy[lastIdx].who === 'AI') {
                    copy[lastIdx] = { who: 'AI', text: (copy[lastIdx].text || '') + chunk }
                }
                return copy
            })
        }
    }, [])

    function cancelStream() {
        abortRef.current?.abort()
        abortRef.current = null
        setPending(false)
        if (updateTimeout.current) { clearTimeout(updateTimeout.current); updateTimeout.current = null }
        flushBuffer()
        setMessages(prev => {
            const copy = [...prev]
            const last = copy[copy.length - 1]
            if (last?.who === 'AI' && !last.text) {
                copy[copy.length - 1] = { who: 'AI', text: '[Cancelled]' }
            }
            return copy
        })
    }

    async function send(text) {
        const msg = (text || input || '').trim()
        if (!msg || pending) return
        setMessages(prev => [...prev, { who: 'You', text: msg }])
        setInput('')
        setPending(true)

        const controller = new AbortController()
        abortRef.current = controller

        try {
            const history = messages.slice(-8).map(m => ({
                role: m.who === 'You' ? 'user' : 'assistant',
                content: m.text
            }))

            setMessages(prev => [...prev, { who: 'AI', text: '' }])

            await api.chatStream(
                { patient_id: chatUserId, prompt: msg, history },
                (chunk) => {
                    chunkBuffer.current += chunk
                    if (!updateTimeout.current) {
                        updateTimeout.current = setTimeout(() => {
                            flushBuffer()
                            updateTimeout.current = null
                        }, 50)
                    }
                },
                controller.signal
            )

            if (updateTimeout.current) {
                clearTimeout(updateTimeout.current)
                updateTimeout.current = null
            }
            flushBuffer()
        } catch (err) {
            if (err?.name !== 'AbortError') {
                setMessages(prev => [...prev, { who: 'AI', text: '[Error] ' + (err.message || 'Request failed') }])
            }
        } finally {
            abortRef.current = null
            setPending(false)
        }
    }

    function handleSubmit(e) {
        e.preventDefault()
        send()
    }

    const displayedSuggestions = useMemo(() => {
        // Show 4 random suggestions each mount
        const shuffled = [...SUGGESTIONS].sort(() => Math.random() - 0.5)
        return shuffled.slice(0, 4)
    }, [])

    const chatHeight = expanded ? 'h-[600px]' : 'h-[440px]'

    return (
        <div className={`relative group ${className}`}>
            {/* Background glow */}
            <div className="absolute inset-0 bg-gradient-to-r from-violet-500/15 to-indigo-500/15 rounded-3xl blur-xl transition-all duration-500 group-hover:blur-2xl opacity-70 pointer-events-none" />

            <div className={`relative bg-white/90 dark:bg-slate-800/90 backdrop-blur-xl border border-white/60 dark:border-slate-700 rounded-3xl shadow-2xl overflow-hidden flex flex-col transition-all duration-300 ${chatHeight}`}>

                {/* Header */}
                <div className="flex items-center justify-between px-5 py-3.5 border-b border-slate-100 dark:border-slate-700 bg-gradient-to-r from-violet-50/80 to-indigo-50/80 dark:from-violet-900/30 dark:to-indigo-900/30">
                    <div className="flex items-center gap-3">
                        <div className="relative">
                            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-violet-500/25">
                                <LuSparkles className="text-white" size={20} />
                            </div>
                            <div className="absolute -bottom-0.5 -right-0.5 w-3 h-3 rounded-full bg-emerald-400 border-2 border-white dark:border-slate-800" />
                        </div>
                        <div>
                            <h3 className="font-bold text-slate-900 dark:text-slate-100 text-sm">DermAI Coach</h3>
                            <p className="text-[11px] text-slate-500 dark:text-slate-400 flex items-center gap-1">
                                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                                Dermatology & skin-care specialist
                            </p>
                        </div>
                    </div>
                    <div className="flex items-center gap-1">
                        <button
                            onClick={() => setExpanded(!expanded)}
                            className="p-2 rounded-lg hover:bg-white/80 dark:hover:bg-slate-700 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 transition-colors"
                            aria-label={expanded ? 'Collapse chat' : 'Expand chat'}
                        >
                            {expanded ? <LuMinimize2 size={16} /> : <LuMaximize2 size={16} />}
                        </button>
                        {messages.length > 0 && (
                            <button
                                onClick={() => { setMessages([]); inputRef.current?.focus() }}
                                className="p-2 rounded-lg hover:bg-white/80 dark:hover:bg-slate-700 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 transition-colors"
                                aria-label="Clear chat"
                            >
                                <LuX size={16} />
                            </button>
                        )}
                    </div>
                </div>

                {/* Messages area */}
                <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-4 space-y-4 scroll-smooth">
                    {/* Empty state */}
                    {messages.length === 0 && !pending && (
                        <div className="h-full flex flex-col items-center justify-center text-center px-4">
                            <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-violet-100 to-indigo-100 dark:from-violet-900/50 dark:to-indigo-900/50 flex items-center justify-center mb-4">
                                <LuMessageCircle className="text-violet-500" size={24} />
                            </div>
                            <h4 className="text-base font-bold text-slate-800 dark:text-slate-200 mb-1">Ask anything about skin</h4>
                            <p className="text-xs text-slate-400 dark:text-slate-500 mb-5 max-w-[260px]">
                                Ingredients, routines, conditions, UV protection — I'm here to help.
                            </p>
                            <div className="grid grid-cols-2 gap-2 w-full max-w-sm">
                                {displayedSuggestions.map((s, i) => (
                                    <motion.button
                                        key={i}
                                        whileHover={{ scale: 1.02 }}
                                        whileTap={{ scale: 0.97 }}
                                        onClick={() => send(s.replace(/^[^\w]*/, ''))}
                                        className="px-3 py-2.5 text-xs text-left bg-white dark:bg-slate-700/60 border border-slate-200 dark:border-slate-600 rounded-xl hover:border-violet-300 dark:hover:border-violet-500 hover:shadow-sm transition-all text-slate-600 dark:text-slate-300 leading-snug"
                                    >
                                        {s}
                                    </motion.button>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Chat messages */}
                    <AnimatePresence>
                        {messages.map((m, i) => (
                            <motion.div
                                key={i}
                                initial={{ opacity: 0, y: 8 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ duration: 0.15 }}
                                className={`flex gap-2.5 ${m.who === 'You' ? 'flex-row-reverse' : ''}`}
                            >
                                {/* Avatar */}
                                <div className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5 ${m.who === 'You'
                                        ? 'bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-600'
                                        : 'bg-gradient-to-br from-violet-500 to-indigo-600 text-white shadow-sm'
                                    }`}>
                                    {m.who === 'You' ? <LuUser size={12} /> : <LuBot size={14} />}
                                </div>
                                {/* Bubble */}
                                <div className={`max-w-[80%] px-3.5 py-2.5 rounded-2xl text-sm leading-relaxed ${m.who === 'You'
                                        ? 'bg-violet-600 text-white rounded-tr-sm shadow-sm'
                                        : 'bg-white dark:bg-slate-700/80 text-slate-700 dark:text-slate-200 rounded-tl-sm border border-slate-100 dark:border-slate-600 shadow-sm'
                                    }`}>
                                    {m.who === 'AI'
                                        ? <ReactMarkdown className="prose prose-sm prose-slate dark:prose-invert max-w-none [&>p]:mb-1 [&>ul]:mb-1 [&>ol]:mb-1">{m.text || ' '}</ReactMarkdown>
                                        : <span>{m.text}</span>
                                    }
                                </div>
                            </motion.div>
                        ))}
                    </AnimatePresence>

                    {/* Typing indicator */}
                    {pending && messages[messages.length - 1]?.text === '' && (
                        <div className="flex gap-2.5">
                            <div className="w-7 h-7 rounded-full bg-gradient-to-br from-violet-500 to-indigo-600 text-white flex items-center justify-center flex-shrink-0 animate-pulse shadow-sm">
                                <LuBot size={14} />
                            </div>
                            <div className="bg-white dark:bg-slate-700/80 border border-slate-100 dark:border-slate-600 px-4 py-3 rounded-2xl rounded-tl-sm shadow-sm">
                                <div className="flex gap-1">
                                    <div className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-bounce" style={{ animationDelay: '0ms' }} />
                                    <div className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-bounce" style={{ animationDelay: '150ms' }} />
                                    <div className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-bounce" style={{ animationDelay: '300ms' }} />
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Input area */}
                <div className="px-4 py-3 border-t border-slate-100 dark:border-slate-700 bg-white/80 dark:bg-slate-800/80">
                    {/* Stop generating button */}
                    {pending && (
                        <div className="flex justify-center mb-2">
                            <button
                                type="button"
                                onClick={cancelStream}
                                className="flex items-center gap-1.5 px-3 py-1 text-[11px] font-medium text-slate-500 hover:text-rose-600 bg-slate-100 dark:bg-slate-700 hover:bg-rose-50 dark:hover:bg-rose-900/30 border border-slate-200 dark:border-slate-600 hover:border-rose-200 dark:hover:border-rose-700 rounded-lg transition-all"
                            >
                                <LuSquare size={10} /> Stop generating
                            </button>
                        </div>
                    )}
                    <form onSubmit={handleSubmit} className="relative">
                        <input
                            ref={inputRef}
                            className="w-full pl-4 pr-12 py-3 bg-slate-50 dark:bg-slate-700/60 border border-slate-200 dark:border-slate-600 rounded-xl focus:outline-none focus:ring-2 focus:ring-violet-500/20 focus:border-violet-400 dark:focus:border-violet-500 transition-all text-sm text-slate-900 dark:text-slate-100 placeholder:text-slate-400"
                            placeholder="Ask about skin conditions, ingredients..."
                            value={input}
                            onChange={e => setInput(e.target.value)}
                            disabled={pending}
                        />
                        <button
                            type="submit"
                            disabled={pending || !input.trim()}
                            className="absolute right-1.5 top-1.5 bottom-1.5 aspect-square bg-gradient-to-br from-violet-500 to-indigo-600 hover:from-violet-600 hover:to-indigo-700 disabled:from-slate-200 disabled:to-slate-300 dark:disabled:from-slate-600 dark:disabled:to-slate-700 text-white disabled:text-slate-400 rounded-lg flex items-center justify-center transition-all shadow-md shadow-violet-500/20 disabled:shadow-none"
                        >
                            <LuSend size={16} />
                        </button>
                    </form>
                    <p className="text-[10px] text-center text-slate-400 dark:text-slate-500 mt-2">
                        DermAI may make mistakes. Always consult a dermatologist for medical advice.
                    </p>
                </div>
            </div>
        </div>
    )
}
