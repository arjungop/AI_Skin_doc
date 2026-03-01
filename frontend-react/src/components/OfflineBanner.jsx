import { useOnlineStatus } from '../hooks/useOnlineStatus'
import { AnimatePresence, motion } from 'framer-motion'
import { LuWifiOff } from 'react-icons/lu'

export default function OfflineBanner() {
  const isOnline = useOnlineStatus()

  return (
    <AnimatePresence>
      {!isOnline && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: 'auto', opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          className="fixed top-0 left-0 right-0 z-[100] overflow-hidden"
        >
          <div
            className="bg-amber-500 text-white text-center py-2.5 px-4 text-sm font-medium flex items-center justify-center gap-2 shadow-lg"
            role="alert"
            aria-live="assertive"
          >
            <LuWifiOff size={16} />
            You are offline. Some features may not work until your connection is restored.
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
