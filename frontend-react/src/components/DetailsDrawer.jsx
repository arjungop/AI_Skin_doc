export default function DetailsDrawer({ open, title, onClose, children, width=420 }){
  if (!open) return null
  return (
    <div className="fixed inset-0 z-40" aria-modal="true" role="dialog">
      <div className="absolute inset-0 bg-black/30" onClick={onClose} />
      <div className="absolute right-0 top-0 h-full bg-[#F7F6FB] border-l border-[#E8E6EF] shadow-lg p-4" style={{ width }}>
        <div className="flex items-center justify-between mb-2">
          <div className="text-lg font-semibold">{title}</div>
          <button className="button" onClick={onClose} aria-label="Close">×</button>
        </div>
        <div className="overflow-y-auto h-[calc(100%-48px)] pr-2">
          {children}
        </div>
      </div>
    </div>
  )
}

