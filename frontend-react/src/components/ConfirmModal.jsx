import { useEffect } from 'react'

export default function ConfirmModal({ open, title, children, confirmText='Confirm', cancelText='Cancel', onConfirm, onClose }){
  useEffect(()=>{
    function onKey(e){ if(!open) return; if(e.key==='Escape') onClose?.() }
    window.addEventListener('keydown', onKey)
    return ()=> window.removeEventListener('keydown', onKey)
  }, [open, onClose])
  if (!open) return null
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" aria-modal="true" role="dialog">
      <div className="absolute inset-0 bg-black/30" onClick={onClose} />
      <div className="relative bg-[#F7F6FB] border border-[#E8E6EF] rounded-lg shadow-lg w-full max-w-lg p-4">
        <div className="text-lg font-semibold mb-2">{title}</div>
        <div className="text-sm mb-4">{children}</div>
        <div className="flex gap-2 justify-end">
          <button className="button" onClick={onClose}>{cancelText}</button>
          <button className="btn-primary" onClick={onConfirm}>{confirmText}</button>
        </div>
      </div>
    </div>
  )
}

