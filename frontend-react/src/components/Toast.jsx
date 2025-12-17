import { createContext, useContext, useState, useCallback, useEffect } from 'react'

const ToastCtx = createContext(null)

export function ToastProvider({ children }){
  const [toasts, setToasts] = useState([])
  const push = useCallback((msg, kind='info')=>{
    const id = Math.random().toString(36).slice(2)
    setToasts(t=>[...t, { id, msg, kind }])
    setTimeout(()=> setToasts(t=> t.filter(x=>x.id!==id)), 3000)
  },[])
  return (
    <ToastCtx.Provider value={{ push }}>
      {children}
      <div className="fixed right-4 bottom-4 space-y-2 z-50">
        {toasts.map(t=> (
          <div key={t.id} className={"px-4 py-2 rounded-lg shadow text-white "+(t.kind==='error'?'bg-error':t.kind==='success'?'bg-success':'bg-primary')}>{t.msg}</div>
        ))}
      </div>
    </ToastCtx.Provider>
  )
}

export function useToast(){
  return useContext(ToastCtx) || { push: ()=>{} }
}

