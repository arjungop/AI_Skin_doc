import { useState, useEffect } from 'react'
import { api } from '../services/api'
import { motion, AnimatePresence } from 'framer-motion'
import {
  LuUpload, LuScan, LuTriangleAlert, LuZap, LuDownload,
  LuArrowRight, LuHistory, LuCircleCheck, LuShield, LuSparkles, LuX
} from 'react-icons/lu'
import { Card, CardTitle, CardDescription, CardData, CardBadge, IconWrapper } from '../components/Card'

export default function LesionUpload() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [res, setRes] = useState(null)
  const [diag, setDiag] = useState('')
  const [report, setReport] = useState(null)
  const patientId = parseInt(localStorage.getItem('patient_id'))
  const role = (localStorage.getItem('role') || '').toUpperCase()
  const [overridePatientId, setOverridePatientId] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [highSensitivity, setHighSensitivity] = useState(true)
  const [dragActive, setDragActive] = useState(false)

  const onSubmit = async (e) => {
    e.preventDefault()
    setRes(null); setDiag(''); setError(''); setReport(null); setLoading(true)
    const pid = role === 'ADMIN' ? parseInt(overridePatientId || '0') : patientId
    if (!pid) { setError('Select a patient first'); setLoading(false); return }
    if (!file) { setError('Select an image'); setLoading(false); return }
    try {
      const result = await api.predictLesion(pid, file, { sensitivity: highSensitivity ? 'high' : undefined })
      setRes(result)
    } catch (err) { setError(err.message || 'Upload failed') }
    setLoading(false)
  }

  const runDiagnosis = async () => {
    if (!res?.lesion_id) return
    try {
      const pid = role === 'ADMIN' ? parseInt(overridePatientId || '0') : patientId
      const rep = await api.createDiagnosisReport(res.lesion_id, pid)
      setReport(rep)
      setDiag(rep.details || '')
    } catch (err) { setDiag(String(err.message || err)) }
  }

  useEffect(() => {
    if (file) {
      const objectUrl = URL.createObjectURL(file)
      setPreview(objectUrl)
      return () => URL.revokeObjectURL(objectUrl)
    }
  }, [file])

  const handleDrag = (e) => {
    e.preventDefault(); e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') setDragActive(true)
    else if (e.type === 'dragleave') setDragActive(false)
  }

  const handleDrop = (e) => {
    e.preventDefault(); e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0])
  }

  return (
    <div className="min-h-screen pb-20 relative">
      {/* Ambient Background */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute top-1/4 right-1/4 w-[500px] h-[500px] bg-ai-500/10 rounded-full blur-[150px] opacity-40" />
        <div className="absolute bottom-1/4 left-1/4 w-[400px] h-[400px] bg-accent-500/10 rounded-full blur-[120px] opacity-30" />
      </div>

      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-10 relative z-10 max-w-5xl mx-auto pt-6"
      >
        <div className="flex items-center gap-4 mb-4">
          <IconWrapper variant="ai" size="lg">
            <LuScan size={28} />
          </IconWrapper>
          <div>
            <h1 className="text-4xl font-bold text-text-primary">
              AI <span className="text-gradient-ai">Lesion</span> Analysis
            </h1>
            <p className="text-text-secondary">Clinical-grade dermatological classification</p>
          </div>
        </div>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 max-w-5xl mx-auto relative z-10">
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}>
          <Card variant="glass" className="h-full">
            <CardTitle>Upload Scan</CardTitle>
            <CardDescription>Upload a clear, close-up image of the skin area.</CardDescription>

            <form onSubmit={onSubmit} className="mt-6 space-y-6">
              {role === 'ADMIN' && (
                <div>
                  <label className="block text-sm font-medium text-text-secondary mb-2">Patient ID Override</label>
                  <input
                    type="number"
                    value={overridePatientId}
                    onChange={e => setOverridePatientId(e.target.value)}
                    className="w-full bg-surface-elevated border border-white/10 rounded-xl px-4 py-3 text-text-primary focus:border-ai-500/50 focus:ring-2 focus:ring-ai-500/20 outline-none transition-all"
                    placeholder="Enter Patient ID"
                  />
                </div>
              )}

              <div
                className={`relative border-2 border-dashed rounded-2xl p-8 text-center transition-all cursor-pointer ${dragActive ? 'border-ai-500 bg-ai-500/10' : 'border-white/10 hover:border-ai-500/30 hover:bg-white/5'
                  }`}
                onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag} onDrop={handleDrop}
                onClick={() => document.getElementById('lesion-file').click()}
              >
                {preview ? (
                  <div className="relative aspect-video rounded-xl overflow-hidden mx-auto max-w-sm group">
                    <img src={preview} alt="Preview" className="w-full h-full object-cover" />
                    <div className="absolute inset-0 bg-black/50 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                      <p className="text-white font-medium flex items-center gap-2"><LuUpload /> Change Image</p>
                    </div>
                  </div>
                ) : (
                  <div className="py-8">
                    <div className="w-16 h-16 rounded-full bg-white/5 mx-auto flex items-center justify-center mb-4">
                      <LuUpload className="text-ai-400" size={32} />
                    </div>
                    <p className="text-text-primary font-medium">Click to upload or drag & drop</p>
                    <p className="text-xs text-text-muted mt-2">JPG, PNG up to 10MB</p>
                  </div>
                )}
                <input id="lesion-file" type="file" onChange={e => setFile(e.target.files[0])} className="hidden" accept="image/*" />
              </div>

              <div className="flex items-center gap-3 p-3 rounded-xl bg-orange-500/10 border border-orange-500/20">
                <LuTriangleAlert className="text-orange-400 flex-shrink-0" />
                <div className="flex-1">
                  <p className="text-sm font-medium text-text-primary">High Sensitivity Mode</p>
                  <p className="text-xs text-text-muted">May increase false positives, recommended for initial screening.</p>
                </div>
                <input
                  type="checkbox"
                  checked={highSensitivity}
                  onChange={e => setHighSensitivity(e.target.checked)}
                  className="w-5 h-5 rounded border-white/20 bg-white/10 text-ai-500 focus:ring-ai-500/50"
                />
              </div>

              {error && <div className="p-3 rounded-xl bg-danger/10 text-danger text-sm text-center font-medium">{error}</div>}

              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                type="submit"
                disabled={loading}
                className="w-full btn-primary py-4 text-base font-bold shadow-lg shadow-ai-500/20 flex items-center justify-center gap-2"
              >
                {loading ? <LuLoader className="animate-spin" /> : <><LuZap /> Analyze Lesion</>}
              </motion.button>
            </form>
          </Card>
        </motion.div>

        <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}>
          <AnimatePresence mode="wait">
            {!res ? (
              <Card variant="glass" className="h-full flex items-center justify-center text-center p-8 opacity-60">
                <div>
                  <LuSparkles className="text-text-muted mx-auto mb-4" size={48} />
                  <h3 className="text-xl font-bold text-text-primary mb-2">Ready for Analysis</h3>
                  <p className="text-text-tertiary">Results will appear here with AI confidence scores and recommendations.</p>
                </div>
              </Card>
            ) : (
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
                <Card variant="glow-ai">
                  <div className="flex items-start justify-between mb-6">
                    <div>
                      <CardTitle>Analysis Results</CardTitle>
                      <CardDescription>AI Confidence: {(res.confidence * 100).toFixed(1)}%</CardDescription>
                    </div>
                    <CardBadge variant={res.prediction?.toLowerCase().includes('melanoma') ? 'danger' : 'success'}>
                      {res.prediction}
                    </CardBadge>
                  </div>

                  <div className="space-y-4">
                    <CardData label="Lesion Severity" value={res.severity || 'Moderate'} />
                    <CardData label="Risk Assessment" value={res.risk_level || 'Attention Recommended'} />

                    <div className="pt-4 border-t border-white/10">
                      <h4 className="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-2">AI Recommendation</h4>
                      <p className="text-text-primary leading-relaxed">{res.recommendation || 'Consult a dermatologist for further evaluation.'}</p>
                    </div>
                  </div>
                </Card>

                {res.lesion_id && (
                  <Card variant="glass">
                    <CardTitle>Clinical Report</CardTitle>
                    <div className="mt-4">
                      {!report ? (
                        <button
                          onClick={runDiagnosis}
                          className="w-full py-3 bg-surface-elevated hover:bg-white/5 border border-white/10 rounded-xl text-text-primary transition-all flex items-center justify-center gap-2 font-medium"
                        >
                          <LuFileText /> Generate Diagnosis Report
                        </button>
                      ) : (
                        <div className="space-y-4 animate-in fade-in slide-in-from-bottom-4">
                          <div className="p-4 rounded-xl bg-white/5 border border-white/10 text-sm leading-relaxed text-text-secondary max-h-60 overflow-y-auto">
                            {diag}
                          </div>
                          <button className="w-full btn-ghost flex items-center justify-center gap-2">
                            <LuDownload size={18} /> Download PDF Report
                          </button>
                        </div>
                      )}
                    </div>
                  </Card>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </div>
    </div>
  )
}

function LuFileText(props) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="20" height="20" viewBox="0 0 24 24"
      fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
    >
      <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" y1="13" x2="8" y2="13" />
      <line x1="16" y1="17" x2="8" y2="17" />
      <line x1="10" y1="9" x2="8" y2="9" />
    </svg>
  )
}
