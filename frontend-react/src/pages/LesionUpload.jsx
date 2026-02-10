import { useState, useEffect } from 'react'
import { api } from '../services/api'
import { motion, AnimatePresence } from 'framer-motion'
import {
  LuUpload, LuScan, LuTriangleAlert, LuZap, LuDownload,
  LuArrowRight, LuHistory, LuCircleCheck, LuShield, LuSparkles, LuX, LuLoader, LuFileText, LuClock
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

  // Previous scans state
  const [previousScans, setPreviousScans] = useState([])
  const [loadingHistory, setLoadingHistory] = useState(true)

  // Fetch previous scans on mount
  useEffect(() => {
    fetchPreviousScans()
  }, [])

  const fetchPreviousScans = async () => {
    setLoadingHistory(true)
    try {
      const pid = role === 'ADMIN' ? null : patientId
      const reports = await api.listDiagnosisReports(pid)
      setPreviousScans(Array.isArray(reports) ? reports.slice(0, 5) : []) // Show last 5
    } catch (err) {
      console.error('Failed to fetch previous scans:', err)
      setPreviousScans([])
    } finally {
      setLoadingHistory(false)
    }
  }

  const onSubmit = async (e) => {
    e.preventDefault()
    setRes(null); setDiag(''); setError(''); setReport(null); setLoading(true)
    const pid = role === 'ADMIN' ? parseInt(overridePatientId || '0') : patientId
    if (!pid || isNaN(pid)) {
      setError(role === 'ADMIN' ? 'Please enter a valid patient ID' : 'Patient profile not found. Please log in again.')
      setLoading(false)
      return
    }
    if (!file) { setError('Please select an image to analyze'); setLoading(false); return }
    try {
      const result = await api.predictLesion(pid, file, { sensitivity: highSensitivity ? 'high' : undefined })
      setRes(result)
      // Update scan count for dashboard
      const currentCount = parseInt(localStorage.getItem('scan_count') || '0')
      localStorage.setItem('scan_count', (currentCount + 1).toString())
      // Refresh history
      fetchPreviousScans()
    } catch (err) { setError(err.message || 'Analysis failed. Please try again.') }
    setLoading(false)
  }

  const runDiagnosis = async () => {
    if (!res?.lesion_id) return
    try {
      const pid = role === 'ADMIN' ? parseInt(overridePatientId || '0') : patientId
      const rep = await api.createDiagnosisReport(res.lesion_id, pid)
      setReport(rep)
      setDiag(rep.details || '')
      fetchPreviousScans() // Refresh history after creating report
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
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8 relative z-10 max-w-5xl mx-auto pt-6"
      >
        <div className="flex items-center gap-4 mb-4">
          <IconWrapper variant="primary" size="lg" className="bg-primary-50 text-primary-600">
            <LuScan size={28} />
          </IconWrapper>
          <div>
            <h1 className="text-3xl font-bold text-slate-900 tracking-tight">
              AI Lesion Analysis
            </h1>
            <p className="text-slate-500">Clinical-grade dermatological assessment powered by Deep Learning</p>
          </div>
        </div>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 max-w-5xl mx-auto relative z-10">
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="space-y-6">
          <Card>
            <CardTitle>Upload Scan</CardTitle>
            <CardDescription>Upload a clear, close-up image of the skin area.</CardDescription>

            <form onSubmit={onSubmit} className="mt-6 space-y-6">
              {role === 'ADMIN' && (
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">Patient ID Override</label>
                  <input
                    type="number"
                    value={overridePatientId}
                    onChange={e => setOverridePatientId(e.target.value)}
                    className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-slate-900 focus:border-primary-500 focus:ring-2 focus:ring-primary-500/10 outline-none transition-all"
                    placeholder="Enter Patient ID"
                  />
                </div>
              )}

              <div
                className={`relative border-2 border-dashed rounded-2xl p-8 text-center transition-all cursor-pointer ${dragActive ? 'border-primary-500 bg-primary-50' : 'border-slate-200 hover:border-primary-300 hover:bg-slate-50'
                  }`}
                onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag} onDrop={handleDrop}
                onClick={() => document.getElementById('lesion-file').click()}
              >
                {preview ? (
                  <div className="relative aspect-video rounded-xl overflow-hidden mx-auto max-w-sm group shadow-sm border border-slate-200">
                    <img src={preview} alt="Preview" className="w-full h-full object-cover" />
                    <div className="absolute inset-0 bg-slate-900/50 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                      <p className="text-white font-medium flex items-center gap-2"><LuUpload /> Change Image</p>
                    </div>
                  </div>
                ) : (
                  <div className="py-8">
                    <div className="w-16 h-16 rounded-full bg-slate-100 mx-auto flex items-center justify-center mb-4 text-primary-500">
                      <LuUpload size={32} />
                    </div>
                    <p className="text-slate-900 font-medium">Click to upload or drag & drop</p>
                    <p className="text-xs text-slate-400 mt-2">JPG, PNG up to 10MB</p>
                  </div>
                )}
                <input id="lesion-file" type="file" onChange={e => setFile(e.target.files[0])} className="hidden" accept="image/*" />
              </div>

              <div className="flex items-center gap-3 p-4 rounded-xl bg-amber-50 border border-amber-100">
                <LuTriangleAlert className="text-amber-500 flex-shrink-0" size={20} />
                <div className="flex-1">
                  <p className="text-sm font-semibold text-slate-900">High Sensitivity Mode</p>
                  <p className="text-xs text-slate-500">Prioritizes detection of potential issues. Recommended for screening.</p>
                </div>
                <input
                  type="checkbox"
                  checked={highSensitivity}
                  onChange={e => setHighSensitivity(e.target.checked)}
                  className="w-5 h-5 rounded border-slate-300 text-primary-600 focus:ring-primary-500"
                />
              </div>

              {error && <div className="p-3 rounded-xl bg-rose-50 border border-rose-100 text-rose-600 text-sm text-center font-medium">{error}</div>}

              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                type="submit"
                disabled={loading}
                className="w-full btn btn-primary py-4 text-base font-bold shadow-lg shadow-primary-500/20 flex items-center justify-center gap-2"
              >
                {loading ? <LuLoader className="animate-spin" /> : <><LuZap /> Analyze Lesion</>}
              </motion.button>
            </form>
          </Card>

          {/* Previous Scans Section */}
          <Card>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <LuHistory className="text-slate-400" size={18} />
                <CardTitle className="text-base">Previous Scans</CardTitle>
              </div>
              {previousScans.length > 0 && (
                <span className="text-xs text-slate-400 font-medium">{previousScans.length} reports</span>
              )}
            </div>

            {loadingHistory ? (
              <div className="py-8 text-center text-slate-400">
                <LuLoader className="animate-spin mx-auto mb-2" size={24} />
                <p className="text-sm">Loading history...</p>
              </div>
            ) : previousScans.length === 0 ? (
              <div className="py-8 text-center text-slate-400">
                <LuClock className="mx-auto mb-2 opacity-50" size={28} />
                <p className="text-sm">No previous scans yet</p>
                <p className="text-xs mt-1">Your scan history will appear here</p>
              </div>
            ) : (
              <div className="space-y-3">
                {previousScans.map((scan, i) => (
                  <motion.div
                    key={scan.id || i}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.05 }}
                    className="p-3 rounded-xl bg-slate-50 border border-slate-100 hover:border-slate-200 transition-all"
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="font-medium text-slate-900 text-sm">
                          {scan.prediction || scan.summary || 'Lesion Analysis'}
                        </p>
                        <p className="text-xs text-slate-400 mt-0.5">
                          {scan.created_at ? new Date(scan.created_at).toLocaleDateString() : 'Recent'}
                        </p>
                      </div>
                      <CardBadge variant={
                        (scan.prediction || '').toLowerCase().includes('melanoma') ? 'danger' :
                          (scan.prediction || '').toLowerCase().includes('benign') ? 'success' : 'default'
                      } className="text-xs">
                        {scan.risk_level || 'Analyzed'}
                      </CardBadge>
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </Card>
        </motion.div>

        <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}>
          <AnimatePresence mode="wait">
            {!res ? (
              <Card className="h-full flex items-center justify-center text-center p-8 bg-slate-50/50 border-dashed border-2 border-slate-200 shadow-none">
                <div>
                  <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center mx-auto mb-4 text-slate-300 shadow-sm border border-slate-100">
                    <LuSparkles size={32} />
                  </div>
                  <h3 className="text-lg font-bold text-slate-900 mb-2">Ready for Analysis</h3>
                  <p className="text-slate-500 max-w-xs mx-auto">Upload an image to receive instant AI diagnostics and recommendations.</p>
                </div>
              </Card>
            ) : (
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
                <Card className="border-l-4 border-l-primary-500">
                  <div className="flex items-start justify-between mb-6">
                    <div>
                      <CardTitle>Analysis Results</CardTitle>
                      <CardDescription>AI Confidence: {(res.confidence * 100).toFixed(1)}%</CardDescription>
                    </div>
                    <CardBadge variant={res.prediction?.toLowerCase().includes('melanoma') ? 'danger' : 'success'}>
                      {res.prediction}
                    </CardBadge>
                  </div>

                  <div className="space-y-6">
                    <div className="grid grid-cols-2 gap-4">
                      <CardData label="Severity" size="sm">{res.severity || 'Moderate'}</CardData>
                      <CardData label="Risk Level" size="sm">{res.risk_level || 'Attention Recommended'}</CardData>
                    </div>

                    <div className="pt-6 border-t border-slate-100">
                      <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">AI Recommendation</h4>
                      <div className="p-4 bg-slate-50 rounded-xl text-slate-700 leading-relaxed border border-slate-100">
                        {res.recommendation || 'Consult a dermatologist for further evaluation.'}
                      </div>
                    </div>
                  </div>
                </Card>

                {res.lesion_id && (
                  <Card>
                    <div className="flex items-center gap-3 mb-4">
                      <IconWrapper variant="default" size="sm">
                        <LuFileText size={20} />
                      </IconWrapper>
                      <CardTitle>Clinical Report</CardTitle>
                    </div>

                    <div className="mt-2">
                      {!report ? (
                        <button
                          onClick={runDiagnosis}
                          className="w-full py-3 bg-white hover:bg-slate-50 border border-slate-200 rounded-xl text-slate-700 transition-all flex items-center justify-center gap-2 font-medium shadow-sm hover:shadow-md"
                        >
                          <LuSparkles className="text-violet-500" /> Generate Detailed Report
                        </button>
                      ) : (
                        <div className="space-y-4 animate-in fade-in slide-in-from-bottom-4">
                          <div className="p-5 rounded-xl bg-slate-50 border border-slate-100 text-sm leading-relaxed text-slate-600 max-h-60 overflow-y-auto whitespace-pre-wrap font-mono text-xs">
                            {diag}
                          </div>
                          <p className="text-xs text-slate-400 text-center">
                            Share this report with your dermatologist for professional evaluation.
                          </p>
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
