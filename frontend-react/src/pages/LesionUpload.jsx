import { useState, useEffect } from 'react'
import { api } from '../services/api'
import { motion, AnimatePresence } from 'framer-motion'
import {
  LuUpload, LuScan, LuTriangleAlert, LuZap, LuDownload,
  LuArrowRight, LuHistory, LuCircleCheck, LuShield, LuSparkles, LuX, LuLoader
} from 'react-icons/lu'
import { Card, CardTitle, CardDescription, CardData, CardBadge, IconWrapper } from '../components/Card'

const DISEASE_INFO = {
  melanoma: {
    name: 'Melanoma',
    icon: '⚠️',
    severity: 'danger',
    label: 'Potentially Malignant',
    what: 'The most serious form of skin cancer. Develops from pigment-producing cells (melanocytes). Can spread rapidly to other organs if not caught early.',
    looks: 'Asymmetric mole with irregular, notched borders. Often multi-coloured (tan, brown, black, red, white). Usually larger than 6mm and changes over time.',
    urgency: 'See a dermatologist within 2 weeks. Early detection dramatically increases survival rates.',
    action: 'Book an urgent dermatology appointment. Avoid sun exposure on the area. Do not pick or scratch it.',
  },
  ak: {
    name: 'Actinic Keratosis',
    icon: '🔶',
    severity: 'warning',
    label: 'Pre-Cancerous',
    what: 'A pre-cancerous rough, scaly patch caused by years of UV sun exposure. A small percentage can develop into squamous cell carcinoma if left untreated.',
    looks: 'Rough, dry, scaly patch — often red, pink or brown. Feels like sandpaper. Common on the face, ears, scalp, neck and backs of hands.',
    urgency: 'Not immediately dangerous but should be reviewed by a dermatologist within 1–2 months.',
    action: 'Book a dermatology appointment. Apply SPF 50+ sunscreen daily. Treatment options include cryotherapy, topical creams or laser therapy.',
  },
  nevus: {
    name: 'Melanocytic Nevus (Mole)',
    icon: '🟤',
    severity: 'success',
    label: 'Typically Benign',
    what: 'A common mole formed by a cluster of pigmented cells. Most moles are harmless, but they should be monitored monthly for any changes that could indicate melanoma.',
    looks: 'Small, round or oval, uniformly coloured spot (tan to dark brown) with well-defined borders. Usually under 6mm. May be flat or raised.',
    urgency: 'Monitor monthly. See a doctor if it changes in size, colour, shape, or starts bleeding.',
    action: 'No treatment needed unless it changes. Use the ABCDE self-check monthly: Asymmetry, Border, Colour, Diameter, Evolving.',
  },
  seborrheic_keratosis: {
    name: 'Seborrheic Keratosis',
    icon: '🟫',
    severity: 'success',
    label: 'Benign',
    what: 'A very common, non-cancerous skin growth that becomes more frequent with age. Completely harmless with no link to cancer.',
    looks: 'Waxy, wart-like, "stuck-on" appearance. Ranges from light tan to almost black. Can be rough or smooth. Often on the face, chest, shoulders or back.',
    urgency: 'No medical urgency — purely cosmetic.',
    action: 'No treatment required. Can be removed for cosmetic reasons by a dermatologist via cryotherapy or shaving.',
  },
  eczema: {
    name: 'Eczema (Atopic Dermatitis)',
    icon: '🔴',
    severity: 'warning',
    label: 'Chronic Inflammatory',
    what: 'A chronic skin condition causing dry, intensely itchy, inflamed skin. Often linked to asthma and hay fever. Not contagious.',
    looks: 'Red, dry, scaly, itchy patches. May weep fluid or crust over. Common on the backs of knees, inside elbows, face and neck.',
    urgency: 'Seek care if severely flared, showing signs of infection (yellow crust or pus), or not responding to moisturisers.',
    action: 'Moisturise frequently with fragrance-free emollients. Avoid triggers. Use topical steroids for flares as directed. Avoid scratching.',
  },
  psoriasis: {
    name: 'Psoriasis',
    icon: '🟠',
    severity: 'warning',
    label: 'Chronic Autoimmune',
    what: 'An autoimmune condition causing skin cells to multiply 10x faster than normal, leading to thick, scaly build-up. Not contagious.',
    looks: 'Red patches covered with silvery-white scales. Often on elbows, knees, scalp and lower back. May itch or burn. Nails can become pitted.',
    urgency: 'Not an emergency, but widespread or joint-affecting psoriasis needs specialist care.',
    action: 'Topical corticosteroids and vitamin D creams for mild cases. Phototherapy or biologic drugs for moderate-severe disease. Manage stress.',
  },
  fungal: {
    name: 'Fungal Skin Infection',
    icon: '🍄',
    severity: 'success',
    label: 'Infectious — Treatable',
    what: 'Infections caused by various fungi including ringworm (tinea corporis), athlete\'s foot (tinea pedis) and jock itch (tinea cruris). Very common and highly treatable.',
    looks: 'Ring-shaped, red, scaly patches with clearer centres and raised edges. Can affect any skin area, scalp, nails or feet. May be itchy.',
    urgency: 'Low urgency. Treat promptly to prevent spread to others.',
    action: 'Over-the-counter antifungal cream (e.g. clotrimazole) for 2–4 weeks. Keep the area clean and dry. See a GP if not improving.',
  },
  impetigo: {
    name: 'Impetigo',
    icon: '🔶',
    severity: 'warning',
    label: 'Bacterial Infection',
    what: 'A highly contagious bacterial skin infection most common in children. Caused by Staphylococcus aureus or Streptococcus pyogenes. Responds well to antibiotics.',
    looks: 'Red sores that burst and ooze, forming a characteristic honey-coloured crust. Often around the nose and mouth. May itch. Spreads easily.',
    urgency: 'See a GP within a few days. Keep the child away from school or nursery until 48 hours after starting antibiotics.',
    action: 'Topical antibiotic cream for mild cases; oral antibiotics for severe or widespread infection. Keep sores clean and covered.',
  },
  wart: {
    name: 'Wart / Verruca',
    icon: '🌱',
    severity: 'success',
    label: 'Benign — Viral (HPV)',
    what: 'Non-cancerous skin growths caused by the human papillomavirus (HPV). Very common, especially in children and young adults. Spread by direct contact.',
    looks: 'Small, rough, grainy bumps — flesh-coloured, white or tan. Often have black dots (clotted blood vessels). Appear anywhere but especially hands and soles.',
    urgency: 'Not urgent. Many resolve naturally within 2 years. Seek help if painful or spreading rapidly.',
    action: 'Over-the-counter salicylic acid treatments work for most warts. GP can offer cryotherapy. Avoid picking to prevent spread.',
  },
  dermatitis: {
    name: 'Dermatitis',
    icon: '🔴',
    severity: 'warning',
    label: 'Skin Inflammation',
    what: 'An umbrella term for skin inflammation. Contact dermatitis is triggered by an irritant or allergen. Seborrhoeic dermatitis is linked to a yeast and affects oily areas.',
    looks: 'Redness, swelling, itching, blistering or scaling. Contact dermatitis often has a clear boundary at the point of contact. Seborrhoeic type appears on the scalp, face and chest.',
    urgency: 'Seek care if severe, infected, or affecting daily life and sleep.',
    action: 'Identify and avoid triggers. Regular moisturising and mild topical steroids for flares. Antifungal shampoo for seborrhoeic type.',
  },
  drug_eruption: {
    name: 'Drug Eruption',
    icon: '💊',
    severity: 'warning',
    label: 'Medication Reaction',
    what: 'A skin rash caused by an adverse reaction to a medication. Can range from a mild morbilliform rash to severe life-threatening reactions (Stevens-Johnson syndrome).',
    looks: 'Widespread red spots or blotches resembling measles (morbilliform). Fixed drug eruptions reappear as a dark patch in the same location each time.',
    urgency: 'Contact your GP or pharmacist. Seek emergency care immediately if blistering, mouth sores or breathing difficulties occur — these are medical emergencies.',
    action: 'Identify and stop the causative drug with medical guidance. Antihistamines for mild itch. Hospital care for severe reactions.',
  },
  hyperpigmentation: {
    name: 'Hyperpigmentation',
    icon: '🟤',
    severity: 'success',
    label: 'Cosmetic — Benign',
    what: 'Dark patches on the skin caused by excess melanin. Common causes include sun damage (solar lentigines), hormonal changes (melasma), or post-inflammatory marks after acne or injury.',
    looks: 'Flat, darkened patches — tan, brown or grey. Clearly defined boundaries. Common on the face, neck and hands. No pain or texture change.',
    urgency: 'Not a medical emergency. Consult a dermatologist if patches are spreading rapidly or changing in character.',
    action: 'Daily SPF 50+ sunscreen is essential. Topical treatments: vitamin C, niacinamide, retinoids or prescribed hydroquinone. Chemical peels or laser for stubborn cases.',
  },
  alopecia: {
    name: 'Alopecia (Hair Loss)',
    icon: '🪄',
    severity: 'warning',
    label: 'Hair Loss Condition',
    what: 'Hair loss from various causes including alopecia areata (autoimmune), androgenetic alopecia (pattern baldness), stress-related telogen effluvium, or scarring conditions.',
    looks: 'Smooth, round patches of hair loss (areata). Diffuse thinning across the scalp (androgenetic). Scarring alopecia may show redness or scaling around follicles.',
    urgency: 'See a dermatologist if patches are expanding quickly, the scalp is inflamed, or hair loss is sudden and widespread.',
    action: 'Topical minoxidil for androgenetic alopecia. Corticosteroid injections or immunotherapy for areata. Treat any underlying medical causes.',
  },
  angioma: {
    name: 'Angioma (Vascular Lesion)',
    icon: '❤️',
    severity: 'success',
    label: 'Benign — Vascular',
    what: 'Benign overgrowths of blood vessels in the skin. Cherry angiomas are extremely common in adults, especially after age 30. Spider angiomas have a central red dot with radiating vessels.',
    looks: 'Bright cherry-red, smooth, dome-shaped papule. Blanches when pressed. Most common on the trunk, upper arms and shoulders. Usually 1–5mm.',
    urgency: 'No medical urgency. Seek care if it bleeds repeatedly, grows rapidly, or its appearance changes.',
    action: 'No treatment required. Can be easily removed for cosmetic reasons via laser or electrocautery by a dermatologist.',
  },
  lupus: {
    name: 'Cutaneous Lupus',
    icon: '🦋',
    severity: 'warning',
    label: 'Autoimmune — Needs Review',
    what: 'Lupus is a systemic autoimmune disease that can manifest on the skin. UV light commonly triggers flares. Skin involvement may indicate or precede systemic lupus.',
    looks: 'Classic butterfly (malar) rash across the cheeks and nose. May be scaly (discoid lupus). Also: photosensitive rashes elsewhere, mouth ulcers, hair thinning.',
    urgency: 'See a rheumatologist or dermatologist. Lupus can affect the kidneys, heart, joints and lungs if not treated.',
    action: 'Sun protection is critical — SPF 50+, hats, clothing. Hydroxychloroquine (anti-malarial) is first-line. Immunosuppressives for severe cases.',
  },
  vasculitis: {
    name: 'Vasculitis',
    icon: '🔴',
    severity: 'danger',
    label: 'Needs Urgent Review',
    what: 'Inflammation of blood vessel walls causing damage and restricted blood flow. Skin vasculitis often signals an underlying systemic condition affecting other organs.',
    looks: 'Palpable purpura — raised, non-blanching, red or purple spots or patches. Typically on the lower legs. May develop into ulcers in severe cases.',
    urgency: 'See a doctor promptly. Systemic vasculitis can affect the kidneys, lungs and peripheral nerves.',
    action: 'Blood tests and skin biopsy needed for diagnosis. Treatment depends on the underlying cause: corticosteroids and immunosuppressives under specialist supervision.',
  },
  bullous: {
    name: 'Bullous Disease (Blistering Disorder)',
    icon: '💧',
    severity: 'danger',
    label: 'Autoimmune — Potentially Serious',
    what: 'Autoimmune conditions (including bullous pemphigoid and pemphigus vulgaris) where the immune system attacks proteins that hold skin layers together, forming blisters.',
    looks: 'Large, tense, fluid-filled blisters (bullous pemphigoid) or fragile, easily-ruptured blisters leaving raw erosions (pemphigus). Often on the trunk and limbs.',
    urgency: 'Seek prompt medical attention. Widespread blistering can lead to dangerous infections and fluid loss.',
    action: 'Diagnosis confirmed by skin biopsy and blood tests. Treated with systemic corticosteroids and immunosuppressives under specialist dermatology care.',
  },
  scabies: {
    name: 'Scabies',
    icon: '🐛',
    severity: 'warning',
    label: 'Parasitic — Contagious',
    what: 'A highly contagious infestation by the Sarcoptes scabiei mite, which burrows into the outer layer of skin. Causes intense itching, especially at night. Spreads through prolonged skin contact.',
    looks: 'Tiny blisters or pimples and S-shaped burrow marks. Most common between the fingers, on the wrists, armpits, waistline and genitals. Widespread rash from allergic response.',
    urgency: 'Treat as soon as possible to prevent spread to household members and close contacts.',
    action: 'Prescription permethrin 5% cream or oral ivermectin. All household contacts must be treated at the same time. Wash all bedding and clothing at 60°C.',
  },
  viral: {
    name: 'Viral Skin Infection',
    icon: '🦠',
    severity: 'success',
    label: 'Infectious — Usually Treatable',
    what: 'Skin conditions caused by viruses including herpes simplex (cold sores), herpes zoster (shingles), chickenpox, molluscum contagiosum and viral exanthems (rashes).',
    looks: 'Varies: grouped blisters on a red base (herpes/shingles); pink, pearly domed papules with a central dimple (molluscum); widespread red blotchy rash (viral exanthem).',
    urgency: 'Seek care urgently for shingles involving the eye or ear. Herpes in immunocompromised patients needs prompt antivirals.',
    action: 'Antivirals (acyclovir, valacyclovir) for herpes/shingles — most effective when started within 72 hours. Most other viral rashes resolve with rest and symptom management.',
  },
  systemic: {
    name: 'Systemic Disease — Skin Manifestation',
    icon: '🏥',
    severity: 'warning',
    label: 'Requires Investigation',
    what: 'The skin can reflect internal diseases. Conditions like diabetes, thyroid disorders, kidney disease, liver disease and autoimmune diseases often appear on the skin first.',
    looks: 'Varies widely: yellowing (jaundice from liver disease); bronze discolouration (Addison\'s disease); xanthomas (fatty yellow deposits from high lipids); velvety dark patches in folds (acanthosis nigricans).',
    urgency: 'Should be investigated by a GP or internal medicine specialist. The skin finding may be the earliest visible sign of internal disease.',
    action: 'See a GP for a full examination and blood tests. Treatment is directed at the underlying systemic condition.',
  },
}

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

  const MAX_FILE_SIZE = 10 * 1024 * 1024 // 10MB
  const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/heic']

  const validateAndSetFile = (f) => {
    if (!f) return
    if (!ALLOWED_TYPES.includes(f.type) && !f.name.match(/\.(jpg|jpeg|png|webp|heic)$/i)) {
      setError('Please select a valid image file (JPG, PNG, or WebP)')
      return
    }
    if (f.size > MAX_FILE_SIZE) {
      setError(`File too large (${(f.size / 1024 / 1024).toFixed(1)}MB). Maximum is 10MB.`)
      return
    }
    setError('')
    setFile(f)
  }

  const onSubmit = async (e) => {
    e.preventDefault()
    setRes(null); setDiag(''); setError(''); setReport(null); setLoading(true)
    const pid = role === 'ADMIN' ? parseInt(overridePatientId || '0') : patientId
    if (!pid || isNaN(pid)) { setError(role === 'ADMIN' ? 'Enter a valid Patient ID' : 'Patient profile not found. Please log out and log in again.'); setLoading(false); return }
    if (!file) { setError('Select an image'); setLoading(false); return }
    try {
      const result = await api.predictLesion(pid, file, { sensitivity: highSensitivity ? 'high' : undefined })
      setRes(result)
    } catch (err) { setError(err.message || 'Upload failed') }
    setLoading(false)
  }

  const [diagLoading, setDiagLoading] = useState(false)

  const runDiagnosis = async () => {
    if (!res?.lesion_id || diagLoading) return
    setDiagLoading(true)
    try {
      const pid = role === 'ADMIN' ? parseInt(overridePatientId || '0') : patientId
      const rep = await api.createDiagnosisReport(res.lesion_id, pid)
      setReport(rep)
      setDiag(rep.details || '')
    } catch (err) { setDiag(String(err.message || err)) }
    finally { setDiagLoading(false) }
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
    if (e.dataTransfer.files && e.dataTransfer.files[0]) validateAndSetFile(e.dataTransfer.files[0])
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
                <input id="lesion-file" type="file" onChange={e => validateAndSetFile(e.target.files[0])} className="hidden" accept="image/jpeg,image/png,image/webp" />
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

                {/* Low-confidence warning banner */}
                {res.is_low_confidence && (
                  <div className="flex items-start gap-3 p-4 rounded-xl bg-yellow-500/10 border border-yellow-500/30">
                    <LuTriangleAlert className="text-yellow-400 flex-shrink-0 mt-0.5" size={20} />
                    <div>
                      <p className="text-sm font-semibold text-yellow-300">Image quality too low for confident prediction</p>
                      <p className="text-xs text-yellow-200/70 mt-1">The model is uncertain — please retake the photo as a clear, well-lit close-up of the skin lesion. Avoid blurry or full-body shots.</p>
                    </div>
                  </div>
                )}

                <Card variant="glow-ai">
                  <div className="flex items-start justify-between mb-6">
                    <div>
                      <CardTitle>Analysis Results</CardTitle>
                      <CardDescription>
                        Confidence: <span className={res.is_low_confidence ? 'text-yellow-400' : 'text-green-400'}>
                          {res.confidence != null ? (res.confidence * 100).toFixed(1) + '%' : '—'}
                        </span>
                        {res.entropy != null && (
                          <span className="ml-2 text-text-muted text-xs">
                            · Certainty {(100 - res.entropy * 100).toFixed(0)}%
                          </span>
                        )}
                      </CardDescription>
                    </div>
                    <CardBadge variant={
                      res.prediction === 'malignant' ? 'danger' :
                      res.is_low_confidence ? 'warning' : 'success'
                    }>
                      {res.label
                        ? res.label.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
                        : res.prediction}
                    </CardBadge>
                  </div>

                  <div className="space-y-4">
                    <CardData
                      label="Risk Assessment"
                      value={
                        res.prediction === 'malignant' ? 'Needs clinical review' :
                        res.is_low_confidence ? 'Unclear — retake photo' :
                        'Low risk — monitor regularly'
                      }
                    />
                    {res.risk_score != null && (
                      <div>
                        <div className="flex justify-between text-xs text-text-muted mb-1">
                          <span>Malignancy risk score</span>
                          <span>{(res.risk_score * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-2 rounded-full bg-white/10 overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all ${res.risk_score > 0.5 ? 'bg-danger' : res.risk_score > 0.3 ? 'bg-yellow-500' : 'bg-green-500'}`}
                            style={{ width: `${Math.min(100, res.risk_score * 100)}%` }}
                          />
                        </div>
                      </div>
                    )}

                    {/* Top predictions breakdown */}
                    {res.top_probs && Object.keys(res.top_probs).length > 0 && (
                      <div className="pt-4 border-t border-white/10">
                        <h4 className="text-xs font-semibold text-text-secondary uppercase tracking-wider mb-3">Top Predictions</h4>
                        <div className="space-y-2">
                          {Object.entries(res.top_probs).map(([cls, prob], i) => (
                            <div key={cls}>
                              <div className="flex justify-between text-xs mb-0.5">
                                <span className={`${i === 0 ? 'text-text-primary font-medium' : 'text-text-muted'}`}>
                                  {cls.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                                </span>
                                <span className="text-text-muted">{(prob * 100).toFixed(1)}%</span>
                              </div>
                              <div className="h-1.5 rounded-full bg-white/10 overflow-hidden">
                                <div
                                  className={`h-full rounded-full ${i === 0 ? 'bg-ai-500' : 'bg-white/30'}`}
                                  style={{ width: `${Math.min(100, prob * 100)}%` }}
                                />
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    <div className="pt-4 border-t border-white/10">
                      <h4 className="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-2">AI Recommendation</h4>
                      <p className="text-text-primary leading-relaxed">
                        {res.is_low_confidence
                          ? 'Image quality is too low for a reliable result. Please retake with a clear, close-up photo in good lighting.'
                          : res.prediction === 'malignant'
                            ? 'This lesion shows features that warrant clinical attention. Please book a dermatologist review within 1–2 weeks.'
                            : 'No immediate concern detected. Monitor the area monthly and consult a dermatologist if it changes.'}
                      </p>
                    </div>
                  </div>
                </Card>

                {/* Disease info panel */}
                {res.label && DISEASE_INFO[res.label] && (() => {
                  const d = DISEASE_INFO[res.label]
                  const borderCls = d.severity === 'danger'
                    ? 'border-red-500/30 bg-red-500/5'
                    : d.severity === 'warning'
                      ? 'border-yellow-500/30 bg-yellow-500/5'
                      : 'border-green-500/30 bg-green-500/5'
                  const urgencyCls = d.severity === 'danger'
                    ? 'bg-red-500/10 border-red-500/30 text-red-400'
                    : d.severity === 'warning'
                      ? 'bg-yellow-500/10 border-yellow-500/30 text-yellow-400'
                      : 'bg-green-500/10 border-green-500/30 text-green-400'
                  return (
                    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}>
                      <Card variant="glass" className={`border ${borderCls}`}>
                        <div className="flex items-start justify-between gap-3 mb-4">
                          <div className="flex items-center gap-3">
                            <span className="text-2xl">{d.icon}</span>
                            <div>
                              <CardTitle>{d.name}</CardTitle>
                              <CardBadge variant={d.severity}>{d.label}</CardBadge>
                            </div>
                          </div>
                        </div>
                        <div className="space-y-4 text-sm">
                          <div>
                            <p className="text-xs font-semibold text-text-secondary uppercase tracking-wider mb-1">What is it?</p>
                            <p className="text-text-primary leading-relaxed">{d.what}</p>
                          </div>
                          <div>
                            <p className="text-xs font-semibold text-text-secondary uppercase tracking-wider mb-1">What does it look like?</p>
                            <p className="text-text-primary leading-relaxed">{d.looks}</p>
                          </div>
                          <div className={`p-3 rounded-xl border ${urgencyCls}`}>
                            <p className="text-xs font-semibold uppercase tracking-wider mb-1">When to seek care</p>
                            <p className="leading-relaxed">{d.urgency}</p>
                          </div>
                          <div>
                            <p className="text-xs font-semibold text-text-secondary uppercase tracking-wider mb-1">What to do</p>
                            <p className="text-text-primary leading-relaxed">{d.action}</p>
                          </div>
                          <p className="text-xs text-text-muted italic border-t border-white/10 pt-3">
                            ⚕️ This information is educational only. Always consult a qualified dermatologist for diagnosis and treatment.
                          </p>
                        </div>
                      </Card>
                    </motion.div>
                  )
                })()}

                {res.lesion_id && (
                  <Card variant="glass">
                    <CardTitle>Clinical Report</CardTitle>
                    <div className="mt-4">
                      {!report ? (
                        <button
                          onClick={runDiagnosis}
                          disabled={diagLoading}
                          className="w-full py-3 bg-surface-elevated hover:bg-white/5 border border-white/10 rounded-xl text-text-primary transition-all flex items-center justify-center gap-2 font-medium disabled:opacity-50"
                        >
                          {diagLoading ? <LuLoader className="animate-spin" /> : <LuFileText />} {diagLoading ? 'Generating...' : 'Generate Diagnosis Report'}
                        </button>
                      ) : (
                        <div className="space-y-4 animate-in fade-in slide-in-from-bottom-4">
                          <div className="p-4 rounded-xl bg-white/5 border border-white/10 text-sm leading-relaxed text-text-secondary max-h-60 overflow-y-auto">
                            {diag}
                          </div>
                          <button onClick={() => {
                            const text = `Diagnosis Report\nLesion ID: ${res.lesion_id}\n\n${diag}`
                            const blob = new Blob([text], { type: 'text/plain' })
                            const url = URL.createObjectURL(blob)
                            const a = document.createElement('a')
                            a.href = url; a.download = `diagnosis_report_${res.lesion_id}.txt`; a.click()
                            URL.revokeObjectURL(url)
                          }} className="w-full btn-ghost flex items-center justify-center gap-2">
                            <LuDownload size={18} /> Download Report
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
