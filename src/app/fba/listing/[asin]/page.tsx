'use client'

import { useParams, useRouter } from 'next/navigation'
import { useEffect, useState, useCallback } from 'react'
import { isPushableDetail, unpushableReason } from '@/lib/fba/productDetailAttrs'
import { SECTION_WEIGHTS, weightedPoints } from '@/lib/fba/scoreWeights'
// Using <img> instead of next/image to avoid domain config issues with Amazon CDN

// ─── Types (mirrored from fba/page.tsx) ─────────────────────────────────────

interface SeoIssue { field: string; message: string; severity: 'error' | 'warning' | 'info'; auto_fixable?: boolean }

interface ChildContentRow {
  sku: string; asin: string; parent_asin: string
  title: string | null; bullet_1: string | null; bullet_2: string | null
  bullet_3: string | null; bullet_4: string | null; bullet_5: string | null
  description: string | null; backend_keywords: string | null
  image_count: number; has_aplus: boolean; aplus_module_count: number
  aplus_has_brand_story: boolean; aplus_has_headline: boolean
  aplus_images_missing_alt: number; content_synced_at: string
}

interface SeoScoreRow {
  parent_asin: string; title_score: number; bullet_score: number
  keyword_score: number; aplus_score: number
  description_score: number | null; features_score: number | null; overall_score: number
  issues: SeoIssue[]; child_count: number; child_override_count: number
  top_child_asin: string | null; product_title: string | null
  image_url: string | null; total_units_30d: number; scored_at: string
  children: ChildContentRow[]
}

interface KeywordReconciliation {
  keyword: string; action_type: 'CRITICAL' | 'UPGRADE' | 'REINFORCE'
  search_volume: number; placed_in: string[]; exact_text: string; why: string
}

interface PerChildKeywords { sku: string; asin: string; keywords: string }

interface VariantCorrection { sku: string; field: string; current: string; replace_with: string; reason: string }
interface CannibalizationWarning { keyword: string; affected_skus: string[]; issue: string; recommendation: string }
interface ProductDetailImprovement { field_name: string; current_value: string | null; recommended_value: string; reason: string }

interface AplusModuleAction {
  module_type: string; action: 'ADD' | 'EDIT' | 'KEEP'
  content_brief: string; position: number
}

interface ActionPlanItem {
  element: string; level: 'parent' | 'per_child'
  verdict: 'REPLACE' | 'EDIT' | 'CREATE' | 'DONE' | 'SKIP'
  priority: 'HIGH' | 'MEDIUM' | 'LOW' | 'NONE'
  current_status: string; instruction: string
  replacement_content?: string | string[] | Record<string, unknown>[] | null
  seller_central_path?: string; content_ready?: boolean
  notes?: string; aplus_modules?: AplusModuleAction[]
}

interface PerChildTitle { sku: string; asin: string; title: string }

interface AiRecommendations {
  parent_asin: string; recommended_title: string; recommended_bullets: string[]
  recommended_keywords: string; per_child_keywords?: PerChildKeywords[]
  /** Per-child titles for capacity variation families (SD cards 64/128/256GB). When present,
   *  each child carries its own capacity instead of a single broadcast title. */
  per_child_titles?: PerChildTitle[]
  recommended_description: string; variant_corrections: VariantCorrection[]
  cannibalization_warnings?: CannibalizationWarning[]
  product_details_improvements?: ProductDetailImprovement[]
  keyword_reconciliation?: KeywordReconciliation[]
  action_plan?: ActionPlanItem[]
  generated_at: string; keyword_opportunities_used?: number
}

interface AnalyzedKeyword {
  keyword: string; opportunityScore: number
  actionType: 'CRITICAL' | 'UPGRADE' | 'REINFORCE' | 'DEFENDED' | 'OPTIMIZED' | 'IRRELEVANT'
  actionText: string; rationale: string; urgency: string; estimatedImpact: string
  searchVolume: number; keywordSales: number; competingProducts: number
  asinImpressionShare: number; asinClickShare: number; asinPurchaseShare: number
  inTitle: boolean; inBullets: boolean; inDescription: boolean; inBackend: boolean
  dataSource: string
}

interface KeywordIntelligenceResult {
  asin: string; analyzedAt: string; dataSource: string
  totalKeywordsAnalyzed: number; topOpportunities: AnalyzedKeyword[]
  summary: { critical: number; upgrade: number; reinforce: number; defended: number; optimized: number }
  apiUsage?: { used: number; limit: number; remaining: number; provider: string }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

function barColor(score: number, max: number) {
  const pct = score / max
  if (pct >= 0.8) return 'bg-green-500'
  if (pct >= 0.6) return 'bg-amber-500'
  return 'bg-red-500'
}

function issueBorder(field: string) {
  if (field.includes('title')) return 'border-l-blue-500'
  if (field.includes('bullet') || field.includes('description')) return 'border-l-green-500'
  if (field.includes('keyword') || field.includes('backend')) return 'border-l-amber-500'
  if (field.includes('aplus') || field.includes('image')) return 'border-l-purple-500'
  return 'border-l-slate-400'
}

function copyToClipboard(text: string) {
  if (typeof window !== 'undefined') navigator.clipboard.writeText(text)
}

// Strip Amazon's appended variant dimensions (" -Light Green-XX-Large") so the header and the
// cohesion comparison use the seller's BASE title, not a child's suffixed one.
const SIZE_TOKEN = "(?:XS|S|M|L|XL|XXL|XXXL|[2-5]XL|X-?Small|XX?X?-?Large|Small|Medium|Large|One[ -]?Size)"
function stripVariantSuffix(title: string | null | undefined): string {
  return (title ?? '')
    .replace(new RegExp(`\\s*[-–—|]\\s*[A-Za-z][\\w /&'-]*?\\s*[-–—|]\\s*${SIZE_TOKEN}\\s*$`, 'i'), '')
    .replace(new RegExp(`\\s*[-–—|]\\s*${SIZE_TOKEN}\\s*$`, 'i'), '')
    .trim()
}

// Strip a storage-capacity token ("64GB", "1 TB") from a title — used to render the capacity-
// AGNOSTIC parent / variation-hub title for capacity-variation families (SD cards by GB). The
// stored product_title is the best-seller CHILD's title, so it carries that child's capacity; the
// parent header must not. Shared by the page header and the TITLES card's PARENT row so the two
// can never drift.
function stripCapacityToken(title: string | null | undefined): string {
  return (title ?? '')
    .replace(/\b\d{1,4}\s?(?:GB|TB|MB)\b/gi, '')
    .replace(/\s{2,}/g, ' ')
    .trim()
}

// Dedup variant rows by ASIN (prefer the FBA SKU). The same ASIN can have both an FBA and an
// FBM SKU; backend keywords are per-ASIN, so the page (and the push) should treat them as one.
function dedupByAsin<T extends { sku: string; asin: string }>(rows: T[]): T[] {
  const byAsin = new Map<string, T>()
  for (const c of rows) {
    const existing = byAsin.get(c.asin)
    if (!existing || c.sku.endsWith('-FBA')) byAsin.set(c.asin, c)
  }
  return [...byAsin.values()].sort((a, b) => a.sku.localeCompare(b.sku))
}

// ─── Inline icons (SVG, not emoji — crisp at any size, consistent stroke) ─────
type IconProps = { className?: string }
const Icon = {
  ArrowLeft: (p: IconProps) => (<svg viewBox="0 0 24 24" fill="none" className={p.className} stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M19 12H5M12 19l-7-7 7-7" /></svg>),
  Sparkles: (p: IconProps) => (<svg viewBox="0 0 24 24" fill="none" className={p.className} stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9L12 3zM5 19l.6 1.6L7 21l-1.4.6L5 23l-.6-1.4L3 21l1.4-.4L5 19z" /></svg>),
  Send: (p: IconProps) => (<svg viewBox="0 0 24 24" fill="none" className={p.className} stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" /></svg>),
  Clipboard: (p: IconProps) => (<svg viewBox="0 0 24 24" fill="none" className={p.className} stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" /><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" /></svg>),
  External: (p: IconProps) => (<svg viewBox="0 0 24 24" fill="none" className={p.className} stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6M15 3h6v6M10 14L21 3" /></svg>),
  Layers: (p: IconProps) => (<svg viewBox="0 0 24 24" fill="none" className={p.className} stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" /></svg>),
  Tag: (p: IconProps) => (<svg viewBox="0 0 24 24" fill="none" className={p.className} stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M20.59 13.41l-7.17 7.17a2 2 0 01-2.83 0L2 12V2h10l8.59 8.59a2 2 0 010 2.82z" /><circle cx="7" cy="7" r="1.4" fill="currentColor" stroke="none" /></svg>),
  Check: (p: IconProps) => (<svg viewBox="0 0 24 24" fill="none" className={p.className} stroke="currentColor" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round"><path d="M20 6L9 17l-5-5" /></svg>),
}

/** Animated SVG progress ring for the overall 0-100 listing score. */
function ScoreRing({ score }: { score: number }) {
  const r = 32
  const c = 2 * Math.PI * r
  const pct = Math.max(0, Math.min(1, score / 100))
  const stroke = score >= 80 ? '#16a34a' : score >= 60 ? '#d97706' : '#dc2626'
  return (
    <div className="relative flex-shrink-0" style={{ width: 84, height: 84 }}>
      <svg width="84" height="84" viewBox="0 0 84 84" className="-rotate-90">
        <circle cx="42" cy="42" r={r} fill="none" stroke="#e2e8f0" strokeWidth="7" />
        <circle cx="42" cy="42" r={r} fill="none" stroke={stroke} strokeWidth="7" strokeLinecap="round"
          strokeDasharray={c} strokeDashoffset={c * (1 - pct)} style={{ transition: 'stroke-dashoffset 700ms ease' }} />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-2xl font-bold text-slate-900 leading-none tabular-nums">{score}</span>
      </div>
    </div>
  )
}

// ─── Page Component ─────────────────────────────────────────────────────────

export default function ListingDetailPage() {
  const params = useParams()
  const router = useRouter()
  const asin = params.asin as string

  const [score, setScore] = useState<SeoScoreRow | null>(null)
  const [aiRecs, setAiRecs] = useState<AiRecommendations | null>(null)
  const [kwData, setKwData] = useState<KeywordIntelligenceResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [aiLoading, setAiLoading] = useState(false)
  const [aiError, setAiError] = useState<string | null>(null)
  const [aiProgress, setAiProgress] = useState<string>('')
  const [copied, setCopied] = useState<string | null>(null)
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['apply']))
  const [competitorAsin, setCompetitorAsin] = useState<string>('')
  const [competitorSaving, setCompetitorSaving] = useState(false)
  const [orphans, setOrphans] = useState<{ orphanCount: number; children: { sku: string; asin: string; liveParent: string | null; status: string }[] } | null>(null)

  // ── Ship optimized content to Amazon — per section (title / bullets / description / keywords / details) ──
  // 'details' is a single-attribute push (one detail per click): Material, Brand, Fit Type, etc.
  // The seller picks WHICH detail in the UI; the route resolves the friendly name to an SP-API
  // attribute key (see lib/fba/productDetailAttrs.ts).
  type PushField = 'title' | 'bullets' | 'description' | 'keywords' | 'details'
  const FIELD_LABEL: Record<PushField, string> = { title: 'Title', bullets: 'Bullets', description: 'Description', keywords: 'Backend Keywords', details: 'Product Detail' }
  interface PushDiffRow { sku: string; current: string; proposed: string; bytes: number; chars: number; changed: boolean; isParent?: boolean; asin?: string }
  interface PushResultRow { sku: string; status: string; submissionId: string | null; error?: string }
  interface PushPreview {
    field: PushField; label: string; broadcast: boolean; count: number; changed: number;
    proposedValue: string | string[] | null; diff: PushDiffRow[]
    /** Only set when field='details': the friendly attribute name (e.g. "Material") and SP-API key. */
    detail_field?: string; attribute_key?: string
    /** Feature B — for enum attributes: Amazon's accepted vocabulary and the value we
     *  normalized FROM (e.g. "Unisex Adult") when the audit's value wasn't accepted. */
    acceptedValues?: string[] | null; normalizedFrom?: string | null
  }
  const [pushField, setPushField] = useState<PushField>('keywords')
  /** Only set when pushField='details': which detail attribute is being pushed (Material, etc.). */
  const [pushDetailField, setPushDetailField] = useState<string | null>(null)
  const [pushPreview, setPushPreview] = useState<PushPreview | null>(null)
  const [pushLoading, setPushLoading] = useState(false)
  const [pushError, setPushError] = useState<string | null>(null)
  const [pushResults, setPushResults] = useState<{ field?: PushField; pushed: number; failed: number; total: number; message: string; results: PushResultRow[] } | null>(null)
  // ── "Verify on Amazon" — fresh getListingsItem per SKU after a push, so the seller can
  // tell whether Amazon APPLIED the patch (vs just ACCEPTED it). Submissions can sit in
  // Amazon's queue for 15min–6hr; "I pushed an hour ago and nothing changed" needs an answer.
  interface VerifyResultRow { sku: string; asin: string; isParent: boolean; currentLive: string; expected: string; matches: boolean; lastUpdatedDate: string | null }
  interface VerifyPayload { total: number; matched: number; stale: number; results: VerifyResultRow[]; attribute_key?: string }
  const [verifyLoading, setVerifyLoading] = useState(false)
  const [verifyResults, setVerifyResults] = useState<VerifyPayload | null>(null)
  const [verifyError, setVerifyError] = useState<string | null>(null)
  // ── Live push progress — populated from NDJSON stream events. Each entry tracks one SKU's
  // state as the route patches it in the loop. Replaces the old all-or-nothing 'Pushing…'
  // spinner with a per-SKU readout, AND eliminates the proxy-502 failure mode entirely
  // (each progress emit() keeps the connection warm).
  interface PushProgressRow { sku: string; status: 'validating' | 'accepted' | 'failed'; error?: string; submissionId?: string | null }
  const [pushProgress, setPushProgress] = useState<PushProgressRow[]>([])
  const [pushPhase, setPushPhase] = useState<'idle' | 'starting' | 'pushing' | 'rescoring' | 'done'>('idle')
  // ── Family-SKUs view — full set of FBA + FBM twins + variation parent SKU. The DB cache
  // (listing_content) historically deduped some FBA/FBM pairs, so cards that render from
  // it alone hid the FBM twins (the seller saw "3 children" but the push hit 6).
  // /family-skus discovers them live so the displayed list matches the push reality.
  interface FamilySkuRow { sku: string; asin: string; fulfillment: 'FBA' | 'FBM' | 'unknown'; base_name: string }
  interface FamilySkus { parent: { sku: string; asin: string } | null; children: FamilySkuRow[]; count: number }
  const [familySkus, setFamilySkus] = useState<FamilySkus | null>(null)
  const [showPushModal, setShowPushModal] = useState(false)
  const [fetchedImage, setFetchedImage] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<string>('apply')

  const copy = (text: string, label: string) => {
    copyToClipboard(text)
    setCopied(label)
    setTimeout(() => setCopied(null), 2000)
  }

  const toggle = (key: string) => {
    setExpandedSections(prev => {
      const next = new Set(prev)
      next.has(key) ? next.delete(key) : next.add(key)
      return next
    })
  }

  // Fetch score data
  useEffect(() => {
    (async () => {
      try {
        const resp = await fetch('/api/fba/listing-optimizer')
        if (!resp.ok) throw new Error('Failed to load')
        const data = await resp.json()
        const found = data.scores?.find((s: SeoScoreRow) => s.parent_asin === asin)
        if (found) setScore(found)
      } catch { /* ignore */ }
      setLoading(false)
    })()
  }, [asin])

  // If the synced score has no product image, fetch it on-demand from the Amazon catalog.
  useEffect(() => {
    if (!score || score.image_url) return
    const imgAsin = score.top_child_asin || asin
    ;(async () => {
      try {
        const resp = await fetch(`/api/fba/product-image?asin=${imgAsin}`)
        if (resp.ok) { const d = await resp.json(); if (d.image_url) setFetchedImage(d.image_url) }
      } catch { /* ignore — placeholder stays */ }
    })()
  }, [score, asin])

  // Orphan check — flag children whose live Amazon variation link to this parent is broken
  // (was part of the family, got disconnected). Best-effort; never blocks the page.
  const runOrphanCheck = useCallback(async () => {
    if (!asin) return
    try {
      const r = await fetch(`/api/fba/listing-optimizer/orphan-check?parent_asin=${asin}`)
      if (r.ok) { const d = await r.json(); if (d && typeof d.orphanCount === 'number') setOrphans(d) }
    } catch { /* best-effort */ }
  }, [asin])
  useEffect(() => { runOrphanCheck() }, [runOrphanCheck])

  // ─── Related orphans — orphan SKUs (no live parent) that share a SKU prefix with this
  // parent's children. Surfaces "this orphan probably belongs HERE — re-link it" prompts on the
  // CORRECT parent's page, not on the stale one where the orphan currently sits. Includes this
  // parent's seller SKU so the Re-link modal pre-fills the target. Best-effort.
  type RelatedOrphans = { parent_sku: string | null; prefix: string; candidates: { sku: string; asin: string; storedParent: string | null }[] }
  const [relatedOrphans, setRelatedOrphans] = useState<RelatedOrphans | null>(null)
  const runRelatedOrphans = useCallback(async () => {
    if (!asin) return
    try {
      const r = await fetch(`/api/fba/listing-optimizer/related-orphans?parent_asin=${asin}`)
      if (r.ok) { const d = await r.json(); if (d && Array.isArray(d.candidates)) setRelatedOrphans(d) }
    } catch { /* best-effort */ }
  }, [asin])
  useEffect(() => { runRelatedOrphans() }, [runRelatedOrphans])

  // (Re-link submission status state + helper live AFTER openRelink is declared — see below.)

  // ─── Capacity attribute check — detect children whose live capacity disagrees with SKU ──
  type CapRow = {
    sku: string; asin: string; productType: string | null
    attributeName: string | null
    live: { value: number; unit: string } | null
    expected: { value: number; unit: string } | null
    liveLabel: string; expectedLabel: string
    mismatch: boolean
    reason: 'no_attribute' | 'no_sku_capacity' | 'match' | 'mismatch' | 'fetch_failed'
  }
  const [capacityCheck, setCapacityCheck] = useState<{ mismatchCount: number; children: CapRow[] } | null>(null)
  const runCapacityCheck = useCallback(async () => {
    if (!asin) return
    try {
      const r = await fetch(`/api/fba/listing-optimizer/capacity-check?parent_asin=${asin}`)
      if (r.ok) { const d = await r.json(); if (d && typeof d.mismatchCount === 'number') setCapacityCheck(d) }
    } catch { /* best-effort */ }
  }, [asin])
  useEffect(() => { runCapacityCheck() }, [runCapacityCheck])

  // ─── Fix a single child's capacity attribute (preview → confirm → live) ──────────
  type CapIssue = { code?: string; message?: string; severity?: string }
  const [fixCapTarget, setFixCapTarget] = useState<{ row: CapRow } | null>(null)
  const [fixCapLoading, setFixCapLoading] = useState(false)
  const [fixCapError, setFixCapError] = useState<string | null>(null)
  const [fixCapPreview, setFixCapPreview] = useState<{ ok: boolean; attributeName: string; current: { value: number | string | undefined; unit: string | undefined }; proposed: { value: number; unit: string }; issues: CapIssue[] } | null>(null)
  const [fixCapResult, setFixCapResult] = useState<{ submitted: boolean; status: string | null; submissionId: string | null; issues: CapIssue[]; message: string } | null>(null)
  const openFixCap = useCallback((row: CapRow) => {
    setFixCapTarget({ row }); setFixCapPreview(null); setFixCapResult(null); setFixCapError(null)
  }, [])
  const previewFixCap = useCallback(async () => {
    const row = fixCapTarget?.row; if (!row?.expected) return
    setFixCapLoading(true); setFixCapError(null); setFixCapPreview(null); setFixCapResult(null)
    try {
      const r = await fetch(`/api/fba/listing-optimizer/fix-capacity?child_sku=${encodeURIComponent(row.sku)}&value=${row.expected.value}&unit=${encodeURIComponent(row.expected.unit)}`)
      const d = await r.json()
      if (!r.ok) throw new Error(d.error || 'Preview failed')
      setFixCapPreview(d)
    } catch (e) { setFixCapError(e instanceof Error ? e.message : 'Preview failed') }
    setFixCapLoading(false)
  }, [fixCapTarget])
  const confirmFixCap = useCallback(async () => {
    const row = fixCapTarget?.row; if (!row?.expected) return
    setFixCapLoading(true); setFixCapError(null)
    try {
      const r = await fetch('/api/fba/listing-optimizer/fix-capacity', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ child_sku: row.sku, value: row.expected.value, unit: row.expected.unit, confirm: true }),
      })
      const d = await r.json()
      if (!r.ok) throw new Error(d.error || 'Fix failed')
      setFixCapResult(d); setFixCapPreview(null)
    } catch (e) { setFixCapError(e instanceof Error ? e.message : 'Fix failed') }
    setFixCapLoading(false)
  }, [fixCapTarget])

  // ─── Re-link an orphan/re-parented child into a target parent SKU ────────────────
  type RelinkIssue = { code?: string; message?: string; severity?: string }
  const [relinkTarget, setRelinkTarget] = useState<{ childSku: string; childAsin: string } | null>(null)
  const [relinkParentSku, setRelinkParentSku] = useState<string>('')
  const [relinkPreview, setRelinkPreview] = useState<{ ok: boolean; productType: string; variation_theme: string | null; issues: RelinkIssue[] } | null>(null)
  const [relinkLoading, setRelinkLoading] = useState(false)
  const [relinkError, setRelinkError] = useState<string | null>(null)
  const [relinkResult, setRelinkResult] = useState<{ submitted: boolean; status: string | null; submissionId: string | null; issues: RelinkIssue[]; message: string } | null>(null)
  const openRelink = useCallback((childSku: string, childAsin: string, parentSku?: string) => {
    setRelinkTarget({ childSku, childAsin })
    // Pre-fill with the caller's parent SKU when known (Related Orphans on a parent page),
    // otherwise leave blank so the user must enter it (no more hard-coded "Memory-Card-P").
    setRelinkParentSku(parentSku || '')
    setRelinkPreview(null); setRelinkResult(null); setRelinkError(null)
  }, [])

  // ─── Re-link submission status — track pending Amazon submissions so we don't keep offering
  //  the Re-link button on a SKU the seller already pushed. The /relink-status endpoint also
  //  lazily auto-resolves pending rows to applied when Amazon's catalog reports the new parent.
  type RelinkStatusRow = { child_sku: string; target_parent_sku: string; status: 'pending' | 'applied' | 'failed'; submitted_at: string; applied_at: string | null; submission_id: string | null }
  const [relinkStatus, setRelinkStatus] = useState<Record<string, RelinkStatusRow>>({})
  const runRelinkStatus = useCallback(async () => {
    const skus = new Set<string>()
    for (const c of orphans?.children ?? []) skus.add(c.sku)
    for (const c of relatedOrphans?.candidates ?? []) skus.add(c.sku)
    if (skus.size === 0) return
    try {
      const r = await fetch(`/api/fba/listing-optimizer/relink-status?child_skus=${[...skus].join(',')}`)
      if (!r.ok) return
      const d = await r.json() as { statuses?: RelinkStatusRow[] }
      const map: Record<string, RelinkStatusRow> = {}
      for (const s of d.statuses ?? []) map[s.child_sku] = s
      setRelinkStatus(map)
    } catch { /* best-effort */ }
  }, [orphans, relatedOrphans])
  useEffect(() => { runRelinkStatus() }, [runRelinkStatus])
  const renderRelinkAction = useCallback((c: { sku: string; asin: string }, parentSku: string | undefined, style: 'amber' | 'sky' | 'violet') => {
    const status = relinkStatus[c.sku]
    if (status?.status === 'pending') {
      const mins = Math.max(1, Math.floor((Date.now() - new Date(status.submitted_at).getTime()) / 60000))
      return (
        <span className="ml-auto inline-flex items-center gap-2 text-[11px]">
          <span className="px-2 py-0.5 rounded bg-slate-700 text-white font-semibold">Submitted {mins}m ago</span>
          <button onClick={runRelinkStatus} className="text-[11px] underline text-slate-600 hover:text-slate-900 cursor-pointer" title="Re-check Amazon for the applied state">Re-check</button>
        </span>
      )
    }
    if (status?.status === 'applied') {
      return <span className="ml-auto text-[11px] font-semibold text-green-700">✓ Applied on Amazon</span>
    }
    const cls = style === 'amber' ? 'bg-amber-700 hover:bg-amber-800 text-white'
      : style === 'sky' ? 'bg-white border border-sky-300 hover:bg-sky-100 text-sky-700'
      : 'bg-violet-700 hover:bg-violet-800 text-white'
    return (
      <button onClick={() => openRelink(c.sku, c.asin, parentSku)} disabled={style === 'violet' && !parentSku}
        className={`ml-auto inline-flex items-center gap-1 text-[11px] font-semibold ${cls} px-2.5 py-1 rounded-md transition-colors cursor-pointer disabled:opacity-50`}>
        {style === 'sky' ? 'Move to this parent' : style === 'violet' ? 'Re-link to this parent' : 'Re-link'}
      </button>
    )
  }, [relinkStatus, runRelinkStatus, openRelink])
  const previewRelink = useCallback(async () => {
    if (!relinkTarget || !relinkParentSku) return
    setRelinkLoading(true); setRelinkError(null); setRelinkPreview(null); setRelinkResult(null)
    try {
      const r = await fetch(`/api/fba/listing-optimizer/relink?child_sku=${encodeURIComponent(relinkTarget.childSku)}&parent_sku=${encodeURIComponent(relinkParentSku)}`)
      const d = await r.json()
      if (!r.ok) throw new Error(d.error || 'Preview failed')
      setRelinkPreview(d)
    } catch (e) { setRelinkError(e instanceof Error ? e.message : 'Preview failed') }
    setRelinkLoading(false)
  }, [relinkTarget, relinkParentSku])
  const confirmRelink = useCallback(async () => {
    if (!relinkTarget || !relinkParentSku) return
    setRelinkLoading(true); setRelinkError(null)
    try {
      const r = await fetch('/api/fba/listing-optimizer/relink', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ child_sku: relinkTarget.childSku, parent_sku: relinkParentSku, confirm: true }),
      })
      const d = await r.json()
      if (!r.ok) throw new Error(d.error || 'Re-link failed')
      setRelinkResult(d); setRelinkPreview(null)
    } catch (e) { setRelinkError(e instanceof Error ? e.message : 'Re-link failed') }
    setRelinkLoading(false)
  }, [relinkTarget, relinkParentSku])

  // Fetch AI recommendations (cached)
  useEffect(() => {
    if (!asin) return
    ;(async () => {
      try {
        const resp = await fetch(`/api/fba/listing-optimizer/ai-recommendations?parent_asin=${asin}`)
        if (resp.ok) {
          const data = await resp.json()
          if (data.recommendations) setAiRecs(data.recommendations)
        }
      } catch { /* ignore */ }
    })()
  }, [asin])

  // Fetch family SKUs (FBA + FBM twins + variation parent) — used by the TITLES card to show
  // every SKU the push will hit, not just what's in the DB cache. Read-only, best-effort.
  useEffect(() => {
    if (!asin) return
    ;(async () => {
      try {
        const resp = await fetch(`/api/fba/listing-optimizer/family-skus?parent_asin=${asin}`)
        if (resp.ok) {
          const data = await resp.json()
          if (data?.children) setFamilySkus(data)
        }
      } catch { /* ignore — UI falls back to per_child_titles only */ }
    })()
  }, [asin])

  // 👻 GHOST-PARENT HARD REDIRECT (PR #89). B0F8WYNVPJ-class ASINs are "half-ghosts":
  // they still carry stale recommendations + score (so the #83 empty-recs banner never
  // fires) but have ZERO live children on Amazon (the variation family migrated to a twin
  // parent, e.g. B0GCF11RKL). Landing here is a dead end — the seller's repeated "clicking
  // a card doesn't open the parent". The authoritative signal is orphan-check (live catalog
  // relationships). If it returns 0 live children AND twin-parent resolves a DIFFERENT real
  // ASIN, hard-redirect to the live parent. Deterministic, runs on load, independent of
  // stale recs / rollup child_count / async dashboard staleStatus. Loop-guarded.
  const [ghostRedirecting, setGhostRedirecting] = useState(false)
  useEffect(() => {
    if (!asin) return
    let cancelled = false
    ;(async () => {
      try {
        const oc = await fetch(`/api/fba/listing-optimizer/orphan-check?parent_asin=${asin}`)
        if (!oc.ok || cancelled) return
        const ocd = await oc.json()
        const liveChildren = (ocd.children ?? []).filter((c: { status?: string }) => c.status !== 'orphan')
        if (liveChildren.length > 0) return // genuine parent — never redirect
        // 0 live children → ghost. Resolve the live twin and route there.
        const tr = await fetch(`/api/fba/listing-optimizer/twin-parent?parent_asin=${asin}`)
        if (!tr.ok || cancelled) return
        const td = await tr.json()
        if (td.twinParent && td.twinParent !== asin) {
          setGhostRedirecting(true)
          router.replace(`/fba/listing/${td.twinParent}`)
        }
      } catch { /* best-effort — page renders normally if checks fail */ }
    })()
    return () => { cancelled = true }
  }, [asin, router])

  // Fetch keyword intelligence
  useEffect(() => {
    if (!score?.top_child_asin) return
    ;(async () => {
      try {
        const resp = await fetch(`/api/fba/intelligence/${score.top_child_asin}?stored=true`)
        if (resp.ok) {
          const data = await resp.json()
          if (data.totalKeywordsAnalyzed > 0) setKwData(data)
        }
      } catch { /* ignore */ }
    })()
  }, [score?.top_child_asin])

  // Fetch competitor ASIN
  useEffect(() => {
    if (!asin) return
    ;(async () => {
      try {
        const resp = await fetch(`/api/fba/competitor-asin?parentAsin=${asin}`)
        if (resp.ok) {
          const data = await resp.json()
          if (data.competitorAsin) setCompetitorAsin(data.competitorAsin)
        }
      } catch { /* ignore */ }
    })()
  }, [asin])

  const saveCompetitorAsin = async () => {
    if (!competitorAsin || !/^[A-Z0-9]{10}$/.test(competitorAsin.toUpperCase())) return
    setCompetitorSaving(true)
    try {
      await fetch('/api/fba/competitor-asin', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ parentAsin: asin, competitorAsin: competitorAsin.toUpperCase() }),
      })
    } catch { /* ignore */ }
    setCompetitorSaving(false)
  }

  // Generate AI recommendations (streaming)
  const generateAiRecs = useCallback(async () => {
    setAiLoading(true)
    setAiError(null)
    setAiProgress('Starting AI audit...')
    try {
      const resp = await fetch('/api/fba/listing-optimizer/ai-recommendations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ parent_asin: asin }),
      })
      if (!resp.ok) throw new Error('AI generation failed')

      // Handle NDJSON streaming response
      const reader = resp.body?.getReader()
      if (!reader) throw new Error('No response stream')

      const decoder = new TextDecoder()
      let buffer = ''
      let finalResult: { recommendations?: AiRecommendations } | null = null

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || '' // Keep incomplete line in buffer

        for (const line of lines) {
          if (!line.trim()) continue
          try {
            const msg = JSON.parse(line)
            if (msg.type === 'progress') {
              setAiProgress(msg.message || 'Processing...')
            } else if (msg.type === 'result') {
              finalResult = msg
            } else if (msg.type === 'error') {
              throw new Error(msg.error || 'AI generation failed')
            }
          } catch (parseErr) {
            // Skip malformed lines
            if (parseErr instanceof Error && parseErr.message !== 'AI generation failed' && !parseErr.message.includes('AI returned')) continue
            throw parseErr
          }
        }
      }

      // Process any remaining buffer
      if (buffer.trim()) {
        try {
          const msg = JSON.parse(buffer)
          if (msg.type === 'result') finalResult = msg
          else if (msg.type === 'error') throw new Error(msg.error)
        } catch { /* ignore */ }
      }

      if (finalResult?.recommendations) {
        setAiRecs(finalResult.recommendations)
      } else {
        throw new Error('No recommendations received')
      }
    } catch (e: unknown) {
      setAiError(e instanceof Error ? e.message : 'Failed')
    }
    setAiLoading(false)
    setAiProgress('')
  }, [asin])

  /** Read a JSON response defensively. Coolify/nginx returns plain-text '502 Bad Gateway'
   *  HTML when the upstream Next process times out, restarts, or OOMs mid-request — calling
   *  `await resp.json()` on that throws the classic 'Unexpected token "B" is not valid JSON'.
   *  This wraps the read so we surface a clean, actionable message instead. */
  const readJsonOrThrowGateway = async (resp: Response, when: 'preview' | 'push'): Promise<unknown> => {
    const ct = resp.headers.get('content-type') || ''
    const text = await resp.text()
    if (!ct.includes('application/json') || /^\s*<|^\s*5\d{2}\s+|Bad Gateway|Gateway Time-?out/i.test(text)) {
      // It's HTML / plain text — a gateway / proxy error from Coolify, NOT a JSON response.
      const code = resp.status || 502
      const hint = when === 'push'
        ? `Server returned ${code} mid-push. The submission likely still reached Amazon — use Verify on Amazon to confirm before retrying (a retry could submit a duplicate patch).`
        : `Server returned ${code}. The app may be restarting (Coolify deploy?) — retry in ~30 seconds.`
      throw new Error(hint)
    }
    try { return JSON.parse(text) }
    catch {
      throw new Error(`Server returned malformed JSON (${resp.status}). ${when === 'push' ? 'Check Verify on Amazon before retrying.' : 'Retry shortly.'}`)
    }
  }

  // ─── Ship a content section to Amazon (preview → confirm) ─────────────────
  // detailField is only used for field='details' (one detail per click).
  const openPushPreview = useCallback(async (field: PushField, detailField?: string) => {
    setPushField(field)
    setPushDetailField(field === 'details' ? (detailField ?? null) : null)
    setPushError(null); setPushResults(null); setPushPreview(null); setVerifyResults(null); setVerifyError(null); setPushProgress([]); setPushPhase('idle'); setShowPushModal(true); setPushLoading(true)
    try {
      const qs = field === 'details' && detailField
        ? `&detail_field=${encodeURIComponent(detailField)}`
        : ''
      const resp = await fetch(`/api/fba/listing-optimizer/push-content?parent_asin=${asin}&field=${field}${qs}`)
      const data = await readJsonOrThrowGateway(resp, 'preview') as { error?: string }
      if (!resp.ok) throw new Error(data.error || 'Preview failed')
      setPushPreview(data as PushPreview)
    } catch (e) {
      setPushError(e instanceof Error ? e.message : 'Preview failed')
    }
    setPushLoading(false)
  }, [asin])

  /**
   * Stream-consume the NDJSON push response. Each emit() from the server arrives as a
   * newline-delimited JSON line; we accumulate the read buffer until a newline, parse,
   * dispatch, then continue. The per-SKU 'progress' events keep the proxy connection
   * warm so the push survives container restarts and nginx idle-timeouts (the original
   * 502 Bad Gateway failure mode).
   */
  const confirmPush = useCallback(async () => {
    setPushError(null); setPushLoading(true); setPushPhase('starting')
    setPushProgress([])
    let finalResult: { pushed: number; failed: number; total: number; message: string; results: PushResultRow[]; field?: PushField } | null = null
    let streamError: string | null = null
    try {
      const body: Record<string, unknown> = { parent_asin: asin, field: pushField, confirm: true }
      if (pushField === 'details' && pushDetailField) body.detail_field = pushDetailField
      const resp = await fetch('/api/fba/listing-optimizer/push-content', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })

      // Non-OK BEFORE the stream opens means a body-validation error (we still send JSON
      // for those — see route.ts). Read it the defensive way to also catch 502 HTML in
      // the rare case the proxy throws before the stream gets going.
      if (!resp.ok) {
        const data = await readJsonOrThrowGateway(resp, 'push') as { error?: string }
        throw new Error(data.error || `Push failed (HTTP ${resp.status})`)
      }
      if (!resp.body) throw new Error('Server returned no body — connection dropped before stream.')

      const reader = resp.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      const handleLine = (line: string) => {
        if (!line.trim()) return
        let msg: { type?: string; sku?: string; status?: string; error?: string; submissionId?: string | null; pushed?: number; failed?: number; total?: number; message?: string; results?: PushResultRow[]; field?: PushField }
        try { msg = JSON.parse(line) } catch { return }
        if (msg.type === 'started') {
          setPushPhase('pushing')
        } else if (msg.type === 'progress' && msg.sku && msg.status) {
          // Upsert per-SKU progress row. Validating overwritten by accepted/failed as the
          // SKU moves through the loop.
          setPushProgress((prev) => {
            const idx = prev.findIndex((p) => p.sku === msg.sku)
            const row: PushProgressRow = { sku: msg.sku!, status: msg.status as PushProgressRow['status'], error: msg.error, submissionId: msg.submissionId }
            if (idx === -1) return [...prev, row]
            const next = prev.slice(); next[idx] = row; return next
          })
        } else if (msg.type === 'rescore') {
          setPushPhase('rescoring')
        } else if (msg.type === 'result') {
          finalResult = {
            field: msg.field,
            pushed: msg.pushed ?? 0,
            failed: msg.failed ?? 0,
            total: msg.total ?? 0,
            message: msg.message ?? 'Push completed.',
            results: msg.results ?? [],
          }
        } else if (msg.type === 'error') {
          streamError = msg.error || 'Push failed mid-stream.'
        }
      }
      // Read pump: decode, split on newlines, dispatch each complete line, keep the
      // partial tail in the buffer for the next read.
      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''
        for (const line of lines) handleLine(line)
      }
      // Flush whatever's left (server should always terminate with a newline, but defend).
      if (buffer.trim()) handleLine(buffer)

      if (streamError) throw new Error(streamError)
      if (!finalResult) throw new Error('Stream ended without a result event.')
      // TS loses narrowing across the closure-mutated `finalResult` — re-assert the shape.
      const data: { pushed: number; failed: number; total: number; message: string; results: PushResultRow[]; field?: PushField } = finalResult

      setPushResults(data)
      setPushPreview(null)
      setPushPhase('done')

      // Refresh score (push re-scored server-side).
      if (data.pushed > 0) {
        try {
          // no-store: a push just mutated listing_content; the refetch MUST be fresh so the
          // cohesion rows (which compare per-child live backend vs recommendation) flip green.
          const sresp = await fetch('/api/fba/listing-optimizer', { cache: 'no-store' })
          const sdata = await sresp.json()
          const found = sdata.scores?.find((s: SeoScoreRow) => s.parent_asin === asin)
          if (found) setScore(found)
        } catch { /* best-effort — the score still updates on next load */ }

        // Mark matching action_plan items DONE locally so the card collapses to the
        // ✓ Pushed state immediately (audit JSON is frozen until Regenerate).
        const matchesPushedField = (elem: string): boolean => {
          if (pushField === 'title') return elem === 'title'
          if (pushField === 'description') return elem === 'description'
          if (pushField === 'keywords') return elem === 'backend_keywords'
          if (pushField === 'details') return elem === 'product_details'
          if (pushField === 'bullets') return /^bullet/.test(elem)
          return false
        }
        const pushedAt = new Date().toISOString()
        const pushedLabel = pushField === 'details' && pushDetailField ? pushDetailField : FIELD_LABEL[pushField]
        setAiRecs((prev) => {
          if (!prev) return prev
          const action_plan = (prev.action_plan ?? []).map((it) => {
            if (!matchesPushedField(it.element)) return it
            return {
              ...it,
              verdict: 'DONE' as const,
              current_status: `✓ Pushed ${pushedLabel} to Amazon (${data.pushed}/${data.total} variants)`,
              notes: `${it.notes ? it.notes + ' · ' : ''}Pushed at ${pushedAt}. Submissions are ACCEPTED — Amazon applies in 15min–6hr. Use Verify on Amazon to confirm.`,
            }
          })
          return { ...prev, action_plan }
        })
      }
    } catch (e) {
      // If we have any progress rows, the seller knows what landed (the stream told them
      // per-SKU). We do NOT clear them on error — they're the rollback evidence.
      setPushError(e instanceof Error ? e.message : 'Push failed')
      setPushPhase('idle')
    }
    setPushLoading(false)
  }, [asin, pushField, pushDetailField])

  /** Verify what Amazon ACTUALLY has live right now for the just-pushed field.
   *  Useful when a push was Accepted but the seller doesn't see it on Seller Central
   *  or the PDP — answers "is Amazon still processing, or did the push silently fail?". */
  const runVerify = useCallback(async () => {
    setVerifyError(null); setVerifyResults(null); setVerifyLoading(true)
    try {
      const qs = pushField === 'details' && pushDetailField
        ? `&detail_field=${encodeURIComponent(pushDetailField)}`
        : ''
      const resp = await fetch(`/api/fba/listing-optimizer/verify-push?parent_asin=${asin}&field=${pushField}${qs}`)
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.error || 'Verify failed')
      setVerifyResults(data)
    } catch (e) {
      setVerifyError(e instanceof Error ? e.message : 'Verify failed')
    }
    setVerifyLoading(false)
  }, [asin, pushField, pushDetailField])

  // ─── Grouped Reconciliation Logic ─────────────────────────────────────────

  const placementGroups = (() => {
    if (!aiRecs?.keyword_reconciliation?.length) return null
    const groups: Record<string, { text: string; keywords: { keyword: string; action_type: string; search_volume: number; why: string }[] }> = {}
    for (const kr of aiRecs.keyword_reconciliation) {
      const key = [...kr.placed_in].sort().join(' + ')
      if (!groups[key]) groups[key] = { text: kr.exact_text, keywords: [] }
      groups[key].keywords.push({ keyword: kr.keyword, action_type: kr.action_type, search_volume: kr.search_volume, why: kr.why })
    }
    const sortedKeys = Object.keys(groups).sort((a, b) => {
      if (a.includes('title') && !b.includes('title')) return -1
      if (!a.includes('title') && b.includes('title')) return 1
      if (a.includes('bullet') && !b.includes('bullet')) return -1
      if (!a.includes('bullet') && b.includes('bullet')) return 1
      return a.localeCompare(b)
    })
    return { groups, sortedKeys, total: aiRecs.keyword_reconciliation.length }
  })()

  // ─── Loading / Not Found ──────────────────────────────────────────────────

  // Ghost-parent redirect in flight (PR #89) — show a clear message instead of flashing
  // the dead listing's stale content before router.replace swaps the page.
  if (ghostRedirecting) return (
    <div className="flex items-center justify-center min-h-[60vh]">
      <div className="text-center">
        <div className="animate-spin w-8 h-8 border-2 border-amber-500 border-t-transparent rounded-full mx-auto mb-3" />
        <p className="text-sm text-slate-600">This ASIN&apos;s variation family moved — opening the live parent…</p>
      </div>
    </div>
  )

  if (loading) return (
    <div className="flex items-center justify-center min-h-[60vh]">
      <div className="text-center">
        <div className="animate-spin w-8 h-8 border-2 border-violet-500 border-t-transparent rounded-full mx-auto mb-3" />
        <p className="text-sm text-slate-500">Loading listing data...</p>
      </div>
    </div>
  )

  if (!score) return (
    <div className="max-w-4xl mx-auto p-8 text-center">
      <p className="text-lg text-slate-600 mb-4">Listing not found: {asin}</p>
      <button onClick={() => router.push('/fba')} className="text-violet-600 hover:underline text-sm">
        &larr; Back to Listing Optimizer
      </button>
    </div>
  )

  const displayImage = score.image_url || fetchedImage
  // Capacity-family parents (e.g. SD cards by GB): the stored product_title is the best-seller
  // CHILD's title, so it carries that child's capacity ("...64GB..."). The parent / variation-hub
  // header must be capacity-AGNOSTIC — strip the GB token, mirroring the TITLES card's PARENT row.
  // Gated on per_child_titles.length > 1 (only built for non-apparel capacity families), so apparel
  // and single-capacity products are untouched.
  const isCapacityFamily = Array.isArray(aiRecs?.per_child_titles) && (aiRecs?.per_child_titles?.length ?? 0) > 1
  const headerTitle = isCapacityFamily
    ? stripCapacityToken(stripVariantSuffix(score.product_title))
    : stripVariantSuffix(score.product_title)
  // Tab definitions for the dashboard-style section nav (one section visible at a time).
  const TABS = [
    { id: 'apply', label: 'Apply Changes', count: (aiRecs?.action_plan ?? []).filter(a => a.verdict !== 'DONE' && a.verdict !== 'SKIP').length },
    { id: 'placement', label: 'Keyword Plan', count: aiRecs?.keyword_reconciliation?.length ?? 0 },
    { id: 'issues', label: 'Diagnostics', count: score.issues.length },
    { id: 'variants', label: 'Variants', count: dedupByAsin(score.children).length },
    ...(kwData && kwData.topOpportunities.length > 0 ? [{ id: 'kwintel', label: 'Intelligence', count: kwData.totalKeywordsAnalyzed }] : []),
  ]
  // Each card shows its IMPORTANCE-WEIGHTED points (max = the section's weight). The six maxes
  // sum to 100, so the cards add up to the listing score — and a perfect listing reads 100, not
  // 150. The % (points/weight) still reflects how good the section is. Weights: scoreWeights.ts.
  const bars = [
    { label: 'Title', score: weightedPoints(score.title_score, SECTION_WEIGHTS.title), max: SECTION_WEIGHTS.title },
    { label: 'Bullets', score: weightedPoints(score.bullet_score, SECTION_WEIGHTS.bullets), max: SECTION_WEIGHTS.bullets },
    { label: 'Keywords', score: weightedPoints(score.keyword_score, SECTION_WEIGHTS.keyword), max: SECTION_WEIGHTS.keyword },
    // Description + Features are NULL on older rows until re-scored — show the cards only once
    // populated so we never render a phantom 0%.
    ...(score.description_score != null ? [{ label: 'Description', score: weightedPoints(score.description_score, SECTION_WEIGHTS.description), max: SECTION_WEIGHTS.description }] : []),
    ...(score.features_score != null ? [{ label: 'Features', score: weightedPoints(score.features_score, SECTION_WEIGHTS.features), max: SECTION_WEIGHTS.features }] : []),
    { label: 'A+', score: weightedPoints(score.aplus_score, SECTION_WEIGHTS.aplus), max: SECTION_WEIGHTS.aplus },
  ]

  // ─── Render ───────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-gradient-to-b from-violet-100 via-slate-50 to-slate-50">
    <div className="max-w-6xl mx-auto px-4 py-6 space-y-5">

      {/* ── Back link ── */}
      <button onClick={() => router.push('/fba')} className="inline-flex items-center gap-1.5 text-sm font-medium text-slate-500 hover:text-slate-800 transition-colors cursor-pointer">
        <Icon.ArrowLeft className="w-4 h-4" /> Back to Listing Optimizer
      </button>

      {/* ══════════════════════════════════════════════════════════════════════
          HEADER — Product info + Score
          ══════════════════════════════════════════════════════════════════════ */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5 sm:p-6">
        <div className="flex flex-col sm:flex-row gap-5">
          {/* Image */}
          <div className="flex-shrink-0 w-20 h-20 bg-slate-100 rounded-2xl overflow-hidden ring-1 ring-slate-200 flex items-center justify-center">
            {displayImage ? (
              <img src={displayImage} alt="" className="w-full h-full object-cover" />
            ) : (
              <svg viewBox="0 0 24 24" className="w-7 h-7 text-slate-300" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="8.5" cy="8.5" r="1.5" /><path d="M21 15l-5-5L5 21" /></svg>
            )}
          </div>

          {/* Title + meta chips */}
          <div className="flex-1 min-w-0">
            <h1 className="text-lg font-semibold text-slate-900 leading-snug line-clamp-2">{headerTitle || asin}</h1>
            <div className="flex flex-wrap items-center gap-1.5 mt-2">
              <span className="font-mono text-[11px] text-slate-600 bg-slate-100 rounded-md px-2 py-0.5">{asin}</span>
              <span className="text-[11px] font-medium text-slate-600 bg-slate-100 rounded-md px-2 py-0.5">{score.child_count} variant{score.child_count !== 1 ? 's' : ''}</span>
              <span className="text-[11px] font-medium text-slate-600 bg-slate-100 rounded-md px-2 py-0.5">{score.total_units_30d.toLocaleString()} units / 30d</span>
            </div>
          </div>

          {/* Overall score ring */}
          <div className="flex-shrink-0 flex flex-row sm:flex-col items-center justify-center gap-2 sm:gap-1">
            <ScoreRing score={score.overall_score} />
            <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wide">Listing score</span>
          </div>
        </div>

        {/* Primary actions */}
        <div className="flex flex-wrap items-center gap-2 mt-5 pt-4 border-t border-slate-100">
          <a href={`https://sellercentral.amazon.com/abis/listing/edit?asin=${asin}&ref_=xx_addlisting_dnav_xx`}
            target="_blank" rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-200 hover:bg-slate-50 hover:border-slate-300 rounded-lg px-3 py-2 transition-colors cursor-pointer">
            <Icon.External className="w-3.5 h-3.5" /> Edit in Seller Central
          </a>
          <a href={`https://sellercentral.amazon.com/enhanced-content/edit?asin=${asin}`}
            target="_blank" rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-200 hover:bg-slate-50 hover:border-slate-300 rounded-lg px-3 py-2 transition-colors cursor-pointer">
            <Icon.External className="w-3.5 h-3.5" /> Edit A+ Content
          </a>
          <button
            onClick={generateAiRecs}
            disabled={aiLoading}
            className="ml-auto inline-flex items-center gap-1.5 text-xs font-semibold text-white bg-violet-600 hover:bg-violet-700 rounded-lg px-4 py-2 disabled:opacity-50 transition-colors cursor-pointer shadow-sm shadow-violet-200">
            <Icon.Sparkles className="w-3.5 h-3.5" /> {aiLoading ? 'Generating…' : aiRecs ? 'Regenerate AI Audit' : 'Run AI Audit'}
          </button>
        </div>
        {aiError && <p className="text-xs text-red-600 mt-2">{aiError}</p>}

        {/* Competitor ASIN input for reverse keyword lookup */}
        <div className="mt-4 pt-4 border-t border-slate-100 flex flex-wrap items-center gap-2">
          <label className="text-xs font-medium text-slate-500 whitespace-nowrap">Competitor ASIN</label>
          <input
            type="text"
            value={competitorAsin}
            onChange={(e) => setCompetitorAsin(e.target.value.toUpperCase())}
            placeholder="B0XXXXXXXXX"
            className="text-xs border border-slate-200 rounded-lg px-2.5 py-1.5 w-36 font-mono uppercase focus:outline-none focus:ring-2 focus:ring-violet-200 focus:border-violet-400 transition-shadow"
            maxLength={10}
          />
          <button
            onClick={saveCompetitorAsin}
            disabled={competitorSaving || !competitorAsin || competitorAsin.length !== 10}
            className="text-xs font-medium bg-slate-100 hover:bg-slate-200 text-slate-700 px-3 py-1.5 rounded-lg disabled:opacity-50 transition-colors cursor-pointer">
            {competitorSaving ? 'Saving…' : 'Save'}
          </button>
          <span className="text-[11px] text-slate-400">Used for Jungle Scout lookup when your ASIN has no data</span>
        </div>
      </div>

      {/* ══ ORPHAN / REPARENTED — split by severity ══
          Orphan: truly disconnected (no parent on Amazon) → Re-link button (writes via SP-API).
          Re-parented: child IS linked, just to a DIFFERENT parent than we have stored. Offering
          a Re-link here would MOVE the child away from its real family, which is wrong. Show
          as "your data is stale" with a low-key "Move to this parent" only if you really meant it. */}
      {orphans && orphans.children.some((c) => c.status === 'orphan') && (() => { const trueOrphans = orphans.children.filter((c) => c.status === 'orphan'); return (
        <div className="bg-amber-50 border border-amber-300 rounded-2xl shadow-sm p-4 flex items-start gap-3">
          <svg viewBox="0 0 24 24" className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" /><path d="M12 9v4M12 17h.01" /></svg>
          <div className="min-w-0 flex-1">
            <p className="text-sm font-semibold text-amber-900">{trueOrphans.length} variant{trueOrphans.length === 1 ? '' : 's'} disconnected from this parent</p>
            <p className="text-xs text-amber-800 mt-0.5">Stored under this parent, but Amazon has no parent link for them — pooled reviews &amp; ranking are lost. Click <b>Re-link</b> to write the variation relationship back to Amazon.</p>
            <ul className="mt-2 space-y-1">
              {trueOrphans.map((c) => (
                <li key={c.asin} className="text-xs text-amber-900 flex items-center gap-2 flex-wrap">
                  <span className="font-mono">{c.asin}</span>
                  <span className="text-amber-700">({c.sku})</span>
                  <span className="text-amber-900">— no parent link on Amazon</span>
                  {renderRelinkAction(c, undefined, 'amber')}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )})()}

      {/* Re-parented — if MOST/ALL children live under the same OTHER parent on Amazon, we have
          the WRONG parent open. Show a clear "you opened the wrong page" prompt that links to the
          right one instead of a vague "Move to this parent" suggestion that's almost always wrong. */}
      {orphans && (() => {
        const reparented = orphans.children.filter((c) => c.status === 'reparented')
        if (reparented.length === 0) return null
        // If a single live parent dominates (≥1 child and ≥half of all detected reparented rows),
        // assume our portal opened the wrong family — point the seller there.
        const counts = new Map<string, number>()
        for (const c of reparented) if (c.liveParent) counts.set(c.liveParent, (counts.get(c.liveParent) ?? 0) + 1)
        const dominantEntry = [...counts.entries()].sort((a, b) => b[1] - a[1])[0]
        const dominantParent = dominantEntry?.[0]
        const dominantCount = dominantEntry?.[1] ?? 0
        const looksLikeWrongParent = !!dominantParent && dominantCount >= Math.max(1, Math.ceil(reparented.length / 2))
        return (
          <div className="bg-sky-50 border border-sky-200 rounded-2xl shadow-sm p-4 flex items-start gap-3">
            <svg viewBox="0 0 24 24" className="w-5 h-5 text-sky-600 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><line x1="12" y1="16" x2="12" y2="12" /><line x1="12" y1="8" x2="12.01" y2="8" /></svg>
            <div className="min-w-0 flex-1">
              {looksLikeWrongParent ? (
                <>
                  <p className="text-sm font-semibold text-sky-900">You may have the wrong parent open</p>
                  <p className="text-xs text-sky-800 mt-0.5">
                    On Amazon, these variants live under <span className="font-mono font-semibold">{dominantParent}</span> — that&apos;s likely the real current family.
                    This page (<span className="font-mono">{asin}</span>) is probably a stale grouping in our records.
                  </p>
                  <div className="mt-2 flex items-center gap-2 flex-wrap">
                    <button onClick={() => router.push(`/fba/listing/${dominantParent}`)} className="inline-flex items-center gap-1.5 text-xs font-semibold bg-sky-600 hover:bg-sky-700 text-white px-3 py-1.5 rounded-lg transition-colors cursor-pointer">
                      Open {dominantParent}
                    </button>
                    <span className="text-[11px] text-sky-700">Then re-run <b>Sync Now</b> to refresh our records.</span>
                  </div>
                  <details className="mt-2">
                    <summary className="text-[11px] text-sky-700 cursor-pointer hover:underline">show details</summary>
                    <ul className="mt-1 space-y-0.5">
                      {reparented.map((c) => (
                        <li key={c.sku} className="text-[11px] text-sky-900">
                          <span className="font-mono">{c.asin}</span> <span className="text-sky-700">({c.sku})</span> → <span className="font-mono">{c.liveParent}</span>
                        </li>
                      ))}
                    </ul>
                  </details>
                </>
              ) : (
                <>
                  <p className="text-sm font-semibold text-sky-900">{reparented.length} variant{reparented.length === 1 ? '' : 's'} live under a different parent on Amazon</p>
                  <p className="text-xs text-sky-800 mt-0.5">Re-run <b>Sync Now</b> to refresh our records. Only <b>Move to this parent</b> if you intend to move the child here.</p>
                  <ul className="mt-2 space-y-1">
                    {reparented.map((c) => (
                      <li key={c.sku} className="text-xs text-sky-900 flex items-center gap-2 flex-wrap">
                        <span className="font-mono">{c.asin}</span>
                        <span className="text-sky-700">({c.sku})</span>
                        <span className="text-sky-900">— linked to <span className="font-mono">{c.liveParent}</span></span>
                        {renderRelinkAction(c, undefined, 'sky')}
                      </li>
                    ))}
                  </ul>
                </>
              )}
            </div>
          </div>
        )
      })()}

      {/* ══ RELATED ORPHANS — orphan SKUs that share this family's SKU prefix and could be linked HERE ══ */}
      {relatedOrphans && relatedOrphans.candidates.length > 0 && (
        <div className="bg-violet-50 border border-violet-200 rounded-2xl shadow-sm p-4 flex items-start gap-3">
          <svg viewBox="0 0 24 24" className="w-5 h-5 text-violet-600 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /><path d="M12 7v6l3 2" /></svg>
          <div className="min-w-0 flex-1">
            <p className="text-sm font-semibold text-violet-900">{relatedOrphans.candidates.length} orphan{relatedOrphans.candidates.length === 1 ? '' : 's'} could be linked to this family</p>
            <p className="text-xs text-violet-800 mt-0.5">
              These SKUs share the <span className="font-mono">{relatedOrphans.prefix}</span> prefix and have <b>no parent on Amazon</b>. Re-link them here {relatedOrphans.parent_sku ? <>(target parent SKU: <span className="font-mono">{relatedOrphans.parent_sku}</span>) </> : null}to restore pooled reviews &amp; ranking.
            </p>
            <ul className="mt-2 space-y-1">
              {relatedOrphans.candidates.map((c) => (
                <li key={c.sku} className="text-xs text-violet-900 flex items-center gap-2 flex-wrap">
                  <span className="font-mono">{c.asin}</span>
                  <span className="text-violet-700">({c.sku})</span>
                  <span className="text-violet-900">— no parent link on Amazon</span>
                  {renderRelinkAction(c, relatedOrphans.parent_sku ?? undefined, 'violet')}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* ══ WRONG CAPACITY ATTRIBUTE — a child's live capacity disagrees with its SKU ══ */}
      {capacityCheck && capacityCheck.mismatchCount > 0 && (
        <div className="bg-amber-50 border border-amber-300 rounded-2xl shadow-sm p-4 flex items-start gap-3">
          <svg viewBox="0 0 24 24" className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" /><path d="M12 9v4M12 17h.01" /></svg>
          <div className="min-w-0 flex-1">
            <p className="text-sm font-semibold text-amber-900">{capacityCheck.mismatchCount} variant{capacityCheck.mismatchCount === 1 ? '' : 's'} with wrong capacity attribute on Amazon</p>
            <p className="text-xs text-amber-800 mt-0.5">A child SKU encodes one capacity but Amazon stores another (e.g. a 32GB SKU listed as 128GB). Click <b>Fix</b> to write the correct value via SP-API — Amazon validates first, then accepts. <span className="font-medium">Heads-up:</span> changing a variation axis can re-validate the family.</p>
            <ul className="mt-2 space-y-1">
              {capacityCheck.children.filter((c) => c.mismatch).map((c) => (
                <li key={c.sku} className="text-xs text-amber-900 flex items-center gap-2 flex-wrap">
                  <span className="font-mono">{c.asin}</span>
                  <span className="text-amber-700">({c.sku})</span>
                  <span className="text-amber-900">— live <span className="font-mono font-semibold">{c.liveLabel}</span> → should be <span className="font-mono font-semibold">{c.expectedLabel}</span></span>
                  <button onClick={() => openFixCap(c)} className="ml-auto inline-flex items-center gap-1 text-[11px] font-semibold bg-amber-700 hover:bg-amber-800 text-white px-2.5 py-1 rounded-md transition-colors cursor-pointer">
                    Fix
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* ══ KPI ROW — the six sub-scores as their own cards ══ */}
      <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
        {bars.map(b => {
          const pct = Math.round((b.score / b.max) * 100)
          const tone = b.score / b.max >= 0.8 ? 'text-green-600' : b.score / b.max >= 0.6 ? 'text-amber-600' : 'text-red-600'
          return (
            <div key={b.label} className="bg-white rounded-2xl border border-slate-200 shadow-sm p-4 transition-shadow hover:shadow-md">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-500 uppercase tracking-wide">{b.label}</span>
                <span className={`text-[11px] font-bold ${tone}`}>{pct}%</span>
              </div>
              <div className="mt-2 flex items-baseline gap-1">
                <span className="text-2xl font-bold text-slate-900 tabular-nums">{b.score}</span>
                <span className="text-xs text-slate-400">/ {b.max}</span>
              </div>
              <div className="mt-2.5 h-1.5 bg-slate-100 rounded-full overflow-hidden">
                <div className={`h-full rounded-full ${barColor(b.score, b.max)}`} style={{ width: `${pct}%`, transition: 'width 600ms ease' }} />
              </div>
            </div>
          )
        })}
      </div>

      {/* ══ TAB NAV — defined dashboard sections, one at a time (no infinite scroll) ══ */}
      {aiRecs && (
        <div className="flex items-center gap-1 overflow-x-auto bg-white rounded-2xl border border-slate-200 shadow-sm p-1.5">
          {TABS.map(t => (
            <button key={t.id} onClick={() => setActiveTab(t.id)}
              className={`relative inline-flex items-center gap-1.5 px-4 py-2 rounded-xl text-sm font-medium whitespace-nowrap transition-colors cursor-pointer ${activeTab === t.id ? 'bg-violet-600 text-white shadow-sm' : 'text-slate-500 hover:text-slate-800 hover:bg-slate-50'}`}>
              {t.label}
              {t.count > 0 && <span className={`text-[10px] px-1.5 py-0.5 rounded-full ${activeTab === t.id ? 'bg-white/20 text-white' : 'bg-slate-100 text-slate-500'}`}>{t.count}</span>}
            </button>
          ))}
        </div>
      )}

      {/* Tab content — only the active section renders, so inactive ones add no gap */}
      <div>
      {/* No audit yet — single prompt (tab bar is hidden until an audit exists) */}
      {!aiRecs && (
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-10 text-center">
          <p className="text-sm text-slate-500 mb-4">{aiLoading ? (aiProgress || 'Running AI audit…') : 'Run an AI audit to see the recommended changes for this listing.'}</p>
          {!aiLoading && (
            <button onClick={generateAiRecs} className="inline-flex items-center gap-1.5 text-xs font-semibold text-white bg-violet-600 hover:bg-violet-700 rounded-lg px-4 py-2 transition-colors cursor-pointer">
              <Icon.Sparkles className="w-3.5 h-3.5" /> Run AI Audit
            </button>
          )}
        </div>
      )}

      {/* ══════════════════════════════════════════════════════════════════════
          ACTION PLAN — Comprehensive listing review with verdicts
          ══════════════════════════════════════════════════════════════════════ */}
      {aiRecs?.action_plan && aiRecs.action_plan.length > 0 && (() => {
        // Calm, SaaS-style cards: white with a colored LEFT accent per verdict (no full-bleed fills).
        const verdictStyles: Record<string, string> = {
          REPLACE: 'border-l-red-400',
          EDIT: 'border-l-amber-400',
          CREATE: 'border-l-blue-400',
          DONE: 'border-l-green-400',
          SKIP: 'border-l-slate-300',
        }
        const verdictDot: Record<string, string> = {
          REPLACE: 'bg-red-500', EDIT: 'bg-amber-500', CREATE: 'bg-blue-500', DONE: 'bg-green-500', SKIP: 'bg-slate-400',
        }
        const priorityBadge: Record<string, string> = {
          HIGH: 'bg-red-100 text-red-700',
          MEDIUM: 'bg-amber-100 text-amber-700',
          LOW: 'bg-slate-100 text-slate-600',
          NONE: 'bg-green-100 text-green-700',
        }
        const recs = aiRecs!
        // One row per ASIN (FBA+FBM SKUs of the same ASIN collapse to one) so the page matches
        // the per-ASIN recommendations/push.
        const variants = dedupByAsin(score.children)
        const parentItems = (recs.action_plan ?? []).filter(a => a.element !== 'backend_keywords')
        const backendItem = (recs.action_plan ?? []).find(a => a.element === 'backend_keywords')
        const recMap = new Map((recs.per_child_keywords ?? []).map(p => [p.sku, p.keywords]))
        const perChildRows = variants.map(c => {
          const recommended = (recMap.get(c.sku) ?? '').trim()
          const current = (c.backend_keywords ?? '').trim()
          return { sku: c.sku, current, recommended, changed: recommended !== '' && recommended !== current }
        })
        const needsUpdate = perChildRows.filter(r => r.changed).length
        // ── Per-field variant cohesion (client-side; "should-match" fields only) ──
        // Groups each child's CURRENT value to show whether the variants are consistent or split,
        // how many need updating, and which SKUs hold which version.
        const normV = (s: string | null | undefined) => (s ?? '').replace(/\s+/g, ' ').trim()
        const fieldCohesion = (getCurrent: (c: ChildContentRow) => string | null | undefined, recommended: string, optimal: boolean) => {
          const groups = new Map<string, string[]>()
          for (const c of variants) {
            const v = normV(getCurrent(c))
            if (!groups.has(v)) groups.set(v, [])
            groups.get(v)!.push(c.sku)
          }
          const versions = [...groups.entries()].map(([value, skus]) => ({ value, skus })).sort((a, b) => b.skus.length - a.skus.length)
          const rec = normV(recommended)
          // When the section already scores MAX it's optimal — don't flag "N need update" against a
          // fresh AI draft that's never byte-identical (same reason the action item becomes DONE not
          // REPLACE). Keeps this row consistent with a 25/25 score instead of contradicting it.
          const needUpdate = optimal ? 0 : variants.filter(c => normV(getCurrent(c)) !== rec).length
          return { versions, distinct: versions.length, needUpdate, total: variants.length, recommended, optimal }
        }
        const cohFields = [
          { key: 'title', label: 'Title', coh: fieldCohesion(c => stripVariantSuffix(c.title), recs.recommended_title, score.title_score >= 23), copyVal: recs.recommended_title },
          { key: 'bullets', label: 'Bullets', coh: fieldCohesion(c => [c.bullet_1, c.bullet_2, c.bullet_3, c.bullet_4, c.bullet_5].filter(Boolean).join('\n'), (recs.recommended_bullets ?? []).join('\n'), score.bullet_score >= 23), copyVal: (recs.recommended_bullets ?? []).join('\n') },
          { key: 'description', label: 'Description', coh: fieldCohesion(c => c.description, recs.recommended_description, (score.description_score ?? 0) >= 23), copyVal: recs.recommended_description },
        ]
        return (
        <section>
          {activeTab === 'apply' && (
            <div className="space-y-6">
              {/* ── VARIANT COHESION — how the variants compare per field ── */}
              <div className="border border-slate-200 rounded-2xl overflow-hidden">
                <div className="px-4 py-2 bg-slate-50 border-b border-slate-200 flex items-baseline gap-2">
                  <span className="text-xs font-bold text-slate-700 uppercase tracking-wide">Variant Cohesion</span>
                  <span className="text-[11px] text-slate-400">how your {variants.length} variants compare today</span>
                </div>
                <div className="divide-y divide-slate-100">
                  {cohFields.map(f => {
                    const split = f.coh.distinct > 1
                    const open = expandedSections.has(`coh-${f.key}`)
                    return (
                      <div key={f.key}>
                        <button onClick={() => toggle(`coh-${f.key}`)} className="w-full flex items-center gap-2 px-4 py-2 text-left hover:bg-slate-50 transition-colors">
                          <span className="text-xs font-semibold text-slate-800 w-20 flex-shrink-0">{f.label}</span>
                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-100 text-slate-500 flex-shrink-0 hidden sm:inline">should match</span>
                          {split
                            ? <span className="text-[11px] text-purple-700 flex items-center gap-1">{f.coh.distinct} versions live</span>
                            : <span className="text-[11px] text-green-700 flex items-center gap-1"><span aria-hidden>✓</span>all {f.coh.total} identical</span>}
                          <span className="ml-auto text-[11px] flex-shrink-0">
                            {f.coh.needUpdate > 0
                              ? <span className="text-amber-700 flex items-center gap-1">{f.coh.needUpdate} need update</span>
                              : <span className="text-green-700">up to date</span>}
                          </span>
                          <span className="text-xs text-slate-400 flex-shrink-0">{open ? '▾' : '▸'}</span>
                        </button>
                        {open && (
                          <div className="px-4 pb-3 pt-1 bg-slate-50/60 space-y-2">
                            <div className="flex items-start justify-between gap-2 bg-green-50 border border-green-200 rounded p-2">
                              <div className="min-w-0">
                                <p className="text-[10px] font-bold text-green-800 uppercase">Update all {f.coh.total} variants to:</p>
                                <p className="text-xs text-slate-800 whitespace-pre-wrap break-words mt-0.5">{f.copyVal || '(none)'}</p>
                              </div>
                              <div className="flex items-center gap-1.5 flex-shrink-0">
                                <button onClick={() => copy(f.copyVal || '', `coh-${f.key}`)} className="text-[10px] bg-green-600 hover:bg-green-700 text-white px-2 py-0.5 rounded">{copied === `coh-${f.key}` ? 'Copied!' : 'Copy'}</button>
                                {/* Ship the broadcast value to Amazon — same push the Edit-Once card uses, surfaced here
                                    so "N need update" always has a one-click path (the seller's "no way to update it"). */}
                                {f.coh.needUpdate > 0 && (
                                  <button onClick={() => openPushPreview(f.key as 'title' | 'bullets' | 'description')} className="text-[10px] bg-emerald-600 hover:bg-emerald-700 text-white px-2 py-0.5 rounded font-medium whitespace-nowrap">Ship →</button>
                                )}
                              </div>
                            </div>
                            <p className="text-[10px] font-medium text-slate-500 uppercase">Current values across your variants{split ? ' — these diverge:' : ':'}</p>
                            {f.coh.versions.map((v, vi) => (
                              <details key={vi} className="bg-white border border-slate-200 rounded">
                                <summary className="cursor-pointer px-2 py-1 text-[11px] flex items-center gap-2">
                                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-100 text-slate-600 flex-shrink-0">{v.skus.length} variant{v.skus.length === 1 ? '' : 's'}</span>
                                  <span className="truncate text-slate-500">{v.value ? (v.value.length > 90 ? v.value.slice(0, 90) + '…' : v.value) : '(empty)'}</span>
                                </summary>
                                <p className="px-2 pb-2 text-[10px] text-slate-400 font-mono break-words">{v.skus.join(', ')}</p>
                              </details>
                            ))}
                          </div>
                        )}
                      </div>
                    )
                  })}
                  <div className="flex items-center gap-2 px-4 py-2">
                    <span className="text-xs font-semibold text-slate-800 w-20 flex-shrink-0">Backend</span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-indigo-50 text-indigo-600 flex-shrink-0 hidden sm:inline">unique each</span>
                    <span className="text-[11px] text-slate-500">each variant gets its own color-specific terms</span>
                    {needsUpdate > 0 ? (
                      <span className="ml-auto flex items-center gap-2 flex-shrink-0">
                        <span className="text-[11px] text-amber-700">{needsUpdate} need update</span>
                        <button onClick={() => openPushPreview('keywords')} className="text-[11px] bg-emerald-600 hover:bg-emerald-700 text-white px-2 py-0.5 rounded-md font-medium">Ship →</button>
                      </span>
                    ) : (
                      <span className="ml-auto text-[11px] text-emerald-600 font-medium flex-shrink-0">✓ up to date</span>
                    )}
                  </div>
                </div>
              </div>

              {/* ── TIER 1 — Edit Once (Parent Level) ── */}
              <div>
                <div className="flex items-baseline gap-2 mb-2">
                  <h3 className="flex items-center gap-1.5 text-xs font-bold text-slate-700 uppercase tracking-wide"><Icon.Layers className="w-3.5 h-3.5 text-violet-500" /> Edit Once</h3>
                  <span className="text-[11px] text-slate-400">Parent level — applies to all {variants.length} variants</span>
                </div>
                <div className="space-y-2">
                  {parentItems.map((item, idx) => {
                    const style = verdictStyles[item.verdict] || verdictStyles.SKIP
                    return (
                  <div key={idx} className={`rounded-2xl border border-slate-200 border-l-4 bg-white shadow-sm p-4 ${style}`}>
                    {/* Row 1: Element + Verdict + Priority + Level */}
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className={`inline-block w-2.5 h-2.5 rounded-full flex-shrink-0 ${verdictDot[item.verdict] || verdictDot.SKIP}`} />
                      <span className="font-semibold text-sm uppercase text-slate-900">{item.element.replace(/_/g, ' ')}</span>
                      <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${priorityBadge[item.priority] || ''}`}>
                        {item.priority}
                      </span>
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-100 border border-slate-200 text-slate-600 font-medium">
                        {item.level === 'parent' ? 'Parent Level' : 'Per Child'}
                      </span>
                      <span className="ml-auto text-[10px] font-mono text-slate-400">{item.verdict}</span>
                    </div>

                    {/* Row 2: Current Status */}
                    <p className="text-xs mt-1.5 text-slate-500">
                      <span className="font-medium text-slate-600">Current:</span> {item.current_status}
                    </p>

                    {/* Row 3: Instruction */}
                    {item.verdict !== 'DONE' && item.verdict !== 'SKIP' && (
                      <div className="mt-2 bg-slate-50 rounded-lg p-2.5 border border-slate-200">
                        <p className="text-xs font-medium mb-0.5 text-slate-700">What to do:</p>
                        <p className="text-xs leading-relaxed text-slate-600">{item.instruction}</p>
                      </div>
                    )}

                    {/* Row 4: Seller Central Path */}
                    {item.seller_central_path && item.verdict !== 'DONE' && item.verdict !== 'SKIP' && (
                      <p className="text-[10px] mt-1.5 text-slate-400">
                        {item.seller_central_path}
                      </p>
                    )}

                    {/* Row 5a: PER-CHILD title table (capacity families like SD cards) — overrides
                        the single Copy & Paste box for title only. Apparel & single-capacity
                        products keep the broadcast card below (per_child_titles is empty).
                        Display is enriched with FBM TWINS from /family-skus so the seller sees
                        every SKU the push will hit (the audit pipeline only sees FBA in
                        listing_content, but the push discovers FBM twins live). */}
                    {item.element === 'title' && Array.isArray(recs.per_child_titles) && recs.per_child_titles.length > 1 && item.verdict !== 'DONE' && item.verdict !== 'SKIP' && (() => {
                      // Parent (variation hub) title = capacity-agnostic. Strip any GB/TB/MB
                      // capacity token from the broadcast recommended_title so the parent SKU
                      // (e.g. Memory-Card-P) carries a generic family title without any specific
                      // capacity — that's what shows on the variation hub before a child is picked.
                      const stripSfx = (sku: string) => sku.replace(/[-_](?:FBA|FBM|AFN|MFN|FN)$/i, '')
                      const parentTitle = stripCapacityToken(recs.recommended_title || '') || (recs.per_child_titles[0]?.title ? stripCapacityToken(recs.per_child_titles[0].title) : '')

                      // Compose the display list: every per_child_titles entry, plus every FBM
                      // twin discovered live (inherits its FBA sibling's title — same as push).
                      // Each row carries an `inherited` flag so the UI can show a subtle hint.
                      type DisplayTitleRow = { sku: string; title: string; fulfillment: 'FBA' | 'FBM' | 'unknown'; inherited: boolean }
                      const titlesBySku = new Map<string, string>()
                      for (const t of recs.per_child_titles ?? []) titlesBySku.set(t.sku, t.title)
                      const titlesByBase = new Map<string, string>()
                      for (const t of recs.per_child_titles ?? []) titlesByBase.set(stripSfx(t.sku), t.title)

                      const display: DisplayTitleRow[] = (recs.per_child_titles ?? []).map((t) => ({
                        sku: t.sku, title: t.title,
                        fulfillment: /[-_]FBA$/i.test(t.sku) ? 'FBA' : (/[-_]FBM$/i.test(t.sku) ? 'FBM' : 'unknown'),
                        inherited: false,
                      }))
                      if (familySkus?.children?.length) {
                        for (const f of familySkus.children) {
                          if (titlesBySku.has(f.sku)) continue
                          // Only show a discovered SKU when we have a sibling title to inherit.
                          const inheritedTitle = titlesByBase.get(f.base_name)
                          if (!inheritedTitle) continue
                          display.push({ sku: f.sku, title: inheritedTitle, fulfillment: f.fulfillment, inherited: true })
                        }
                      }
                      // Group by base_name (capacity), FBA before FBM, parent SKU rendered separately.
                      display.sort((a, b) => {
                        const baseA = stripSfx(a.sku), baseB = stripSfx(b.sku)
                        if (baseA !== baseB) return baseA.localeCompare(baseB)
                        const order = { FBA: 0, FBM: 1, unknown: 2 } as const
                        return order[a.fulfillment] - order[b.fulfillment]
                      })

                      return (
                        <div className="mt-2 bg-white rounded-md border-2 border-green-300 p-3">
                          <div className="flex items-center justify-between mb-1.5 flex-wrap gap-1">
                            <span className="flex items-center gap-1 text-[10px] font-bold text-green-800 uppercase">
                              <Icon.Clipboard className="w-3 h-3" /> Titles ({display.length + 1}: parent + {display.length} variants{display.some(d => d.inherited) ? ` incl. FBM twins` : ''}):
                            </span>
                            <span className="text-[10px] text-slate-500">Each child carries its own capacity (Amazon SEO best-practice). FBM twins inherit their FBA sibling&apos;s title.</span>
                          </div>

                          {/* PARENT ROW — the variation hub title, capacity-agnostic */}
                          {parentTitle && (
                            <div className="flex items-center gap-2 bg-violet-50 p-1.5 rounded border border-violet-200 mb-1">
                              <span className="text-[10px] font-bold text-violet-700 uppercase tracking-wide flex-shrink-0 w-24">PARENT</span>
                              <span className="text-xs leading-relaxed text-slate-800 flex-1 min-w-0 break-words">{parentTitle}</span>
                              <button
                                onClick={() => { navigator.clipboard.writeText(parentTitle); setCopied('pct-PARENT'); setTimeout(() => setCopied(null), 2000) }}
                                className="text-[10px] px-2 py-0.5 bg-violet-600 text-white rounded hover:bg-violet-700 font-medium flex-shrink-0">
                                {copied === 'pct-PARENT' ? 'Copied!' : 'Copy'}
                              </button>
                            </div>
                          )}

                          <div className="space-y-1">
                            {display.map((t) => (
                              <div key={t.sku} className={`flex items-center gap-2 p-1.5 rounded border ${t.inherited ? 'bg-sky-50 border-sky-200' : 'bg-green-50 border-green-200'}`}>
                                <span className="text-[10px] font-mono text-slate-500 flex-shrink-0">{t.sku}</span>
                                <span className={`text-[9px] font-semibold px-1 rounded flex-shrink-0 ${t.fulfillment === 'FBA' ? 'bg-emerald-200 text-emerald-800' : t.fulfillment === 'FBM' ? 'bg-sky-200 text-sky-800' : 'bg-slate-200 text-slate-700'}`}>
                                  {t.fulfillment === 'unknown' ? '?' : t.fulfillment}
                                </span>
                                <span className="text-xs leading-relaxed text-slate-800 flex-1 min-w-0 break-words">{t.title}</span>
                                {t.inherited && <span className="text-[9px] text-sky-600 italic flex-shrink-0">inherits FBA twin</span>}
                                <button
                                  onClick={() => { navigator.clipboard.writeText(t.title); setCopied(`pct-${t.sku}`); setTimeout(() => setCopied(null), 2000) }}
                                  className={`text-[10px] px-2 py-0.5 text-white rounded font-medium flex-shrink-0 ${t.inherited ? 'bg-sky-600 hover:bg-sky-700' : 'bg-green-600 hover:bg-green-700'}`}>
                                  {copied === `pct-${t.sku}` ? 'Copied!' : 'Copy'}
                                </button>
                              </div>
                            ))}
                          </div>

                          {parentTitle && (
                            <p className="text-[10px] text-slate-500 mt-2 leading-relaxed">
                              <span className="font-semibold text-slate-700">Parent</span> is the variation hub title (no specific capacity — it&apos;s the shared family title sellers see in Seller Central before drilling into a variant).
                              <span className="font-semibold text-slate-700"> Variants</span> are the actual buyable child SKUs, each carrying its own capacity.
                              {display.some(d => d.inherited) && <> <span className="font-semibold text-slate-700">FBM twins</span> share the title of their FBA sibling (same product, different fulfillment).</>}
                            </p>
                          )}
                        </div>
                      )
                    })()}

                    {/* Row 5b: Replacement Content (the actual fix) — broadcast card. Hidden when
                        the per-child title table above is showing. */}
                    {item.replacement_content && item.verdict !== 'DONE' && item.verdict !== 'SKIP' && !(item.element === 'title' && Array.isArray(recs.per_child_titles) && recs.per_child_titles.length > 1) && (
                      <div className="mt-2 bg-white rounded-md border-2 border-green-300 p-3">
                        <div className="flex items-center justify-between mb-1.5">
                          <span className="flex items-center gap-1 text-[10px] font-bold text-green-800 uppercase"><Icon.Clipboard className="w-3 h-3" /> Copy & Paste This:</span>
                          <button
                            onClick={() => {
                              const text = Array.isArray(item.replacement_content)
                                ? item.replacement_content.map(line => typeof line === 'string' ? line : (line as Record<string, unknown>).keywords ? `${(line as Record<string, unknown>).sku}: ${(line as Record<string, unknown>).keywords}` : JSON.stringify(line)).join('\n')
                                : (typeof item.replacement_content === 'string' ? item.replacement_content : JSON.stringify(item.replacement_content)) || ''
                              navigator.clipboard.writeText(text)
                            }}
                            className="text-[10px] px-2 py-0.5 bg-green-600 text-white rounded hover:bg-green-700 font-medium"
                          >
                            Copy
                          </button>
                        </div>
                        {Array.isArray(item.replacement_content) ? (
                          <div className="space-y-1">
                            {item.replacement_content.map((line, li) => (
                              <p key={li} className="text-xs leading-relaxed font-mono bg-green-50 p-1.5 rounded border border-green-200">
                                {typeof line === 'string' ? line : (line as Record<string, unknown>).keywords ? `${(line as Record<string, unknown>).sku}: ${(line as Record<string, unknown>).keywords}` : JSON.stringify(line)}
                              </p>
                            ))}
                          </div>
                        ) : (
                          <p className="text-xs leading-relaxed font-mono bg-green-50 p-2 rounded border border-green-200 whitespace-pre-wrap">{typeof item.replacement_content === 'string' ? item.replacement_content : JSON.stringify(item.replacement_content)}</p>
                        )}
                      </div>
                    )}

                    {/* Row 5c: Ship this section to Amazon — each section has its own approval.
                        Audit emits element names: 'title' / 'description' / 'bullet_1..5'
                        (one consolidated bullets card, not five). Old code matched the literal
                        'bullet_1' only, which never appears — so the Ship button never showed up
                        on the bullets card. Match anything starting with 'bullet'. */}
                    {(() => {
                      const shipField: PushField | null =
                        item.element === 'title' ? 'title'
                        : item.element === 'description' ? 'description'
                        : /^bullet/.test(item.element) ? 'bullets'
                        : null
                      if (!shipField || item.verdict === 'DONE' || item.verdict === 'SKIP') return null
                      // Title is per-child for capacity families; everything else is broadcast.
                      const perChildTitle = shipField === 'title' && Array.isArray(recs.per_child_titles) && recs.per_child_titles.length > 1
                      return (
                        <div className="mt-2.5 flex items-center gap-2 flex-wrap border-t border-current/10 pt-2.5">
                          <button
                            onClick={() => openPushPreview(shipField)}
                            disabled={pushLoading}
                            title={perChildTitle ? 'Each variant gets its own capacity-specific title' : `Write the recommended ${FIELD_LABEL[shipField].toLowerCase()} directly to Amazon for every variant`}
                            className="inline-flex items-center gap-1.5 text-[11px] bg-emerald-600 hover:bg-emerald-700 text-white px-3 py-1.5 rounded-lg font-semibold disabled:opacity-50 transition-colors cursor-pointer">
                            <Icon.Send className="w-3.5 h-3.5" /> Ship {shipField === 'bullets' ? 'all 5 bullets' : FIELD_LABEL[shipField].toLowerCase()} to Amazon
                          </button>
                          <span className="text-[10px] text-slate-600 inline-flex items-center gap-1 flex-wrap">
                            {perChildTitle ? (
                              <>
                                <span className="px-1.5 py-0.5 rounded bg-amber-50 text-amber-700 font-medium">Per child</span>
                                each variant gets its own capacity-specific title
                              </>
                            ) : (
                              <>
                                <span className="px-1.5 py-0.5 rounded bg-indigo-50 text-indigo-700 font-medium">Parent</span>
                                same value written to all {variants.length} variants
                              </>
                            )}
                          </span>
                        </div>
                      )
                    })()}

                    {/* Row 6: Notes */}
                    {item.notes && (
                      <p className="text-[10px] mt-1.5 italic opacity-70">{item.notes}</p>
                    )}

                    {/* A+ Module Details */}
                    {item.aplus_modules && item.aplus_modules.length > 0 && (
                      <div className="mt-2 space-y-1">
                        <p className="text-[10px] font-medium">A+ Modules to configure:</p>
                        {item.aplus_modules.map((mod, mi) => (
                          <div key={mi} className="text-[10px] bg-white/80 rounded p-1.5 border border-current/10 flex gap-2">
                            <span className="font-mono text-[9px] opacity-50">#{mod.position}</span>
                            <span className={`px-1 rounded text-[9px] ${
                              mod.action === 'ADD' ? 'bg-blue-100 text-blue-700' :
                              mod.action === 'EDIT' ? 'bg-amber-100 text-amber-700' : 'bg-green-100 text-green-700'
                            }`}>{mod.action}</span>
                            <span className="font-medium">{mod.module_type}</span>
                            <span className="opacity-70">— {mod.content_brief}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                    )
                  })}
                </div>

                {/* Recommended Product Detail values (folded from AI Recommendations) */}
                {/* Each detail ships INDEPENDENTLY (Material, Brand, Fit Type, …). The map in
                    lib/fba/productDetailAttrs.ts decides which friendly names are pushable —
                    parent-shared attributes get a Push button, per-variant ones (Color/Size/
                    Capacity) and unmapped names keep Copy + a tooltip explaining why. */}
                {recs.product_details_improvements && recs.product_details_improvements.length > 0 && (
                  <div className="mt-3 bg-white border border-slate-200 rounded-2xl p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-semibold text-slate-700">Recommended Product Detail values</span>
                      <span className="text-[10px] text-slate-400">
                        {recs.product_details_improvements.filter((pd) => isPushableDetail(pd.field_name)).length} pushable · {recs.product_details_improvements.length} total
                      </span>
                    </div>
                    <div className="grid sm:grid-cols-2 gap-2">
                      {recs.product_details_improvements.map((pd, i) => {
                        const pushable = isPushableDetail(pd.field_name)
                        const blockedReason = pushable ? null : unpushableReason(pd.field_name)
                        return (
                          <div key={i} className={`rounded-lg p-2.5 ${pushable ? 'bg-emerald-50/40 border border-emerald-100' : 'bg-slate-50'}`}>
                            <div className="flex items-center justify-between mb-0.5 gap-2">
                              <span className="text-xs font-semibold text-slate-800">{pd.field_name}</span>
                              <div className="flex items-center gap-2 shrink-0">
                                <button onClick={() => copy(pd.recommended_value, `pd-${i}`)} className="text-[10px] text-violet-600 hover:underline">{copied === `pd-${i}` ? 'Copied!' : 'Copy'}</button>
                                {pushable ? (
                                  <button
                                    onClick={() => openPushPreview('details', pd.field_name)}
                                    className="text-[10px] bg-emerald-600 hover:bg-emerald-700 text-white px-2 py-0.5 rounded font-medium whitespace-nowrap"
                                    title={`Push ${pd.field_name} to Amazon for every child SKU`}
                                  >
                                    Push →
                                  </button>
                                ) : (
                                  <span
                                    className="text-[10px] text-slate-400 cursor-help"
                                    title={blockedReason ?? 'Set this in Seller Central.'}
                                  >
                                    Manual
                                  </span>
                                )}
                              </div>
                            </div>
                            <p className="text-xs text-slate-700">{pd.recommended_value}</p>
                            {pd.current_value && pd.current_value !== pd.recommended_value && (
                              <p className="text-[10px] text-slate-400 line-through mt-1 break-words">{pd.current_value}</p>
                            )}
                            {!pushable && blockedReason && (
                              <p className="text-[10px] text-slate-500 italic mt-1">{blockedReason}</p>
                            )}
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}
              </div>

              {/* ── TIER 2 — Edit Per Variant (Per Child) ── */}
              <div>
                <div className="flex items-baseline gap-2 mb-2 flex-wrap">
                  <h3 className="flex items-center gap-1.5 text-xs font-bold text-slate-700 uppercase tracking-wide"><Icon.Tag className="w-3.5 h-3.5 text-violet-500" /> Edit Per Variant</h3>
                  <span className="text-[11px] text-slate-400">Backend search terms — unique per color/size</span>
                  {needsUpdate > 0
                    ? <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-100 text-amber-700 font-medium">{needsUpdate} of {perChildRows.length} need update</span>
                    : <span className="text-[10px] px-1.5 py-0.5 rounded bg-green-100 text-green-700 font-medium">all {perChildRows.length} match</span>}
                </div>
                {backendItem?.instruction && <p className="text-xs text-slate-600 mb-2">{backendItem.instruction}</p>}
                {perChildRows.length === 0 ? (
                  <p className="text-xs text-slate-400">No variant data yet.</p>
                ) : (
                  <details className="bg-white border border-slate-200 rounded-2xl overflow-hidden group">
                    <summary className="cursor-pointer select-none px-4 py-2.5 text-xs font-medium text-slate-600 hover:bg-slate-50 flex items-center gap-2">
                      <span className="text-slate-400 transition-transform group-open:rotate-90" aria-hidden>▸</span>
                      View per-variant backend terms — {perChildRows.length} SKUs{needsUpdate > 0 ? `, ${needsUpdate} need update` : ''}
                    </summary>
                    <div className="overflow-x-auto border-t border-slate-100">
                    <table className="w-full text-xs">
                      <thead className="bg-slate-50 border-b border-slate-200">
                        <tr>
                          <th className="text-left px-3 py-2 font-medium text-slate-500">SKU</th>
                          <th className="text-left px-3 py-2 font-medium text-slate-500">Status</th>
                          <th className="text-left px-3 py-2 font-medium text-slate-500">Recommended backend search terms</th>
                          <th className="px-3 py-2"></th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {perChildRows.map(r => (
                          <tr key={r.sku} className={r.changed ? 'bg-amber-50/40' : ''}>
                            <td className="px-3 py-2 font-mono text-slate-700 align-top whitespace-nowrap">{r.sku}</td>
                            <td className="px-3 py-2 align-top whitespace-nowrap">
                              {r.recommended === ''
                                ? <span className="text-slate-400">—</span>
                                : r.changed
                                  ? <span className="text-amber-600 font-medium">Update</span>
                                  : <span className="text-green-600">✓ OK</span>}
                            </td>
                            <td className="px-3 py-2 align-top">
                              {r.changed && r.current && <p className="text-[10px] text-slate-400 line-through mb-0.5 whitespace-pre-wrap break-words">{r.current}</p>}
                              <p className="text-slate-700 font-mono leading-relaxed whitespace-pre-wrap break-words">{r.recommended || '(no recommendation)'}</p>
                            </td>
                            <td className="px-3 py-2 align-top">
                              {r.recommended && (
                                <button onClick={() => copy(r.recommended, `kw-${r.sku}`)} className="text-[10px] text-violet-600 hover:underline whitespace-nowrap">{copied === `kw-${r.sku}` ? 'Copied!' : 'Copy'}</button>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    </div>
                  </details>
                )}
                {needsUpdate > 0 && (
                  <button onClick={() => openPushPreview('keywords')} className="mt-2 text-xs bg-emerald-600 hover:bg-emerald-700 text-white px-3 py-1.5 rounded-lg font-medium">
                    Push {needsUpdate} update{needsUpdate === 1 ? '' : 's'} to Amazon →
                  </button>
                )}

                {/* Variant-specific corrections (folded from AI Recommendations) */}
                {recs.variant_corrections && recs.variant_corrections.length > 0 && (
                  <div className="mt-3 bg-white border border-slate-200 rounded-2xl p-4">
                    <span className="text-xs font-semibold text-slate-700 block mb-2">Variant-specific corrections</span>
                    <div className="space-y-2">
                      {recs.variant_corrections.map((vc, i) => (
                        <div key={i} className="bg-slate-50 rounded-lg p-3">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-xs font-mono text-slate-600">{vc.sku}</span>
                            <span className="text-[10px] bg-blue-100 text-blue-700 px-1.5 py-0.5 rounded">{vc.field}</span>
                          </div>
                          <p className="text-[10px] text-slate-400 line-through">{vc.current.length > 100 ? vc.current.slice(0, 100) + '...' : vc.current}</p>
                          <p className="text-xs text-slate-800 mt-0.5">{vc.replace_with}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </section>
        )
      })()}

      {/* Apply-tab fallback — recommendations loaded but the audit returned an empty action plan
          (otherwise the tab renders blank — the "not loading" report). */}
      {activeTab === 'apply' && aiRecs && !(aiRecs.action_plan && aiRecs.action_plan.length > 0) && (
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-10 text-center">
          <p className="text-sm text-slate-500 mb-1">The recommendations loaded, but the action plan came back empty.</p>
          <p className="text-xs text-slate-400 mb-4">Regenerate to rebuild the per-section changes.</p>
          <button onClick={generateAiRecs} disabled={aiLoading} className="inline-flex items-center gap-1.5 text-xs font-semibold text-white bg-violet-600 hover:bg-violet-700 rounded-lg px-4 py-2 transition-colors cursor-pointer disabled:opacity-50">
            <Icon.Sparkles className="w-3.5 h-3.5" /> {aiLoading ? 'Regenerating…' : 'Regenerate'}
          </button>
        </div>
      )}

      {/* ══════════════════════════════════════════════════════════════════════
          SECTION 1 — Keyword Placement Plan (grouped by placement)
          ══════════════════════════════════════════════════════════════════════ */}
      <section>
        {activeTab === 'placement' && (
          <>
            {!aiRecs && !aiLoading && (
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-6 text-center">
                <p className="text-sm text-slate-500 mb-3">No AI audit yet. Generate one to see the keyword placement plan.</p>
                <button onClick={generateAiRecs} className="text-xs bg-violet-600 hover:bg-violet-700 text-white px-4 py-2 rounded-lg">
                  Run AI Audit
                </button>
              </div>
            )}
            {aiLoading && (
              <div className="bg-violet-50 border border-violet-200 rounded-lg p-6 text-center">
                <div className="animate-spin w-6 h-6 border-2 border-violet-500 border-t-transparent rounded-full mx-auto mb-2" />
                <p className="text-sm text-violet-600">{aiProgress || 'Generating keyword placement plan...'}</p>
              </div>
            )}
            {placementGroups && (
              <div className="space-y-4">
                {placementGroups.sortedKeys.map(groupKey => {
                  const group = placementGroups.groups[groupKey]
                  const placements = groupKey.split(' + ')
                  const totalVol = group.keywords.reduce((s, k) => s + (k.search_volume || 0), 0)
                  const hasCritical = group.keywords.some(k => k.action_type === 'CRITICAL')
                  const hasUpgrade = group.keywords.some(k => k.action_type === 'UPGRADE' || k.action_type === 'TITLE UPGRADE')
                  const borderClass = hasCritical ? 'border-red-300 bg-red-50/30' : hasUpgrade ? 'border-amber-300 bg-amber-50/30' : 'border-green-300 bg-green-50/30'
                  const copyLabel = `placement-${groupKey}`

                  return (
                    <div key={groupKey} className={`rounded-2xl border-2 p-4 ${borderClass}`}>
                      {/* Placement header */}
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex flex-wrap gap-1.5">
                          {placements.map((loc, j) => (
                            <span key={j} className="text-xs font-bold bg-violet-700 text-white px-2.5 py-1 rounded uppercase tracking-wide">
                              {loc.replace(/_/g, ' ')}
                            </span>
                          ))}
                        </div>
                        <div className="flex items-center gap-3">
                          <span className="text-xs text-slate-500">{group.keywords.length} keywords &middot; {totalVol.toLocaleString()} searches/mo</span>
                          <button
                            onClick={() => copy(group.text, copyLabel)}
                            className="text-[10px] bg-white border border-slate-300 hover:bg-slate-50 text-slate-700 px-2 py-1 rounded transition-colors">
                            {copied === copyLabel ? 'Copied!' : 'Copy'}
                          </button>
                        </div>
                      </div>

                      {/* The actual text */}
                      <div className="bg-white rounded-lg border border-slate-200 p-3 mb-3">
                        <p className="text-sm text-slate-800 leading-relaxed">{group.text}</p>
                      </div>

                      {/* Keyword pills */}
                      <div className="flex flex-wrap gap-2">
                        {group.keywords.sort((a, b) => (b.search_volume || 0) - (a.search_volume || 0)).map((kw, i) => (
                          <span key={i} className={`inline-flex items-center gap-1.5 text-xs font-medium px-2.5 py-1 rounded-full border ${
                            kw.action_type === 'CRITICAL' ? 'bg-red-100 border-red-300 text-red-800'
                            : (kw.action_type === 'UPGRADE' || kw.action_type === 'TITLE UPGRADE') ? 'bg-amber-100 border-amber-300 text-amber-800'
                            : 'bg-green-100 border-green-300 text-green-800'
                          }`}>
                            {kw.keyword}
                            {kw.search_volume > 0 && <span className="opacity-60">({kw.search_volume.toLocaleString()})</span>}
                          </span>
                        ))}
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </>
        )}
      </section>

      {/* ══════════════════════════════════════════════════════════════════════
          SECTION 2 — Issues to Fix
          ══════════════════════════════════════════════════════════════════════ */}
      <section>
        {activeTab === 'issues' && (
          <div className="space-y-2">
            {score.issues.length === 0 ? (
              <p className="text-sm text-green-600 bg-green-50 border border-green-200 rounded-lg p-3">No issues found. This listing looks great!</p>
            ) : (
              score.issues.map((issue, i) => (
                <div key={i} className={`border-l-4 ${issueBorder(issue.field)} bg-white border border-slate-200 rounded-r-lg p-3`}>
                  <div className="flex items-start gap-2">
                    <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded flex-shrink-0 ${
                      issue.severity === 'error' ? 'bg-red-100 text-red-700'
                      : issue.severity === 'warning' ? 'bg-amber-100 text-amber-700'
                      : 'bg-blue-100 text-blue-700'
                    }`}>{issue.field}</span>
                    <p className="text-sm text-slate-700 leading-relaxed">{issue.message}</p>
                  </div>
                </div>
              ))
            )}
          </div>
        )}
      </section>

      {/* ══════════════════════════════════════════════════════════════════════
          SECTION 3 — Variant Breakdown
          ══════════════════════════════════════════════════════════════════════ */}
      <section>
        {activeTab === 'variants' && (
          <div className="bg-white border border-slate-200 rounded-2xl overflow-hidden">
            <table className="w-full text-xs">
              <thead className="bg-slate-50 border-b border-slate-200">
                <tr>
                  <th className="text-left px-3 py-2 font-medium text-slate-500">SKU</th>
                  <th className="text-left px-3 py-2 font-medium text-slate-500">A+</th>
                  <th className="text-left px-3 py-2 font-medium text-slate-500">Bullets</th>
                  <th className="text-left px-3 py-2 font-medium text-slate-500">Keywords</th>
                  <th className="text-left px-3 py-2 font-medium text-slate-500">Images</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {dedupByAsin(score.children).map(child => (
                  <tr key={child.sku} className="hover:bg-slate-50">
                    <td className="px-3 py-2">
                      <div className="font-mono text-slate-700">{child.sku}</div>
                      <div className="text-[10px] text-slate-400 truncate max-w-[300px]">{child.title}</div>
                    </td>
                    <td className="px-3 py-2">
                      {child.has_aplus ? (
                        <span className="text-green-600">&#10003; ({child.aplus_module_count}m)</span>
                      ) : (
                        <span className="text-red-500">&#10007;</span>
                      )}
                    </td>
                    <td className="px-3 py-2 text-slate-700">
                      {[child.bullet_1, child.bullet_2, child.bullet_3, child.bullet_4, child.bullet_5].filter(Boolean).length}/5
                    </td>
                    <td className="px-3 py-2 text-slate-700">
                      {child.backend_keywords ? `${child.backend_keywords.length}/250` : '0/250'}
                    </td>
                    <td className="px-3 py-2 text-slate-700">{child.image_count}/7</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* ══════════════════════════════════════════════════════════════════════
          SECTION 4 — (removed: consolidated into "Apply These Changes")
          ══════════════════════════════════════════════════════════════════════ */}
      {/* AI Recommendations consolidated into "Apply These Changes" above (Batch 4). */}

      {/* ══════════════════════════════════════════════════════════════════════
          SECTION 5 — Keyword Intelligence (if available)
          ══════════════════════════════════════════════════════════════════════ */}
      {kwData && kwData.topOpportunities.length > 0 && (
        <section>
          {activeTab === 'kwintel' && (
            <div className="bg-white border border-slate-200 rounded-2xl overflow-hidden">
              {/* Summary badges */}
              <div className="flex gap-3 p-3 border-b border-slate-100 bg-slate-50">
                {kwData.summary.critical > 0 && <span className="text-xs bg-red-100 text-red-700 px-2 py-0.5 rounded-full">{kwData.summary.critical} Critical</span>}
                {kwData.summary.upgrade > 0 && <span className="text-xs bg-amber-100 text-amber-700 px-2 py-0.5 rounded-full">{kwData.summary.upgrade} Upgrade</span>}
                {kwData.summary.reinforce > 0 && <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full">{kwData.summary.reinforce} Reinforce</span>}
                {kwData.summary.defended > 0 && <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full">{kwData.summary.defended} Defended</span>}
              </div>
              {/* Top 20 keywords table */}
              <table className="w-full text-xs">
                <thead className="bg-slate-50 border-b border-slate-200">
                  <tr>
                    <th className="text-left px-3 py-2 font-medium text-slate-500">Keyword</th>
                    <th className="text-right px-3 py-2 font-medium text-slate-500">Vol</th>
                    <th className="text-left px-3 py-2 font-medium text-slate-500">Action</th>
                    <th className="text-left px-3 py-2 font-medium text-slate-500">Present In</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {kwData.topOpportunities.slice(0, 20).map((kw, i) => (
                    <tr key={i} className="hover:bg-slate-50">
                      <td className="px-3 py-2 text-slate-800">{kw.keyword}</td>
                      <td className="px-3 py-2 text-right text-slate-600">{kw.searchVolume.toLocaleString()}</td>
                      <td className="px-3 py-2">
                        <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                          kw.actionType === 'CRITICAL' ? 'bg-red-100 text-red-700'
                          : kw.actionType === 'UPGRADE' ? 'bg-amber-100 text-amber-700'
                          : kw.actionType === 'REINFORCE' ? 'bg-green-100 text-green-700'
                          : kw.actionType === 'IRRELEVANT' ? 'bg-slate-100 text-slate-500'
                          : 'bg-blue-100 text-blue-700'
                        }`}>{kw.actionType === 'IRRELEVANT' ? 'OFF-PRODUCT' : kw.actionType}</span>
                      </td>
                      <td className="px-3 py-2">
                        <div className="flex gap-1">
                          {kw.inTitle && <span className="text-[9px] bg-blue-50 text-blue-600 px-1 rounded">T</span>}
                          {kw.inBullets && <span className="text-[9px] bg-green-50 text-green-600 px-1 rounded">B</span>}
                          {kw.inDescription && <span className="text-[9px] bg-purple-50 text-purple-600 px-1 rounded">D</span>}
                          {kw.inBackend && <span className="text-[9px] bg-slate-100 text-slate-600 px-1 rounded">K</span>}
                          {!kw.inTitle && !kw.inBullets && !kw.inDescription && !kw.inBackend && (
                            <span className="text-[9px] text-red-500">nowhere</span>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}

      </div>

      {/* ══════════════════════════════════════════════════════════════════════
          FIX CAPACITY ATTRIBUTE — preview → confirm → live PATCH (same chain as re-link)
          ══════════════════════════════════════════════════════════════════════ */}
      {fixCapTarget && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" onClick={() => !fixCapLoading && setFixCapTarget(null)}>
          <div className="bg-white rounded-2xl shadow-xl max-w-2xl w-full max-h-[85vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between px-5 py-3 border-b border-slate-200 sticky top-0 bg-white">
              <h3 className="text-sm font-bold text-slate-900">Fix capacity for <span className="font-mono">{fixCapTarget.row.sku}</span></h3>
              <button onClick={() => !fixCapLoading && setFixCapTarget(null)} className="text-slate-400 hover:text-slate-600 text-lg leading-none">&times;</button>
            </div>
            <div className="p-5">
              <p className="text-xs text-slate-600 mb-3">
                This patches the <span className="font-mono">{fixCapTarget.row.attributeName ?? 'capacity'}</span> attribute via SP-API. We <b>validation-preview first</b> — only an Amazon-validated change reaches Live. Amazon returns ACCEPTED before the change is actually visible, so re-run the capacity check after a few minutes. <b>Heads-up:</b> changing a variation axis value can cause Amazon to re-validate the variation family.
              </p>
              <div className="text-xs bg-slate-50 border border-slate-200 rounded-lg p-3 space-y-1">
                <div className="flex justify-between"><span className="text-slate-500">SKU</span><span className="font-mono">{fixCapTarget.row.sku}</span></div>
                <div className="flex justify-between"><span className="text-slate-500">ASIN</span><span className="font-mono">{fixCapTarget.row.asin}</span></div>
                <div className="flex justify-between"><span className="text-slate-500">Attribute</span><span className="font-mono">{fixCapTarget.row.attributeName ?? '—'}</span></div>
                <div className="flex justify-between"><span className="text-slate-500">Live (wrong)</span><span className="font-mono text-red-600">{fixCapTarget.row.liveLabel}</span></div>
                <div className="flex justify-between"><span className="text-slate-500">Will become</span><span className="font-mono text-emerald-700 font-semibold">{fixCapTarget.row.expectedLabel}</span></div>
              </div>

              {fixCapError && <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg p-3 mt-3">{fixCapError}</p>}

              {fixCapPreview && !fixCapResult && (
                <div className={`mt-4 rounded-lg p-3 ${fixCapPreview.ok ? 'bg-emerald-50 border border-emerald-200' : 'bg-red-50 border border-red-200'}`}>
                  <p className={`text-xs font-semibold ${fixCapPreview.ok ? 'text-emerald-800' : 'text-red-800'}`}>
                    {fixCapPreview.ok ? 'Amazon validated — ready to write.' : 'Amazon rejected the planned change.'}
                  </p>
                  {fixCapPreview.issues.length > 0 && (
                    <ul className="mt-2 space-y-1">
                      {fixCapPreview.issues.map((i, idx) => (
                        <li key={idx} className="text-[11px]">
                          <span className={`px-1.5 py-0.5 rounded font-bold mr-1 ${i.severity === 'ERROR' ? 'bg-red-100 text-red-700' : 'bg-amber-100 text-amber-700'}`}>{i.severity}</span>
                          <span className="text-slate-700">{i.message}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              )}

              {fixCapResult && (
                <div className={`mt-4 rounded-lg p-3 ${fixCapResult.submitted ? 'bg-emerald-50 border border-emerald-200' : 'bg-red-50 border border-red-200'}`}>
                  <p className={`text-xs font-semibold ${fixCapResult.submitted ? 'text-emerald-800' : 'text-red-800'}`}>
                    {fixCapResult.submitted ? 'Submitted to Amazon.' : 'Amazon rejected the capacity fix.'}
                  </p>
                  <p className="text-[11px] text-slate-700 mt-1">{fixCapResult.message}</p>
                  {fixCapResult.submissionId && <p className="text-[11px] text-slate-500 mt-1">submissionId: <span className="font-mono">{fixCapResult.submissionId}</span></p>}
                  {fixCapResult.issues.length > 0 && (
                    <ul className="mt-2 space-y-1">
                      {fixCapResult.issues.map((i, idx) => (
                        <li key={idx} className="text-[11px]">
                          <span className={`px-1.5 py-0.5 rounded font-bold mr-1 ${i.severity === 'ERROR' ? 'bg-red-100 text-red-700' : 'bg-amber-100 text-amber-700'}`}>{i.severity}</span>
                          <span className="text-slate-700">{i.message}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              )}

              <div className="mt-5 flex items-center gap-2">
                <button onClick={() => setFixCapTarget(null)} className="text-xs px-4 py-2 rounded-lg border border-slate-300 text-slate-700 hover:bg-slate-50 cursor-pointer">Cancel</button>
                {!fixCapResult && (
                  <button onClick={previewFixCap} disabled={fixCapLoading}
                    className="text-xs px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-900 text-white disabled:opacity-50 cursor-pointer">
                    {fixCapLoading && !fixCapPreview ? 'Validating…' : 'Validate with Amazon'}
                  </button>
                )}
                {fixCapPreview?.ok && !fixCapResult && (
                  <button onClick={confirmFixCap} disabled={fixCapLoading}
                    className="ml-auto text-xs px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white font-semibold disabled:opacity-50 cursor-pointer">
                    {fixCapLoading ? 'Writing…' : 'Confirm & Fix'}
                  </button>
                )}
                {fixCapResult && (
                  <button onClick={() => { setFixCapTarget(null); runCapacityCheck() }}
                    className="ml-auto text-xs px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-900 text-white cursor-pointer">
                    Done (re-check)
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ══════════════════════════════════════════════════════════════════════
          RE-LINK ORPHAN — preview (VALIDATION_PREVIEW) → confirm → live PATCH
          ══════════════════════════════════════════════════════════════════════ */}
      {relinkTarget && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" onClick={() => !relinkLoading && setRelinkTarget(null)}>
          <div className="bg-white rounded-2xl shadow-xl max-w-2xl w-full max-h-[85vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between px-5 py-3 border-b border-slate-200 sticky top-0 bg-white">
              <h3 className="text-sm font-bold text-slate-900">Re-link <span className="font-mono">{relinkTarget.childSku}</span> to a parent</h3>
              <button onClick={() => !relinkLoading && setRelinkTarget(null)} className="text-slate-400 hover:text-slate-600 text-lg leading-none">&times;</button>
            </div>
            <div className="p-5">
              <p className="text-xs text-slate-600 mb-3">This writes the variation relationship directly to Amazon. We <b>validation-preview first</b> — only an Amazon-validated change reaches Live. After Live, Amazon returns ACCEPTED before actually applying the change, so confirm via the orphan check again in a few minutes.</p>
              <label className="block text-xs font-medium text-slate-700 mb-1">Target parent SKU</label>
              <input
                type="text" value={relinkParentSku} onChange={(e) => { setRelinkParentSku(e.target.value); setRelinkPreview(null); setRelinkResult(null) }}
                className="w-full text-xs border border-slate-200 rounded-lg px-2.5 py-1.5 font-mono focus:outline-none focus:ring-2 focus:ring-violet-200 focus:border-violet-400"
                placeholder="e.g. Memory-Card-P"
              />
              <p className="text-[11px] text-slate-400 mt-1">The parent SKU (not ASIN). Must already exist as a non-buyable variation parent of the same productType.</p>

              {relinkError && <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg p-3 mt-3">{relinkError}</p>}

              {/* Preview */}
              {relinkPreview && !relinkResult && (
                <div className={`mt-4 rounded-lg p-3 ${relinkPreview.ok ? 'bg-emerald-50 border border-emerald-200' : 'bg-red-50 border border-red-200'}`}>
                  <p className={`text-xs font-semibold ${relinkPreview.ok ? 'text-emerald-800' : 'text-red-800'}`}>
                    {relinkPreview.ok ? 'Amazon validated — ready to write.' : 'Amazon rejected the planned change.'}
                  </p>
                  <p className="text-[11px] text-slate-600 mt-1">productType: <span className="font-mono">{relinkPreview.productType}</span> · variation_theme: <span className="font-mono">{relinkPreview.variation_theme ?? '(none from parent)'}</span></p>
                  {relinkPreview.issues.length > 0 && (
                    <ul className="mt-2 space-y-1">
                      {relinkPreview.issues.map((i, idx) => (
                        <li key={idx} className="text-[11px]">
                          <span className={`px-1.5 py-0.5 rounded font-bold mr-1 ${i.severity === 'ERROR' ? 'bg-red-100 text-red-700' : 'bg-amber-100 text-amber-700'}`}>{i.severity}</span>
                          <span className="text-slate-700">{i.message}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              )}

              {/* Result */}
              {relinkResult && (
                <div className={`mt-4 rounded-lg p-3 ${relinkResult.submitted ? 'bg-emerald-50 border border-emerald-200' : 'bg-red-50 border border-red-200'}`}>
                  <p className={`text-xs font-semibold ${relinkResult.submitted ? 'text-emerald-800' : 'text-red-800'}`}>
                    {relinkResult.submitted ? 'Submitted to Amazon.' : 'Amazon rejected the re-link.'}
                  </p>
                  <p className="text-[11px] text-slate-700 mt-1">{relinkResult.message}</p>
                  {relinkResult.submissionId && <p className="text-[11px] text-slate-500 mt-1">submissionId: <span className="font-mono">{relinkResult.submissionId}</span></p>}
                  {relinkResult.issues.length > 0 && (
                    <ul className="mt-2 space-y-1">
                      {relinkResult.issues.map((i, idx) => (
                        <li key={idx} className="text-[11px]">
                          <span className={`px-1.5 py-0.5 rounded font-bold mr-1 ${i.severity === 'ERROR' ? 'bg-red-100 text-red-700' : 'bg-amber-100 text-amber-700'}`}>{i.severity}</span>
                          <span className="text-slate-700">{i.message}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              )}

              <div className="mt-5 flex items-center gap-2">
                <button onClick={() => setRelinkTarget(null)} className="text-xs px-4 py-2 rounded-lg border border-slate-300 text-slate-700 hover:bg-slate-50 cursor-pointer">Cancel</button>
                {!relinkResult && (
                  <button onClick={previewRelink} disabled={relinkLoading || !relinkParentSku}
                    className="text-xs px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-900 text-white disabled:opacity-50 cursor-pointer">
                    {relinkLoading && !relinkPreview ? 'Validating…' : 'Validate with Amazon'}
                  </button>
                )}
                {relinkPreview?.ok && !relinkResult && (
                  <button onClick={confirmRelink} disabled={relinkLoading}
                    className="ml-auto text-xs px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white font-semibold disabled:opacity-50 cursor-pointer">
                    {relinkLoading ? 'Writing…' : 'Confirm & Re-link'}
                  </button>
                )}
                {relinkResult && (
                  <button onClick={() => { setRelinkTarget(null); runOrphanCheck(); runRelinkStatus() }}
                    className="ml-auto text-xs px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-900 text-white cursor-pointer">
                    Done (re-check orphans)
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ══════════════════════════════════════════════════════════════════════
          SHIP CONTENT TO AMAZON — per-section preview → confirm modal
          ══════════════════════════════════════════════════════════════════════ */}
      {showPushModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" onClick={() => !pushLoading && setShowPushModal(false)}>
          <div className="bg-white rounded-2xl shadow-xl max-w-3xl w-full max-h-[85vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between px-5 py-3 border-b border-slate-200 sticky top-0 bg-white">
              <h3 className="flex items-center gap-2 text-sm font-bold text-slate-900">
                <Icon.Send className="w-4 h-4 text-emerald-600" />
                {pushField === 'details' && pushDetailField
                  ? <>Ship Detail · <span className="font-semibold">{pushDetailField}</span> to Amazon</>
                  : <>Ship {FIELD_LABEL[pushField]} to Amazon</>}
                {pushField === 'details' && pushPreview?.attribute_key && (
                  <span className="text-[10px] text-slate-500 font-mono ml-1">/attributes/{pushPreview.attribute_key}</span>
                )}
              </h3>
              <button onClick={() => !pushLoading && setShowPushModal(false)} className="text-slate-400 hover:text-slate-600 text-lg leading-none">&times;</button>
            </div>

            <div className="p-5">
              {pushError && <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg p-3 mb-3">{pushError}</p>}

              {pushLoading && !pushResults && (
                <div className="py-4">
                  <div className="text-center mb-3">
                    <div className="animate-spin w-6 h-6 border-2 border-emerald-500 border-t-transparent rounded-full mx-auto mb-2" />
                    <p className="text-sm text-slate-500">
                      {pushPhase === 'rescoring' ? 'Re-scoring listing…'
                        : pushPhase === 'pushing' ? 'Streaming pushes to Amazon — one SKU at a time…'
                        : pushPhase === 'starting' ? 'Connecting to Amazon…'
                        : pushPreview ? 'Pushing to Amazon…'
                        : 'Loading preview…'}
                    </p>
                  </div>
                  {/* Live per-SKU stream — visible during the push, kept after on success
                       so the seller can scroll back to see what each SKU did. The stream
                       eliminates the proxy-502 failure mode by keeping the connection
                       warm with one event per SKU (250ms each). */}
                  {pushProgress.length > 0 && (
                    <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 max-w-md mx-auto">
                      <p className="text-[10px] font-semibold text-slate-600 uppercase mb-1.5">Per-SKU progress · {pushProgress.filter(p => p.status === 'accepted').length} accepted · {pushProgress.filter(p => p.status === 'failed').length} failed · {pushProgress.filter(p => p.status === 'validating').length} pending</p>
                      <div className="border border-slate-200 rounded divide-y divide-slate-100 max-h-[35vh] overflow-y-auto bg-white">
                        {pushProgress.map((p) => (
                          <div key={p.sku} className="px-2 py-1.5 text-[11px] flex items-center gap-2">
                            <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                              p.status === 'accepted' ? 'bg-green-100 text-green-700'
                              : p.status === 'failed' ? 'bg-red-100 text-red-700'
                              : 'bg-slate-100 text-slate-600'
                            }`}>
                              {p.status === 'accepted' ? '✓ accepted' : p.status === 'failed' ? '✗ failed' : '⏳ validating'}
                            </span>
                            <span className="font-mono text-slate-700 flex-1 min-w-0 truncate">{p.sku}</span>
                            {p.error && <span className="text-[10px] text-red-600 truncate">{p.error.slice(0, 60)}</span>}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Preview (pre-confirm) */}
              {pushPreview && !pushResults && !pushLoading && (
                <>
                  {/* Scope banner — Parent (broadcast, same to all) vs Per-child (unique each) */}
                  {pushPreview.broadcast ? (
                    <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-3 mb-3">
                      <p className="text-xs text-indigo-900 leading-relaxed">
                        <span className="px-1.5 py-0.5 rounded bg-indigo-600 text-white font-semibold mr-1.5 whitespace-nowrap">Parent</span>
                        {pushPreview.field === 'details' ? (
                          <>The same <b>{pushPreview.detail_field}</b> value is written to <b>all {pushPreview.count}</b> SKUs (each ASIN&apos;s FBA + FBM + the variation parent).{' '}
                          <b>{pushPreview.changed}</b> currently differ and will change.</>
                        ) : (
                          <>The same {pushPreview.label.toLowerCase()} is written to <b>all {pushPreview.count}</b> SKUs — including each ASIN&apos;s matching FBA + FBM.{' '}
                          <b>{pushPreview.changed}</b> currently differ and will change.</>
                        )}
                      </p>
                    </div>
                  ) : (
                    <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 mb-3">
                      <p className="text-xs text-amber-900 leading-relaxed">
                        <span className="px-1.5 py-0.5 rounded bg-amber-600 text-white font-semibold mr-1.5 whitespace-nowrap">Per-child</span>
                        {pushPreview.field === 'keywords'
                          ? <>Each of <b>{pushPreview.count}</b> SKUs (incl. matching FBA + FBM) gets its <b>own</b> backend search terms. <b>{pushPreview.changed}</b> will change — not customer-visible.</>
                          : pushPreview.field === 'title'
                          ? <>Each of <b>{pushPreview.count}</b> SKUs gets its <b>own</b> title — different SKUs get different titles (e.g. each capacity variant carries its own GB). Review each row below before confirming. <b>{pushPreview.changed}</b> will change.</>
                          : <>Each of <b>{pushPreview.count}</b> SKUs gets its <b>own</b> {pushPreview.label.toLowerCase()}. Review each row below. <b>{pushPreview.changed}</b> will change.</>}
                      </p>
                    </div>
                  )}
                  <p className="text-xs text-slate-500 mb-3">
                    Every value is checked with Amazon (VALIDATION_PREVIEW) before any live write, and the previous value is saved for rollback.
                    {pushPreview.field === 'keywords' && ' Backend strings are capped at 250 bytes.'}
                  </p>

                  {pushPreview.broadcast ? (
                    /* Broadcast: show the single new value once, then which children currently differ */
                    <>
                      <div className="bg-white rounded-md border-2 border-emerald-300 p-3 mb-3">
                        <p className="text-[10px] font-bold text-emerald-800 uppercase mb-1.5">
                          {pushPreview.field === 'details' && pushPreview.detail_field
                            ? <>New {pushPreview.detail_field} → all {pushPreview.count} SKUs</>
                            : <>New {pushPreview.label.toLowerCase()} → all {pushPreview.count} SKUs</>}
                        </p>
                        {pushPreview.field === 'bullets' && Array.isArray(pushPreview.proposedValue) ? (
                          <ul className="list-disc pl-5 space-y-1">
                            {pushPreview.proposedValue.map((b, i) => <li key={i} className="text-xs text-slate-800 break-words">{b}</li>)}
                          </ul>
                        ) : pushPreview.field === 'description' ? (
                          <div className="text-xs text-slate-800 max-h-56 overflow-y-auto leading-relaxed [&_li]:ml-4 [&_li]:list-disc [&_p]:mb-2 [&_b]:font-semibold"
                               dangerouslySetInnerHTML={{ __html: String(pushPreview.proposedValue ?? '') }} />
                        ) : (
                          <p className="text-xs text-slate-800 break-words whitespace-pre-wrap">{String(pushPreview.proposedValue ?? '')}</p>
                        )}
                      </div>
                      {/* Feature B — show Amazon's accepted vocabulary for enum attributes so the
                          seller can see the system knows the valid terms (and what we normalized). */}
                      {pushPreview.field === 'details' && pushPreview.acceptedValues && pushPreview.acceptedValues.length > 0 && (
                        <div className="bg-sky-50 border border-sky-200 rounded-md p-2.5 mb-3">
                          {pushPreview.normalizedFrom && (
                            <p className="text-[11px] text-sky-900 mb-1.5">
                              Normalized <span className="font-mono line-through text-slate-500">{pushPreview.normalizedFrom}</span> → <span className="font-mono font-semibold text-emerald-700">{String(pushPreview.proposedValue ?? '')}</span> to match Amazon&apos;s accepted terms.
                            </p>
                          )}
                          <p className="text-[10px] text-sky-800 leading-relaxed">
                            <span className="font-bold uppercase mr-1">Amazon accepts</span>
                            {pushPreview.acceptedValues.slice(0, 12).map((v, i) => (
                              <span key={i} className={`inline-block mr-1 mb-1 px-1.5 py-0.5 rounded border ${String(pushPreview.proposedValue ?? '').toLowerCase() === v.toLowerCase() ? 'bg-emerald-100 border-emerald-300 text-emerald-800 font-semibold' : 'bg-white border-slate-200 text-slate-600'}`}>{v}</span>
                            ))}
                            {pushPreview.acceptedValues.length > 12 && <span className="text-slate-400">+{pushPreview.acceptedValues.length - 12} more</span>}
                          </p>
                        </div>
                      )}
                      {pushPreview.changed > 0 && (
                        <details className="mb-4">
                          <summary className="text-xs text-slate-600 cursor-pointer hover:text-slate-800">
                            {pushPreview.changed} of {pushPreview.count} SKUs currently differ — view which
                          </summary>
                          <div className="mt-2 border border-slate-200 rounded-lg divide-y divide-slate-100 max-h-[35vh] overflow-y-auto">
                            {pushPreview.diff.filter(d => d.changed).map((d) => (
                              <div key={d.sku} className="p-2.5 text-xs">
                                <div className="font-mono text-slate-700 mb-0.5">{d.sku}</div>
                                <p className="text-slate-400 line-through break-words">{d.current || '(empty)'}</p>
                              </div>
                            ))}
                          </div>
                        </details>
                      )}
                    </>
                  ) : (
                    /* Per-child: full current → proposed diff per SKU. Sort parent row to the top,
                       then by SKU for stable ordering, so the modal mirrors the section card above. */
                    <div className="border border-slate-200 rounded-lg divide-y divide-slate-100 mb-4 max-h-[45vh] overflow-y-auto">
                      {pushPreview.diff
                        .filter(d => d.changed)
                        .slice()
                        .sort((a, b) => (a.isParent ? -1 : b.isParent ? 1 : a.sku.localeCompare(b.sku)))
                        .map((d) => {
                        // Field-aware size readout: keywords are byte-capped (250B), titles char-capped
                        // (~200), description char-capped (~2000). Show the meaningful unit per field.
                        const sizeLabel = pushPreview.field === 'keywords' ? `${d.bytes}/250 bytes`
                          : pushPreview.field === 'title' ? `${d.chars}/200 chars`
                          : pushPreview.field === 'description' ? `${d.chars} chars`
                          : `${d.chars} chars`
                        return (
                          <div key={d.sku} className={`p-3 text-xs ${d.isParent ? 'bg-violet-50' : ''}`}>
                            <div className="font-mono text-slate-700 mb-1 flex items-center gap-2">
                              {d.isParent && <span className="text-[10px] font-bold text-violet-700 uppercase tracking-wide bg-violet-100 px-1.5 py-0.5 rounded">PARENT</span>}
                              <span>{d.sku}</span>
                              <span className="text-slate-400">({sizeLabel})</span>
                            </div>
                            <p className="text-slate-400 line-through mb-0.5 break-words">{d.current || '(empty — Amazon will validate the new value)'}</p>
                            <p className={`break-words ${d.isParent ? 'text-violet-800' : 'text-emerald-700'}`}>{d.proposed}</p>
                          </div>
                        )
                      })}
                    </div>
                  )}

                  {/* ── Verify on Amazon (also on preview, so a seller who pushed earlier
                       can check "did Amazon apply it?" without re-pushing). Shows when
                       changed === 0 (cache matches rec → push was Accepted earlier) too. */}
                  <div className="mb-3 -mt-1">
                    <button
                      onClick={runVerify}
                      disabled={verifyLoading}
                      className="text-xs text-indigo-700 hover:text-indigo-900 underline disabled:opacity-50"
                    >
                      {verifyLoading ? 'Checking Amazon live…' : verifyResults ? 'Re-check Amazon live' : 'Verify on Amazon → is it actually applied?'}
                    </button>
                    {verifyError && <p className="text-xs text-red-600 mt-1">{verifyError}</p>}
                    {verifyResults && (
                      <div className="mt-2 bg-slate-50 border border-slate-200 rounded-lg p-2.5">
                        <p className="text-xs text-slate-700 mb-1.5">
                          <span className="inline-flex items-center gap-1 mr-3"><span className="w-2 h-2 rounded-full bg-green-500" /> <b>{verifyResults.matched}</b> applied</span>
                          <span className="inline-flex items-center gap-1 mr-3"><span className="w-2 h-2 rounded-full bg-amber-500" /> <b>{verifyResults.stale}</b> stale</span>
                          <span className="text-slate-500">· {verifyResults.total} SKUs checked</span>
                        </p>
                        <div className="border border-slate-200 rounded divide-y divide-slate-100 max-h-[25vh] overflow-y-auto bg-white">
                          {verifyResults.results.map((v) => (
                            <div key={v.sku} className={`px-2 py-1.5 text-[11px] flex items-center gap-2 ${v.isParent ? 'bg-violet-50' : ''}`}>
                              <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${v.matches ? 'bg-green-100 text-green-700' : 'bg-amber-100 text-amber-700'}`}>
                                {v.matches ? '✓ applied' : 'stale'}
                              </span>
                              <span className="font-mono text-slate-700">{v.sku}</span>
                              {v.isParent && <span className="text-[10px] px-1 rounded bg-violet-200 text-violet-800">PARENT</span>}
                              {v.lastUpdatedDate && (
                                <span className="text-[10px] text-slate-400 ml-auto whitespace-nowrap">{new Date(v.lastUpdatedDate).toLocaleString()}</span>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="flex justify-end gap-2">
                    <button onClick={() => setShowPushModal(false)} className="text-xs px-4 py-2 rounded-lg border border-slate-300 text-slate-600 hover:bg-slate-50">Cancel</button>
                    <button onClick={confirmPush} disabled={pushPreview.changed === 0}
                      className="text-xs px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white disabled:opacity-50">
                      Confirm &amp; Ship {pushPreview.field === 'details' && pushPreview.detail_field
                        ? pushPreview.detail_field
                        : pushPreview.label.toLowerCase()} to {pushPreview.changed} SKU{pushPreview.changed !== 1 ? 's' : ''}
                    </button>
                  </div>
                </>
              )}

              {/* Results (post-push) */}
              {pushResults && (
                <>
                  <p className="text-sm text-slate-800 mb-3">{pushResults.message}</p>
                  <div className="border border-slate-200 rounded-lg divide-y divide-slate-100 mb-4 max-h-[40vh] overflow-y-auto">
                    {pushResults.results.map((r) => (
                      <div key={r.sku} className="p-2.5 text-xs flex items-center gap-2">
                        <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${r.status === 'accepted' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>{r.status}</span>
                        <span className="font-mono text-slate-700">{r.sku}</span>
                        {r.error && <span className="text-red-600 truncate">{r.error}</span>}
                      </div>
                    ))}
                  </div>

                  {/* ── Verify on Amazon (live getListingsItem per SKU) ─────────────────
                       Amazon returns ACCEPTED, not "applied". This calls /verify-push to
                       read the LIVE attribute and tell the seller whether each SKU is
                       APPLIED, STALE (Amazon still processing or rejected), or matches. */}
                  <div className="mb-3 bg-slate-50 border border-slate-200 rounded-lg p-3">
                    <div className="flex items-center justify-between gap-2 flex-wrap">
                      <div className="text-xs text-slate-700">
                        <span className="font-semibold">Did Amazon apply it?</span>{' '}
                        <span className="text-slate-500">Submissions are ACCEPTED, then processed asynchronously (15min–6hr for variation families). Check what&apos;s actually live:</span>
                      </div>
                      <button
                        onClick={runVerify}
                        disabled={verifyLoading}
                        className="text-xs px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white disabled:opacity-50 whitespace-nowrap"
                      >
                        {verifyLoading ? 'Checking Amazon…' : verifyResults ? 'Re-check live' : 'Verify on Amazon →'}
                      </button>
                    </div>
                    {verifyError && <p className="text-xs text-red-600 mt-2">{verifyError}</p>}
                    {verifyResults && (
                      <div className="mt-3">
                        <p className="text-xs text-slate-700 mb-2">
                          <span className="inline-flex items-center gap-1 mr-3">
                            <span className="w-2 h-2 rounded-full bg-green-500" /> <b>{verifyResults.matched}</b> applied
                          </span>
                          <span className="inline-flex items-center gap-1 mr-3">
                            <span className="w-2 h-2 rounded-full bg-amber-500" /> <b>{verifyResults.stale}</b> still stale (Amazon processing or rejected)
                          </span>
                          <span className="text-slate-500">· {verifyResults.total} SKUs checked{verifyResults.attribute_key ? ` · /attributes/${verifyResults.attribute_key}` : ''}</span>
                        </p>
                        <div className="border border-slate-200 rounded-lg divide-y divide-slate-100 max-h-[35vh] overflow-y-auto bg-white">
                          {verifyResults.results.map((v) => (
                            <div key={v.sku} className={`p-2.5 text-xs ${v.isParent ? 'bg-violet-50' : ''}`}>
                              <div className="flex items-center gap-2 mb-1">
                                <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${v.matches ? 'bg-green-100 text-green-700' : 'bg-amber-100 text-amber-700'}`}>
                                  {v.matches ? '✓ applied' : 'stale'}
                                </span>
                                <span className="font-mono text-slate-700">{v.sku}</span>
                                {v.isParent && <span className="text-[10px] px-1 rounded bg-violet-200 text-violet-800">PARENT</span>}
                                {v.lastUpdatedDate && (
                                  <span className="text-[10px] text-slate-400 ml-auto">
                                    Amazon updated {new Date(v.lastUpdatedDate).toLocaleString()}
                                  </span>
                                )}
                              </div>
                              {!v.matches && (
                                <div className="grid sm:grid-cols-2 gap-2 mt-1">
                                  <div>
                                    <p className="text-[10px] font-semibold text-amber-700 uppercase mb-0.5">Live on Amazon</p>
                                    <p className="text-[10px] text-slate-600 whitespace-pre-wrap break-words">{v.currentLive || <em className="text-slate-400">(empty)</em>}</p>
                                  </div>
                                  <div>
                                    <p className="text-[10px] font-semibold text-emerald-700 uppercase mb-0.5">Expected (pushed)</p>
                                    <p className="text-[10px] text-slate-700 whitespace-pre-wrap break-words">{v.expected || <em className="text-slate-400">(empty)</em>}</p>
                                  </div>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                        {verifyResults.stale > 0 && (
                          <p className="text-[10px] text-slate-500 mt-2 italic">
                            Stale = Amazon hasn&apos;t applied the new value yet. If it&apos;s been &gt; 6 hours, the submission may have been silently rejected — re-push from the section above.
                          </p>
                        )}
                      </div>
                    )}
                  </div>

                  <div className="flex justify-end">
                    <button onClick={() => setShowPushModal(false)} className="text-xs px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-900 text-white">Done</button>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
    </div>
  )
}
