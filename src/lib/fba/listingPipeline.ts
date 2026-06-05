/**
 * listingPipeline.ts — Multi-agent Amazon listing optimization pipeline (PR 2)
 * ─────────────────────────────────────────────────────────────────────────────
 * Replaces the single monolithic LLM call with four focused agents:
 *
 *   Stage 0  Candidate prep      (code)         — score-rank, seasonal-filter, dedup, role-tag keywords
 *   Stage 1  Title Agent         (gpt-4.1-mini) — picks 2-3 candidates, writes title, code validates + retries
 *   Stage 2  Bullets Agent       (gpt-4.1-mini) — receives FINAL title, writes 5 bullets from remaining keywords
 *   Stage 3  Backend Agent       (code + gpt-4.1-mini) — color-group keywords, exclude title/bullet words, byte-cap
 *   Stage 4  Audit Agent         (o4-mini)      — action plan, reconciliation, variant health, product details
 *
 * Why split: one focused prompt per task beats an 860-line prompt doing 11 jobs.
 * The Title/Bullets/Backend outputs are CANONICAL; the Audit agent builds the
 * action plan AROUND them (never regenerates them), and the orchestrator overwrites
 * action_plan replacement_content with the canonical text — so the UI "Copy & Paste"
 * box can never drift from recommended_title/bullets/keywords (the read-path bug class).
 *
 * Each agent is reachable within the Cloudflare 100s idle budget because the caller
 * emits an NDJSON keepalive (onProgress) before every stage.
 */

import OpenAI from 'openai'
import type { AnalyzedKeyword } from '@/lib/keyword-engine'

// ─── Shared output types (structurally identical to the route's interfaces) ────

export interface PipelinePerChildKeywords { sku: string; asin: string; keywords: string }
export interface PipelineVariantCorrection { sku: string; field: string; current: string; replace_with: string; reason: string }
export interface PipelineCannibalizationWarning { keyword: string; affected_skus: string[]; issue: string; recommendation: string }
export interface PipelineProductDetailImprovement { field_name: string; current_value: string | null; recommended_value: string; reason: string }
export interface PipelineKeywordReconciliation { keyword: string; action_type: 'CRITICAL' | 'UPGRADE' | 'REINFORCE'; search_volume: number; placed_in: string[]; exact_text: string; why: string }
export interface PipelineAplusModuleAction { module_type: string; action: 'ADD' | 'EDIT' | 'KEEP'; content_brief: string; position: number }
export interface PipelineActionPlanItem {
  element: string
  level: 'parent' | 'per_child'
  verdict: 'REPLACE' | 'EDIT' | 'CREATE' | 'DONE' | 'SKIP'
  priority: 'HIGH' | 'MEDIUM' | 'LOW' | 'NONE'
  current_status: string
  instruction: string
  replacement_content?: string | string[] | null
  seller_central_path?: string
  notes?: string | null
  aplus_modules?: PipelineAplusModuleAction[]
}

export interface PipelineChild { sku: string; asin: string; color: string | null; size: string | null; title?: string | null }

export interface PipelineInput {
  openai: OpenAI
  brandName: string
  category: string
  analysis: AnalyzedKeyword[]
  children: PipelineChild[]
  /** Current title of the representative child — used for product-name token extraction */
  repTitle: string | null
  /** Per-variant content block (for the audit's variant-health check) */
  variantDetails: string
  /** Keyword intelligence context block (reused for the audit agent) */
  keywordContext: string
  hasAplus: boolean
  /** Whether the account already has an A+ Brand Story (EMC) module — gates the
   *  brand_story CREATE recommendation so we never tell a seller to create one twice. */
  hasBrandStory: boolean
  /** Reasoning-class model for the audit step, e.g. 'o4-mini' */
  auditModel: string
  /** NDJSON keepalive emitter — called before each stage */
  onProgress: (msg: string) => void
}

export interface PipelineResult {
  recommended_title: string
  recommended_bullets: string[]
  per_child_keywords: PipelinePerChildKeywords[]
  /** Per-child titles for capacity/size-spec variation families (e.g. SD cards by GB). Undefined
   *  for apparel and single-capacity products, which use the one shared recommended_title. */
  per_child_titles?: { sku: string; asin: string; title: string }[]
  recommended_description: string
  variant_corrections: PipelineVariantCorrection[]
  cannibalization_warnings: PipelineCannibalizationWarning[]
  product_details_improvements: PipelineProductDetailImprovement[]
  keyword_reconciliation: PipelineKeywordReconciliation[]
  action_plan: PipelineActionPlanItem[]
  debug: { titleProblems: string[]; candidatesUsed: string[]; titleRetried: boolean }
}

// ─── Constants / small helpers ────────────────────────────────────────────────

const SEASONAL_TERMS = [
  'christmas', 'xmas', 'halloween', 'valentines', 'valentine', 'easter',
  'thanksgiving', 'mothers day', 'mother day', 'fathers day', 'father day',
  'back to school', 'last day of school', 'schools out', 'school out',
  'independence day', '4th of july', 'fourth of july', 'july 4th',
  'st patrick', 'new year', 'new years', 'memorial day', 'labor day',
  'spring break', 'summer break', 'winter break', 'black friday',
  'cyber monday', 'prime day', 'hanukkah',
]
const MINOR_WORDS = new Set(['a', 'an', 'the', 'and', 'or', 'for', 'in', 'on', 'with', 'of', 'to', 'at', 'by'])
const KIDS_AUDIENCE = ['kids', 'kid', 'toddler', 'toddlers', 'baby', 'babies', 'infant', 'youth', 'boys', 'girls', 'children']
const ADULT_AUDIENCE = ['men', 'mens', 'women', 'womens', 'man', 'woman', 'adult', 'adults']
const GENERIC_APPAREL = new Set([
  'shirt', 'shirts', 'tshirt', 'tshirts', 'tee', 'tees', 'top', 'hoodie', 'sweatshirt',
  'tank', 'crewneck', 'pullover', 'graphic', 'vintage', 'retro', 'funny', 'novelty',
  'classic', 'comfort', 'cotton', 'soft', 'color', 'colors', 'cute', 'cool', 'gift',
])
// Meaningless filler + Amazon-prohibited subjective/temporal claims. These are never
// real search terms; they leak from low-quality keyword pulls (e.g. "interest",
// "full transparency") and from claim words Amazon forbids in search terms. Used as a
// DETERMINISTIC backstop to the non-deterministic LLM relevance gate, and to keep the
// code-built backend core clean.
const JUNK_WORDS = new Set([
  'interest', 'interested', 'transparency', 'full', 'thing', 'things', 'stuff',
  'item', 'items', 'product', 'products', 'misc', 'general', 'various', 'etc',
  'best', 'bestseller', 'bestselling', 'cheap', 'cheapest', 'discount', 'sale',
  'free', 'new', 'hot', 'popular', 'amazing', 'awesome', 'guaranteed', 'official',
])
/** True if every meaningful (non-minor) word in the phrase is junk filler. */
function isAllJunk(phrase: string): boolean {
  const words = phrase.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter((w) => w && !MINOR_WORDS.has(w))
  return words.length > 0 && words.every((w) => JUNK_WORDS.has(w))
}
// Profession/role words a graphic tee usually ISN'T about. Dropped from the backend core
// UNLESS the word is in the title (a genuine "Best Teacher" shirt keeps "teacher"). Stops
// weak-relevance leaks like "...gator teacher..." on a "see you later alligator" design.
const ROLE_WORDS = new Set([
  'teacher', 'teachers', 'nurse', 'nurses', 'doctor', 'doctors', 'educator', 'educators',
  'coach', 'coaches', 'professor', 'professors', 'principal', 'student', 'students',
])

/**
 * Third-party brand names that REQUIRE 'for [Brand]' or 'compatible with [Brand]' framing
 * in titles and bullets. Amazon's Jan 2025 enforcement (tightened Q4 2025): bare third-party
 * brand references in titles trigger listing suppression and can lead to ASIN takedown.
 * Sources: DAM Law Firm 2026 Q4 enforcement report; Amazon Seller Central Product Title
 * Guidelines effective Jan 21, 2025.
 *
 * The seller's own brand (input.brandName) is exempted at runtime — this list is
 * COMPETITORS / accessories ecosystems the seller's product is compatible WITH, not made by.
 *
 * Apparel "blank" brands (Comfort Colors, Bella Canvas, Gildan…) are deliberately NOT here.
 * Amazon has long tolerated them as material/style descriptors and the existing pipeline
 * handles them via `attributePin`. This list focuses on actively-enforcing trademark holders.
 */
const THIRD_PARTY_BRANDS = new Set([
  // Cameras & imaging
  'canon', 'nikon', 'sony', 'fujifilm', 'fuji', 'olympus', 'panasonic', 'pentax', 'leica',
  'kodak', 'gopro', 'insta360', 'dji', 'ricoh', 'sigma', 'tamron',
  // Memory / storage manufacturers
  'sandisk', 'samsung', 'lexar', 'kingston', 'pny', 'toshiba', 'transcend', 'adata', 'patriot',
  'crucial', 'seagate', 'maxell', 'micron',
  // Phones & computing
  'apple', 'iphone', 'ipad', 'macbook', 'imac', 'galaxy', 'pixel', 'microsoft', 'surface',
  'huawei', 'xiaomi', 'oneplus', 'motorola',
  // Drones
  'parrot', 'autel', 'skydio', 'yuneec',
  // Gaming
  'nintendo', 'playstation', 'xbox', 'switch',
  // Audio
  'bose', 'beats', 'jbl', 'sennheiser',
])

/** Multi-word brand phrases (checked verbatim, not per-word). */
const THIRD_PARTY_BRAND_PHRASES = [
  'western digital', 'audio technica', 'sea gate', 'go pro',
]

/**
 * Sports teams, college athletic programs, media franchises, and other licensed
 * trademark phrases. **Different semantics from THIRD_PARTY_BRANDS**: there's no safe
 * `'for [Brand]'` framing for these — an unlicensed seller can't write "for Florida
 * Gators" because the WORD MARK itself is protected (separately from the design mark
 * / logo). They must be DROPPED entirely from the keyword pool BEFORE any agent sees
 * them. The LLM relevance gate (filterRelevantKeywords) is supposed to catch these
 * but is non-deterministic; this is the deterministic backstop.
 *
 * Live-verified gap (B0G884ZJ27): regen emitted "Florida Gators" bare in title +
 * bullets + description. UF actively enforces this mark.
 *
 * Curated list focuses on actively-enforced marks where licensing is required for
 * apparel/accessory resale. Generic words ("alligator", "gators", "lions", "bears")
 * are NOT here — only the multi-word PHRASES that constitute the registered marks.
 */
const TRADEMARK_PHRASES = new Set([
  // NCAA football/basketball (top enforced)
  'florida gators', 'tennessee volunteers', 'tennessee vols', 'texas longhorns',
  'alabama crimson tide', 'lsu tigers', 'georgia bulldogs', 'auburn tigers',
  'oklahoma sooners', 'ohio state buckeyes', 'michigan wolverines',
  'penn state nittany lions', 'notre dame fighting irish', 'usc trojans', 'ucla bruins',
  'oregon ducks', 'washington huskies', 'miami hurricanes', 'florida state seminoles',
  'clemson tigers', 'virginia tech hokies', 'north carolina tar heels',
  'duke blue devils', 'kentucky wildcats', 'arkansas razorbacks',
  'kansas jayhawks', 'syracuse orange', 'villanova wildcats', 'arizona wildcats',
  'gonzaga bulldogs', 'michigan state spartans',
  // NFL
  'dallas cowboys', 'new york giants', 'new england patriots', 'kansas city chiefs',
  'green bay packers', 'pittsburgh steelers', 'philadelphia eagles', 'chicago bears',
  'san francisco 49ers', 'seattle seahawks', 'tampa bay buccaneers', 'miami dolphins',
  'new orleans saints', 'denver broncos', 'baltimore ravens', 'los angeles rams',
  // NBA
  'los angeles lakers', 'boston celtics', 'chicago bulls', 'golden state warriors',
  'miami heat', 'new york knicks', 'brooklyn nets', 'philadelphia 76ers',
  // MLB
  'new york yankees', 'boston red sox', 'los angeles dodgers', 'chicago cubs',
  'san francisco giants', 'st louis cardinals', 'houston astros', 'atlanta braves',
  // Major media franchises (word marks)
  'star wars', 'harry potter', 'lord of the rings', 'game of thrones',
  'the simpsons', 'family guy', 'south park', 'breaking bad',
])

/** Single-word trademark tokens (registered marks where the single word is unambiguous). */
const TRADEMARK_TOKENS = new Set([
  // Media / entertainment
  'marvel', 'disney', 'pixar', 'dreamworks', 'lucasfilm',
  // Universities (apparel context — Brown, Yale, Harvard etc. license their names)
  'harvard', 'stanford', 'yale', 'princeton',
  // Major leagues
  'nfl', 'nba', 'mlb', 'nhl', 'ncaa',
])

/**
 * Crude singular-ization for proximity matching ("gators" → "gator", "cowboys" → "cowboy").
 * Doesn't try to be a real stemmer — normalizes the trademark word forms most likely to
 * evade an exact-phrase check.
 */
function singularize(w: string): string {
  if (w.endsWith('ies') && w.length > 4) return w.slice(0, -3) + 'y'
  if (w.endsWith('es') && w.length > 3) return w.slice(0, -2)
  if (w.endsWith('s') && w.length > 3) return w.slice(0, -1)
  return w
}

/**
 * Find trademark phrases (sports teams, universities, media franchises) in `text`.
 * Returns matched phrases — these should NEVER appear in customer-facing copy.
 *
 * Two detection paths (PR #79 widened from exact-phrase only):
 *   1. EXACT phrase match — verbatim "florida gators", "dallas cowboys".
 *   2. TOKEN-PROXIMITY match — for each multi-word phrase, tokenize, normalize stems
 *      (gators → gator), then if ALL phrase tokens appear inside any 4-token window
 *      of `text` (in either order), flag the phrase.
 *
 * Catches the live-verified evasions PR #77 missed on B0G884ZJ27:
 *   - "vintage 90s Florida gator shirt"  → flagged via 'florida' + 'gator' in window
 *   - "Vintage 90s Gator Florida Tee"    → reversed order → flagged
 *   - "love for Florida gators"          → plural caught by exact match
 *
 * Generic words alone ("alligator", "lions", "gator") are NOT flagged — only the
 * COMBINATION of trademark tokens within the proximity window triggers.
 */
export function findTrademarkPhrases(text: string): string[] {
  const lc = ` ${text.toLowerCase()} `
  const found = new Set<string>()

  // 1. Exact phrase match.
  for (const phrase of TRADEMARK_PHRASES) {
    if (lc.includes(` ${phrase} `) || lc.includes(` ${phrase}.`) || lc.includes(` ${phrase},`) || lc.includes(`${phrase}'`)) {
      found.add(phrase)
    } else if (lc.includes(phrase)) {
      found.add(phrase)
    }
  }

  // 2. Token-proximity match — handles singular/plural and reversed word order.
  const textTokens = lc.split(/[^a-z0-9]+/).filter(Boolean)
  const textStems = textTokens.map(singularize)
  for (const phrase of TRADEMARK_PHRASES) {
    if (found.has(phrase)) continue
    const phraseStems = phrase.split(/\s+/).filter(Boolean).map(singularize)
    if (phraseStems.length < 2) continue
    const windowSize = Math.max(4, phraseStems.length + 2)
    for (let i = 0; i <= textStems.length - phraseStems.length; i++) {
      const window = new Set(textStems.slice(i, i + windowSize))
      if (phraseStems.every((s) => window.has(s))) { found.add(phrase); break }
    }
  }

  // 3. Single-word trademark tokens (Marvel/Disney/Harvard/etc.).
  for (const w of textTokens) {
    if (TRADEMARK_TOKENS.has(w)) found.add(w)
  }
  return [...found]
}

/** Find every third-party brand token in `text`, excluding the seller's own brand. */
export function findThirdPartyBrands(text: string, ownBrandTokens: Set<string>): string[] {
  const lc = text.toLowerCase()
  const found = new Set<string>()
  for (const w of lc.split(/[^a-z0-9]+/).filter(Boolean)) {
    if (ownBrandTokens.has(w)) continue
    if (THIRD_PARTY_BRANDS.has(w)) found.add(w)
  }
  for (const phrase of THIRD_PARTY_BRAND_PHRASES) {
    if (lc.includes(phrase)) found.add(phrase)
  }
  return [...found]
}

/**
 * True if EVERY occurrence of `brandToken` in `text` is properly preceded by a framing
 * word ('for' / 'compatible with' / 'works with' / 'fits') within the prior 2-3 content
 * tokens (commas / 'and' / 'or' don't count as content — they're list connectors).
 *
 * Per-occurrence check (PR #77). The previous one-shot check would mark a bullet like
 *   "GoPro is great. Compatible with GoPro Hero."
 * as framed because SOMEWHERE a framing word touches the brand — but the FIRST occurrence
 * is still bare and that's the policy-violating one.
 *
 * Handles list-with-shared-framing correctly:
 *   "for Canon, Nikon, and Sony cameras"      → all three framed via the leading "for"
 *   "compatible with GoPro Hero and Insta360" → both framed via "compatible with"
 */
export function isBrandProperlyFramed(text: string, brandToken: string): boolean {
  const lc = text.toLowerCase()
  const tokRe = brandToken.replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace(/\s+/g, '\\s+')
  const occurrences = [...lc.matchAll(new RegExp(`\\b${tokRe}\\b`, 'gi'))]
  if (occurrences.length === 0) return true   // nothing to check
  const FRAMING_RE = /^(?:for|compatible\s+with|works?\s+with|fits?(?:\s+for)?)\s*$/
  const CONNECTOR_RE = /^(?:and|or|,|\/|&)$/
  for (const m of occurrences) {
    // Walk backwards through up to 5 preceding tokens (skipping commas/and/or as list
    // connectors). If we find a framing token within that window → this occurrence is
    // framed. Otherwise → bare.
    const before = lc.slice(Math.max(0, (m.index ?? 0) - 80), m.index ?? 0)
    const beforeTokens = before.split(/[\s]+/).filter(Boolean)
    let framed = false
    let contentTokensSeen = 0
    for (let i = beforeTokens.length - 1; i >= 0 && contentTokensSeen < 4; i--) {
      const t = beforeTokens[i]
      // Try multi-word framing patterns first ("compatible with", "works with").
      if (i > 0) {
        const pair = `${beforeTokens[i - 1]} ${t}`
        if (FRAMING_RE.test(pair)) { framed = true; break }
      }
      if (FRAMING_RE.test(t)) { framed = true; break }
      if (CONNECTOR_RE.test(t)) continue            // list connector — keep walking, don't count
      // Allow up to 3 brand-related content tokens between the framing word and this
      // brand (covers "for action camera GoPro" + "for Canon EOS GoPro").
      contentTokensSeen++
    }
    if (!framed) return false
  }
  return true
}

/** Get the seller's own brand tokens for exemption from brand checks. */
function ownBrandTokenSet(brandName: string): Set<string> {
  return new Set(brandName.toLowerCase().split(/\s+/).filter(Boolean))
}
// Product-type words capped at 2 total in the backend core (Amazon's bag-of-words already
// has them from the title; >2 is the "shirt ×7" waste the PO flagged).
const PRODUCT_TYPE_WORDS = new Set(['shirt', 'shirts', 'tshirt', 'tshirts', 'tee', 'tees'])

// Is this an APPAREL product? The title/bullet/description framing (graphic tee, shirt, garment
// brand, men/women audience, fabric/fit specs) only makes sense for clothing. For non-apparel
// (memory cards, mugs, mounts…) the old hardcoded framing produced nonsense — e.g. "Graphic Tee
// for Men" on an SD card — so every agent branches on this.
function looksApparel(category?: string | null, repTitle?: string | null): boolean {
  const hay = ` ${category ?? ''} ${repTitle ?? ''} `.toLowerCase()
  // Strong non-apparel nouns WIN: a memory-card / mug listing templated from a shirt still has
  // leftover "shirt" in its data, but it is NOT clothing. The product noun decides.
  if (/\b(?:memory\s?cards?|sd\s?cards?|micro\s?sd|usb|flash\s?drives?|mugs?|cups?|tumblers?|bottles?|mounts?|holders?|stands?|chargers?|cables?|adapters?|cases?|stickers?|decals?|posters?|prints?|canvas|mousepads?|keychains?|magnets?|earbuds?|headphones?|speakers?|watch(?:es)?|necklaces?|bracelets?|earrings?|candles?|blankets?|pillows?|towels?|backpacks?|wallets?|notebooks?|journals?|toys?|puzzles?|ornaments?|mats?|signs?)\b/.test(hay)) return false
  return /\b(?:t[-\s]?shirts?|tees?|shirts?|hoodie|sweat\s?shirt|sweater|apparel|clothing|tank\s?top|dress|leggings|pajama|garment|jersey|crew\s?neck|long\s?sleeve|onesie|bodysuit|romper|blouse|cardigan|socks?|jacket|beanie|crop\s?top)\b/.test(hay)
}

// Apparel words that contaminate a NON-apparel listing (one templated from a shirt). Dropped
// from the keyword pool + specs so a memory card / mug never inherits "graphic tee",
// "ring-spun cotton", "for men", etc. Only applied when the product is non-apparel.
const APPAREL_CONTAMINANTS = /\b(?:t[-\s]?shirts?|tees?|shirts?|graphic\s*tees?|hoodie|sweat\s?shirts?|sweater|apparel|clothing|garments?|fabric|cotton|ring[-\s]?spun|jersey|knit(?:ted)?|relaxed\s*fit|regular\s*fit|comfort\s*colors|bella\s*canvas|gildan|next\s*level|unisex|m[ae]ns?|wom[ae]ns?|fashion|outfit|wardrobe|sleeves?|crew\s?neck|tank\s?tops?|garment[-\s]?dyed|\bdye\b|wear|wearable)\b/i

// A storage-capacity token ("128GB", "1 TB"). When children span >=2 distinct capacities the
// title is per-child (each carries its own capacity) — NOT a concept that ever matches apparel.
const CAPACITY_RE = /\b(\d{1,4})\s?(t|g)b?\b/i // GB/TB only — "MB" is usually a transfer speed, not capacity
function capacityOf(s: string | null | undefined): string | null {
  const m = (s ?? '').match(CAPACITY_RE)
  // "32G"/"64G." -> 32GB/64GB, "128GB" -> 128GB, "1T"/"1TB" -> 1TB
  return m ? `${m[1]}${m[2].toUpperCase()}B` : null
}

const isSeasonal = (kw: string) => SEASONAL_TERMS.some((t) => kw.toLowerCase().includes(t))

function wordOverlapRatio(a: string, b: string): number {
  const wa = new Set(a.toLowerCase().split(/\s+/).filter(Boolean))
  const wb = new Set(b.toLowerCase().split(/\s+/).filter(Boolean))
  if (wa.size === 0 || wb.size === 0) return 0
  const inter = [...wa].filter((w) => wb.has(w)).length
  return inter / Math.min(wa.size, wb.size)
}

function getByteLength(str: string): number {
  return new TextEncoder().encode(str).length
}

function truncateToBytes(str: string, maxBytes: number): string {
  if (getByteLength(str) <= maxBytes) return str
  let low = 0, high = str.length
  while (low < high) {
    const mid = Math.ceil((low + high) / 2)
    if (getByteLength(str.slice(0, mid)) <= maxBytes) low = mid
    else high = mid - 1
  }
  const truncated = str.slice(0, low)
  const lastSpace = truncated.lastIndexOf(' ')
  return lastSpace > low * 0.7 ? truncated.slice(0, lastSpace).trim() : truncated.trim()
}

/** Robust JSON extraction from an LLM response (strips fences, trailing prose, repairs trailing commas). */
function parseJsonLoose<T>(raw: string): T {
  let cleaned = raw.replace(/^```json\s*/i, '').replace(/^```\s*/i, '').replace(/\s*```$/i, '').trim()
  const first = cleaned.indexOf('{')
  const last = cleaned.lastIndexOf('}')
  if (first > 0) cleaned = cleaned.slice(first)
  if (last >= 0 && last < cleaned.length - 1) cleaned = cleaned.slice(0, last + 1)
  try {
    return JSON.parse(cleaned) as T
  } catch {
    const repaired = cleaned.replace(/,\s*([}\]])/g, '$1')
    return JSON.parse(repaired) as T
  }
}

// ─── Title validation (shared with the route's PR1 validator semantics) ────────

export function validateTitle(title: string, brandName: string, mustInclude?: string, attributePin?: string, upgradeKws?: string[]): string[] {
  const problems: string[] = []
  const len = title.length
  if (len > 150) problems.push(`Title is ${len} characters; Amazon's hard limit is 150 — shorten it (aim 80-125 for apparel).`)
  else if (len < 80) problems.push(`Title is only ${len} characters; use at least 80 to capture more keyword space.`)

  const counts = new Map<string, number>()
  title.toLowerCase().split(/\s+/).forEach((w) => {
    const base = w.replace(/[^a-z0-9]/g, '')
    if (base && !MINOR_WORDS.has(base)) {
      let norm = base.replace(/s$/, '')
      // Unify the product-type word so "T-Shirt" + "Shirt" + "Shirts" counts as THREE
      // "shirt"s (Amazon's max-2 rule). Without this, "tshirt" and "shirt" were treated
      // as different tokens and the 3× "shirt" repetition slipped through.
      if (norm === 'tshirt') norm = 'shirt'
      counts.set(norm, (counts.get(norm) ?? 0) + 1)
    }
  })
  const repeated = [...counts.entries()].filter(([, c]) => c > 2).map(([w]) => w)
  if (repeated.length) problems.push(`These words appear more than twice (Amazon allows max 2 each): ${repeated.join(', ')}. Do NOT append "shirt"/"tee" to every keyphrase — name the product type once or twice total.`)

  const lc = title.toLowerCase()
  const season = SEASONAL_TERMS.find((s) => lc.includes(s))
  if (season) problems.push(`Remove the seasonal term "${season}" — evergreen product; seasonal keywords belong in backend terms.`)

  const words = new Set(lc.replace(/[^a-z0-9\s]/g, ' ').split(/\s+/))
  const hasKids = KIDS_AUDIENCE.some((t) => words.has(t))
  const hasAdult = ADULT_AUDIENCE.some((t) => words.has(t))
  if (hasKids && hasAdult) problems.push('Title mixes kids and adult audiences (e.g. "kids" with "men"/"women") — pick ONE consistent audience.')

  // The single highest-search-volume keyword must survive into the title (the money term).
  if (mustInclude) {
    const phrase = mustInclude.toLowerCase()
    const phraseWords = phrase.replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter((w) => w && !MINOR_WORDS.has(w))
    const present = lc.includes(phrase) || (phraseWords.length > 0 && phraseWords.every((w) => words.has(w)))
    if (!present) problems.push(`Title MUST contain the highest-volume keyword "${mustInclude}" (or all of its key words) — front-load it.`)
  }

  // The blank/garment brand attribute (e.g. "comfort colors") is a strategic ranking term
  // the seller wants in the title alongside the money keyword — enforce its presence.
  if (attributePin) {
    const attrWords = attributePin.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter((w) => w && !MINOR_WORDS.has(w))
    if (attrWords.length > 0 && !attrWords.every((w) => words.has(w))) {
      problems.push(`Title MUST also contain the blank-brand attribute "${attributePin}" — it's a strategic ranking term.`)
    }
  }

  if (brandName && !lc.includes(brandName.toLowerCase())) problems.push(`Title must start with the brand "${brandName}".`)

  // 🚫 BRAND-NAME SAFETY (Amazon Jan 2025 policy, Q4 2025 enforcement).
  // Bare third-party brand names in titles trigger listing suppression. Only allowed with
  // 'for [Brand]' or 'compatible with [Brand]' framing. The seller's own brand is exempted
  // via ownBrandTokenSet. Critical — fail validation hard so the retry fires the fix.
  const ownBrands = ownBrandTokenSet(brandName)
  const brandsInTitle = findThirdPartyBrands(title, ownBrands)
  const bareRefs = brandsInTitle.filter((b) => !isBrandProperlyFramed(title, b))
  if (bareRefs.length > 0) {
    problems.push(`🚫 LISTING-SUPPRESSION RISK: title contains bare third-party brand name${bareRefs.length === 1 ? '' : 's'} ${bareRefs.map((b) => `"${b}"`).join(', ')} without 'for' or 'compatible with' framing. Amazon's Jan 2025 policy (Q4 2025 enforcement) suppresses listings with bare brand references. Rewrite using "for ${bareRefs[0]}" or "compatible with ${bareRefs[0]}" — never the brand name standing alone.`)
  }

  // 🚫 TRADEMARK PHRASES (sports teams / universities / media franchises). Different
  // semantics from THIRD_PARTY_BRANDS — no compatibility framing can rescue them. The
  // keyword pool already drops these (PR #77 filterRelevantKeywords backstop) but
  // this is the validator's last-mile check.
  const trademarksInTitle = findTrademarkPhrases(title)
  if (trademarksInTitle.length > 0) {
    problems.push(`🚫 TRADEMARK INFRINGEMENT RISK: title contains registered trademark phrase${trademarksInTitle.length === 1 ? '' : 's'} ${trademarksInTitle.map((t) => `"${t}"`).join(', ')}. Sports teams, universities, and media franchises (Florida Gators, Dallas Cowboys, Marvel, Star Wars, etc.) cannot be used in product titles unless you hold an official license. Remove these words entirely and rewrite the title with generic descriptors (e.g. "alligator graphic" not "Florida Gators", "superhero tee" not "Marvel").`)
  }

  // UPGRADE keyword coverage — the scorer docks 5 points when 7+ UPGRADE keywords appear
  // in bullets but NOT in the title (3 points when 3-6 miss). Fail fast in validation so the
  // existing retry loop pulls more of them into the title before we commit.
  // Threshold of 3+ missing matches the scorer's mid-tier penalty so we don't over-retry
  // on titles that are already 'good enough' (1-2 missing).
  if (upgradeKws && upgradeKws.length >= 3) {
    const missing = upgradeKws.filter((kw) => {
      const phrase = kw.toLowerCase()
      // Either the verbatim phrase or every meaningful word survives — same matching the
      // mustInclude check uses, so the validator and scorer agree on what 'present' means.
      if (lc.includes(phrase)) return false
      const phraseWords = phrase.replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter((w) => w && !MINOR_WORDS.has(w))
      return !(phraseWords.length > 0 && phraseWords.every((w) => words.has(w)))
    })
    if (missing.length >= 3) {
      // Show the first 5 missing keywords — enough for the retry prompt to act on without
      // overwhelming the agent.
      const sample = missing.slice(0, 5).map((k) => `"${k}"`).join(', ')
      problems.push(`Title is missing ${missing.length} top UPGRADE keywords that appear in your bullets — Amazon weights title keywords 3-5× more than bullets. Pull at least ${Math.max(0, missing.length - 2)} of these in: ${sample}.`)
    }
  }

  return problems
}

/**
 * Validate the generated bullets against scorer-equivalent rules + Amazon brand-safety.
 * Mirrors the validateTitle contract: returns problems for the bullets-agent retry loop to
 * resolve. Each check matches a real penalty in scoreListingContent() so passing here means
 * passing the scorer too — the "follow the recommendation → perfect score" promise extends
 * to bullets the same way PR #73 extended it to title.
 *
 * Checks (all match scoreListingContent.ts:654+):
 *   - length < 100 chars per bullet (scorer -5 per short bullet, capped at -15)
 *   - bare third-party brand in any bullet (Amazon Jan 2025 listing-suppression risk —
 *     parallel to validateTitle's check; PR #74 closed it for title, this closes it for bullets)
 *   - 3+ top opportunity keywords (CRITICAL ∪ UPGRADE) missing across the whole bullet set
 *     (scorer -2 per missing, capped at -12; threshold matches scorer's "2+" trigger)
 */
export function validateBullets(
  bullets: string[],
  brandName: string,
  opportunityKws: string[] = [],
  /** Distinct capacity tokens across the family (e.g. ['32GB','64GB','128GB']). Non-empty
   *  means broadcast bullets must NOT hardcode any specific capacity — they ship to every
   *  child. Live-verified bug at B0GCF11RKL: bullets 2 and 3 hardcoded "128 GB" though the
   *  family has 32/64/128 SKUs. The agent prompt already forbids this — validation
   *  enforces it through the retry loop. */
  capacityFamily: string[] = [],
): string[] {
  const problems: string[] = []
  if (bullets.length === 0) {
    problems.push('No bullets generated.')
    return problems
  }

  // Length: bullets <100 chars get docked by the scorer.
  const shortBullets = bullets
    .map((b, i) => ({ i, b, len: b.length }))
    .filter((x) => x.len < 100)
  if (shortBullets.length > 0) {
    const names = shortBullets.map((x) => `bullet ${x.i + 1} (${x.len} chars)`).join(', ')
    problems.push(`${shortBullets.length} bullet${shortBullets.length === 1 ? '' : 's'} under 100 chars: ${names}. Expand each to 100-200 chars with a "so that" benefit, a compatible-device example, and a long-tail keyword.`)
  }

  // CAPS-hook check (live-verified bug at B0G884ZJ27: bullets opened with sentence prose
  // "Made from soft, breathable..." instead of "PREMIUM FABRIC - Made from..."). Pattern:
  // 2-4 ALL-CAPS tokens (length ≥2 each, hyphens/digits allowed) followed by " - "/" – "/" — ".
  const capsHookRe = /^(?:[A-Z][A-Z0-9-]+(?:\s+[A-Z][A-Z0-9-]+){0,3})\s*[-–—]\s+/
  const noHook: { i: number }[] = []
  for (let i = 0; i < bullets.length; i++) {
    if (!capsHookRe.test(bullets[i].trim())) noHook.push({ i })
  }
  if (noHook.length > 0) {
    const names = noHook.map((x) => `bullet ${x.i + 1}`).join(', ')
    problems.push(`${noHook.length} bullet${noHook.length === 1 ? '' : 's'} missing the CAPS benefit hook (${names}). Each bullet must start with a 2-3 word ALL-CAPS benefit followed by " - " then the explanation. Example: "HIGH-SPEED PERFORMANCE - Class 10 UHS-I technology ensures fast read/write speeds...".`)
  }

  // Awkward-forced-keyword check (live-verified bug at B0GCF11RKL: "CRITICAL UPGRADE -
  // Upgrade your storage..." — the agent literalized PR #73's MANDATORY #3 prompt term as
  // the benefit hook). CAPS hooks should be BENEFIT phrases, not pipeline action-type
  // labels that name the SEO mechanic itself.
  const FORBIDDEN_HOOK_WORDS = new Set(['CRITICAL', 'UPGRADE', 'REINFORCE', 'DEFENDED', 'KEYWORD', 'KEYWORDS', 'SEO', 'OPPORTUNITY'])
  const awkwardHooks: { i: number; forbidden: string[] }[] = []
  for (let i = 0; i < bullets.length; i++) {
    const m = bullets[i].trim().match(capsHookRe)
    if (!m) continue
    const hookOnly = m[0].replace(/\s*[-–—]\s+$/, '').trim()
    const forbidden = hookOnly.split(/\s+/).filter((w) => FORBIDDEN_HOOK_WORDS.has(w))
    if (forbidden.length > 0) awkwardHooks.push({ i, forbidden })
  }
  if (awkwardHooks.length > 0) {
    const detail = awkwardHooks.map((x) => `bullet ${x.i + 1} ("${x.forbidden.join('/')}" in hook)`).join(', ')
    problems.push(`${awkwardHooks.length} bullet${awkwardHooks.length === 1 ? '' : 's'} have a CAPS hook that literalizes a pipeline action-type label (${detail}). Hooks must name a real BENEFIT shoppers care about ("HIGH-SPEED PERFORMANCE", "DURABLE DESIGN", "WIDE COMPATIBILITY") — never "CRITICAL", "UPGRADE", "KEYWORD" or other SEO mechanic words. Rewrite the hook with a benefit phrase that fits the bullet body.`)
  }

  // 🚫 CAPACITY-FAMILY check (live-verified bug at B0GCF11RKL: bullets 2 and 3 hardcoded
  // "128 GB" though the family spans 32GB/64GB/128GB — broadcast bullets ship to every
  // child, so 32GB shoppers would see "this 128 GB SD card").
  if (capacityFamily.length >= 2) {
    const capacityRe = /\b\d{1,4}\s?(?:GB|TB|MB)\b/gi
    const hardcoded: { i: number; matches: string[] }[] = []
    for (let i = 0; i < bullets.length; i++) {
      const ms = bullets[i].match(capacityRe)
      if (ms && ms.length > 0) hardcoded.push({ i, matches: [...new Set(ms.map((m) => m.replace(/\s+/g, ' ').toUpperCase()))] })
    }
    if (hardcoded.length > 0) {
      const detail = hardcoded.map((x) => `bullet ${x.i + 1} ("${x.matches.join(', ')}")`).join('; ')
      problems.push(`🚫 CAPACITY-FAMILY VIOLATION: ${hardcoded.length} bullet${hardcoded.length === 1 ? '' : 's'} hardcode a specific capacity (${detail}). This family has multiple capacities (${capacityFamily.join(', ')}) and bullets are SHARED across all variants — saying "128GB" in a bullet ships to your 32GB and 64GB SKUs and misleads shoppers. Use capacity-agnostic phrasing: "ample capacity", "high-capacity storage", "available in multiple sizes". Each variant's specific capacity already lives in its TITLE.`)
    }
  }

  // 🚫 BRAND-NAME SAFETY — same rule that PR #74 enforced for title. Each bullet checked
  // independently because a bare brand reference in ANY bullet is a listing-suppression risk.
  const ownBrands = ownBrandTokenSet(brandName)
  const bareByBullet: { i: number; bare: string[] }[] = []
  for (let i = 0; i < bullets.length; i++) {
    const found = findThirdPartyBrands(bullets[i], ownBrands)
    const bare = found.filter((b) => !isBrandProperlyFramed(bullets[i], b))
    if (bare.length > 0) bareByBullet.push({ i, bare })
  }
  if (bareByBullet.length > 0) {
    const detail = bareByBullet
      .map(({ i, bare }) => `bullet ${i + 1}: ${bare.map((b) => `"${b}"`).join(', ')}`)
      .join('; ')
    problems.push(`🚫 LISTING-SUPPRESSION RISK: bare third-party brand name(s) in ${detail}. Amazon's Jan 2025 policy requires 'for [Brand]', 'compatible with [Brand]', or 'works with [Brand]' framing — never bare. Rewrite those bullets to wrap each brand mention in compatibility language.`)
  }

  // 🚫 TRADEMARK PHRASES per bullet (sports teams / universities / media franchises).
  const tmByBullet: { i: number; tm: string[] }[] = []
  for (let i = 0; i < bullets.length; i++) {
    const tms = findTrademarkPhrases(bullets[i])
    if (tms.length > 0) tmByBullet.push({ i, tm: tms })
  }
  if (tmByBullet.length > 0) {
    const detail = tmByBullet
      .map(({ i, tm }) => `bullet ${i + 1}: ${tm.map((t) => `"${t}"`).join(', ')}`)
      .join('; ')
    problems.push(`🚫 TRADEMARK INFRINGEMENT RISK: ${detail}. Sports teams, universities, and media franchises cannot be used in customer-facing copy unless you hold an official license. Remove these phrases entirely and rewrite with generic descriptors.`)
  }

  // Opportunity-keyword coverage. The scorer aggregates CRITICAL ∪ UPGRADE and penalizes
  // when 2+ are missing across the bullets joined together. We use 3+ as the retry trigger
  // (1-2 missing is below the scorer's first penalty tier and not worth a retry round).
  if (opportunityKws.length >= 3) {
    const joined = bullets.join(' ').toLowerCase()
    const joinedWords = new Set(joined.replace(/[^a-z0-9\s]/g, ' ').split(/\s+/))
    const missing = opportunityKws.filter((kw) => {
      const phrase = kw.toLowerCase()
      if (joined.includes(phrase)) return false
      const words = phrase.replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter((w) => w && !MINOR_WORDS.has(w))
      return !(words.length > 0 && words.every((w) => joinedWords.has(w)))
    })
    if (missing.length >= 3) {
      const sample = missing.slice(0, 5).map((k) => `"${k}"`).join(', ')
      problems.push(`Bullets are missing ${missing.length} of your top opportunity keywords (CRITICAL + UPGRADE). Weave at least ${Math.max(0, missing.length - 2)} of these naturally into the bullet body text: ${sample}.`)
    }
  }

  return problems
}

/**
 * Validate the generated description. Mirrors validateBullets: returns problems for the
 * description-agent retry loop. Checks match scoreListingContent.ts:724+ penalties.
 *
 * Checks:
 *   - plain text < 200 chars (scorer -4) — description below the meaningful-indexing floor
 *   - bare third-party brand (Amazon Jan 2025 listing-suppression risk — parallel to title)
 *
 * Note: the scorer's "<3 shared title keywords" penalty isn't enforced here because the
 * agent already gets the title as input and is told to reinforce its themes. If that gap
 * shows up in practice we can add a check that takes title tokens as a param.
 */
export function validateDescription(
  description: string,
  brandName: string,
): string[] {
  const problems: string[] = []
  // Strip HTML the same way the scorer does (loose tag strip — agent returns <p>/<ul>/<li>).
  const plain = description.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim()
  if (plain.length === 0) {
    problems.push('No description generated.')
    return problems
  }
  if (plain.length < 200) {
    problems.push(`Description is only ${plain.length} chars — expand to 800-2000 chars with use cases, target audience, technical specs, and long-tail keywords. Amazon indexes the full text.`)
  }

  const ownBrands = ownBrandTokenSet(brandName)
  const bare = findThirdPartyBrands(plain, ownBrands).filter((b) => !isBrandProperlyFramed(plain, b))
  if (bare.length > 0) {
    problems.push(`🚫 LISTING-SUPPRESSION RISK: description contains bare third-party brand name(s) ${bare.map((b) => `"${b}"`).join(', ')} without 'for' or 'compatible with' framing. Rewrite every brand mention as 'for [Brand]', 'compatible with [Brand]', or 'works with [Brand]'.`)
  }

  // 🚫 TRADEMARK PHRASES in description.
  const tmInDesc = findTrademarkPhrases(plain)
  if (tmInDesc.length > 0) {
    problems.push(`🚫 TRADEMARK INFRINGEMENT RISK: description contains registered trademark phrase${tmInDesc.length === 1 ? '' : 's'} ${tmInDesc.map((t) => `"${t}"`).join(', ')}. Sports teams, universities, and media franchises cannot be used in customer-facing copy unless you hold an official license. Rewrite using generic descriptors.`)
  }

  return problems
}

// ─── Stage 0 — candidate preparation (code only) ───────────────────────────────

interface TitleCandidate { keyword: string; opportunityScore: number; role: 'keyphrase' | 'descriptive' | 'audience' }

function extractProductNameTokens(repTitle: string | null): string[] {
  return (repTitle ?? '')
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter((w) => w.length >= 5 && !GENERIC_APPAREL.has(w))
    .slice(0, 3)
}

function selectTitleCandidates(analysis: AnalyzedKeyword[], brandName: string, repTitle: string | null): TitleCandidate[] {
  const brandTokens = brandName.toLowerCase().split(/\s+/).filter(Boolean)
  const productTokens = extractProductNameTokens(repTitle)

  const eligible = analysis
    .filter((k) => ['CRITICAL', 'UPGRADE', 'DEFENDED', 'REINFORCE'].includes(k.actionType))
    .filter((k) => !isSeasonal(k.keyword))
    .sort((a, b) => b.opportunityScore - a.opportunityScore)

  // Dedup by >=60% word overlap, keeping the higher-scoring keyword
  const deduped: AnalyzedKeyword[] = []
  for (const k of eligible) {
    if (!deduped.some((d) => wordOverlapRatio(d.keyword, k.keyword) >= 0.6)) deduped.push(k)
  }

  const roleOf = (kw: string): TitleCandidate['role'] => {
    const lc = kw.toLowerCase()
    const kwWords = new Set(lc.replace(/[^a-z0-9\s]/g, ' ').split(/\s+/))
    if (brandTokens.some((t) => lc.includes(t)) || productTokens.some((t) => kwWords.has(t))) return 'keyphrase'
    if (ADULT_AUDIENCE.some((t) => kwWords.has(t)) || /\bfor (men|women|kids)\b/.test(lc)) return 'audience'
    return 'descriptive'
  }

  return deduped.slice(0, 7).map((k) => ({ keyword: k.keyword, opportunityScore: k.opportunityScore, role: roleOf(k.keyword) }))
}

// ─── Stage 1 — Title Agent ─────────────────────────────────────────────────────

async function runTitleAgent(
  input: PipelineInput,
  candidates: TitleCandidate[],
  attributes: string[],
  mustInclude: string | undefined,
  preferredAudience: string,
  attributePin: string | undefined,
  /** Top UPGRADE-tagged keywords (present in bullets but not title — see scorer's
   *  title.upgradeCount penalty). The agent is told to pull as many in as fit; the
   *  validator below fails the title if 3+ are still missing, triggering a retry. */
  upgradeKws: string[] = [],
): Promise<{ title: string; problems: string[]; retried: boolean }> {
  const { openai, brandName, category, repTitle } = input
  const apparel = looksApparel(category, repTitle)
  const candidateList = candidates
    .map((c) => `  - "${c.keyword}" (opportunity ${c.opportunityScore}, role: ${c.role})`)
    .join('\n')
  const attrLine = attributes.length
    ? `\nSearchable product keyphrases shoppers actually type — work them in AFTER the mandatory keyword${apparel ? ' (e.g. a blank-brand term like "comfort colors graphic tee")' : ' (titles under 110 chars have room for more — keep adding while you have budget)'}:\n  ${attributes.join(', ')}\n`
    : ''
  const mustLine = mustInclude
    ? `\n🔴 MANDATORY #1 — the title MUST contain this highest-search-volume keyword VERBATIM and FRONT-LOADED (it is your single biggest money term — never drop it): "${mustInclude}"\n`
    : ''
  const attrPinLine = attributePin
    ? `\n🔴 MANDATORY #2 — the title MUST ALSO contain the blank/garment brand "${attributePin}" (a strategic ranking attribute the seller ranks for). Place it after the #1 keyword.\n`
    : ''
  // UPGRADE keywords = ranking signals the seller has demonstrated traffic on (present in
  // bullets, missing from title). Amazon weights title 3-5× over bullets, so dragging these
  // INTO the title is the highest-leverage move available. The scorer penalizes the title
  // when 3+ are missing; the validator below fails on the same threshold so the retry loop
  // is responsible for covering them, not the seller.
  const upgradeLine = upgradeKws.length >= 3
    ? `\n🟡 MANDATORY #3 — these UPGRADE keywords already drive your bullets' search traffic but are MISSING from your live title. Amazon weights title keywords 3-5× more than bullets — folding them into the title is your single highest-leverage SEO move. Include AT LEAST ${Math.max(3, upgradeKws.length - 2)} of these (more is better, fit as many as the budget allows):\n  ${upgradeKws.map((k) => `"${k}"`).join(', ')}\n`
    : upgradeKws.length > 0
      ? `\nTry to include these UPGRADE keywords too (they drive bullet traffic but are missing from the title): ${upgradeKws.map((k) => `"${k}"`).join(', ')}\n`
      : ''
  const audienceLine = preferredAudience
    ? `\nAUDIENCE: end with "for ${preferredAudience}" (this product is for ${preferredAudience} — do NOT narrow it to a single gender if it says Men and Women).\n`
    : ''

  const system = `You are an Amazon SEO title writer${apparel ? ' specializing in apparel' : ''}. Write a title for the ACTUAL product described below — never reframe it as something it is not. Output ONLY the final title string — no quotes, no markdown, no explanation.`
  const user = `Brand: ${brandName}
Category: ${category}
${mustLine}${attrPinLine}${upgradeLine}
Pre-filtered keyword candidates (already de-duplicated and seasonal-stripped — use as many as fit naturally, ${apparel ? 'typically 1-2 beyond the mandatory keyword' : 'aim for 3-5 of these alongside the mandatory keyword'}):
${candidateList}
${attrLine}${audienceLine}
Write ONE product title as NATURAL, readable language — NOT dash-separated sections.
Order: ${brandName}, then the MANDATORY #1 keyword, then ${attributePin ? `the MANDATORY #2 blank-brand "${attributePin}", then an optional supporting keyphrase` : `${apparel ? 'ONE supporting keyphrase' : 'multiple supporting keyphrases/specs from above (fill the title)'}`}, then the audience. It should read like a human-written phrase.

Rules:
- FRONT-LOAD the mandatory keyword in the first ~80 characters (that's all mobile shows).
- Do NOT use " - " dashes or " | " pipes to separate sections — flow as natural language (a single comma is OK only if it genuinely reads better). Amazon indexes the title as a bag of words, so separators add nothing and only cost characters.
- ${apparel ? '80-125 characters' : 'TARGET 110-125 characters (Amazon indexes every word — use the budget; titles under 100 chars are leaving ranking on the table)'}. Title Case. ONE consistent audience (never mix kids with men/women).
- ${apparel ? 'Use the product-type word ("shirt"/"tee"/"t-shirt") AT MOST TWICE in the WHOLE title. Do NOT append "Shirt" to every keyphrase (no "Comfort Colors Shirt Vintage 90s Shirt Cool T Shirts").' : 'Name the product type concisely (once or twice total). Do NOT reframe the product as apparel / a t-shirt / "graphic tee" / clothing unless it genuinely is one.'}
- ${apparel
  ? 'Include the searchable keyphrases above when they fit. Do NOT put dry product SPECS (material, fabric, fit, weight, dye) in the title — those are not search terms.'
  : 'Include the searchable keyphrases above. **Technical/feature identifiers that shoppers actually search for ARE search terms** — include them when in the candidate pool above (e.g. UHS-I, Class 10, Bluetooth 5.0, USB-C, IP68, speed ratings like "90MB/s", capacity like "256GB"). Only EXCLUDE dry physical specs shoppers do not search (raw inch dimensions, gram weights, internal model codes).'}
- 🚫 BRAND-NAME SAFETY (Amazon Jan 2025 policy — bare brand references SUPPRESS listings): If any keyword above is a third-party brand name (e.g. Canon, Nikon, Sony, GoPro, SanDisk, Kingston, Lexar, Samsung, Apple, iPhone, Galaxy, DJI, Bose, etc. — anything that isn't your own brand "${brandName}"), use it ONLY in 'for [Brand]' or 'compatible with [Brand]' phrasing. Examples: ✓ 'for GoPro Hero', ✓ 'compatible with Canon EOS', ✗ 'GoPro SD Card' (bare reference — listing gets suppressed). Same rule for model names (iPhone 14, DSLR camera brands, etc.).
- ${apparel ? '' : 'PREFER concrete keyphrases over filler descriptors. NEVER add empty marketing words like "Durable", "Reliable", "Solution", "Premium", "High-Quality", "Versatile", "Versatile Options" — every word should be either a search term, a real product attribute shoppers type, or an essential connector. If you have budget left, add another keyphrase from the candidate pool, not filler.'}
- Must read like a human wrote it. Return ONLY the title.`

  const completion = await openai.chat.completions.create({
    model: 'gpt-4.1-mini',
    messages: [{ role: 'system', content: system }, { role: 'user', content: user }],
    temperature: 0.3,
    max_tokens: 120,
  })
  let title = (completion.choices[0]?.message?.content || '').trim().replace(/^["']+|["']+$/g, '')
  let problems = title ? validateTitle(title, brandName, mustInclude, attributePin, upgradeKws) : ['No title generated.']
  let retried = false

  // Up to 2 corrective passes — the mandatory-keyword + max-2 rules are non-negotiable.
  for (let attempt = 0; attempt < 2 && title && problems.length > 0; attempt++) {
    retried = true
    const fix = await openai.chat.completions.create({
      model: 'gpt-4.1-mini',
      messages: [
        { role: 'system', content: `You are an Amazon SEO title editor${apparel ? ' for apparel' : ''}. Output ONLY the corrected title string.` },
        { role: 'user', content: `Fix this title. Brand: ${brandName}\nTitle: ${title}\n\nProblems:\n- ${problems.join('\n- ')}\n\nWrite it as natural readable language (NO " - " dashes or pipes): ${brandName} then ${mustInclude ? `the MANDATORY keyword "${mustInclude}"` : 'the top keyphrase'}${attributePin ? ` then the blank-brand "${attributePin}"` : ''} then ${apparel ? 'an optional supporting keyphrase' : 'multiple supporting keyphrases (fill toward 110-125 chars)'}${preferredAudience ? ` then "for ${preferredAudience}"` : ''}. Front-load the mandatory keyword. ${apparel ? '80-125 chars' : 'TARGET 110-125 chars — use the budget'}. ${apparel ? 'Product-type word ("shirt"/"tee") used AT MOST twice total. ' : 'Name the product type once or twice; do NOT reframe it as apparel. Include technical search terms (UHS-I/Class N/USB-C/Bluetooth/MB-per-s/capacity/model identifiers) when present in the keyword pool — they ARE search terms. NO filler words ("Durable", "Reliable", "Solution", "Premium", "Versatile"). '}No seasonal terms. No dry physical specs shoppers don\\'t search.${apparel ? ' ONE audience.' : ''} Return ONLY the corrected title.` },
      ],
      temperature: 0.2,
      max_tokens: 120,
    })
    const corrected = (fix.choices[0]?.message?.content || '').trim().replace(/^["']+|["']+$/g, '')
    if (corrected) {
      const cp = validateTitle(corrected, brandName, mustInclude, attributePin, upgradeKws)
      if (cp.length <= problems.length) { title = corrected; problems = cp }
    }
  }

  // Compliance guarantee: brand must lead.
  if (title && brandName && !title.toLowerCase().includes(brandName.toLowerCase())) {
    const prefixed = `${brandName} ${title}`.trim()
    if (prefixed.length <= 150) { title = prefixed; problems = validateTitle(title, brandName, mustInclude, attributePin, upgradeKws) }
  }

  // Audience guarantee: never silently narrow a unisex product to one gender.
  if (preferredAudience === 'Men and Women' && title) {
    const lc = title.toLowerCase()
    if (/\bm[ae]n\b/.test(lc) && !/\bwom[ae]n\b/.test(lc)) {
      const swapped = title
        .replace(/\bfor Men\b/i, 'for Men and Women')
        .replace(/\bMen'?s\b/i, "Men's and Women's")
      if (swapped !== title && swapped.length <= 150) { title = swapped; problems = validateTitle(title, brandName, mustInclude, attributePin, upgradeKws) }
    }
  }

  // 🛟 LLM brand-safety judge — final catch-net (PR #80). Same shape as the bullets
  // judge; catches third-party brands/TMs the curated list can't enumerate. One
  // corrective rewrite if flagged. Fail-open on LLM error.
  try {
    const judged = await judgeBrandSafetyLLM(title, brandName, openai)
    if (judged.detected.length > 0) {
      const phrasesList = judged.detected.map((d) => `"${d.phrase}"`).join(', ')
      const fix = await openai.chat.completions.create({
        model: 'gpt-4.1-mini',
        messages: [
          { role: 'system', content: system },
          { role: 'user', content: `Rewrite the title to REMOVE these third-party brand/trademark references entirely: ${phrasesList}. Keep the brand "${brandName}" and the mandatory keyword${mustInclude ? ` "${mustInclude}"` : ''}. Use generic descriptors in place of any flagged term. 80-150 chars. Return ONLY the title string, no quotes or markdown.` },
        ],
        temperature: 0.2,
        max_tokens: 120,
      })
      const corrected = (fix.choices[0]?.message?.content || '').trim().replace(/^["']+|["']+$/g, '')
      if (corrected && corrected.length <= 200) {
        title = corrected
        problems = validateTitle(title, brandName, mustInclude, attributePin, upgradeKws)
      }
    }
  } catch { /* fail-open */ }

  return { title, problems, retried }
}

// ─── Stage 2 — Bullets Agent ───────────────────────────────────────────────────

async function runBulletsAgent(
  input: PipelineInput,
  finalTitle: string,
  remaining: AnalyzedKeyword[],
  attributes: string[],
  /** Top opportunity keywords (CRITICAL ∪ UPGRADE) the bullets MUST cover. Passed to the
   *  validation retry below — if 3+ are missing across all bullets, the corrective prompt
   *  fires with the missing list. Same contract as runTitleAgent's upgradeKws. */
  opportunityKws: string[] = [],
  /** Capacity tokens across the family (e.g. ['32GB','64GB','128GB']). When ≥2 distinct
   *  capacities, the validator rejects any hardcoded capacity in a broadcast bullet
   *  (PR #76 fix for B0GCF11RKL "this 128 GB SD card" bug). */
  capacityFamilyTokens: string[] = [],
): Promise<string[]> {
  const { openai, brandName, category, repTitle, children } = input
  const apparel = looksApparel(category, repTitle)
  // Capacity family detection: do any 2 children carry different capacity tokens in their SKUs?
  // If yes, the SHARED bullets must NOT hardcode a specific GB value — each variant's title
  // carries its own capacity; shared bullets that say "128GB" mislead the 32GB and 64GB rows.
  const childCaps = new Set<string>()
  for (const c of children) { const cap = capacityOf(c.sku) || capacityOf(c.title); if (cap) childCaps.add(cap) }
  const capacityFamily = !apparel && childCaps.size >= 2
  const familyCapList = [...childCaps].join(', ')
  const topKeyphrases = remaining.slice(0, 3).map((k) => k.keyword)
  const kwList = remaining.slice(0, 8).map((k) => `  - "${k.keyword}"`).join('\n')
  const topLine = topKeyphrases.length
    ? `\n🔴 TOP SEARCH KEYPHRASES — weave ONE of these into EACH of bullets 1, 2, and 3 (verbatim or lightly reworded), ONLY where it reads naturally and accurately. These are the highest-volume terms not already in the title; bullets must reinforce them for ranking cohesion:\n${topKeyphrases.map((k) => `  - "${k}"`).join('\n')}\n`
    : ''
  const attrLine = attributes.length
    ? `\nKNOWN PRODUCT ATTRIBUTES — real product facts; mention ${apparel ? 'the garment brand and material' : 'the key specs'} in ONE bullet${apparel ? ' (e.g. "comfort colors", "ring-spun cotton")' : ''}. Do NOT let specs crowd out the top keyphrases above:\n  ${attributes.join(', ')}\n`
    : ''
  const system = `You are an Amazon SEO copywriter${apparel ? ' for apparel' : ''}. Return ONLY valid JSON: {"bullets": ["b1","b2","b3","b4","b5"]}. Accuracy to the actual product is non-negotiable — never invent an audience, profession, occasion, or product type the product is not explicitly about.`
  const user = `The title is FINAL (do not change it): "${finalTitle}"

🚫 ACCURACY IS THE #1 RULE — violating it is a failure:
- ${apparel ? 'This is a GRAPHIC TEE; its design is ONLY what the title above says.' : 'This product is EXACTLY what the title above describes — do NOT reframe it as apparel, a t-shirt, "graphic tee", clothing, or "fashion" unless the title literally says so.'} Do NOT claim it is FOR a profession, role, or audience not explicitly named in the title. NEVER write "teacher", "nurse", "mom", "dad", "coach", "student", "educator", "boss", or any job/role word unless that exact word is in the title.
- A keyword being in the candidate list does NOT make it usable — SKIP any keyword that forces an inaccurate or awkward claim. Fewer-but-accurate beats more-but-wrong.
- Before returning, RE-READ each bullet: if any implies the product is for a specific job/role/occasion NOT named in the title — or reframes it as a product type it is not — REWRITE it to describe the actual product instead.
${topLine}
These are ADDITIONAL candidate keywords you MAY weave into the bullet body text (not the hook) — only when they fit naturally and accurately:
${kwList || '  (none — focus on benefits)'}
${attrLine}
- Never stuff a long-tail phrase (e.g. "later gator after while crocodile shirt") verbatim if it reads unnaturally — paraphrase or skip it.
- 🚫 BRAND-NAME SAFETY (Amazon Jan 2025 policy, Q4 2025 enforcement): If any candidate keyword above is a third-party brand name (Canon, Nikon, Sony, GoPro, SanDisk, Kingston, Lexar, Samsung, Apple, iPhone, DJI, Bose, etc. — anything that isn't your own brand), use it ONLY in 'for [Brand]', 'compatible with [Brand]', or 'works with [Brand]' phrasing. Examples: ✓ 'Compatible with GoPro Hero 11', ✗ 'Sandisk Standard Speed'. Bare third-party brand references in bullets risk listing suppression and trademark complaints. Same rule for model names (iPhone 14, EOS R5, etc.).

Rules per bullet:
- Start with a 2-3 WORD BENEFIT HOOK in ALL CAPS, then " - ", then the benefit sentence.
- The hook is a benefit (e.g. RETRO STYLE VIBES), NOT a keyword phrase.
- 80-200 characters each. Generic for ALL variants (no specific size/color).${capacityFamily ? `
- 🚫 CAPACITY: this family has MULTIPLE capacities (${familyCapList}) — each variant carries its own GB in its own TITLE. The bullets are SHARED across all variants. NEVER hardcode a specific capacity value (e.g. "128GB SD card", "128GB and 64GB capacities"). Use capacity-agnostic phrasing ("ample capacity", "available in multiple capacities", "high-capacity storage") instead. If a candidate keyword contains a specific GB number, paraphrase it without that number, or skip it.` : ''}
- Bullets 1-3 carry the top keyphrases; bullets 4-5 may focus on ${apparel ? 'material/comfort/care/gifting' : 'features/quality/use/gifting'}.
Return ONLY the JSON object.`

  const completion = await openai.chat.completions.create({
    model: 'gpt-4.1-mini',
    messages: [{ role: 'system', content: system }, { role: 'user', content: user }],
    temperature: 0.4,
    max_tokens: 1200,
    response_format: { type: 'json_object' },
  })
  const parsed = parseJsonLoose<{ bullets?: string[] }>(completion.choices[0]?.message?.content || '{}')
  let bullets = Array.isArray(parsed.bullets) ? parsed.bullets.filter((b) => typeof b === 'string').map((b) => b.trim()).filter(Boolean).slice(0, 5) : []

  // Deterministic role-leak guard. The prompt forbids profession claims, but gpt-4.1-mini
  // occasionally slips ("PLAYFUL TEACHER VIBE"). Detect role words not in the title, retry
  // once with a pointed correction, then strip any residual role tokens as a hard backstop.
  const titleWords = new Set(finalTitle.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/))
  const leakedRoles = (bs: string[]): string[] => {
    const found = new Set<string>()
    for (const b of bs) for (const w of b.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/)) {
      if (ROLE_WORDS.has(w) && !titleWords.has(w)) found.add(w)
    }
    return [...found]
  }
  let leaked = leakedRoles(bullets)
  if (leaked.length > 0) {
    try {
      const fix = await openai.chat.completions.create({
        model: 'gpt-4.1-mini',
        messages: [
          { role: 'system', content: system },
          { role: 'user', content: `Your previous bullets WRONGLY implied this product is for: ${leaked.join(', ')}. It is NOT — it is "${finalTitle}". Rewrite ALL 5 bullets describing ONLY the actual product; NEVER use the words ${leaked.join(', ')} or any profession/role word. Return ONLY {"bullets":["b1","b2","b3","b4","b5"]}.` },
        ],
        temperature: 0.3,
        max_tokens: 1200,
        response_format: { type: 'json_object' },
      })
      const reparsed = parseJsonLoose<{ bullets?: string[] }>(fix.choices[0]?.message?.content || '{}')
      const rb = Array.isArray(reparsed.bullets) ? reparsed.bullets.filter((b) => typeof b === 'string').map((b) => b.trim()).filter(Boolean).slice(0, 5) : []
      if (rb.length > 0 && leakedRoles(rb).length < leaked.length) { bullets = rb; leaked = leakedRoles(rb) }
    } catch { /* keep best-so-far */ }
  }
  if (leaked.length > 0) {
    // Hard backstop: remove residual role tokens (rare) — drops the bad word, keeps the sentence.
    const roleRe = new RegExp(`\\b(?:${leaked.join('|')})\\b`, 'gi')
    bullets = bullets.map((b) => b.replace(roleRe, '').replace(/\s{2,}/g, ' ').replace(/\s+([.,!])/g, '$1').trim())
  }

  // ── Brand-safety + length + opportunity coverage retry (validateBullets) ─────
  // Same shape as runTitleAgent's corrective loop in PR #73/#74: up to 2 attempts. The
  // role-leak guard above runs first because its check is cheap and deterministic; this
  // pass costs one more LLM call only when validateBullets actually finds problems.
  if (bullets.length > 0 && brandName) {
    let bProblems = validateBullets(bullets, brandName, opportunityKws, capacityFamilyTokens)
    for (let attempt = 0; attempt < 2 && bProblems.length > 0; attempt++) {
      try {
        const capacityClause = capacityFamilyTokens.length >= 2
          ? `\n- 🚫 CAPACITY: this family has multiple capacities (${capacityFamilyTokens.join(', ')}). The bullets are SHARED across all variants — NEVER hardcode a specific GB number ("128GB", "this 128 GB SD card"). Use capacity-agnostic phrasing only.`
          : ''
        const fix = await openai.chat.completions.create({
          model: 'gpt-4.1-mini',
          messages: [
            { role: 'system', content: system },
            { role: 'user', content: `Rewrite ALL 5 bullets to fix these problems. The product is "${finalTitle}" — describe ONLY that.

Problems:
- ${bProblems.join('\n- ')}

Rules to honor on rewrite:
- Each bullet 100-200 chars, starting with a 2-3 word BENEFIT HOOK in ALL CAPS then " - ". The hook is a real BENEFIT ("HIGH-SPEED PERFORMANCE", "DURABLE DESIGN") — never a pipeline label like "CRITICAL UPGRADE" or "KEYWORD".
- Any third-party brand name (Canon/Nikon/Sony/GoPro/SanDisk/Kingston/Lexar/Samsung/Apple/iPhone/DJI/Bose etc. — anything not "${brandName}") appears ONLY as 'for [Brand]', 'compatible with [Brand]', or 'works with [Brand]'.${capacityClause}
- Weave in any missing opportunity keywords listed above where they fit naturally.
Return ONLY {"bullets":["b1","b2","b3","b4","b5"]}.` },
          ],
          temperature: 0.3,
          max_tokens: 1200,
          response_format: { type: 'json_object' },
        })
        const fp = parseJsonLoose<{ bullets?: string[] }>(fix.choices[0]?.message?.content || '{}')
        const fb = Array.isArray(fp.bullets) ? fp.bullets.filter((b) => typeof b === 'string').map((b) => b.trim()).filter(Boolean).slice(0, 5) : []
        if (fb.length === 0) break
        const fbProblems = validateBullets(fb, brandName, opportunityKws, capacityFamilyTokens)
        // Accept criteria (PR #79 strictened after live audit found the loose count-only
        // check kept original bullets when rewrite traded one issue type for another):
        //   - take the rewrite when total count drops, OR
        //   - take the rewrite when CRITICAL-class problems (suppression / trademark /
        //     capacity-family) strictly decrease — even at the cost of more lower-tier
        //     issues (length, hook). Critical violations are the seller-visible legal
        //     risk; lower-tier are polish.
        const criticalCount = (ps: string[]) => ps.filter((p) =>
          /LISTING-SUPPRESSION|TRADEMARK INFRINGEMENT|CAPACITY-FAMILY VIOLATION/.test(p)
        ).length
        const prevCrit = criticalCount(bProblems)
        const newCrit = criticalCount(fbProblems)
        if (newCrit < prevCrit || (newCrit === prevCrit && fbProblems.length < bProblems.length)) {
          bullets = fb; bProblems = fbProblems
        } else break
      } catch { break /* keep best-so-far */ }
    }

    // 🛟 Programmatic capacity backstop (PR #79). The agent retry sometimes keeps a "this
    // 128 GB SD card" string in a broadcast bullet (live-verified on B0GCF11RKL bullet 3
    // even after the validator flagged it). If capacity tokens are still present after
    // the retry, strip them deterministically — better awkward phrasing than a 32GB
    // shopper reading "this 128 GB SD card" on their PDP.
    if (capacityFamilyTokens.length >= 2) {
      const capRe = /\b\d{1,4}\s?(?:GB|TB|MB)\b/gi
      bullets = bullets.map((b) =>
        b
          // Strip "this <N>GB <noun>" → "this <noun>" (common pattern).
          .replace(/\bthis\s+\d{1,4}\s?(?:GB|TB|MB)\s+/gi, 'this ')
          // Any other "<N>GB sd card alternative" → "sd card alternative".
          .replace(/\b\d{1,4}\s?(?:GB|TB|MB)\s+(sd\s+card|memory\s+card|micro\s*sd)/gi, '$1')
          // Catch-all: any remaining capacity token → "ample capacity".
          .replace(capRe, 'ample capacity')
          // Tidy double spaces left behind.
          .replace(/\s{2,}/g, ' ')
          .trim()
      )
    }

    // 🛟 Role-leak final pass (PR #79). The pre-validation role-leak strip runs once
    // before the brand-safety/coverage retry loop; the retry can REINTRODUCE role
    // words (live-verified: bullet 2 of B0G884ZJ27 said "later gator teacher shirt"
    // after the brand retry). Strip any residual role words against the final title.
    {
      const finalTitleWords = new Set(finalTitle.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/))
      const residual = new Set<string>()
      for (const b of bullets) for (const w of b.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/)) {
        if (ROLE_WORDS.has(w) && !finalTitleWords.has(w)) residual.add(w)
      }
      if (residual.size > 0) {
        const roleRe = new RegExp(`\\b(?:${[...residual].join('|')})\\b`, 'gi')
        bullets = bullets.map((b) => b.replace(roleRe, '').replace(/\s{2,}/g, ' ').replace(/\s+([.,!])/g, '$1').trim())
      }
    }

    // 🛟 LLM brand-safety judge — final catch-net (PR #80, hybrid with curated list).
    // After all deterministic checks pass, an LLM judges the bullets for third-party
    // brand/trademark refs the curated TRADEMARK_PHRASES list can't enumerate
    // (Ripcurl, Homelander, Spaceballs, Iration etc. — all live-verified leaks on
    // B0G884ZJ27). One corrective rewrite if flagged; fail-open on LLM error.
    try {
      const joined = bullets.map((b, i) => `${i + 1}. ${b}`).join('\n')
      const judged = await judgeBrandSafetyLLM(joined, brandName, openai)
      if (judged.detected.length > 0) {
        const phrasesList = judged.detected.map((d) => `"${d.phrase}"`).join(', ')
        const fix = await openai.chat.completions.create({
          model: 'gpt-4.1-mini',
          messages: [
            { role: 'system', content: system },
            { role: 'user', content: `Rewrite ALL 5 bullets to REMOVE these third-party brand/trademark references entirely: ${phrasesList}. The product is "${finalTitle}" — describe ONLY that with generic descriptors. Keep the 2-3 word ALL-CAPS BENEFIT HOOK + " - " format. Each bullet 100-200 chars. Replace any brand reference with a generic descriptor (e.g. "Ripcurl design" → "bold graphic design", "Homelander shirt" → "superhero-style shirt"). Return ONLY {"bullets":["b1","b2","b3","b4","b5"]}.` },
          ],
          temperature: 0.3,
          max_tokens: 1200,
          response_format: { type: 'json_object' },
        })
        const fp = parseJsonLoose<{ bullets?: string[] }>(fix.choices[0]?.message?.content || '{}')
        const fb = Array.isArray(fp.bullets) ? fp.bullets.filter((b) => typeof b === 'string').map((b) => b.trim()).filter(Boolean).slice(0, 5) : []
        if (fb.length === 5) bullets = fb
      }
    } catch { /* fail-open — keep best-so-far */ }
  }

  return bullets
}

// ─── Stage 3 — Backend keywords (code grouping + LLM aesthetic assignment) ──────

async function runBackendAgent(
  input: PipelineInput,
  finalTitle: string,
  bullets: string[],
  remaining: AnalyzedKeyword[],
): Promise<PipelinePerChildKeywords[]> {
  const { openai, children, brandName, category, repTitle } = input
  const apparel = looksApparel(category, repTitle)

  // Words already in title/bullets/brand — Amazon auto-indexes those, so exclude from backend.
  const excludeWords = new Set(
    `${finalTitle} ${bullets.join(' ')} ${brandName}`.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter(Boolean),
  )
  // Color names are auto-indexed from the variant attribute — never repeat them in backend.
  const colors = [...new Set(children.map((c) => (c.color || 'default').toLowerCase()))]
  colors.forEach((c) => excludeWords.add(c))
  // Title-only word set. The role-word exception ("keep 'teacher' only if this IS a teacher
  // product") must check the TITLE, not bullets — a bullet that wrongly slips "teacher" must
  // not license it back into the backend.
  const titleWords = new Set(finalTitle.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter(Boolean))

  // ── SHARED CORE: the product's biggest-opportunity keywords, intelligently de-duped ──
  // The core IS the bulk of the 250 bytes and carries the TOP opportunity-score keywords
  // (the pool is opportunity-sorted). Each meaningful word appears once (no "cool … cool"
  // repeats); JUNK, weak-relevance ROLE words ("teacher"), and WRONG-AUDIENCE words (kids/
  // youth/toddler — unless the product is actually for them, i.e. the word is in the title) are
  // dropped; the product-type word is capped at 2; minor words survive only as connectors. The
  // product's audience tokens (men/women, from the title) are GUARANTEED present.
  const kidsWords = new Set(KIDS_AUDIENCE)
  // Backend keywords are space-separated tokens — you CAN'T express "for [Brand]" framing
  // in that format. Bare third-party brand tokens in backend draw the same Amazon
  // trademark-complaint risk as in titles. Drop them entirely. The seller's compatibility
  // signal still lands via the title and bullets ("for GoPro Hero 11") which Amazon DOES
  // index. (Jan 2025 policy + Q4 2025 enforcement.)
  const ownBrandsForBackend = ownBrandTokenSet(brandName)
  const corePhrases: string[] = []
  const coreWordSet = new Set<string>()
  let productTypeCount = 0
  for (const k of remaining) {
    const raw = k.keyword.toLowerCase().replace(/[^a-z0-9]+/g, ' ').replace(/\s+/g, ' ').trim()
    if (!raw || isAllJunk(raw)) continue
    const toks: ({ w: string; minor: boolean } | null)[] = []
    for (const w of raw.split(' ')) {
      if (JUNK_WORDS.has(w)) { toks.push(null); continue }
      if (ROLE_WORDS.has(w) && !titleWords.has(w)) { toks.push(null); continue }            // weak-relevance role
      if (kidsWords.has(w) && !titleWords.has(w)) { toks.push(null); continue }             // wrong audience (kids)
      if (THIRD_PARTY_BRANDS.has(w) && !ownBrandsForBackend.has(w)) { toks.push(null); continue }  // 3P brand: trademark risk in backend
      if (MINOR_WORDS.has(w)) { toks.push({ w, minor: true }); continue }
      if (PRODUCT_TYPE_WORDS.has(w)) {
        if (productTypeCount >= 2) { toks.push(null); continue }
        productTypeCount++; toks.push({ w, minor: false }); continue
      }
      if (coreWordSet.has(w)) { toks.push(null); continue }            // content word already placed — drop repeat
      coreWordSet.add(w); toks.push({ w, minor: false })
    }
    // Keep a minor word only when it connects two surviving content words (no orphan/leading/trailing).
    const out: string[] = []
    for (let i = 0; i < toks.length; i++) {
      const t = toks[i]
      if (!t) continue
      if (t.minor) {
        const prevContent = out.length > 0 && !MINOR_WORDS.has(out[out.length - 1])
        let nextContent = false
        for (let j = i + 1; j < toks.length; j++) { const n = toks[j]; if (n) { nextContent = !n.minor; break } }
        if (prevContent && nextContent) out.push(t.w)
      } else out.push(t.w)
    }
    if (out.length === 0 || out.every((w) => MINOR_WORDS.has(w))) continue
    corePhrases.push(out.join(' '))
    if (getByteLength(corePhrases.join(' ')) >= 215) break
  }
  // Guarantee the product's audience tokens (PO wants Men AND Women in the backend).
  for (const a of ['men', 'women']) {
    if (titleWords.has(a) && !coreWordSet.has(a)) { corePhrases.push(a); coreWordSet.add(a) }
  }
  // FILL: a small product's opportunity pool can run dry well under 250 bytes, leaving the
  // search-term field half-empty (PO: "keywords are 150 chars"). Top it up with LLM long-tail
  // BUYER search words (gifts / occasions / recipients / themes) — run through the SAME junk /
  // role / kids / dedup filters as the core, so it fills with real terms, not rejected junk.
  if (getByteLength(corePhrases.join(' ')) < 205) {
    try {
      const fillSys = 'You generate ADDITIONAL Amazon backend search keywords (long-tail buyer phrases) to fill the search-term field. Return ONLY JSON: {"keywords":"lowercase space-separated search words"}.'
      const fillUsr = `Product: ${finalTitle}
List ~25 ADDITIONAL real search terms a shopper would TYPE to find this product — gift occasions, recipients, styles, themes, related concepts (e.g. "fathers day gift", "summer vacation tee", "novelty graphic", "animal lover gift", "back to school").
ONLY real buyer search words. NO brand, NO color names, NO sizes, NO moods/adjectives ("elegant", "timeless", "premium", "cozy"). lowercase, space-separated, no commas/quotes.
Avoid reusing: ${[...coreWordSet, ...titleWords].slice(0, 60).join(' ')}
Return ONLY the JSON.`
      const fc = await openai.chat.completions.create({
        model: 'gpt-4.1-mini',
        messages: [{ role: 'system', content: fillSys }, { role: 'user', content: fillUsr }],
        temperature: 0.6,
        max_tokens: 300,
        response_format: { type: 'json_object' },
      })
      const fillParsed = parseJsonLoose<{ keywords?: string }>(fc.choices[0]?.message?.content || '{}')
      const fillOut: string[] = []
      for (const w of (fillParsed.keywords || '').toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/)) {
        if (!w || JUNK_WORDS.has(w) || MINOR_WORDS.has(w)) continue
        if (ROLE_WORDS.has(w) && !titleWords.has(w)) continue          // weak-relevance role
        if (kidsWords.has(w) && !titleWords.has(w)) continue           // wrong audience
        if (THIRD_PARTY_BRANDS.has(w) && !ownBrandsForBackend.has(w)) continue  // 3P brand: trademark risk
        if (coreWordSet.has(w) || excludeWords.has(w)) continue        // already covered / auto-indexed
        if (PRODUCT_TYPE_WORDS.has(w)) { if (productTypeCount >= 2) continue; productTypeCount++ }
        coreWordSet.add(w); fillOut.push(w)
        if (getByteLength([...corePhrases, fillOut.join(' ')].join(' ')) >= 224) break
      }
      if (fillOut.length) corePhrases.push(fillOut.join(' '))
    } catch { /* fill is best-effort; the opportunity core still ships */ }
  }
  // The core is the opportunity keywords + long-tail fill — most of the 250 bytes (NOT colors).
  const core = truncateToBytes(corePhrases.join(' '), 228)

  // ── PER-COLOR TAIL: just the 2-3 top shade synonyms for THIS variant's color (not 10) ──
  const system = 'You generate a SHORT Amazon backend color tail per color variant. Return ONLY valid JSON: {"groups":[{"color":"<color>","keywords":"2-3 lowercase color words"}]}.'
  const user = `Color variants: ${colors.join(', ')}

For EACH color, output ONLY the 2-3 MOST-SEARCHED shade synonyms a buyer would type — no more than 3. Examples:
  light green -> sage olive
  ivory -> cream off white
  pepper -> charcoal heather
Use ONLY real color/shade SEARCH words — NEVER moods/feelings ("serene", "calm", "whimsical", "elegant", "timeless"). Max 3 words per color.

Do NOT use any of these words (already covered in title/bullets/core/color names):
${[...excludeWords].slice(0, 50).join(' ')}

Rules: lowercase, space-separated, NO commas, NO quotes, no brand or size words, 2-3 words ONLY.
Return ONLY the JSON object.`

  const tailMap = new Map<string, string>()
  if (apparel) try {  // color shade synonyms are an apparel concept — skip for non-apparel
    const completion = await openai.chat.completions.create({
      model: 'gpt-4.1-mini',
      messages: [{ role: 'system', content: system }, { role: 'user', content: user }],
      temperature: 0.5,
      max_tokens: 800,
      response_format: { type: 'json_object' },
    })
    const parsed = parseJsonLoose<{ groups?: { color: string; keywords: string }[] }>(completion.choices[0]?.message?.content || '{}')
    for (const g of parsed.groups ?? []) {
      if (g?.color && typeof g.keywords === 'string') tailMap.set(g.color.toLowerCase(), g.keywords.trim())
    }
  } catch {
    /* tail is best-effort; core still ships */
  }

  // ── Per-child CAPACITY awareness ──
  // For a capacity variation family (e.g. SD cards 64/128/256GB), every child currently received
  // the IDENTICAL core, leading with whichever capacity ranked highest in the keyword pool
  // (usually 128GB). That means the 32G SKU's backend says "128 gb sd card..." first — Amazon
  // ranks it for the wrong capacity. Per Amazon SEO research, each child's backend should
  // emphasize ITS OWN capacity. Detect distinct child capacities and, when present, build a
  // per-child core: strip OTHER capacity tokens, prepend the child's own capacity.
  const backendChildCaps = new Map<string, string>()
  const liveCaps = new Set<string>()
  for (const c of children) {
    const cap = capacityOf(c.sku) || capacityOf(c.title)
    if (cap) { backendChildCaps.set(c.sku, cap); liveCaps.add(cap) }
  }
  const backendCapacityFamily = !apparel && liveCaps.size >= 2

  // Build a regex that matches ANY GB/TB token (e.g. "128 gb", "128gb", "1tb"). Used to strip
  // wrong-capacity tokens from the broadcast core before re-prepending the child's own.
  const anyCapRe = /\b\d{1,4}\s?(?:tb|gb|mb)\b/gi

  // ── Combine: opportunity-keyword core + AT MOST 3 net-new color words, byte-cap to 250 ──
  const buildString = (tail: string, customCore?: string): string => {
    const effectiveCore = customCore ?? core
    const effectiveCoreWords = customCore ? new Set(effectiveCore.toLowerCase().split(/\s+/).filter(Boolean)) : new Set(core.toLowerCase().split(/\s+/).filter(Boolean))
    const tailWords = tail.toLowerCase().split(/\s+/)
      .filter((w) => w && !effectiveCoreWords.has(w) && !excludeWords.has(w) && !MINOR_WORDS.has(w))
      .slice(0, 3)   // at most 3 color words — the PO does NOT want 10 color synonyms
    return truncateToBytes(`${effectiveCore} ${tailWords.join(' ')}`.trim(), 250)
  }

  return children.map((c) => {
    const color = (c.color || 'default').toLowerCase()
    let childCore: string | undefined
    if (backendCapacityFamily) {
      const childCap = backendChildCaps.get(c.sku)
      if (childCap) {
        const childCapLc = childCap.toLowerCase()                  // e.g. "32gb"
        const childCapSpaced = childCapLc.replace(/(\d+)([a-z]+)/, '$1 $2')  // "32 gb"
        // 1) Strip every capacity token in the core (we'll re-add this child's first).
        let stripped = core.replace(anyCapRe, ' ').replace(/\s{2,}/g, ' ').trim()
        // 2) Prepend this child's own capacity (twice — both unspaced and spaced — so it ranks
        //    for "32gb" and "32 gb" search variants without burning much budget).
        stripped = `${childCapSpaced} ${childCapLc} ${stripped}`.replace(/\s{2,}/g, ' ').trim()
        childCore = truncateToBytes(stripped, 228)
      }
    }
    return { sku: c.sku, asin: c.asin, keywords: buildString(tailMap.get(color) || '', childCore) }
  })
}

// ─── Stage 4 — Audit Agent (o4-mini, reasoning) ────────────────────────────────

interface AuditResult {
  variant_corrections?: PipelineVariantCorrection[]
  cannibalization_warnings?: PipelineCannibalizationWarning[]
  product_details_improvements?: PipelineProductDetailImprovement[]
  keyword_reconciliation?: PipelineKeywordReconciliation[]
  action_plan?: PipelineActionPlanItem[]
}

async function runAuditAgent(
  input: PipelineInput,
  finalTitle: string,
  bullets: string[],
  perChild: PipelinePerChildKeywords[],
  description: string,
  specs: string[],
): Promise<AuditResult> {
  const { openai, auditModel, variantDetails, keywordContext, hasAplus, category, repTitle } = input
  const apparel = looksApparel(category, repTitle)
  const backendSummary = perChild.slice(0, 3).map((p) => `  ${p.sku}: ${p.keywords}`).join('\n')
  const specsLine = specs.length
    ? `\n=== KNOWN PRODUCT SPECS (use these to fill structured Product-Detail fields with REAL values — e.g. ${apparel ? 'Fabric Type, Material, Fit Type, Department' : 'Material, Capacity, Compatibility, Item Dimensions'}) ===\n${specs.join(', ')}\n`
    : ''

  const system = `You are a senior Amazon SEO auditor. Return ONLY valid JSON.
The recommended title, bullets, backend keywords, and description below are ALREADY FINALIZED — do NOT rewrite them. Build the action plan and reconciliation AROUND them.`

  const user = `=== FINALIZED CONTENT (do not change) ===
TITLE: ${finalTitle}
BULLETS:
${bullets.map((b, i) => `  ${i + 1}. ${b}`).join('\n')}
BACKEND (sample of per-child):
${backendSummary}
DESCRIPTION: ${description ? 'provided (HTML)' : '(none)'}
A+ exists: ${hasAplus}
${specsLine}
=== CURRENT LISTING (for variant health + product details) ===
${variantDetails}

=== KEYWORD INTELLIGENCE ===
${keywordContext}

Produce a JSON object with these keys:
{
  "action_plan": [ { "element": "title|bullet_1..5|backend_keywords|description|aplus_modules|brand_story|product_details|images", "level": "parent|per_child", "verdict": "REPLACE|EDIT|CREATE|DONE|SKIP", "priority": "HIGH|MEDIUM|LOW|NONE", "current_status": "...", "instruction": "...", "replacement_content": "string|null", "seller_central_path": "...", "notes": "...", "aplus_modules": [ { "module_type": "...", "action": "ADD|EDIT|KEEP", "content_brief": "...", "position": 1 } ] } ],
  "keyword_reconciliation": [ { "keyword": "...", "action_type": "CRITICAL|UPGRADE|REINFORCE", "search_volume": 0, "placed_in": ["title|bullet_1..5|backend_keywords"], "exact_text": "...", "why": "..." } ],
  "variant_corrections": [ { "sku": "...", "field": "title|bullets|keywords|description", "current": "...", "replace_with": "...", "reason": "..." } ],
  "cannibalization_warnings": [ { "keyword": "...", "affected_skus": ["..."], "issue": "...", "recommendation": "..." } ],
  "product_details_improvements": [ { "field_name": "...", "current_value": "string|null", "recommended_value": "...", "reason": "..." } ]
}

Rules:
- Review EVERY element in the action plan (title, bullet_1..5, backend_keywords, description, aplus_modules, brand_story, product_details, images). For title/bullets/backend/description, the replacement_content is the FINALIZED content above — restate it, do not invent new copy.
- DESCRIPTION: even if A+ exists, the field is still indexed for search — mark CREATE/EDIT (not SKIP) and note that customers see A+ but Amazon indexes this field.
- CANNIBALIZATION: children in ONE variation family do NOT compete in search. Leave cannibalization_warnings empty unless the SAME backend string is duplicated identically across many children. Never report cross-listing cannibalization (not assessable here).
- PRODUCT DETAILS: suggest only genuinely missing structured attributes appropriate to THIS product type (${apparel ? 'e.g. Fabric Type, Material, Fit Type, Department' : 'e.g. Material, Capacity, Compatibility, Dimensions, Special Features — NOT apparel fields like Fabric Weight or Fit Type'}). 3-10 max.
- A+ modules: more modules lift conversion and dwell time; A+ body text is not a confirmed ranking field, so recommend filling image ALT-TEXT for discoverability.
Return ONLY the JSON object.`

  const completion = await openai.chat.completions.create({
    model: auditModel,
    messages: [{ role: 'system', content: system }, { role: 'user', content: user }],
    max_completion_tokens: 8000,
    response_format: { type: 'json_object' },
  })
  return parseJsonLoose<AuditResult>(completion.choices[0]?.message?.content || '{}')
}

// ─── Description (code-triggered LLM, always generated — field is indexed) ──────

async function runDescriptionAgent(input: PipelineInput, finalTitle: string, bullets: string[], attributes: string[]): Promise<string> {
  const { openai, category, repTitle, children } = input
  const apparel = looksApparel(category, repTitle)
  // Capacity-family detection (mirrors bullets): shared description must NOT hardcode a
  // capacity that doesn't match every variant.
  const descChildCaps = new Set<string>()
  for (const c of children) { const cap = capacityOf(c.sku) || capacityOf(c.title); if (cap) descChildCaps.add(cap) }
  const descCapacityFamily = !apparel && descChildCaps.size >= 2
  const descFamilyCapList = [...descChildCaps].join(', ')
  const attrLine = attributes.length
    ? `\nNaturally mention these known product attributes (real facts from the listing${apparel ? ' — e.g. garment brand, material, fit' : ''}): ${attributes.join(', ')}.`
    : ''
  const system = `You are an Amazon SEO copywriter${apparel ? ' for apparel' : ''}. Return ONLY the HTML description (no markdown, no JSON). Describe ONLY the actual product — never invent an audience, profession, occasion, or product type the product is not explicitly about.`
  const user = `Write a SUBSTANTIAL 270-330 word HTML product description (generic for all variants) using <p>, <b>, <ul>, <li>. Use most of Amazon's ~2000-character budget — do NOT write a short blurb; expand on ${apparel ? 'the design, materials, fit, styling, and use cases' : "the product's features, specs, quality, and use cases"}.
Title: ${finalTitle}
Bullet themes: ${bullets.map((b) => b.split(' - ')[0]).join(', ')}${attrLine}

🚫 ACCURACY: describe ONLY what the title says this product is${apparel ? '' : ' — do NOT reframe it as apparel / a t-shirt / clothing unless it genuinely is one'}. Do NOT claim it is for a profession/role/occasion not named in the title — never write "teacher", "nurse", "mom", "educator", "coach", etc. unless that word is in the title. If a bullet theme above implies such a claim, ignore that theme and describe the actual product instead.

🚫 BRAND-NAME SAFETY (Amazon Jan 2025 policy): any third-party brand name (Canon, Nikon, Sony, GoPro, SanDisk, Kingston, Lexar, Samsung, Apple, iPhone, DJI, Bose, etc. — anything not your own brand) appears ONLY in 'for [Brand]', 'compatible with [Brand]', or 'works with [Brand]' phrasing. Examples: ✓ 'compatible with Canon EOS R5 and Sony Alpha cameras', ✗ 'Sandisk-quality storage' (bare brand reference — risks listing suppression and trademark complaints).${descCapacityFamily ? `

🚫 CAPACITY: this family has MULTIPLE capacities (${descFamilyCapList}). The description is SHARED across all variants — NEVER hardcode a specific GB number in any paragraph or bullet (no "128GB and 64GB capacities", no "this 128GB SD card", no "Available in 128GB and 64GB"). Use capacity-agnostic phrasing: "available in multiple capacities", "high-capacity storage", "ample space for your needs". The capacity-specific text already lives in each variant's TITLE.` : ''}

Structure: hook -> <ul> of key features -> use cases/audience -> short closing line. Return ONLY the HTML.`
  const completion = await openai.chat.completions.create({
    model: 'gpt-4.1-mini',
    messages: [{ role: 'system', content: system }, { role: 'user', content: user }],
    temperature: 0.5,
    max_tokens: 1200,
  })
  let description = (completion.choices[0]?.message?.content || '').replace(/^```html\s*/i, '').replace(/\s*```$/i, '').trim()

  // ── Brand-safety + length retry (validateDescription) ─────────────────────────
  // Same shape as runTitleAgent / runBulletsAgent: up to 2 corrective passes. Closes the
  // last surface in PR #75 — title (#74), bullets (above), and now description all share
  // the parent/child coverage standard for validate+retry.
  const { brandName: descBrand } = input
  if (description && descBrand) {
    let dProblems = validateDescription(description, descBrand)
    for (let attempt = 0; attempt < 2 && dProblems.length > 0; attempt++) {
      try {
        const fix = await openai.chat.completions.create({
          model: 'gpt-4.1-mini',
          messages: [
            { role: 'system', content: system },
            { role: 'user', content: `Rewrite the description to fix these problems. The product is "${finalTitle}" — describe ONLY that.

Problems:
- ${dProblems.join('\n- ')}

Rules to honor on rewrite:
- 270-330 words HTML using <p>, <b>, <ul>, <li>.
- Any third-party brand name (Canon/Nikon/Sony/GoPro/SanDisk/Kingston/Lexar/Samsung/Apple/iPhone/DJI/Bose etc. — anything not "${descBrand}") appears ONLY as 'for [Brand]', 'compatible with [Brand]', or 'works with [Brand]'.
- Return ONLY the HTML.` },
          ],
          temperature: 0.4,
          max_tokens: 1200,
        })
        const corrected = (fix.choices[0]?.message?.content || '').replace(/^```html\s*/i, '').replace(/\s*```$/i, '').trim()
        if (!corrected) break
        const cdProblems = validateDescription(corrected, descBrand)
        if (cdProblems.length < dProblems.length) { description = corrected; dProblems = cdProblems }
        else break
      } catch { break /* keep best-so-far */ }
    }

    // 🛟 LLM brand-safety judge — final catch-net (PR #80). Strip HTML for the judge
    // (don't ask it to reason about markup); rewrite if flagged.
    try {
      const plainForJudge = description.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim()
      const judged = await judgeBrandSafetyLLM(plainForJudge, descBrand, openai)
      if (judged.detected.length > 0) {
        const phrasesList = judged.detected.map((d) => `"${d.phrase}"`).join(', ')
        const fix = await openai.chat.completions.create({
          model: 'gpt-4.1-mini',
          messages: [
            { role: 'system', content: system },
            { role: 'user', content: `Rewrite the HTML description to REMOVE these third-party brand/trademark references entirely: ${phrasesList}. The product is "${finalTitle}" — describe ONLY that with generic descriptors. 270-330 words HTML using <p>, <b>, <ul>, <li>. Return ONLY the HTML.` },
          ],
          temperature: 0.4,
          max_tokens: 1200,
        })
        const corrected = (fix.choices[0]?.message?.content || '').replace(/^```html\s*/i, '').replace(/\s*```$/i, '').trim()
        if (corrected) description = corrected
      }
    } catch { /* fail-open */ }
  }

  return description
}

/**
 * LLM brand-safety judge (PR #80, hybrid with curated list).
 *
 * The curated TRADEMARK_PHRASES + token-proximity check (PR #77/#79) catches the
 * enumerated marks. But the keyword pool has thousands of brand/trademark queries
 * we'll never exhaustively list — live-verified B0G884ZJ27 regen leaked Ripcurl /
 * Homelander / Spaceballs / Ella Fella / Iration (band) into title + bullets, all
 * outside our set. This judge is the catch-net: one cheap LLM pass per agent output
 * that flags any third-party brand / trademark / proper-noun reference, with the
 * seller's own brand exempt.
 *
 * Returns `{ detected: [{phrase, reason}] }`. Conservative — better false-positive
 * than miss a TM. Fail-open: on error returns empty so the agent ships its current
 * output (don't block a regen on a transient LLM failure).
 */
export interface BrandSafetyFinding { phrase: string; reason: string }
export async function judgeBrandSafetyLLM(
  text: string,
  brandName: string,
  openai: OpenAI,
): Promise<{ detected: BrandSafetyFinding[] }> {
  if (!text || !text.trim()) return { detected: [] }
  // PR #81: judge model is env-configurable. Default 'gpt-5' (much better recall on
  // proper-noun classification than gpt-4.1-mini, which returned empty findings on
  // live B0G884ZJ27 even with Homelander/Ripcurl/Iration listed as examples). Manus
  // must enable GPT-5 access on the OpenAI org for the default to work; otherwise
  // override via BRAND_SAFETY_JUDGE_MODEL env var. A runtime failure here is logged
  // (so Coolify shows whether it's an access issue vs a prompt issue) and fails open.
  const judgeModel = process.env.BRAND_SAFETY_JUDGE_MODEL || 'gpt-5'
  const system = `You are a STRICT Amazon trademark-safety judge. Find every third-party brand, registered trademark, sports team, university, media franchise, character name, band/musician, or other proper-noun reference that this seller cannot legally use without a license.

Rules:
1. Flag ANY proper-noun phrase that COULD be a registered mark — band names, movies, TV characters, sports teams, clothing/electronics brands, university nicknames, video-game IPs.
2. If you do not recognize a capitalized phrase AND cannot confirm it is generic English, you MUST flag it.
3. NEVER flag generic English words (alligator, lion, gator, eagle, cool, vintage, classic, retro).
4. NEVER flag the seller's own brand "${brandName}".
5. NEVER flag pure descriptors (XL, Black, JPEG).
6. When uncertain, FLAG. False positives are recoverable; missed trademarks cause listing suppression.

Return ONLY {"detected":[{"phrase":"<exact substring>","reason":"<one line>"}]}. Empty array only if certain there is no trademarked reference.`
  const user = `Seller brand: ${brandName}

Text to review:
"""
${text.slice(0, 1800)}
"""

Identify EVERY third-party brand/trademark/proper-noun reference. Be paranoid.

Items that MUST be flagged when present:
- Canon, Sony, GoPro, SanDisk, Lexar (electronics)
- Marvel, Disney, Pixar, Star Wars, Harry Potter (media)
- Homelander, Iron Man, Spider-Man (characters)
- Ripcurl, Rip Curl, Quiksilver, Hurley (apparel brands)
- Iration, Sublime, Metallica (musicians)
- Ella Fella, Janie & Jack, Mini Boden (kids brands)
- Florida Gators, Dallas Cowboys (sports teams)
- iPhone, PlayStation, Xbox (product names)

If you see a CAPITALIZED phrase that is NOT obvious generic English AND NOT "${brandName}", flag it.

Return ONLY the JSON object.`
  try {
    const r = await openai.chat.completions.create({
      model: 'gpt-5',
      messages: [{ role: 'system', content: system }, { role: 'user', content: user }],
      max_completion_tokens: 600,
      response_format: { type: 'json_object' },
    })
    const parsed = parseJsonLoose<{ detected?: BrandSafetyFinding[] }>(r.choices[0]?.message?.content || '{}')
    const detected = Array.isArray(parsed.detected) ? parsed.detected
      .filter((f) => f && typeof f.phrase === 'string' && f.phrase.trim().length > 0)
      .slice(0, 15) : []
    return { detected }
  } catch {
    return { detected: [] }
  }
}

// ─── Stage 0b — Relevance gate (PR17) ─────────────────────────────────────────
// Jungle Scout / competitor pulls contaminate the keyword set with terms that are
// not about THIS product (other brands, trademarks, bands, character/personal names).
// Left unfiltered, they leak into the backend core (code-built) and reconciliation —
// and pushing a trademark like "florida gators" to live listings is a TOS risk.
// One cheap classifier pass drops clearly-unrelated terms. Conservative + fail-open.
async function filterRelevantKeywords(input: PipelineInput, analysis: AnalyzedKeyword[]): Promise<AnalyzedKeyword[]> {
  if (analysis.length === 0) return analysis
  const { openai, brandName, repTitle, category } = input
  const list = analysis.map((k, i) => `${i}: ${k.keyword}`).join('\n')
  const system = 'You are an Amazon SEO relevance filter. Return ONLY valid JSON: {"drop":[<indices to drop>]}.'
  const user = `Product brand: ${brandName}
Category: ${category}
Current product title (context only): ${repTitle ?? '(unknown)'}

Keywords (index: phrase):
${list}

Return the indices of keywords to DROP:
1. Other companies' brands or TRADEMARKS (sports teams, bands, other sellers), unrelated products, or personal/character names with no connection to this product.
2. VAGUE, non-descriptive filler that does not describe a product attribute, style, design, audience, occasion, or use case (e.g. "interest", "full transparency", "high quality", "best seller").
KEEP anything plausibly about this product, including broad descriptors, audiences, occasions, gift terms, and seasonal terms (relevant even when broad). Be CONSERVATIVE — only drop clearly-unrelated or clearly-meaningless terms. Return ONLY {"drop":[...]}.`
  // Deterministic backstops: ALWAYS drop these regardless of the LLM gate, which is
  // non-deterministic and let them through in live testing.
  //   1. all-junk keywords ("interest", "full transparency", "best seller")
  //   2. trademark phrases (sports teams, universities, media franchises) — PR #77
  //      after live B0G884ZJ27 audit leaked "Florida Gators" into a recommended title.
  //      Generic team-mascot words ("alligator", "gators", "lions") still pass through
  //      — only the multi-word REGISTERED phrases are dropped.
  const dropJunkAndTrademarks = (kws: AnalyzedKeyword[]) => kws.filter((k) => {
    if (isAllJunk(k.keyword)) return false
    if (findTrademarkPhrases(k.keyword).length > 0) return false
    return true
  })
  try {
    const completion = await openai.chat.completions.create({
      model: 'gpt-4.1-mini',
      messages: [{ role: 'system', content: system }, { role: 'user', content: user }],
      temperature: 0,
      max_tokens: 500,
      response_format: { type: 'json_object' },
    })
    const parsed = parseJsonLoose<{ drop?: number[] }>(completion.choices[0]?.message?.content || '{}')
    const drop = new Set((parsed.drop ?? []).filter((n) => Number.isInteger(n)))
    const filtered = analysis.filter((_, i) => !drop.has(i))
    // Fail-open: if the gate dropped (nearly) everything it likely misfired — keep the original pool.
    if (filtered.length < Math.max(3, Math.floor(analysis.length * 0.3))) return dropJunkAndTrademarks(analysis)
    return dropJunkAndTrademarks(filtered)
  } catch (err) {
    console.warn('[pipeline] relevance gate failed, using unfiltered pool:', err)
    return dropJunkAndTrademarks(analysis)
  }
}

// ─── Stage 0c — Known-attribute extraction (PR: known-attribute-injection) ─────
// Jungle Scout seeds the keyword pool from the IMAGE/title, so seller-known product
// FACTS that aren't search-derived — the garment/blank brand ("Comfort Colors"),
// material, fit — fall out entirely. They live in the CURRENT listing (title/bullets/
// backend) but no code path surfaces them. This reads them back from the existing
// listing text and returns them so they can be reinforced across every surface.
interface ProductAttributes {
  /** Real search terms a shopper would type — e.g. "comfort colors graphic tee". Title-eligible. */
  searchKeyphrases: string[]
  /** Specs that are NOT search terms — e.g. "garment dyed", "ring spun cotton", "relaxed fit".
   *  Go in bullets / description / structured Product-Detail fields, NEVER the title. */
  specs: string[]
}

async function extractProductAttributes(input: PipelineInput): Promise<ProductAttributes> {
  const { openai, repTitle, variantDetails, category } = input
  const apparel = looksApparel(category, repTitle)
  const text = `${repTitle ?? ''}\n${variantDetails}`.slice(0, 4000).trim()
  if (!text) return { searchKeyphrases: [], specs: [] }
  const system = `You extract product attributes from an existing Amazon${apparel ? ' apparel' : ''} listing, split into searchable keyphrases vs specs. Return ONLY valid JSON: {"searchKeyphrases":["..."],"specs":["..."]}.`
  const user = `From the listing text, extract TWO groups:

1. searchKeyphrases — terms a shopper would actually TYPE into Amazon search.${apparel ? ' In particular, if a recognizable garment BLANK BRAND is present (e.g. comfort colors, bella canvas, gildan, next level, american apparel), output it COMBINED with the product type as a real query — e.g. "comfort colors graphic tee", "comfort colors shirt".' : ' Combine the product type with its key descriptors as real buyer queries (e.g. "64gb sd card", "high speed memory card").'} 2-4 words each. These are TITLE-eligible.

2. specs — concrete product SPECIFICATIONS that nobody searches: ${apparel ? 'material/fabric, weight, fit/cut, dye method (e.g. "ring spun cotton", "6.1 oz", "garment dyed", "relaxed fit", "unisex")' : 'material, dimensions, capacity, compatibility, technical ratings (e.g. "class 10", "uhs-i", "waterproof")'}. These go in bullets/description/structured fields, NOT the title.

Only include attributes actually stated or strongly implied — do NOT invent. Lowercase. Max 4 per group.

Listing text:
${text}

Return ONLY {"searchKeyphrases":[...],"specs":[...]}.`
  const clean = (arr?: string[]) =>
    Array.isArray(arr) ? arr.filter((a) => typeof a === 'string' && a.trim()).map((a) => a.trim().toLowerCase()).slice(0, 4) : []
  try {
    const completion = await openai.chat.completions.create({
      model: 'gpt-4.1-mini',
      messages: [{ role: 'system', content: system }, { role: 'user', content: user }],
      temperature: 0,
      max_tokens: 250,
      response_format: { type: 'json_object' },
    })
    const parsed = parseJsonLoose<{ searchKeyphrases?: string[]; specs?: string[] }>(completion.choices[0]?.message?.content || '{}')
    return { searchKeyphrases: clean(parsed.searchKeyphrases), specs: clean(parsed.specs) }
  } catch (err) {
    console.warn('[pipeline] product attribute extraction failed:', err)
    return { searchKeyphrases: [], specs: [] }
  }
}

/** Turn an attribute phrase into a synthetic high-opportunity keyword so it flows through
 * the title-candidate / bullets / backend pools via the existing selection logic. */
function attributeAsKeyword(attr: string): AnalyzedKeyword {
  return {
    keyword: attr,
    // Secondary, NOT top-tier: a known attribute ("comfort colors shirt") should be
    // included in the title when it fits, but must never out-rank and crowd out the
    // highest SEARCH-VOLUME real keyword (that regression dropped "see you later
    // alligator shirt", 22.7k/mo, from the title). 35 keeps it above filler, below the
    // genuine money keywords. The orchestrator separately PINS the top-volume keyword.
    opportunityScore: 35,
    actionType: 'CRITICAL',
    actionText: '', rationale: '', urgency: 'high', estimatedImpact: '',
    searchVolume: 0, keywordSales: 0, competingProducts: 0,
    asinImpressionShare: 0, asinClickShare: 0, asinPurchaseShare: 0,
    inTitle: false, inBullets: false, inDescription: false, inBackend: false,
    dataSource: 'jungle_scout',
  } as AnalyzedKeyword
}

// ─── Orchestrator ──────────────────────────────────────────────────────────────

export async function runListingPipeline(input: PipelineInput): Promise<PipelineResult> {
  const { brandName, repTitle, onProgress } = input

  // Stage 0a — relevance gate: drop keywords that are not about this product
  // (competitor brands, trademarks, unrelated names) before anything downstream uses them.
  onProgress('Filtering keywords for relevance...')
  const gated = await filterRelevantKeywords(input, input.analysis)

  // Stage 0c — surface seller-known product attributes (garment brand "Comfort Colors",
  // material, fit) from the existing listing. JS never captures these, so without this
  // they're invisible to the optimizer. Inject as high-opportunity keywords so they flow
  // into the title-candidate / bullets / backend pools, and reinforce them in the prompts.
  onProgress('Extracting product attributes...')
  const attrs = await extractProductAttributes(input)
  const apparelProduct = looksApparel(input.category, repTitle)
  // Non-apparel listings are sometimes templated from a shirt, so their data carries apparel
  // contamination ("graphic tee", "ring-spun cotton", "for men"). Strip it for non-apparel so a
  // memory card / mug never inherits clothing language in the keyword pool, specs, or title.
  if (!apparelProduct) {
    const clean = (s: string) => !APPAREL_CONTAMINANTS.test(s)
    attrs.searchKeyphrases = attrs.searchKeyphrases.filter(clean)
    attrs.specs = attrs.specs.filter(clean)
  }
  const cleanGated = apparelProduct ? gated : gated.filter((k) => !APPAREL_CONTAMINANTS.test(k.keyword))
  // Only SEARCHABLE keyphrases (e.g. "comfort colors graphic tee") become title-eligible
  // keywords. Specs (garment-dyed, ring-spun cotton, relaxed fit) are NOT search terms and
  // must NOT enter the title — they go to bullets/description/structured fields only.
  const analysis = [...attrs.searchKeyphrases.map(attributeAsKeyword), ...cleanGated]
  const bulletAttrs = [...attrs.searchKeyphrases, ...attrs.specs]

  // PIN the single highest SEARCH-VOLUME real keyword (not a synthetic attribute, not
  // seasonal — seasonal belongs in backend) so the title agent can never drop the money
  // term. This is what stopped "see you later alligator shirt" (22.7k/mo) from surviving.
  const mustIncludeKw = cleanGated
    .filter((k) => ['CRITICAL', 'UPGRADE', 'DEFENDED', 'REINFORCE'].includes(k.actionType))
    .filter((k) => !isSeasonal(k.keyword))
    .filter((k) => k.keyword.split(/\s+/).length <= 6)
    .sort((a, b) => (b.searchVolume || 0) - (a.searchVolume || 0) || (b.opportunityScore || 0) - (a.opportunityScore || 0))[0]
  const mustInclude = mustIncludeKw?.keyword

  // Determine the product's true audience from the existing listing + specs, so the title
  // never silently narrows a unisex product to one gender (the "for Men" regression).
  // Apparel only — a memory card, mug, or mount has no gendered audience; forcing "for Men"
  // on an SD card is exactly the non-apparel mess this guards against (apparelProduct computed above).
  const audienceText = `${repTitle ?? ''} ${attrs.specs.join(' ')} ${cleanGated.map((k) => k.keyword).join(' ')}`.toLowerCase()
  const mentionsWomen = /\bwom[ae]n\b|womens|ladies|female/.test(audienceText)
  const mentionsMen = /\bm[ae]n\b|mens|male/.test(audienceText)
  const preferredAudience = !apparelProduct ? ''
    : /\bunisex\b/.test(audienceText) || (mentionsWomen && mentionsMen) ? 'Men and Women'
    : mentionsWomen ? 'Women'
    : mentionsMen ? 'Men'
    : ''

  // SECOND pin: the blank/garment brand (e.g. "comfort colors") — a strategic attribute the
  // seller ranks for. Now that the #1 money keyword is guaranteed, this can be re-elevated
  // into the title without crowding it out. Derived from the top searchKeyphrase with the
  // product-type word stripped ("comfort colors shirt" -> "comfort colors").
  let attributePin = ''
  if (apparelProduct) {
    attributePin = (attrs.searchKeyphrases[0] || '')
      .replace(/\b(graphic\s+tee|t[-\s]?shirts?|tees?|shirts?|graphic|tops?)\b/gi, '')
      .replace(/\s+/g, ' ')
      .trim()
  }
  // Drop it if empty or already fully covered by the #1 pin (avoids a redundant constraint).
  if (attributePin && mustInclude) {
    const miWords = new Set(mustInclude.toLowerCase().split(/\s+/))
    if (attributePin.toLowerCase().split(/\s+/).every((w) => miWords.has(w))) attributePin = ''
  }
  const attributePinFinal = attributePin || undefined

  // Stage 0b — candidates (code)
  const candidates = selectTitleCandidates(analysis, brandName, repTitle)

  // Stage 0c — top UPGRADE keywords for explicit title-coverage. UPGRADE = ranking
  // signal already present in bullets but absent from the title. The scorer in
  // syncListingContent.ts docks 5 points when 7+ of these are missing (3 when 3-6
  // miss). We feed them to the title agent as MANDATORY #3 and fail validation when
  // 3+ still aren't in the title, so the existing retry loop is on the hook for
  // covering them — not the seller.
  const topUpgradeKws = cleanGated
    .filter((k) => k.actionType === 'UPGRADE')
    .filter((k) => !isSeasonal(k.keyword))
    .filter((k) => k.keyword.split(/\s+/).length <= 6)  // skip long-tail phrases that wouldn't fit
    .sort((a, b) => (b.opportunityScore || 0) - (a.opportunityScore || 0))
    .slice(0, 10)                                        // matches the scorer's top-10 cap
    .map((k) => k.keyword)

  // Stage 1 — Title
  onProgress('Writing title...')
  const { title: finalTitle, problems: titleProblems, retried } = await runTitleAgent(input, candidates, attrs.searchKeyphrases, mustInclude, preferredAudience, attributePinFinal, topUpgradeKws)

  // Per-child capacity titles — ONLY for non-apparel families whose children span >=2 distinct
  // capacities (e.g. SD cards 64/128/256GB). Researched Amazon best practice: each child must
  // carry its OWN capacity in the title; broadcasting one capacity to the others risks search
  // suppression. Apparel is excluded by apparelProduct AND never matches the capacity pattern,
  // so its title stays the single shared/broadcast value untouched.
  let perChildTitles: { sku: string; asin: string; title: string }[] | undefined
  if (!apparelProduct) {
    const childCap = new Map<string, string>()
    // Capacity from the SKU first (reliable — e.g. "...-32G-FBA"); the child's CURRENT title is
    // an unreliable fallback (it can be a templated/wrong title — part of why we're optimizing).
    for (const c of input.children) { const cap = capacityOf(c.sku) || capacityOf(c.title); if (cap) childCap.set(c.sku, cap) }
    if (new Set(childCap.values()).size >= 2) {
      const baseCap = capacityOf(finalTitle)
      perChildTitles = input.children.map((c) => {
        const cap = childCap.get(c.sku)
        if (!cap) return { sku: c.sku, asin: c.asin, title: finalTitle }
        let t = finalTitle
        if (baseCap && cap !== baseCap) t = finalTitle.replace(new RegExp(`\\b${baseCap}\\b`, 'gi'), cap)
        else if (!baseCap) t = finalTitle.replace(/^(\S+\s+\S+\s+\S+)/, `$1 ${cap}`)
        return { sku: c.sku, asin: c.asin, title: truncateToBytes(t, 200) }
      })
    }
  }

  // Bullets pool (PR15): critical/upgrade/reinforce, not in title, NO seasonal (bullets are
  // customer-facing — a seasonal claim off-season misleads and mis-describes the product),
  // no awkward >5-word composites, deduped. This is the same discipline the title gets.
  const titleLc = finalTitle.toLowerCase()
  const remainingForBullets: AnalyzedKeyword[] = []
  for (const k of analysis) {
    if (!['CRITICAL', 'UPGRADE', 'REINFORCE'].includes(k.actionType)) continue
    if (titleLc.includes(k.keyword.toLowerCase())) continue
    if (isSeasonal(k.keyword)) continue
    if (k.keyword.split(/\s+/).length > 5) continue
    if (remainingForBullets.some((d) => wordOverlapRatio(d.keyword, k.keyword) >= 0.6)) continue
    remainingForBullets.push(k)
  }
  // Highest-opportunity first so bullets 1-3 reinforce the true top keyphrases.
  remainingForBullets.sort((a, b) => b.opportunityScore - a.opportunityScore)

  // Stage 2 — Bullets
  // Opportunity pool for the bullets retry validator: top CRITICAL ∪ UPGRADE keywords (same
  // discipline title gets in Stage 0c, but for bullets we keep BOTH tiers since the bullets
  // scorer penalizes when 2+ CRITICAL-or-UPGRADE keywords are missing across all 5 bullets).
  // Sorted by opportunity, deduped against title (those don't count as bullet gaps).
  const topOpportunityKwsForBullets = cleanGated
    .filter((k) => k.actionType === 'CRITICAL' || k.actionType === 'UPGRADE')
    .filter((k) => !isSeasonal(k.keyword))
    .filter((k) => k.keyword.split(/\s+/).length <= 5)
    .filter((k) => !titleLc.includes(k.keyword.toLowerCase()))   // already in title → not a bullet gap
    .sort((a, b) => (b.opportunityScore || 0) - (a.opportunityScore || 0))
    .slice(0, 10)
    .map((k) => k.keyword)
  // Capacity-family detection — passed to the bullets validator (PR #76) so the retry
  // loop rejects bullets that hardcode a specific GB/TB/MB when the family spans ≥2 capacities.
  // Mirrors the agent prompt's own capacity rule but enforces it through validation, not just instruction.
  const bulletCapTokens = new Set<string>()
  for (const c of input.children) {
    const cap = capacityOf(c.sku) || capacityOf(c.title)
    if (cap) bulletCapTokens.add(cap.toUpperCase())
  }
  const capacityFamilyTokens = bulletCapTokens.size >= 2 ? [...bulletCapTokens] : []
  onProgress('Writing bullets...')
  const bullets = await runBulletsAgent(input, finalTitle, remainingForBullets, bulletAttrs, topOpportunityKwsForBullets, capacityFamilyTokens)

  // Stage 3 — Backend keywords. HYBRID (PO-chosen): include the TOP product keyphrases
  // (even ones in the title — utilize the best Jungle Scout terms) PLUS long-tail /
  // synonyms / occasion / seasonal. Whole coherent phrases, filled toward ~240 bytes.
  // Sorted by opportunity so the highest-value phrases land first.
  onProgress('Distributing backend keywords...')
  const backendPool = analysis
    .filter((k) => ['CRITICAL', 'UPGRADE', 'REINFORCE', 'DEFENDED'].includes(k.actionType))
    .sort((a, b) => b.opportunityScore - a.opportunityScore)
  const perChild = await runBackendAgent(input, finalTitle, bullets, backendPool)

  // Description (always generated — indexed field)
  onProgress('Writing description...')
  const description = await runDescriptionAgent(input, finalTitle, bullets, bulletAttrs)

  // Stage 4 — Audit (reasoning model)
  onProgress('Auditing & building action plan...')
  let audit: AuditResult = {}
  try {
    audit = await runAuditAgent(input, finalTitle, bullets, perChild, description, attrs.specs)
  } catch (err) {
    console.warn('[pipeline] Audit agent failed, returning content without action plan:', err)
  }

  // Guarantee the read path AND that content always renders. The o4-mini audit is
  // non-deterministic about verdicts — it sometimes marks title/bullets DONE ("matches
  // finalized content"), which hides the copy-paste box in the UI. So for every content
  // element we (a) overwrite replacement_content with the CANONICAL pipeline output and
  // (b) force verdict=REPLACE so the seller can always copy it.
  const actionPlan = Array.isArray(audit.action_plan) ? audit.action_plan : []
  // forceReplace also overwrites current_status + instruction. The o4-mini audit is told
  // the content is "ALREADY FINALIZED", so it writes current_status="Finalized title…" and
  // instruction="No changes required" — which directly contradicts the forced REPLACE
  // verdict (a 🔴 REPLACE card that says "no changes required"). Set deterministic,
  // action-oriented copy that describes what the seller must DO.
  const forceReplace = (item: PipelineActionPlanItem, content: string, label: string) => {
    item.replacement_content = content
    item.verdict = 'REPLACE'
    if (item.priority === 'NONE') item.priority = 'HIGH'
    item.current_status = `Your live ${label} is not optimized for the target keywords.`
    item.instruction = `Replace your current ${label} with the optimized version below, then save in Seller Central.`
  }
  for (const item of actionPlan) {
    if (item.element === 'title') forceReplace(item, finalTitle, 'title')
    else if (/^bullet_([1-5])$/.test(item.element)) {
      const n = Number(item.element.split('_')[1])
      if (bullets[n - 1]) forceReplace(item, bullets[n - 1], `bullet ${n}`)
    } else if (item.element === 'description') {
      forceReplace(item, description, 'description')
      item.level = 'parent'   // description is shared across the whole family, never per-child
    } else if (item.element === 'backend_keywords') {
      item.verdict = 'REPLACE'
      if (item.priority === 'NONE') item.priority = 'HIGH'
      item.level = 'per_child'
      // Show the canonical representative string (not o4-mini's fabricated one) so the
      // copy box matches what actually gets pushed; the per-variant list lives below.
      if (perChild[0]?.keywords) item.replacement_content = perChild[0].keywords
      item.current_status = 'Your live backend search terms miss high-value keywords and repeat words already in the title.'
      item.instruction = "Replace each child SKU's backend search terms with its per-variant string below."
    } else if (item.element === 'brand_story' && item.verdict === 'SKIP' && !input.hasBrandStory) {
      // The seller has no Brand Story — recommend creating one (it auto-appears on every ASIN).
      item.verdict = 'CREATE'
      if (item.priority === 'NONE') item.priority = 'LOW'
      item.current_status = 'No Brand Story (EMC) module detected on your account.'
      item.instruction = 'Create a Brand Story in A+ Content Manager — it auto-appears on every ASIN and links shoppers to your full catalog.'
    }
  }

  // GUARANTEE the core elements always exist. The audit (o4-mini) is best-effort and sometimes
  // returns an EMPTY action_plan — which left the "Apply Changes" UI blank despite a full set of
  // recommendations (the "not loading" bug). Synthesize any missing core element from the
  // canonical pipeline output so the page ALWAYS renders its recommendations.
  const present = new Set(actionPlan.map((a) => a.element))
  const synth = (element: string, content: string, level: 'parent' | 'per_child', status: string, instruction: string) => {
    if (present.has(element) || !content) return
    actionPlan.push({
      element, level, verdict: 'REPLACE', priority: 'HIGH',
      current_status: status, instruction, replacement_content: content,
      seller_central_path: 'Manage Inventory > Edit Listing',
    } as PipelineActionPlanItem)
  }
  synth('title', finalTitle, 'parent', 'Your live title is not optimized for the target keywords.', 'Replace your current title with the optimized version below, then save in Seller Central.')
  for (let i = 1; i <= 5; i++) synth(`bullet_${i}`, bullets[i - 1], 'parent', `Your live bullet ${i} is not optimized for the target keywords.`, `Replace your current bullet ${i} with the optimized version below, then save in Seller Central.`)
  synth('description', description, 'parent', 'Your live description is not optimized for the target keywords.', 'Replace your current description with the optimized version below, then save in Seller Central.')
  synth('backend_keywords', perChild[0]?.keywords ?? '', 'per_child', 'Your live backend search terms miss high-value keywords.', "Replace each child SKU's backend search terms with its per-variant string below.")

  return {
    recommended_title: finalTitle,
    recommended_bullets: bullets,
    per_child_keywords: perChild,
    per_child_titles: perChildTitles,
    recommended_description: description,
    variant_corrections: Array.isArray(audit.variant_corrections) ? audit.variant_corrections : [],
    cannibalization_warnings: Array.isArray(audit.cannibalization_warnings) ? audit.cannibalization_warnings : [],
    product_details_improvements: Array.isArray(audit.product_details_improvements) ? audit.product_details_improvements.slice(0, 10) : [],
    keyword_reconciliation: Array.isArray(audit.keyword_reconciliation) ? audit.keyword_reconciliation : [],
    action_plan: actionPlan,
    debug: { titleProblems, candidatesUsed: candidates.map((c) => c.keyword), titleRetried: retried },
  }
}
