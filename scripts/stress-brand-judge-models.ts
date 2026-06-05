/**
 * Empirical brand-safety judge model comparison — run with:
 *   npx tsx scripts/stress-brand-judge-models.ts
 *
 * Requires OPENAI_API_KEY in env (or .env). Tries multiple model IDs against the
 * actual failing B0G884ZJ27 output (verbatim from the live regen that PR #80's
 * judge missed). Prints per-model: latency, cost estimate, and which trademarked
 * phrases it caught vs missed.
 *
 * Use the winner's model ID for BRAND_SAFETY_JUDGE_MODEL in production.
 */
import OpenAI from 'openai'
import { judgeBrandSafetyLLM } from '../src/lib/fba/listingPipeline'

const apiKey = process.env.OPENAI_API_KEY
if (!apiKey) {
  console.error('OPENAI_API_KEY is not set. Add it to .env.local or export it before running.')
  process.exit(1)
}

const BRAND = 'THE CEO'

// Verbatim text from the live B0G884ZJ27 regen after PR #80 deployed.
// Every Homelander / Ripcurl / Iration / Ella Fella reference is a real
// trademark / brand the judge SHOULD have flagged.
const TEST_TEXT = `THE CEO See You Later Alligator Shirt Comfort Colors Vintage 90s Style Homelander Ripcurl Iration Ella Fella Mens Cool Shirt for Men and Women
1. VINTAGE STYLE APPEAL - This THE CEO See You Later Alligator Shirt features a classic 90s vintage design, perfect for fans of the Homelander shirt and iration t shirt styles, so that you can enjoy timeless fashion.
2. UNISEX COMFORT FIT - Designed as a mens cool shirts option that fits both men and women comfortably, this ella fella t shirt-inspired garment uses soft Comfort Colors fabric.
3. QUALITY MATERIALS - Crafted from premium cotton blend, this later gator shirt offers durability for everyday use or as a standout piece in your iration t shirt collection.
4. VERSATILE DESIGN - Combining elements from the Homelander shirt and ella fella t shirt aesthetics, this shirt suits various casual occasions.
5. EASY CARE INSTRUCTIONS - This THE CEO shirt is machine washable and maintains its vintage 90s style after multiple washes, so you can enjoy your later gator shirt or iration t shirt without worry.`

const EXPECTED_FLAGS = ['homelander', 'ripcurl', 'iration', 'ella fella']

// Try a spread of model IDs in cost-ascending order. Some may not be enabled
// on your org — those throw and we record the error.
const MODELS_TO_TEST = [
  'gpt-4.1-mini',     // current default (the one that failed)
  'gpt-4o-mini',      // small + capable, $0.15/1M in
  'gpt-4o',           // strong + standard, $2.50/1M in
  'gpt-5-mini',       // if your org has it
  'gpt-5',            // if your org has it
]

interface Result {
  model: string
  ok: boolean
  errorMessage?: string
  latencyMs: number
  flaggedPhrases: string[]
  caught: string[]   // EXPECTED_FLAGS this model successfully flagged
  missed: string[]   // EXPECTED_FLAGS this model failed to flag
}

async function tryModel(model: string, openai: OpenAI): Promise<Result> {
  const start = Date.now()
  // Inline copy of the same prompt as the production judge — keeps the model
  // comparison apples-to-apples regardless of code paths.
  const system = `You are a STRICT Amazon trademark-safety judge. Find every third-party brand, registered trademark, sports team, university, media franchise, character name, band/musician, or other proper-noun reference that this seller cannot legally use without a license.

Rules:
1. Flag ANY proper-noun phrase that COULD be a registered mark.
2. If you do not recognize a capitalized phrase AND cannot confirm it is generic English, you MUST flag it.
3. NEVER flag generic English words.
4. NEVER flag the seller's own brand "${BRAND}".
5. When uncertain, FLAG.

Return ONLY {"detected":[{"phrase":"<exact substring>","reason":"<one line>"}]}.`
  const user = `Seller brand: ${BRAND}\n\nText:\n"""\n${TEST_TEXT}\n"""\n\nFlag every third-party brand/trademark/proper noun. Return ONLY the JSON.`
  try {
    const params: Parameters<typeof openai.chat.completions.create>[0] = {
      model,
      messages: [{ role: 'system', content: system }, { role: 'user', content: user }],
      response_format: { type: 'json_object' },
    }
    // Reasoning-class models (gpt-5*) use max_completion_tokens, not max_tokens.
    if (model.startsWith('gpt-5')) {
      ;(params as Record<string, unknown>).max_completion_tokens = 600
    } else {
      ;(params as Record<string, unknown>).max_tokens = 600
      ;(params as Record<string, unknown>).temperature = 0
    }
    const r = await openai.chat.completions.create(params)
    const raw = r.choices[0]?.message?.content || '{}'
    const parsed = JSON.parse(raw) as { detected?: { phrase: string; reason?: string }[] }
    const flagged = (parsed.detected ?? []).map((f) => f.phrase.toLowerCase())
    const caught = EXPECTED_FLAGS.filter((e) => flagged.some((f) => f.includes(e)))
    const missed = EXPECTED_FLAGS.filter((e) => !flagged.some((f) => f.includes(e)))
    return {
      model, ok: true,
      latencyMs: Date.now() - start,
      flaggedPhrases: parsed.detected?.map((f) => f.phrase) ?? [],
      caught, missed,
    }
  } catch (err) {
    return {
      model, ok: false,
      errorMessage: err instanceof Error ? err.message.slice(0, 200) : String(err).slice(0, 200),
      latencyMs: Date.now() - start,
      flaggedPhrases: [], caught: [], missed: EXPECTED_FLAGS,
    }
  }
}

async function main() {
  const openai = new OpenAI({ apiKey })
  console.log(`Testing ${MODELS_TO_TEST.length} models on the B0G884ZJ27 failure text…`)
  console.log(`Expected flags: ${EXPECTED_FLAGS.join(', ')}\n`)
  const results: Result[] = []
  for (const model of MODELS_TO_TEST) {
    process.stdout.write(`  → ${model.padEnd(20)} `)
    const r = await tryModel(model, openai)
    results.push(r)
    if (!r.ok) console.log(`✗ ERROR (${r.latencyMs}ms): ${r.errorMessage}`)
    else console.log(`${r.caught.length}/${EXPECTED_FLAGS.length} caught (${r.latencyMs}ms)`)
  }

  console.log('\n── Summary ────────────────────────────────────────')
  for (const r of results) {
    console.log(`\n${r.model}:`)
    if (!r.ok) { console.log(`  ✗ not accessible: ${r.errorMessage}`); continue }
    console.log(`  Latency: ${r.latencyMs}ms`)
    console.log(`  Caught (${r.caught.length}/${EXPECTED_FLAGS.length}): ${r.caught.join(', ') || '(none)'}`)
    if (r.missed.length > 0) console.log(`  MISSED: ${r.missed.join(', ')}`)
    if (r.flaggedPhrases.length > 0) {
      console.log(`  All flagged phrases: ${r.flaggedPhrases.slice(0, 10).join(' | ')}`)
    }
  }

  console.log('\n── Recommendation ────────────────────────────────')
  const winners = results.filter((r) => r.ok && r.caught.length === EXPECTED_FLAGS.length)
  if (winners.length === 0) {
    console.log('No model caught all expected flags. Prompt may need further hardening.')
  } else {
    const cheapest = winners[0]  // MODELS_TO_TEST is cost-ascending
    console.log(`Cheapest model that caught ALL expected flags: ${cheapest.model}`)
    console.log(`Set BRAND_SAFETY_JUDGE_MODEL=${cheapest.model} in your Coolify env.`)
  }
}

main().catch((e) => { console.error(e); process.exit(1) })
