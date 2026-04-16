export function toProbability(value) {
  const numeric = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(numeric)) return null

  // Support both 0-1 probabilities and 0-100 percentage-like values.
  if (numeric >= 0 && numeric <= 1) return numeric
  if (numeric > 1 && numeric <= 100) return numeric / 100
  return null
}

function isProbability(value) {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1
}

function sideFromProbability(value) {
  return value >= 0.5 ? 'ai' : 'human'
}

function directionLabel(side) {
  return side === 'ai' ? 'AI-generated' : 'human-made'
}

function percent(value) {
  return `${Math.round(value * 100)}%`
}

export function deriveSignalConsensus(signals) {
  const vit = toProbability(signals?.vit_probability)
  const lgb = toProbability(signals?.lgb_probability)

  if (!isProbability(vit) || !isProbability(lgb)) {
    return {
      tone: 'neutral',
      badge: 'Signal check unavailable',
      detail: 'ArtLens could not compare the two checks for this image.',
    }
  }

  const vitSide = sideFromProbability(vit)
  const lgbSide = sideFromProbability(lgb)
  const sameDirection = vitSide === lgbSide
  const gap = Math.abs(vit - lgb)
  const direction = directionLabel(vitSide)

  if (!sameDirection) {
    return {
      tone: 'warn',
      badge: 'Signals conflict',
      detail: 'ViT and LGB point in opposite directions. Treat this result as uncertain and verify manually before relying on it.',
    }
  }

  if (gap <= 0.3) {
    return {
      tone: 'good',
      badge: 'Signals aligned',
      detail: `Both checks point to ${direction}. They differ in strength, but they agree on the direction.`,
    }
  }

  return {
    tone: 'caution',
    badge: 'Signals mostly aligned',
    detail: `Both checks point to ${direction}, but the strength gap is large. Use caution and avoid treating this as definitive on its own.`,
  }
}

export function simplifyModelSummary(summary) {
  if (!summary) return summary

  // Remove technical disagreement notes that are confusing to non-technical users.
  const sentences = summary
    .split(/(?<=[.!?])\s+/)
    .filter((sentence) => !/(models?\s+disagree|show\s+some\s+disagreement|interpret\s+with\s+appropriate\s+caution)/i.test(sentence.trim()))

  return sentences.join(' ').trim() || summary
}

export function buildResultNarrative(result) {
  const summary = simplifyModelSummary(result?.explanation?.summary)
  const vit = toProbability(result?.model_signals?.vit_probability)
  const lgb = toProbability(result?.model_signals?.lgb_probability)
  const label = result?.label === 'ai' ? 'AI-generated' : 'human-made'

  if (!isProbability(vit) || !isProbability(lgb)) {
    return summary || 'ArtLens completed the analysis for this image.'
  }

  const consensus = deriveSignalConsensus(result.model_signals)
  const lead = `Final verdict: likely ${label} (${percent(result?.confidence ?? 0)} confidence).`
  const vitText = `Visual pattern check: ${percent(vit)} for ${vit >= 0.5 ? 'AI' : 'human'}.`
  const lgbText = `Texture and frequency check: ${percent(lgb)} for ${lgb >= 0.5 ? 'AI' : 'human'}.`

  if (consensus.tone === 'warn') {
    return `${lead} ViT and LGB disagree, so this result is high-uncertainty and should be manually reviewed. ${vitText} ${lgbText} In simple terms: one detector sees strong AI cues while the other sees stronger human-like cues.`
  }

  if (consensus.tone === 'caution') {
    return `${lead} ViT and LGB lean the same way, but one is much stronger than the other, so this should be treated as cautionary rather than definitive. ${vitText} ${lgbText} In simple terms: the direction is consistent, but confidence support is uneven.`
  }

  return `${lead} Both checks support the same direction. ${vitText} ${lgbText} In simple terms: the evidence is consistent across both detectors.`
}