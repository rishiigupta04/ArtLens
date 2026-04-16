import { createClient } from '@supabase/supabase-js'

export const supabase = createClient(
  import.meta.env.VITE_SUPABASE_URL,
  import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY
)

// Log prediction to Supabase
export async function logPrediction(result, imageHash) {
  const { error } = await supabase.from('predictions').insert({
    image_hash       : imageHash,
    label            : result.label,
    confidence       : result.confidence,
    generator        : result.generator?.name,
    is_unknown_gen   : result.generator?.is_unknown,
    vit_prob         : result.model_signals?.vit_probability,
    lgb_prob         : result.model_signals?.lgb_probability,
    ensemble_prob    : result.model_signals?.ensemble_probability,
    inference_time_s : result.inference_time_s,
  })
  if (error) console.warn('Supabase log failed:', error.message)
}

// Submit community flag
export async function submitFlag(imageHash, modelLabel, userClaim, notes = '') {
  const { error } = await supabase.from('community_flags').insert({
    image_hash  : imageHash,
    model_label : modelLabel,
    user_claim  : userClaim,
    notes,
  })
  if (error) throw error
}