import { useState } from 'react'
import { X, Flag } from 'lucide-react'
import { submitFlag } from '../lib/supabase'

export default function FlagModal({ result, imageHash, onClose }) {
  const [claim, setClaim]   = useState('')
  const [notes, setNotes]   = useState('')
  const [status, setStatus] = useState('idle')  // idle | loading | done | error

  const submit = async () => {
    if (!claim) return
    setStatus('loading')
    try {
      await submitFlag(imageHash, result.label, claim, notes)
      setStatus('done')
    } catch (e) {
      setStatus('error')
    }
  }

  return (
    <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-xl w-full max-w-md p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-amber-600">
            <Flag size={18} />
            <h2 className="font-semibold">Flag this result</h2>
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
            <X size={18} />
          </button>
        </div>

        {status === 'done' ? (
          <div className="text-center py-6 space-y-2">
            <div className="text-3xl">✓</div>
            <p className="font-medium text-gray-700">Thank you for the feedback</p>
            <p className="text-sm text-gray-400">
              This helps improve ArtLens for everyone.
            </p>
            <button
              onClick={onClose}
              className="mt-4 text-sm text-blue-600 hover:underline"
            >
              Close
            </button>
          </div>
        ) : (
          <>
            <p className="text-sm text-gray-600">
              The model predicted this image as{' '}
              <strong>{result.label === 'ai' ? 'AI-generated' : 'human-made'}</strong>.
              If you believe this is wrong, let us know.
            </p>

            <div className="space-y-1">
              <label className="text-xs font-medium text-gray-600">
                What is this image actually?
              </label>
              <div className="flex gap-2">
                {['human', 'ai', 'unsure'].map(c => (
                  <button
                    key={c}
                    onClick={() => setClaim(c)}
                    className={`flex-1 py-2 text-sm rounded-lg border font-medium capitalize transition-colors
                      ${claim === c
                        ? 'bg-gray-900 text-white border-gray-900'
                        : 'border-gray-200 text-gray-600 hover:border-gray-400'
                      }`}
                  >
                    {c === 'ai' ? 'AI-generated' : c === 'human' ? 'Human-made' : 'Unsure'}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-1">
              <label className="text-xs font-medium text-gray-600">
                Notes (optional)
              </label>
              <textarea
                value={notes}
                onChange={e => setNotes(e.target.value)}
                placeholder="e.g. I created this in Procreate in 2023..."
                rows={2}
                className="w-full border border-gray-200 rounded-lg p-2 text-sm
                           resize-none focus:outline-none focus:border-gray-400"
              />
            </div>

            <div className="flex gap-2 pt-1">
              <button
                onClick={onClose}
                className="flex-1 py-2 text-sm border border-gray-200 rounded-lg
                           text-gray-600 hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={submit}
                disabled={!claim || status === 'loading'}
                className="flex-1 py-2 text-sm bg-gray-900 text-white rounded-lg
                           font-medium hover:bg-gray-700 disabled:opacity-40"
              >
                {status === 'loading' ? 'Submitting...' : 'Submit flag'}
              </button>
            </div>
            {status === 'error' && (
              <p className="text-xs text-red-500 text-center">
                Submission failed. Please try again.
              </p>
            )}
          </>
        )}
      </div>
    </div>
  )
}