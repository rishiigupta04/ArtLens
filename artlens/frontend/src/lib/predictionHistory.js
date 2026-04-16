const STORAGE_KEY = 'artlens-recent-results-v1'
const MAX_ITEMS = 5

function isObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value)
}

export function loadHistory() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) return []

    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []

    return parsed
      .filter((item) => isObject(item) && isObject(item.result) && typeof item.id === 'string')
      .slice(0, MAX_ITEMS)
  } catch {
    return []
  }
}

export function saveHistory(items) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(items.slice(0, MAX_ITEMS)))
  } catch {
    // Ignore storage quota or serialization errors.
  }
}

export function pushHistory(items, entry) {
  const deduped = items.filter((item) => item.id !== entry.id)
  return [entry, ...deduped].slice(0, MAX_ITEMS)
}

export function clearHistory() {
  try {
    window.localStorage.removeItem(STORAGE_KEY)
  } catch {
    // Ignore storage errors.
  }
}