const BASE = '/api/ingest'

export async function request(path, options = {}) {
  const response = await fetch(`${BASE}${path}`, {
    ...options,
    headers: options.body instanceof FormData ? options.headers : {
      'Content-Type': 'application/json', ...options.headers,
    },
  })
  const contentType = response.headers.get('content-type') || ''
  if (!contentType.includes('application/json')) {
    throw new Error(response.ok
      ? 'Сервис загрузки не подключён. Проверьте адрес сервера.'
      : `Сервис загрузки недоступен (${response.status}).`)
  }
  const data = await response.json()
  if (!response.ok) {
    const detail = data.detail || data.error || data.message
    throw new Error(typeof detail === 'string' ? detail : `Ошибка сервера (${response.status}).`)
  }
  return data
}

export const api = {
  crawls: () => request('/crawls'),
  crawl: id => request(`/crawls/${encodeURIComponent(id)}`),
  createCrawl: (topic, limit) => request('/crawls', { method: 'POST', body: JSON.stringify({ topic, limit }) }),
  pauseCrawl: id => request(`/crawls/${encodeURIComponent(id)}/pause`, { method: 'POST' }),
  resumeCrawl: id => request(`/crawls/${encodeURIComponent(id)}/resume`, { method: 'POST' }),
  retryFailedCrawl: id => request(`/crawls/${encodeURIComponent(id)}/retry-failed`, { method: 'POST' }),
  materials: (id, status, offset = 0, limit = 100) => request(`/crawls/${encodeURIComponent(id)}/materials?status=${encodeURIComponent(status)}&limit=${limit}&offset=${offset}`),
  status: () => request('/status'),
  settings: body => request('/settings', { method: 'POST', body: JSON.stringify(body) }),
  result: (job, document) => request(`/jobs/${encodeURIComponent(job)}/documents/${encodeURIComponent(document)}/result`),
  downloadUrl: (job, document) => `${BASE}/jobs/${encodeURIComponent(job)}/documents/${encodeURIComponent(document)}/download`,
}
