import { mockSearch, mockGraph } from './mock.js'

// VITE_MOCK=0 → ходим в реальный бэкенд (FastAPI), иначе — рандомные данные для проверки UI.
const MOCK = import.meta.env.VITE_MOCK !== '0'

/*
Контракт бэкенда.

GET /api/search?q=<запрос>  →
{
  query: string,
  stats: { sourcesProcessed: number, candidates: number, confident: number },   // confident = score > 0.75
  signals: [{                         // ТОП-15, отсортированы по score
    id, title, domain,
    score: 0..1,                      // уверенность модели
    stage: string,                    // «исследования» | «прототипы» | «пилоты»
    summary: string,                  // одна строка для таблицы
    predictors: [{ name, weight: -1..1, value: string }],   // вклад признаков (SHAP и т.п.)
    trend: number[],                  // упоминания по кварталам (спарклайн)
    description, advantages: string[], cases: [{ title, text }],
    reports: [{ org, text }],         // оценки в аналитических отчётах
    whyWeak: string,                  // почему это слабый сигнал
    confidenceReason: string,         // почему именно такая уверенность
    sources: [{ title, url, date, type, lang, trust: 'high'|'medium'|'low',
                summaryRu?: string, generated?: bool }]   // generated = автоперевод/LLM-резюме
  }],
  rejected: [{ title, category: 'mature'|'hype'|'standard'|'noise', reason }]
}

GET /api/graph?q=<запрос>[&signal=<id>]  →  (из Neo4j)
{
  nodes: [{ id, label, type: 'query'|'signal'|'source'|'entity', score? }],
  links: [{ source, target, label?, weight: 0..1 }]
}
*/

async function get(path) {
  const r = await fetch(path)
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`)
  return r.json()
}

export const search = (q) =>
  MOCK ? mockSearch(q) : get(`/api/search?q=${encodeURIComponent(q)}`)

export const graph = (q, signal) =>
  MOCK
    ? mockGraph(q, signal)
    : get(`/api/graph?q=${encodeURIComponent(q)}${signal ? `&signal=${encodeURIComponent(signal)}` : ''}`)
