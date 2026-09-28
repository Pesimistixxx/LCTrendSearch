import { mockSearch, mockGraph } from './mock.js'

// VITE_MOCK=0 → ходим в реальный бэкенд (FastAPI), иначе — рандомные данные для проверки UI.
// Docker-сборка фронта задаёт VITE_MOCK=0; `npm run dev` без флага — демо.
const MOCK = import.meta.env.VITE_MOCK !== '0'
// Экраны результатов и справки помечают синтетику плашкой «ДЕМО».
export const DEMO_DATA = MOCK

/*
Контракт бэкенда.

GET /api/search?q=<запрос>[&date=YYYY-MM-DD]  →  (lctrend.graph.search)
{
  query: string,
  snapshot: 'YYYY-MM-DD',             // дата T: всё посчитано по данным ≤ T
  demo: bool,                         // true только у синтетики mock.js
  scope: 'domain'|'label'|'all',      // чем запрос отобрал технологии
  matched: string[],                  // найденные домены (scope = 'domain')
  note: string|null,                  // пояснение, например «показан общий ТОП»
  stats: { sourcesProcessed: number, candidates: number, confident: number },   // confident = score > 0.75
  signals: [{                         // ТОП-15, отсортированы по score
    id, title, domain,
    score: 0..1,                      // скор: логистика от Σ вес × z-оценка (не вероятность)
    stage: string,                    // «прототипы» | «пилоты» | … | «не определена»
    summary: string,                  // одна строка для таблицы
    predictors: [{ name, weight, value: string }],   // топ-3 вклада: вес × z-оценка признака
    trend: number[],                  // упоминания по кварталам до T (спарклайн)
    description, advantages: string[], cases: [{ title, text }],
    reports: [{ org, text }],         // только демо; в реальном ответе []
    quotes: [{ text, title, date, url }],   // 2–3 цитаты из accepted-утверждений графа
    whyWeak: string,                  // почему прошла правило отбора на дату T
    confidenceReason: string,         // как получен скор
    sources: [{ title, url, date, type, lang, trust: 'high'|'medium'|'low',
                summaryRu?: string, generated?: bool }]   // generated = автоперевод/LLM-резюме
  }],
  rejected: [{ title, category: 'mature'|'hype'|'standard'|'noise', reason }]
}

GET /api/graph?q=<запрос>[&signal=<id>]  →  (из Neo4j; на бэкенде пока нет —
экран показывает «Граф недоступен»)
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
