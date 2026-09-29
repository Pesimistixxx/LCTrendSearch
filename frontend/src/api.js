import { mockSearch, mockGraph } from './mock.js'

// VITE_MOCK=0 → ходим в реальный бэкенд (FastAPI), иначе — рандомные данные для проверки UI.
// Docker-сборка фронта задаёт VITE_MOCK=0; `npm run dev` без флага — демо.
const MOCK = import.meta.env.VITE_MOCK !== '0'
// Экраны результатов и справки помечают синтетику плашкой «ДЕМО».
export const DEMO_DATA = MOCK

/*
Контракт бэкенда.

Реальный поиск сочетает BM25 и косинусную близость GigaChat-эмбеддингов.
Оценка слабого сигнала пока эвристическая, не обученная вероятность.

GET /api/search?q=<запрос>[&date=YYYY-MM-DD]  →  (lctrend.ranking.search)
{
  query: string,
  snapshot: 'YYYY-MM-DD',             // дата T: всё посчитано по данным ≤ T
  demo: bool,                         // true только у синтетики mock.js
  scope: 'hybrid'|'lexical',          // семантика + BM25 либо только BM25 (GigaChat недоступен)
  matched: string[],                  // упомянутые в запросе домены
  note: string|null,                  // пояснение о недоступной семантике / пустом результате
  ranking: 'model'|'rule',            // model: сигнальная часть — вероятность обученной модели
                                      // (узлы графа, ноутбук 05); rule: эвристика признаков графа
  stats: { sourcesProcessed: number, candidates: number, confident: number },   // confident = weakSignalScore > 0.75
  signals: [{                         // ТОП-15, отсортированы по общему score
    id, title, domain,
    score: 0..1,                      // 0.75 × relevanceScore + 0.25 × weakSignalScore
    relevanceScore: 0..1,             // 0.65 × cosine + 0.35 × нормализованный BM25, при наличии обеих частей
    weakSignalScore: 0..1,            // вероятность модели (ranking=model) либо эвристика правила
    ruleScore: 0..1,                  // эвристика правила (для сравнения)
    semanticSimilarity: number|null,  // косинус запроса и названия технологии
    bm25Score: number,                // BM25 названия и домена на дату T
    model: null | {                   // оценки с узла Technology (signal_*, llm_*)
      probability, flag, name, snapshot,          // модель
      verdict, llmScore, hype, maturity, rationale // LLM-разметка траектории
    },
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
  rejected: [{ title, category: 'mature'|'hype'|'standard'|'noise', reason }]   // noise: LLM «не технология»
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
  if (!r.ok) {
    const body = await r.json().catch(() => null)
    throw new Error(typeof body?.detail === 'string' ? body.detail : `${r.status} ${r.statusText}`)
  }
  return r.json()
}

export const search = (q) =>
  MOCK ? mockSearch(q) : get(`/api/search?q=${encodeURIComponent(q)}`)

export const graph = (q, signal) =>
  MOCK
    ? mockGraph(q, signal)
    : get(`/api/graph?q=${encodeURIComponent(q)}${signal ? `&signal=${encodeURIComponent(signal)}` : ''}`)
