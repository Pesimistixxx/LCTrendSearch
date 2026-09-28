// Фейковые данные для проверки UI. Детерминированы по тексту запроса.
// Источники и отчёты — обезличенные заглушки: никаких реальных организаций,
// брендов и ссылок на их сайты, чтобы синтетику нельзя было принять за цитату.

const TECH = [
  ['Квантовые сенсоры для мед. диагностики', 'Квантовые технологии'],
  ['Биоразлагаемые полимеры 3-го поколения', 'Материалы'],
  ['Нейроморфные чипы для Edge AI', 'Микроэлектроника'],
  ['Гомоморфное шифрование в скоринге', 'Кибербезопасность'],
  ['Жидкие нейросети (Liquid NN)', 'Искусственный интеллект'],
  ['Фотонные вычисления для инференса', 'Микроэлектроника'],
  ['Синтетические данные с гарантиями приватности', 'Данные'],
  ['Малые языковые модели на устройстве', 'Искусственный интеллект'],
  ['Твердотельные натрий-ионные батареи', 'Энергетика'],
  ['Агентные системы для комплаенса', 'Финтех'],
  ['Программируемые CBDC-смарт-контракты', 'Финтех'],
  ['Постквантовые подписи в платёжных сетях', 'Кибербезопасность'],
  ['Мемристорные массивы памяти', 'Микроэлектроника'],
  ['Цифровые двойники клеточных культур', 'Биотех'],
  ['Федеративное обучение в антифроде', 'Финтех'],
  ['Метаматериалы для 6G-антенн', 'Телеком'],
  ['Каузальный ИИ для кредитных рисков', 'Искусственный интеллект'],
  ['Спинтронные логические элементы', 'Микроэлектроника'],
  ['ДНК-хранилища архивных данных', 'Данные'],
  ['Нейросимвольное рассуждение', 'Искусственный интеллект'],
  ['Прямой захват CO₂ из воздуха на MOF', 'Климатех'],
  ['Zero-knowledge KYC', 'Кибербезопасность'],
]

const PREDICTORS = [
  ['Рост патентной активности', (r) => `+${Math.round(40 + r() * 260)}% за 4 кв.`],
  ['Упоминания в препринтах', (r) => `×${(1.5 + r() * 4).toFixed(1)} год к году`],
  ['Гранты и госпрограммы', (r) => `${Math.round(1 + r() * 9)} новых`],
  ['Стартапы pre-seed/seed', (r) => `${Math.round(2 + r() * 12)} компаний`],
  ['Миграция разработчиков (репозитории кода)', (r) => `+${Math.round(20 + r() * 180)}% звёзд`],
  ['Рост вакансий', (r) => `+${Math.round(10 + r() * 90)}%`],
  ['Доля хайп-лексики', (r) => `${Math.round(r() * 30)}% текстов`],
  ['Присутствие в СМИ', (r) => `${Math.round(r() * 40)} публикаций`],
  ['Наличие рыночных лидеров', () => 'не выявлены'],
]

const SOURCES = [
  ['Демо-препринт', 'Научная публикация', 'EN', 'high', 'https://example.org/demo/preprint'],
  ['Демо-патент (RU)', 'Патентная база', 'RU', 'high', 'https://example.org/demo/patent-ru'],
  ['Демо-патент (международный)', 'Патентная база', 'EN', 'high', 'https://example.org/demo/patent-intl'],
  ['Демо-журнал', 'Научная публикация', 'EN', 'high', 'https://example.org/demo/journal'],
  ['Демо-университет', 'Университет', 'RU', 'high', 'https://example.org/demo/university'],
  ['Демо-международная организация', 'Международная организация', 'EN', 'high', 'https://example.org/demo/intl-org'],
  ['Демо-регулятор', 'Регулятор', 'RU', 'high', 'https://example.org/demo/regulator'],
  ['Демо-конференция', 'Материалы конференции', 'EN', 'high', 'https://example.org/demo/conference'],
  ['Демо-аналитика', 'Аналитический отчёт', 'EN', 'medium', 'https://example.org/demo/analyst'],
  ['Демо-венчурный обзор', 'Аналитический отчёт', 'EN', 'medium', 'https://example.org/demo/venture'],
  ['Демо-отраслевое медиа', 'Отраслевое медиа', 'RU', 'medium', 'https://example.org/demo/media'],
  ['Демо-блог', 'Блог / сообщество', 'RU', 'low', 'https://example.org/demo/blog'],
  ['Демо-канал', 'Социальная сеть', 'RU', 'low', 'https://example.org/demo/channel'],
  ['Демо-пресс-релиз', 'Пресс-релиз', 'EN', 'low', 'https://example.org/demo/press'],
]

const REJECTED = [
  ['Генеративные чат-боты для клиентов', 'mature', 'Массовое внедрение, сформированный рынок и явные лидеры'],
  ['Биометрия по лицу', 'standard', 'Закреплено в отраслевых стандартах'],
  ['NFT-маркетплейсы', 'hype', 'Пик упоминаний в СМИ без роста патентов и научных работ'],
  ['Облачные вычисления', 'mature', 'Устойчивое конкурентное разделение рынка'],
  ['«Революционный ИИ-трейдер»', 'noise', 'Единственный источник — рекламный пресс-релиз'],
  ['Метавселенные для банкинга', 'hype', 'Падение упоминаний на 70% после хайпа, нет пилотов'],
  ['RPA-роботизация бэк-офиса', 'mature', 'Технология массово используется с 2018 года'],
  ['Open Banking API', 'standard', 'Отраслевой стандарт, регулируется отраслевым регулятором'],
]

const STAGES = ['исследования', 'прототипы', 'пилоты']

function rng(seedStr) {
  let h = 2166136261
  for (const c of seedStr) h = Math.imul(h ^ c.charCodeAt(0), 16777619)
  return () => {
    h = Math.imul(h ^ (h >>> 15), 2246822507)
    h = Math.imul(h ^ (h >>> 13), 3266489909)
    return ((h ^= h >>> 16) >>> 0) / 4294967296
  }
}
const pick = (r, arr) => arr[Math.floor(r() * arr.length)]
const shuffle = (r, arr) => arr.map((x) => [r(), x]).sort((a, b) => a[0] - b[0]).map((x) => x[1])
const pad = (n) => String(n).padStart(2, '0')
const date = (r) => `2026-${pad(1 + Math.floor(r() * 9))}-${pad(1 + Math.floor(r() * 28))}`
const wait = (ms) => new Promise((res) => setTimeout(res, ms))

function makeSignal(r, [title, domain], i, q) {
  const score = Math.max(0.52, Math.min(0.97, 0.95 - i * 0.028 + (r() - 0.5) * 0.06))
  const predictors = shuffle(r, PREDICTORS).slice(0, 5).map(([name, val]) => {
    const neg = /хайп|СМИ/.test(name)
    return { name, value: val(r), weight: +((neg ? -1 : 1) * (0.15 + r() * 0.8)).toFixed(2) }
  }).sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
  let v = 2 + r() * 4
  const trend = Array.from({ length: 10 }, (_, k) => Math.round((v *= 1 + r() * 0.25 + k * 0.01)))
  const sources = shuffle(r, SOURCES).slice(0, 3 + Math.floor(r() * 4)).map(([name, type, lang, trust, url]) => ({
    title: `${name}: ${title.toLowerCase()} — ${pick(r, ['обзор', 'новые результаты', 'оценка зрелости', 'пилотный проект'])}`,
    url, date: date(r), type, lang, trust,
    summaryRu: lang === 'EN' ? `Авторы описывают прогресс в направлении «${title}» и отмечают отсутствие коммерческих решений.` : undefined,
    generated: lang === 'EN',
  }))
  const pos = predictors.filter((p) => p.weight > 0).slice(0, 2).map((p) => p.name.toLowerCase())
  return {
    id: `s${i}-${Math.floor(r() * 1e6)}`,
    title, domain, score: +score.toFixed(2),
    stage: pick(r, STAGES),
    summary: predictors.filter((p) => p.weight > 0).slice(0, 2).map((p) => `${p.name}: ${p.value}`).join(' · '),
    predictors, trend, sources,
    description: `«${title}» — направление на стыке фундаментальных исследований и первых прикладных прототипов. По запросу «${q}» технология встречается в свежих научных работах и патентных заявках, но пока не имеет массового рынка и выраженных лидеров.`,
    advantages: [
      'Снижение стоимости операции в 3–10 раз относительно текущих решений',
      'Новые продуктовые сценарии, недоступные на зрелом стеке',
      'Возможность занять позицию до формирования рынка',
    ],
    cases: [
      { title: 'Лабораторный пилот', text: `Исследовательская группа продемонстрировала прототип в области «${domain}» с приростом ключевой метрики на ${Math.round(20 + r() * 60)}%.` },
      { title: 'Ранний стартап', text: `Seed-раунд $${(1 + r() * 8).toFixed(1)} млн на коммерциализацию технологии; первые B2B-пилоты в 2026 году.` },
    ],
    reports: [
      { org: 'Демо-отчёт А (синтетический пример)', text: 'Ранняя стадия; горизонт выхода на рынок — 5–10 лет.' },
      { org: 'Демо-отчёт Б (синтетический пример)', text: 'Направление с высоким потенциалом и низкой концентрацией игроков.' },
    ],
    whyWeak: `Модель относит технологию к слабым сигналам: сильный рост по признакам «${pos.join('», «')}» при низком медийном шуме и отсутствии сформированного рынка.`,
    confidenceReason: `Уверенность ${Math.round(score * 100)}%: ${sources.filter((s) => s.trust === 'high').length} подтверждения из доверенных источников; ${sources.some((s) => s.trust === 'low') ? 'часть сведений из источников пониженной доверенности снижает оценку.' : 'источники пониженной доверенности не использовались.'}`,
  }
}

export async function mockSearch(q) {
  await wait(2600 + Math.random() * 800)
  return build(q)
}

function build(q) {
  const r = rng(q.trim().toLowerCase())
  const signals = shuffle(r, TECH).slice(0, 15).map((t, i) => makeSignal(r, t, i, q)).sort((a, b) => b.score - a.score)
  return {
    query: q,
    stats: {
      sourcesProcessed: Math.round(40000 + r() * 120000),
      candidates: Math.round(300 + r() * 900),
      confident: signals.filter((s) => s.score > 0.75).length,
    },
    signals,
    rejected: shuffle(r, REJECTED).slice(0, 5).map(([title, category, reason]) => ({ title, category, reason })),
  }
}

export async function mockGraph(q, signalId) {
  await wait(300)
  const all = build(q).signals
  const signal = all.find((s) => s.id === signalId)
  const r = rng(q.trim().toLowerCase() + (signalId || ''))
  const nodes = [], links = []
  const add = (n) => (nodes.push(n), n.id)
  const root = add({ id: 'q', label: q, type: 'query' })
  const signals = signal ? [signal] : all
  const entities = ['Патенты', 'Препринты', 'Стартапы', 'Гранты', 'Репозитории кода', 'Вакансии', 'Регуляторы']
  entities.forEach((e) => add({ id: `e:${e}`, label: e, type: 'entity' }))
  for (const s of signals) {
    add({ id: s.id, label: s.title, type: 'signal', score: s.score })
    links.push({ source: root, target: s.id, weight: s.score })
    shuffle(r, entities).slice(0, signal ? 5 : 2).forEach((e) =>
      links.push({ source: s.id, target: `e:${e}`, weight: 0.3 + r() * 0.6, label: 'влияет' }))
    if (signal) s.sources.forEach((src, k) => {
      const id = add({ id: `src${k}`, label: src.title.split(':')[0], type: 'source', trust: src.trust })
      links.push({ source: id, target: s.id, weight: 0.5, label: 'подтверждает' })
    })
  }
  return { nodes, links }
}
