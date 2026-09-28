import { useEffect, useState } from 'react'
import { search, graph, DEMO_DATA } from './api.js'
import Graph from './Graph.jsx'
import Ingestion from './ingest/App.jsx'
import './ingest/styles.css'

const EXAMPLES = ['технологии в ИИ', 'перспективные решения в финтехе', 'слабые сигналы в кибербезопасности', 'новые материалы', 'энергетика будущего']
const STEPS = ['Сбор открытых источников', 'Парсинг патентов, статей, отчётов', 'NLP: извлечение сущностей', 'Инференс и скоринг', 'Фильтрация мейнстрима и хайпа']
const TRUST = { high: 'Высокая', medium: 'Средняя', low: 'Пониженная' }
const REJECT = { mature: 'Зрелая технология', hype: 'Маркетинговый хайп', standard: 'Отраслевой стандарт', noise: 'Инфошум' }
const pct = (x) => Math.round(x * 100)
const level = (s) => (s >= 0.85 ? 'hi' : s >= 0.75 ? 'mid' : 'lo')

// ─── маршрутизация через hash: #q=<запрос>&s=<id сигнала> ───
const readHash = () => Object.fromEntries(new URLSearchParams(location.hash.slice(1)))
const go = (params) => (location.hash = new URLSearchParams(params).toString())

export default function App() {
  const [route, setRoute] = useState(readHash)
  const [res, setRes] = useState(null)
  const [err, setErr] = useState(null)
  const ingestion = route.view === 'ingest'

  useEffect(() => {
    const on = () => setRoute(readHash())
    addEventListener('hashchange', on)
    return () => removeEventListener('hashchange', on)
  }, [])

  useEffect(() => {
    if (ingestion || !route.q || res?.query === route.q) return
    let live = true
    setRes(null); setErr(null)
    search(route.q).then((d) => live && setRes(d), (e) => live && setErr(e.message))
    return () => { live = false }
  }, [route.q, ingestion])

  useEffect(() => { window.scrollTo({ top: 0 }) }, [route.s, route.q])

  const signal = route.s && res?.signals.find((s) => s.id === route.s)

  return (
    <>
      <div className="sky" aria-hidden="true" />
      <Header q={route.q} compact={!!route.q || ingestion} ingestion={ingestion} />
      <main className="wrap">
        {ingestion ? <Ingestion /> : <>
          {!route.q && <Home />}
          {route.q && err && <div className="error card">Не удалось выполнить поиск: {err}</div>}
          {route.q && !res && !err && <Scanning q={route.q} />}
          {route.q && res && !route.s && <Results res={res} />}
          {route.q && res && signal && <Insight s={signal} q={res.query} />}
        </>}
      </main>
      <footer className="foot wrap">
        <span>Сигнал · радар зарождающихся технологий</span>
        <span className="mono">Газпромбанк.Тех · Хакатон 2026</span>
      </footer>
    </>
  )
}

function SearchBox({ initial = '', big }) {
  const [v, setV] = useState(initial)
  useEffect(() => setV(initial), [initial])
  return (
    <form className={`search ${big ? 'search-big' : ''}`} onSubmit={(e) => { e.preventDefault(); v.trim() && go({ q: v.trim() }) }}>
      <svg viewBox="0 0 24 24" className="search-ico" aria-hidden="true"><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></svg>
      <input value={v} onChange={(e) => setV(e.target.value)} placeholder="Технологическое направление, например «финтех»" aria-label="Поисковый запрос" autoFocus={big} />
      <button type="submit">Найти сигналы</button>
    </form>
  )
}

function Header({ q, compact, ingestion }) {
  return (
    <header className={`top ${compact ? 'top-compact' : ''}`}>
      <div className="wrap top-in">
        <a href="#" className="brand" aria-label="На главную">
          <Logo /> <span>Сигнал</span>
        </a>
        {q && !ingestion && <SearchBox initial={q} />}
        <nav className="top-nav" aria-label="Разделы">
          <a href="#" aria-current={!ingestion ? 'page' : undefined}>Демо поиска</a>
          <a href="#view=ingest" aria-current={ingestion ? 'page' : undefined}>Сбор материалов</a>
        </nav>
      </div>
    </header>
  )
}

const Logo = () => (
  <svg viewBox="0 0 32 32" className="logo" aria-hidden="true">
    <circle cx="16" cy="16" r="14" /><circle cx="16" cy="16" r="8" opacity=".5" />
    <path d="M16 16 L28 9" className="logo-beam" /><circle cx="22" cy="11" r="2.6" className="logo-dot" />
  </svg>
)

function Home() {
  return (
    <section className="hero">
      <Radar />
      <p className="eyebrow">Слабые сигналы · научно-технологические тренды</p>
      <h1>Найдите технологию <em>до того,</em><br />как о ней заговорят все</h1>
      <p className="lead">Система сканирует патенты, научные публикации и аналитические отчёты, отсекает зрелые тренды и хайп и показывает ТОП-15 зарождающихся технологий с объяснением каждой оценки.</p>
      <SearchBox big />
      <div className="chips">
        {EXAMPLES.map((e) => <button key={e} className="chip" onClick={() => go({ q: e })}>{e}</button>)}
      </div>
      <p className="lead small">{DEMO_DATA ? 'Поиск показывает демонстрационные данные. ' : 'ТОП-15 считается по графу знаний на текущую дату. '}<a className="ingestion-link" href="#view=ingest">Перейти к сбору и проверке материалов →</a></p>
    </section>
  )
}

function Radar({ small }) {
  const blips = [[0.35, 40], [0.62, 130], [0.8, 215], [0.5, 290], [0.9, 330], [0.25, 180]]
  return (
    <div className={`radar ${small ? 'radar-sm' : ''}`} aria-hidden="true">
      <div className="radar-sweep" />
      {blips.map(([r, a], i) => (
        <i key={i} className="blip" style={{ '--r': r, '--a': `${a}deg`, '--d': `${(a / 360) * 4}s` }} />
      ))}
    </div>
  )
}

function Scanning({ q }) {
  const [step, setStep] = useState(0)
  useEffect(() => {
    const t = setInterval(() => setStep((s) => Math.min(s + 1, STEPS.length - 1)), 620)
    return () => clearInterval(t)
  }, [])
  return (
    <section className="scan">
      <Radar small />
      <h2>Сканирую: «{q}»</h2>
      <ol className="steps">
        {STEPS.map((s, i) => (
          <li key={s} className={i < step ? 'done' : i === step ? 'now' : ''}><span className="step-dot" />{s}</li>
        ))}
      </ol>
    </section>
  )
}

function CountUp({ to }) {
  const [v, setV] = useState(0)
  useEffect(() => {
    let raf, t0
    const f = (t) => {
      t0 ??= t
      const k = Math.min(1, (t - t0) / 1100)
      setV(Math.round(to * (1 - (1 - k) ** 4)))
      if (k < 1) raf = requestAnimationFrame(f)
    }
    raf = requestAnimationFrame(f)
    return () => cancelAnimationFrame(raf)
  }, [to])
  return v.toLocaleString('ru-RU')
}

// Синтетика не должна выглядеть анализом: плашка не скрывается и печатается.
function DemoBanner() {
  if (!DEMO_DATA) return null
  return (
    <div className="demo-banner" role="note">
      <b>ДЕМО: синтетические данные</b>
      <span>Технологии, оценки, числа, источники и цитаты сгенерированы для проверки интерфейса и не являются результатом анализа.</span>
    </div>
  )
}

function Results({ res }) {
  const [onlyConfident, setOnly] = useState(false)
  const [g, setG] = useState(null)
  useEffect(() => { graph(res.query).then(setG, () => setG(false)) }, [res.query])
  const list = onlyConfident ? res.signals.filter((s) => s.score > 0.75) : res.signals

  return (
    <section className="results">
      <DemoBanner />
      <div className="res-head">
        <p className="eyebrow">{DEMO_DATA ? 'Результаты открытого поиска' : `Граф знаний на ${res.snapshot}`}</p>
        <h1 className="res-title">«{res.query}»</h1>
        {res.note && <p className="muted small">{res.note}</p>}
      </div>

      <div className="stats">
        <div className="stat card">
          <span className="stat-k">Обработано источников</span>
          <b className="stat-v"><CountUp to={res.stats.sourcesProcessed} /></b>
          <span className="stat-sub">патенты · статьи · отчёты · реестры</span>
        </div>
        <button className="stat card stat-btn" onClick={() => { setOnly(false); document.getElementById('list').scrollIntoView({ behavior: 'smooth' }) }}>
          <span className="stat-k">Кандидаты в слабые сигналы</span>
          <b className="stat-v c-amber"><CountUp to={res.stats.candidates} /></b>
          <span className="stat-sub">→ к ТОП-15</span>
        </button>
        <button className={`stat card stat-btn ${onlyConfident ? 'is-on' : ''}`} onClick={() => { setOnly(!onlyConfident); document.getElementById('list').scrollIntoView({ behavior: 'smooth' }) }}>
          <span className="stat-k">{DEMO_DATA ? 'Уверенность модели' : 'Скор'} &gt; 75%</span>
          <b className="stat-v c-cyan"><CountUp to={res.stats.confident} /></b>
          <span className="stat-sub">{onlyConfident ? '✓ фильтр включён' : '→ показать только их'}</span>
        </button>
      </div>

      <div className="card list" id="list">
        <div className="list-head">
          <h2>ТОП-{list.length} слабых сигналов</h2>
          <span className="muted">{DEMO_DATA ? 'Отсортировано по уверенности модели' : 'Отсортировано по скору: Σ вес × z-оценка'}</span>
        </div>
        <div className="row row-h" aria-hidden="true">
          <span>#</span><span>Технология</span><span>Скоринг</span><span>Ключевые предикторы</span><span>Динамика</span><span />
        </div>
        {list.map((s, i) => (
          <a key={s.id} className="row" href={`#${new URLSearchParams({ q: res.query, s: s.id })}`} style={{ '--i': i }}>
            <span className="row-n mono">{String(i + 1).padStart(2, '0')}</span>
            <span className="row-t"><b>{s.title}</b><small>{s.domain} · стадия: {s.stage}</small></span>
            <span><Score v={s.score} /></span>
            <span className="row-p">{s.summary}</span>
            <span><Spark data={s.trend} /></span>
            <span className="row-go">Инсайт →</span>
          </a>
        ))}
      </div>

      <div className="grid2">
        <div className="card graph-card">
          <div className="list-head">
            <h2>Карта сигналов</h2>
            <span className="muted">Граф из базы знаний · нажмите на узел</span>
          </div>
          {g ? <Graph data={g} onPick={(id) => go({ q: res.query, s: id })} /> : <div className="graph-empty">{g === false ? 'Граф недоступен' : 'Загружаю граф…'}</div>}
          <Legend />
        </div>
        <div className="card rejected">
          <div className="list-head">
            <h2>Отсеяно фильтром</h2>
            <span className="muted">Не слабые сигналы</span>
          </div>
          <ul>
            {res.rejected.map((r) => (
              <li key={r.title}>
                <span className={`tag tag-${r.category}`}>{REJECT[r.category]}</span>
                <b>{r.title}</b>
                <small>{r.reason}</small>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  )
}

const Legend = () => (
  <div className="legend">
    <span><i className="lg lg-query" />Запрос</span>
    <span><i className="lg lg-signal" />Сигнал (кольцо = уверенность)</span>
    <span><i className="lg lg-entity" />Фактор</span>
    <span><i className="lg lg-source" />Источник</span>
  </div>
)

function Score({ v, big }) {
  const l = level(v)
  if (big) return (
    <div className={`ring lv-${l}`} style={{ '--p': pct(v) }}>
      <svg viewBox="0 0 120 120"><circle cx="60" cy="60" r="52" className="ring-bg" /><circle cx="60" cy="60" r="52" className="ring-fg" pathLength="100" /></svg>
      <b><CountUp to={pct(v)} /><small>%</small></b>
      <span>{DEMO_DATA ? 'уверенность' : 'скор'}</span>
    </div>
  )
  return (
    <span className={`score lv-${l}`}>
      <span className="score-bar"><i style={{ width: `${pct(v)}%` }} /></span>
      <b className="mono">{pct(v)}%</b>
    </span>
  )
}

function Spark({ data }) {
  const max = Math.max(1, ...data), w = 90, h = 28
  const step = data.length > 1 ? w / (data.length - 1) : 0
  const pts = data.map((v, i) => `${i * step},${h - (v / max) * (h - 4) - 2}`).join(' ')
  return (
    <svg className="spark" viewBox={`0 0 ${w} ${h}`} aria-label="Рост упоминаний по кварталам">
      <polyline points={`0,${h} ${pts} ${w},${h}`} className="spark-fill" />
      <polyline points={pts} className="spark-line" />
    </svg>
  )
}

function Insight({ s, q }) {
  const [g, setG] = useState(null)
  useEffect(() => { setG(null); graph(q, s.id).then(setG, () => setG(false)) }, [s.id])
  const maxW = Math.max(1e-9, ...s.predictors.map((p) => Math.abs(p.weight)))

  return (
    <article className="doc">
      <nav className="doc-nav">
        <a href={`#${new URLSearchParams({ q })}`}>← К результатам «{q}»</a>
        <button className="ghost" onClick={() => print()}>Экспорт в PDF</button>
      </nav>
      <DemoBanner />

      <header className="doc-head card">
        <div>
          <p className="eyebrow">Аналитическая справка · {s.domain}</p>
          <h1>{s.title}</h1>
          <div className="doc-meta">
            <span className="pill">Стадия: {s.stage}</span>
            <span className="pill">Источников: {s.sources.length}</span>
            <span className="pill">Статус: слабый сигнал</span>
          </div>
        </div>
        <Score v={s.score} big />
      </header>

      <div className="doc-body">
        <div className="doc-main">
          <Section n="01" t="Описание технологии"><p>{s.description}</p></Section>
          {s.advantages.length > 0 && <Section n="02" t="Потенциальные преимущества">
            <ul className="adv">{s.advantages.map((a) => <li key={a}>{a}</li>)}</ul>
          </Section>}
          {s.cases.length > 0 && <Section n="03" t="Кейс-примеры">
            <div className="cases">{s.cases.map((c) => <div key={c.title} className="case"><b>{c.title}</b><p>{c.text}</p></div>)}</div>
          </Section>}
          {s.reports.length > 0 && <Section n="04" t="Оценки в аналитических отчётах">
            {s.reports.map((r) => <blockquote key={r.org}><p>{r.text}</p><cite>{r.org}</cite></blockquote>)}
          </Section>}
          {s.quotes?.length > 0 && <Section n="04" t="Цитаты из подтверждённых утверждений">
            {s.quotes.map((c, i) => (
              <blockquote key={i}><p>«{c.text}»</p><cite>{c.url ? <a href={c.url} target="_blank" rel="noreferrer">{c.title}</a> : c.title} · {c.date}</cite></blockquote>
            ))}
          </Section>}
          <Section n="05" t="Почему это слабый сигнал">
            <p>{s.whyWeak}</p>
            <div className="preds">
              {s.predictors.map((p) => (
                <div key={p.name} className={`pred ${p.weight < 0 ? 'neg' : 'pos'}`}>
                  <span className="pred-n">{p.name}<small>{p.value}</small></span>
                  <span className="pred-bar"><i style={{ width: `${(Math.abs(p.weight) / maxW) * 50}%` }} /></span>
                  <b className="mono">{p.weight > 0 ? '+' : ''}{p.weight.toFixed(2)}</b>
                </div>
              ))}
            </div>
            <p className="muted small">{DEMO_DATA ? 'Вклад признаков в решение модели: вправо — за слабый сигнал, влево — против (признаки зрелости или хайпа).' : 'Топ-3 вклада в скор: вес признака × его z-оценка среди всех кандидатов; вправо — за слабый сигнал, влево — против.'}</p>
          </Section>
          <Section n="06" t={DEMO_DATA ? 'Почему такая уверенность' : 'Как получен скор'}><p>{s.confidenceReason}</p></Section>
          <Section n="07" t="Источники">
            <div className="srcs">
              {s.sources.map((src, i) => (
                <div key={i} className="src">
                  <div className="src-top">
                    {src.url ? <a href={src.url} target="_blank" rel="noreferrer">{src.title}</a> : <b>{src.title}</b>}
                    <span className={`trust trust-${src.trust}`}>{TRUST[src.trust]} доверенность</span>
                  </div>
                  <div className="src-meta mono">
                    <span>{src.type}</span><span>{src.date}</span><span>язык: {src.lang}</span>
                    {src.generated && <span className="ai">ИИ-резюме / автоперевод</span>}
                  </div>
                  {src.summaryRu && <p className="src-sum">{src.summaryRu}</p>}
                  {src.trust === 'low' && <p className="src-warn">Первичный индикатор — требует подтверждения независимыми источниками.</p>}
                </div>
              ))}
            </div>
          </Section>
        </div>
        <aside className="doc-side">
          <div className="card side-card">
            <h3>Граф связей</h3>
            {g ? <Graph data={g} focusId={s.id} /> : <div className="graph-empty">{g === false ? 'Граф недоступен' : 'Загружаю…'}</div>}
            <Legend />
          </div>
          <div className="card side-card">
            <h3>Динамика упоминаний</h3>
            <Spark data={s.trend} />
            <p className="muted small">{s.trend.length} кварталов, упоминания во всех источниках</p>
          </div>
        </aside>
      </div>
    </article>
  )
}

const Section = ({ n, t, children }) => (
  <section className="sec">
    <h2><span className="mono">{n}</span>{t}</h2>
    {children}
  </section>
)
