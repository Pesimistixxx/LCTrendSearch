import React, { useEffect, useState } from 'react'
import { api } from './api.js'

const ACTIVE = new Set(['queued', 'running', 'cancelling', 'pausing'])
const labels = {
  ok: 'готово', queued: 'в очереди', running: 'обработка', succeeded: 'готово', partial: 'частично',
  failed: 'ошибка', cancelled: 'отменено', cancelling: 'остановка', interrupted: 'прервано',
  disabled: 'отключено', pending: 'ожидание', done: 'завершено', parsing: 'разбор текста',
  fetching: 'получение статьи', downloading: 'загрузка файла', fulltext: 'получение PDF',
  model_loading: 'загрузка GLiNER', neo4j: 'подключение к базе',
  extract: 'извлечение LLM', review: 'проверка LLM', validation: 'проверка цитат',
  plan: 'подготовка пакетов', context: 'дополнительный контекст', assemble: 'сборка результата',
  resolution: 'сопоставление сущностей', publication: 'запись в Neo4j', snapshot: 'сохранение источника',
  discovery: 'получение статей', downloading_snapshot: 'скачивание базы', reading_snapshot: 'чтение базы',
  paused: 'остановлено', pausing: 'остановка', parsed: 'обработано', discovering: 'обход источников',
  completed: 'завершено', processing: 'обработка материала', openalex: 'OpenAlex', github: 'GitHub', pypi: 'PyPI',
  limited: 'выдача ограничена API', idle: 'не начат', unsupported: 'не подключён', epo: 'EPO',
  linked: 'по ссылкам из GitHub', complete: 'завершён',
  hydration: 'получение текста источника', metadata: 'получение связанных материалов',
  accepted: 'принято', rejected: 'отклонено', needs_review: 'нужна проверка',
  solves_task: 'решает задачу', changes_metric: 'изменяет метрику',
}
const label = value => labels[value] || (value?.includes(':') ? value.split(':').map(part => labels[part] || part).join(' · ') : value) || 'ожидание'
const message = error => typeof error === 'string' ? error : error?.message || error?.code || ''

export default function Ingestion() {
  const [service, setService] = useState(null), [crawls, setCrawls] = useState([])
  const [crawlId, setCrawlId] = useState(''), [crawl, setCrawl] = useState(null), [topic, setTopic] = useState('')
  const [busy, setBusy] = useState(false), [error, setError] = useState('')
  const [refresh, setRefresh] = useState(0)
  useEffect(() => {
    let stopped = false, timer
    async function poll() {
      try {
        const [status, listing, selected] = await Promise.all([
          api.status(), api.crawls(), crawlId ? api.crawl(crawlId) : Promise.resolve(null),
        ])
        if (stopped) return
        setService(status); setCrawls(listing.crawls || []); setCrawl(selected); setError('')
        if (!crawlId && listing.crawls?.length) setCrawlId(listing.crawls[0].crawl_id)
        timer = setTimeout(poll, listing.crawls?.some(item => ACTIVE.has(item.status)) ? 2000 : 10000)
      } catch (e) {
        if (!stopped) { setError(e.message); timer = setTimeout(poll, 5000) }
      }
    }
    poll()
    return () => { stopped = true; clearTimeout(timer) }
  }, [crawlId, refresh])

  async function upload(event) {
    event.preventDefault(); setBusy(true); setError('')
    try {
      const created = await api.createCrawl(topic.trim())
      setCrawlId(created.crawl_id); setCrawl(created); setRefresh(value => value + 1)
    } catch (e) { setError(e.message) }
    finally { setBusy(false) }
  }
  async function control(action) {
    setBusy(true)
    try { await (action === 'pause' ? api.pauseCrawl(crawlId) : action === 'retry' ? api.retryFailedCrawl(crawlId) : api.resumeCrawl(crawlId)); setRefresh(value => value + 1) }
    catch (e) { setError(e.message) }
    finally { setBusy(false) }
  }
  const ready = service?.neo4j?.available && service?.llm?.configured && service?.pdf?.installed
  const active = crawls.some(item => ACTIVE.has(item.status))
  const counts = crawl?.counts || {}
  const sources = Array.isArray(crawl?.sources) ? crawl.sources : Object.entries(crawl?.sources || {}).map(([name, value]) => ({ name, ...value }))
  return <section className="ingest">
    <p className="eyebrow">Источники и проверка загрузки</p>
    <h1>Сбор материалов</h1>
    <p className="status">База: {service ? service.neo4j?.available ? 'подключена' : 'недоступна' : 'проверяем'} · LLM: {service?.llm?.configured ? 'настроена' : 'нужна настройка'} · GLiNER: {service?.gliner?.installed ? 'установлен' : 'не установлен'}</p>
    <form onSubmit={upload}>
      <label>Тематика<input value={topic} onChange={event => setTopic(event.target.value)} placeholder="Пусто — все настроенные направления" maxLength={1000} /></label>
      <p className="hint">Отправляем тему в API OpenAlex и GitHub, получаем карточки статей и репозитории, затем извлекаем текст; пакеты PyPI берём из ссылок в README.</p>
      <p className="pipeline">Текст → GLiNER: подсказки → LLM: извлечение и проверка → Neo4j</p>
      <button type="submit" disabled={busy || !ready || active}>{busy ? 'Подождите…' : 'Собирать'}</button>
      {!ready && service && <p className="hint">Перед загрузкой нужно подключить базу и настроить модели.{!service.pdf?.installed && ' Для получения PDF требуется модуль Docling.'}</p>}
    </form>
    <Settings service={service} active={active} onSave={status => { setService(status); setRefresh(value => value + 1) }} />
    {error && <p className="error" role="alert">{error}</p>}
    {crawls.length > 1 && <label className="history">Запуск<select value={crawlId} onChange={event => setCrawlId(event.target.value)}>{crawls.map(item => <option key={item.crawl_id} value={item.crawl_id}>{item.topic || 'Все настроенные направления'} · {new Date(item.created_at).toLocaleString('ru-RU')} · {label(item.status)}</option>)}</select></label>}
    {crawl ? <section className="run">
      <div className="run-heading"><h2>{crawl.topic || 'Все настроенные направления'}</h2>{ACTIVE.has(crawl.status) ? <button className="secondary" disabled={busy || crawl.status === 'pausing'} onClick={() => control('pause')}>Остановить</button> : ['paused', 'interrupted', 'failed'].includes(crawl.status) && <button className="secondary" disabled={busy || !ready} onClick={() => control('resume')}>Продолжить</button>}</div>
      <p>Обработано {counts.parsed || 0} из {counts.discovered || 0} найденных материалов · частично {counts.partial || 0} · ожидают {counts.pending || 0} · в работе {counts.processing || 0} · ошибки {counts.failed || 0}</p>
      <p className="hint">{crawl.status === 'pausing' ? 'Останавливаем после текущего материала.' : `Сейчас: ${label(crawl.stage)} · ${label(crawl.status)}`} Повторов пропущено: {counts.duplicates || 0}.</p>
      {counts.discovered > 0 && <progress value={(counts.parsed || 0) + (counts.partial || 0) + (counts.failed || 0)} max={counts.discovered} />}
      {sources.map(item => <div key={item.source}><p>{label(item.source)}: обработано {item.parsed || 0}, ожидают {item.pending || 0} · {item.complete ? 'обход завершён' : label(item.status)} · всего {item.total ?? 'неизвестно'}</p>{item.limitations?.length > 0 && <p className="hint">{item.limitations.map(message).join(' ')}</p>}</div>)}
      {crawl.error && <p className="error">{message(crawl.error)}</p>}
      <Materials crawlId={crawlId} status="parsed" title="Обработанные материалы" refresh={crawl.updated_at || JSON.stringify(counts)} />
      <Materials crawlId={crawlId} status="pending" title="Ожидают обработки" refresh={crawl.updated_at || JSON.stringify(counts)} />
      <Materials crawlId={crawlId} status="failed" title="Ошибки обработки" refresh={crawl.updated_at || JSON.stringify(counts)} action={!active && counts.failed > 0 && ready ? <button className="secondary" disabled={busy} onClick={() => control('retry')}>Повторить ошибки</button> : null} />
    </section> : <p className="hint">Загрузок пока нет.</p>}
  </section>
}

function Settings({ service, active, onSave }) {
  const [opened, setOpened] = useState(false)
  const [provider, setProvider] = useState('openai_compatible'), [model, setModel] = useState('')
  const [url, setUrl] = useState('https://api.openai.com/v1'), [key, setKey] = useState('')
  const [busy, setBusy] = useState(false), [error, setError] = useState(''), [initialized, setInitialized] = useState(false)
  useEffect(() => {
    if (!service || initialized) return
    setProvider(service.llm?.provider || 'openai_compatible'); setModel(service.llm?.models?.extract || '')
    setUrl(service.llm?.base_url || 'https://api.openai.com/v1'); setInitialized(true)
  }, [service, initialized])
  async function save(event) {
    event.preventDefault(); setBusy(true); setError('')
    try { onSave(await api.settings({ provider, model: model.trim(), base_url: url.trim(), api_key: key.trim() || null })); setKey('') }
    catch (e) { setError(e.message) }
    finally { setBusy(false) }
  }
  return <details className="settings" open={opened}><summary onClick={event => { event.preventDefault(); setOpened(value => !value) }}>Настройки LLM</summary><form onSubmit={save}>
    <label>Провайдер<select disabled={active} value={provider} onChange={event => { const value = event.target.value; setProvider(value); setModel(''); setUrl(value === 'gigachat' ? 'https://api.giga.chat/v1' : 'https://api.openai.com/v1') }}><option value="openai_compatible">OpenAI-совместимый API</option><option value="gigachat">GigaChat</option></select></label>
    <label>Адрес API<input type="url" value={url} onChange={event => setUrl(event.target.value)} required disabled={active} /></label>
    <label>Модель<input value={model} onChange={event => setModel(event.target.value)} placeholder={provider === 'gigachat' ? 'Пустое поле — автоматический выбор' : 'Название модели'} maxLength={200} required={provider !== 'gigachat'} disabled={active} /></label>
    <label>{provider === 'gigachat' ? 'Ключ авторизации GigaChat' : 'Ключ API'}<input type="password" autoComplete="new-password" value={key} onChange={event => setKey(event.target.value)} placeholder={service?.llm?.has_key ? 'Ключ уже задан. Пустое поле сохранит его.' : 'Введите ключ'} disabled={active} /></label>
    {active && <p className="hint">Настройки можно менять после завершения загрузки.</p>}
    {error && <p className="error" role="alert">{error}</p>}
    <button disabled={busy || active}>{busy ? 'Сохраняем…' : 'Сохранить'}</button>
  </form></details>
}

function Materials({ crawlId, status, title, refresh, action }) {
  const [opened, setOpened] = useState(false), [materials, setMaterials] = useState(null), [error, setError] = useState('')
  const [pages, setPages] = useState(1), [more, setMore] = useState(false), [loading, setLoading] = useState(false)
  useEffect(() => {
    setMaterials(null); setPages(1); setMore(false)
  }, [crawlId])
  useEffect(() => {
    if (!opened) return
    let stale = false
    const statuses = status === 'parsed' ? ['parsed', 'partial'] : status === 'pending' ? ['pending', 'processing'] : [status]
    const size = 100 / statuses.length
    setLoading(true)
    Promise.all(statuses.flatMap(value => Array.from({ length: pages }, (_, page) => api.materials(crawlId, value, page * size, size)))).then(results => {
      if (!stale) {
        setMaterials([...new Map(results.flatMap(item => item.materials || []).map(item => [item.material_id, item])).values()]); setError('')
        setMore(results.some((item, index) => index % pages === pages - 1 && (Number.isFinite(item.total) ? item.total > pages * size : item.materials?.length === size)))
      }
    }).catch(e => { if (!stale) setError(e.message) }).finally(() => { if (!stale) setLoading(false) })
    return () => { stale = true }
  }, [opened, crawlId, status, refresh, pages])
  return <details className="materials" open={opened}><summary onClick={event => { event.preventDefault(); setOpened(value => !value) }}>{title}</summary>
    {error && <p className="error">{error}</p>}
    {!materials ? <p className="hint">Загружаем список…</p> : !materials.length ? <p className="hint">Пока нет.</p> : materials.map(item => item.job_id && item.doc_id ? <Document key={item.material_id} jobId={item.job_id} document={{ ...item, result_ready: ['parsed', 'partial', 'failed'].includes(item.status) }} /> : <p key={item.material_id}>{item.title || item.canonical_id} <small>· {label(item.source)} · {label(item.status)}</small>{item.error && <span className="error"> · {message(item.error)}</span>}</p>)}
    {more && <button className="secondary" disabled={loading} onClick={() => setPages(value => value + 1)}>{loading ? 'Загружаем…' : 'Ещё 100'}</button>}
    {action}
  </details>
}

function Document({ jobId, document }) {
  const [opened, setOpened] = useState(false), [result, setResult] = useState(null), [error, setError] = useState('')
  useEffect(() => {
    if (!opened || !document.result_ready) return
    let stale = false
    api.result(jobId, document.doc_id).then(data => { if (!stale) { setResult(data); setError('') } }).catch(e => { if (!stale) setError(e.message) })
    return () => { stale = true }
  }, [opened, jobId, document.doc_id, document.result_ready])
  const extraction = result?.extraction, concepts = extraction?.concepts || []
  const names = Object.fromEntries(concepts.map(item => [item.concept_id, item.preferred_label]))
  return <details className="document" open={opened}><summary onClick={event => { event.preventDefault(); setOpened(value => !value) }}>
    <strong>{document.title || document.canonical_id || document.doc_id}</strong><span>{document.stage ? `${label(document.stage)} · ` : ''}{label(document.source)} · {label(document.status)}</span>
    {(document.llm_status || document.gliner_status) && <small>LLM: {label(document.llm_status)} · GLiNER: {label(document.gliner_status)}</small>}
  </summary>
    {document.error && <p className="error">{message(document.error)}</p>}
    {error && <p className="error">{error}</p>}
    {!document.result_ready ? <p className="hint">Результат ещё не готов.</p> : !result ? <p className="hint">Загружаем результат…</p> : <>
      <p><a href={api.downloadUrl(jobId, document.doc_id)} download>Скачать результат JSON</a></p>
      <p className="hint">LLM: {label(extraction?.run?.status)} · GLiNER: {label(extraction?.run?.metadata?.ner?.status)}</p>
      <h3>Сущности</h3>{concepts.length ? <ul>{concepts.map(item => <li key={item.concept_id}>{item.preferred_label} <small>({item.kind})</small></li>)}</ul> : <p className="hint">Не выделены.</p>}
      <h3>Утверждения</h3>{extraction?.assertions?.length ? extraction.assertions.map(claim => <article key={claim.assertion_id}>
        <p><b>{label(claim.predicate)}</b> · {label(claim.status)}</p>
        <p>{Object.entries(claim.roles || {}).map(([role, id]) => `${role}: ${names[id] || id}`).join('; ')}</p>
        {(claim.evidence || []).map((evidence, index) => <blockquote key={index}>{evidence.quote}</blockquote>)}
      </article>) : <p className="hint">Проверенных утверждений нет.</p>}
    </>}
  </details>
}
