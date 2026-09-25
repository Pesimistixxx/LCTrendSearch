import { useEffect, useMemo, useRef, useState } from 'react'
import { forceSimulation, forceLink, forceManyBody, forceCenter, forceCollide } from 'd3-force'

const R = { query: 26, signal: 13, entity: 9, source: 7 }
const W = 900, H = 560

// Граф связей из Neo4j: запрос → сигналы → сущности / источники.
export default function Graph({ data, onPick, focusId }) {
  const [, tick] = useState(0)
  const [hover, setHover] = useState(null)
  const drag = useRef(null)
  const svg = useRef(null)

  const { nodes, links, sim } = useMemo(() => {
    const nodes = data.nodes.map((n) => ({ ...n }))
    const links = data.links.map((l) => ({ ...l }))
    const sim = forceSimulation(nodes)
      .force('link', forceLink(links).id((d) => d.id).distance((l) => (l.source.type === 'query' ? 170 : 90)).strength(0.6))
      .force('charge', forceManyBody().strength((d) => (d.type === 'query' ? -900 : -260)))
      .force('center', forceCenter(W / 2, H / 2))
      .force('collide', forceCollide((d) => R[d.type] + 22))
    return { nodes, links, sim }
  }, [data])

  useEffect(() => {
    sim.on('tick', () => tick((t) => t + 1))
    return () => sim.stop()
  }, [sim])

  const near = useMemo(() => {
    if (!hover) return null
    const s = new Set([hover])
    links.forEach((l) => {
      if (l.source.id === hover) s.add(l.target.id)
      if (l.target.id === hover) s.add(l.source.id)
    })
    return s
  }, [hover, links, nodes])

  const toSvg = (e) => {
    const p = svg.current.createSVGPoint()
    p.x = e.clientX; p.y = e.clientY
    return p.matrixTransform(svg.current.getScreenCTM().inverse())
  }
  const down = (e, n) => {
    e.currentTarget.setPointerCapture(e.pointerId)
    drag.current = { n, moved: false }
    sim.alphaTarget(0.25).restart()
  }
  const move = (e) => {
    if (!drag.current) return
    const { x, y } = toSvg(e)
    Object.assign(drag.current.n, { fx: x, fy: y })
    drag.current.moved = true
  }
  const up = () => {
    const d = drag.current
    if (!d) return
    d.n.fx = d.n.fy = null
    sim.alphaTarget(0)
    if (!d.moved && d.n.type === 'signal') onPick?.(d.n.id)
    drag.current = null
  }

  const dim = (id) => near && !near.has(id)

  // подгоняем viewBox под фактическое облако узлов (с полями под подписи)
  const xs = nodes.map((n) => n.x || W / 2), ys = nodes.map((n) => n.y || H / 2)
  const pad = 70
  const x0 = Math.min(...xs) - pad, x1 = Math.max(...xs) + pad
  const y0 = Math.min(...ys) - pad, y1 = Math.max(...ys) + pad
  const vw = Math.max(x1 - x0, (y1 - y0) * (W / H)), vh = vw * (H / W)
  const vb = `${(x0 + x1 - vw) / 2} ${(y0 + y1 - vh) / 2} ${vw} ${vh}`

  return (
    <svg ref={svg} className="graph" viewBox={vb} onPointerMove={move} onPointerUp={up} onPointerLeave={up}
      role="img" aria-label="Граф связей слабых сигналов">
      <defs>
        <radialGradient id="g-query"><stop offset="0" stopColor="#fff" /><stop offset=".35" stopColor="var(--cyan)" /><stop offset="1" stopColor="var(--cyan)" stopOpacity="0" /></radialGradient>
        <radialGradient id="g-halo"><stop offset="0" stopColor="var(--cyan)" stopOpacity=".45" /><stop offset="1" stopColor="var(--cyan)" stopOpacity="0" /></radialGradient>
        <radialGradient id="g-halo-v"><stop offset="0" stopColor="var(--violet)" stopOpacity=".4" /><stop offset="1" stopColor="var(--violet)" stopOpacity="0" /></radialGradient>
      </defs>
      {[90, 180, 270].map((r) => (
        <circle key={r} cx={W / 2} cy={H / 2} r={r} className="graph-ring" />
      ))}
      <g>
        {links.map((l, i) => (
          <line key={i} x1={l.source.x} y1={l.source.y} x2={l.target.x} y2={l.target.y}
            className={`graph-link ${l.source.type === 'query' ? 'is-main' : ''} ${dim(l.source.id) || dim(l.target.id) ? 'is-dim' : ''}`}
            strokeWidth={0.6 + (l.weight || 0.5) * 1.8} />
        ))}
      </g>
      <g>
        {nodes.map((n) => {
          const r = R[n.type]
          const cls = `graph-node t-${n.type} ${dim(n.id) ? 'is-dim' : ''} ${focusId === n.id ? 'is-focus' : ''} ${n.trust ? 'trust-' + n.trust : ''}`
          return (
            <g key={n.id} className={cls} transform={`translate(${n.x || 0},${n.y || 0})`}
              onPointerDown={(e) => down(e, n)} onPointerEnter={() => setHover(n.id)} onPointerLeave={() => setHover(null)}>
              {n.type === 'query' && <circle r={r * 2.6} fill="url(#g-query)" opacity=".35" className="pulse" />}
              {n.type === 'signal' && <circle r={r * (1.6 + (n.score || 0.5))} fill={`url(#g-halo${n.score > 0.75 ? '' : '-v'})`} />}
              <circle r={r} className="graph-dot" />
              {n.type === 'signal' && (
                <circle r={r + 4} className="graph-score" pathLength="100"
                  strokeDasharray={`${Math.round((n.score || 0) * 100)} 100`} transform="rotate(-90)" />
              )}
              <text y={n.type === 'query' ? 5 : r + 16} className="graph-label">
                {n.type === 'query' ? '⌕' : n.label.length > 28 && hover !== n.id ? n.label.slice(0, 26) + '…' : n.label}
              </text>
              {n.type === 'signal' && hover === n.id && (
                <text y={-r - 10} className="graph-label graph-tip">{Math.round(n.score * 100)}% · открыть →</text>
              )}
            </g>
          )
        })}
      </g>
    </svg>
  )
}
