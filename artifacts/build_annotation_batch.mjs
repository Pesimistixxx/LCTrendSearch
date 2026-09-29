import fs from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = "/Users/komputer/PycharmProjects/LCTrendSearch";
const outputDir = `${root}/outputs/annotation-batch-2026-09-29`;
const python = "/Users/komputer/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3";
const { review, supplied } = JSON.parse(
  execFileSync(python, [`${root}/artifacts/annotation_batch_data.py`], {
    encoding: "utf8",
    maxBuffer: 8 * 1024 * 1024,
  }),
);
const wb = Workbook.create();
const guide = wb.worksheets.add("Правила");
const reviewSheet = wb.worksheets.add("Исторические срезы");
const sourceSheet = wb.worksheets.add("Список 100");

const guideRows = [
  ["Разметка слабых сигналов", "300 исторических срезов и 100 записей внешнего списка"],
  ["Единица разметки", "Одна технология на конкретную дату snapshot_date. Одна и та же технология может менять статус со временем."],
  ["reviewer_state", "weak — ранний достоверный сигнал на дату T; trend — уже заметный тренд; mature — зрелая технология; faded — не развилась; insufficient — не хватает данных; rejected — неверно выделенная технология."],
  ["reviewer_signal_12m / 36m", "1 — на T это был слабый сигнал, впоследствии подтверждённый независимыми источниками в указанный срок. 0 — проверенный отрицательный исход. Пусто — неизвестно или охват источников недостаточен."],
  ["reviewer_trend_12m / 36m", "1 — за указанный срок технология стала заметным трендом; 0 — проверенный отрицательный исход; пусто — неизвестно. Не подменяйте слабый сигнал одним только ростом публикаций."],
  ["Временная граница", "Для оценки статуса на T смотрите только документы, датированные не позже T. Последующие документы нужны отдельно для проверки исхода за 12 или 36 месяцев."],
  ["Неполный горизонт", "Если calendar_complete_36m = FALSE, оставьте обе 36-месячные метки пустыми. Календарное завершение само по себе не доказывает полноту источников."],
  ["evidence_notes", "Укажите URL/ID решающих источников и коротко объясните 1 или 0. Если вывод невозможен, укажите, каких данных не хватает."],
  ["Отбор строк", "В партии 300 разных технологических семейств. Часть строк намеренно отобрана по наличию последующих документов; эта партия не отражает долю слабых сигналов во всём графе."],
  ["Ссылки в графе", "В колонках источников показано до четырёх документов за период. Это подсказки для проверки, а не полный обзор литературы. Пустая колонка не доказывает отсутствие результата."],
  ["Ретроспектива", "Срезы строились по датам документов из графа 2026 года. Документ мог быть получен системой позднее даты T; для строгого backtest потребуется аудит времени получения."],
  ["Список 100", "Это предоставленные кандидаты на сентябрь 2026, не готовые метки исхода через 12/36 месяцев. Проверьте сопоставление с графом; похожее название не означает тождественную технологию."],
  ["Источники", "Исторические срезы: выгрузка графа 2026-09-29. Список 100: 100_слабых_технологических_сигналов_сентябрь_2026.xlsx, лист «Слабые сигналы». Ссылки на публикации есть в колонках источников."],
];
guide.getRange(`A1:B${guideRows.length}`).values = guideRows;
guide.getRange(`A1:B${guideRows.length}`).format.font = { name: "Arial", size: 11, color: "#1F2937" };
guide.getRange("A1:B1").format.font = { name: "Arial", size: 13, bold: true, color: "#111827" };
guide.getRange(`A2:A${guideRows.length}`).format.font = { name: "Arial", size: 11, bold: true, color: "#1F2937" };
guide.getRange(`A1:A${guideRows.length}`).format.columnWidth = 30;
guide.getRange(`B1:B${guideRows.length}`).format.columnWidth = 105;
guide.getRange(`B1:B${guideRows.length}`).format.wrapText = true;
guide.getRange(`A2:B${guideRows.length}`).format.rowHeight = 55;
guide.getRange("A1:B1").format.rowHeight = 35;
guide.showGridLines = false;

const reviewFields = [
  "technology_id", "technology", "snapshot_date", "reviewer_state",
  "reviewer_signal_12m", "reviewer_signal_36m", "reviewer_trend_12m",
  "reviewer_trend_36m", "evidence_notes", "family_id", "first_seen_date",
  "horizon_12m_end", "horizon_end", "calendar_complete_12m",
  "calendar_complete_36m", "document_count", "source_families",
  "organizations", "pre_t_sources", "followup_0_12m_sources",
  "followup_13_36m_sources", "sample_group",
];
const dateFields = new Set(["snapshot_date", "first_seen_date", "horizon_12m_end", "horizon_end"]);
const boolFields = new Set(["calendar_complete_12m", "calendar_complete_36m"]);
const reviewMatrix = review.map((row) => reviewFields.map((field) => {
  const value = row[field];
  if (dateFields.has(field)) return value ? new Date(`${value}T00:00:00Z`) : null;
  if (boolFields.has(field)) return value === "True";
  if (field === "document_count") return Number(value);
  return value || null;
}));
reviewSheet.getRange(`A1:V1`).values = [reviewFields];
reviewSheet.getRangeByIndexes(1, 0, review.length, reviewFields.length).values = reviewMatrix;
reviewSheet.getRange(`A1:V${review.length + 1}`).format.font = { name: "Arial", size: 10, color: "#1F2937" };
reviewSheet.getRange("A1:V1").format = {
  fill: "#24324A", font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
};
reviewSheet.getRange("A1:V1").format.wrapText = true;
reviewSheet.getRange("A1:V1").format.rowHeight = 36;
reviewSheet.getRange(`D2:I${review.length + 1}`).format.fill = "#FFF4CE";
reviewSheet.getRange(`D2:D${review.length + 1}`).dataValidation = {
  rule: { type: "list", values: ["weak", "trend", "mature", "faded", "insufficient", "rejected"] },
};
reviewSheet.getRange(`E2:H${review.length + 1}`).dataValidation = {
  rule: { type: "list", values: ["0", "1"] },
};
for (const letter of ["C", "K", "L", "M"]) {
  reviewSheet.getRange(`${letter}2:${letter}${review.length + 1}`).setNumberFormat("yyyy-mm-dd");
}
reviewSheet.getRange(`A1:A${review.length + 1}`).format.columnWidth = 29;
reviewSheet.getRange(`B1:B${review.length + 1}`).format.columnWidth = 46;
reviewSheet.getRange(`C1:C${review.length + 1}`).format.columnWidth = 17;
reviewSheet.getRange(`D1:H${review.length + 1}`).format.columnWidth = 20;
reviewSheet.getRange(`I1:I${review.length + 1}`).format.columnWidth = 53;
reviewSheet.getRange(`J1:R${review.length + 1}`).format.columnWidth = 22;
reviewSheet.getRange(`S1:U${review.length + 1}`).format.columnWidth = 68;
reviewSheet.getRange(`V1:V${review.length + 1}`).format.columnWidth = 22;
reviewSheet.getRange(`I2:I${review.length + 1}`).format.wrapText = true;
reviewSheet.getRange(`S2:U${review.length + 1}`).format.wrapText = true;
reviewSheet.getRange(`A2:V${review.length + 1}`).format.rowHeight = 80;
reviewSheet.freezePanes.freezeRows(1);
reviewSheet.freezePanes.freezeColumns(3);
reviewSheet.tables.add(`A1:V${review.length + 1}`, true, "HistoricalAnnotation");
reviewSheet.showGridLines = false;

const sourceFields = [
  "source_no", "technology", "domain", "companies", "why_weak", "stage",
  "trend", "source_score", "source_urls", "graph_technology_id",
  "match_status", "match_notes",
];
sourceSheet.getRange("A1:L1").values = [sourceFields];
sourceSheet.getRangeByIndexes(1, 0, supplied.length, sourceFields.length).values = supplied.map(
  (row) => sourceFields.map((field) => row[field] || null),
);
sourceSheet.getRange(`A1:L${supplied.length + 1}`).format.font = { name: "Arial", size: 10, color: "#1F2937" };
sourceSheet.getRange("A1:L1").format = {
  fill: "#24324A", font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
};
sourceSheet.getRange("A1:L1").format.wrapText = true;
sourceSheet.getRange("A1:L1").format.rowHeight = 36;
sourceSheet.getRange(`J2:L${supplied.length + 1}`).format.fill = "#FFF4CE";
sourceSheet.getRange(`K2:K${supplied.length + 1}`).dataValidation = {
  rule: { type: "list", values: ["confirmed", "not_in_graph", "ambiguous"] },
};
for (const [column, width] of Object.entries({
  A: 12, B: 54, C: 18, D: 42, E: 65, F: 28, G: 50, H: 15,
  I: 70, J: 30, K: 22, L: 45,
})) {
  sourceSheet.getRange(`${column}1:${column}${supplied.length + 1}`).format.columnWidth = width;
}
sourceSheet.getRange(`B2:L${supplied.length + 1}`).format.wrapText = true;
sourceSheet.getRange(`A2:L${supplied.length + 1}`).format.rowHeight = 76;
sourceSheet.freezePanes.freezeRows(1);
sourceSheet.tables.add(`A1:L${supplied.length + 1}`, true, "ProvidedSignals");
sourceSheet.showGridLines = false;

for (const [name, range] of [
  ["Исторические срезы", "A1:I5"],
  ["Список 100", "A1:I5"],
  ["Правила", "A1:B10"],
]) {
  const result = await wb.inspect({ kind: "table", range: `${name}!${range}`, include: "values,formulas", tableMaxRows: 10, tableMaxCols: 12, maxChars: 2300 });
  console.log(name, result.ndjson.slice(0, 2300));
  const preview = await wb.render({ sheetName: name, range, scale: 1.4, format: "png" });
  await fs.writeFile(`/private/tmp/annotation-${name.replaceAll(" ", "-")}.png`, new Uint8Array(await preview.arrayBuffer()));
}
const errors = await wb.inspect({
  kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 100 }, summary: "formula error scan",
});
console.log("errors", errors.ndjson.slice(0, 1000));
await fs.mkdir(outputDir, { recursive: true });
const output = await SpreadsheetFile.exportXlsx(wb);
const target = `${outputDir}/weak-signals-annotation-batch.xlsx`;
await output.save(target);
console.log(JSON.stringify({ target, review_rows: review.length, supplied_rows: supplied.length }));
