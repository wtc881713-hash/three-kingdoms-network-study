import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const root = "E:/three-kingdoms-network";
const inputPath = `${root}/outputs/few_shot_pilot/annotation_batch_01.xlsx`;
const reportPath = `${root}/outputs/reports/annotation_validation_report.json`;
const outputDir = `${root}/outputs/few_shot_pilot`;
const outputPath = `${outputDir}/annotation_batch_01_corrections_needed.xlsx`;
const report = JSON.parse(await fs.readFile(reportPath, "utf8"));
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));
const annotation = workbook.worksheets.getItem("Pilot Annotation");
const summary = workbook.worksheets.getOrAdd("Validation Summary");

const values = annotation.getRange("A1:X21").values;
const rowById = new Map();
for (let index = 1; index < values.length; index += 1) {
  rowById.set(String(values[index][0]), index + 1);
}

const issueByCode = new Map(report.issues.map((item) => [item.code, item]));
for (const id of issueByCode.get("numeric_only_evidence")?.rows ?? []) {
  const row = rowById.get(id);
  annotation.getRange(`U${row}`).format = {
    fill: "#F4CCCC",
    font: { bold: true, color: "#9C0006" },
    borders: { preset: "outside", style: "medium", color: "#C00000" },
  };
}
for (const id of issueByCode.get("evidence_not_exact_substring")?.rows ?? []) {
  const row = rowById.get(id);
  annotation.getRange(`U${row}`).format = {
    fill: "#FCE4D6",
    font: { color: "#9C5700" },
    borders: { preset: "outside", style: "thin", color: "#ED7D31" },
  };
}
for (const id of issueByCode.get("duplicate_primary_secondary")?.rows ?? []) {
  const row = rowById.get(id);
  annotation.getRange(`P${row}`).format = {
    fill: "#FCE4D6",
    font: { color: "#9C5700" },
    borders: { preset: "outside", style: "thin", color: "#ED7D31" },
  };
}
for (const id of issueByCode.get("uncertain_primary_with_specific_secondary")?.rows ?? []) {
  const row = rowById.get(id);
  annotation.getRange(`O${row}:P${row}`).format = {
    fill: "#FCE4D6",
    font: { color: "#9C5700" },
    borders: { preset: "outside", style: "thin", color: "#ED7D31" },
  };
}
annotation.getRange("N2:N21").format = {
  fill: "#E7E6E6",
  font: { color: "#666666", italic: true },
};

summary.showGridLines = false;
summary.getRange("A1:E1").merge();
summary.getRange("A1").values = [["Pilot Annotation Validation — Corrections Needed"]];
summary.getRange("A1:E1").format = {
  fill: "#17365D",
  font: { name: "Aptos Display", size: 16, bold: true, color: "#FFFFFF" },
  rowHeight: 34,
  verticalAlignment: "center",
};
summary.getRange("A3:B8").values = [
  ["Status", report.status],
  ["Reviewed rows", report.reviewed_rows],
  ["Critical issue groups", report.critical_issue_count],
  ["Warning groups", report.warning_count],
  ["Primary labels covered", report.covered_primary_labels.length],
  ["Unique character pairs", report.unique_character_pairs],
];
summary.getRange("A3:A8").format = { fill: "#D9EAF7", font: { bold: true, color: "#17365D" } };
summary.getRange("B3:B8").format = { fill: "#F5F8FC" };

summary.getRange("A10:E10").values = [["Level", "Issue", "Affected instances", "What to do", "Required before next batch?"]];
summary.getRange("A10:E10").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
  rowHeight: 32,
};
const actions = {
  numeric_only_evidence: "Replace 42 with a short exact quotation from the passage.",
  evidence_not_exact_substring: "Check whether the quotation was combined or shortened; prefer one exact passage span.",
  duplicate_primary_secondary: "Leave secondary_relation blank unless it adds a different relation.",
  uncertain_primary_with_specific_secondary: "Confirm whether the specific secondary label should become the primary label.",
  unexpected_model_confidence: "No action needed. This column is ignored until a classifier exists.",
};
const rows = report.issues.map((item) => [
  item.level,
  item.code,
  item.rows.join(", ") || "Batch-level",
  actions[item.code] ?? item.message,
  item.level === "critical" ? "Yes" : "No — review recommended",
]);
summary.getRange(`A11:E${10 + rows.length}`).values = rows;
summary.getRange(`A11:E${10 + rows.length}`).format = {
  wrapText: true,
  verticalAlignment: "top",
  borders: { insideHorizontal: { style: "thin", color: "#D9E2F3" } },
};
for (let index = 0; index < rows.length; index += 1) {
  const row = 11 + index;
  summary.getRange(`A${row}:E${row}`).format.fill = rows[index][0] === "critical" ? "#F4CCCC" : "#FCE4D6";
}
summary.getRange("A18:E20").values = [
  ["Colour guide", "Red", "Must correct", "Orange", "Review recommended"],
  ["After correction", "Keep all rows marked reviewed.", "Save this workbook.", "Return it to Codex.", "Do not edit model_confidence."],
  ["Next gate", "Validation must have no critical issues before expanding the dataset.", "", "", ""],
];
summary.getRange("A18:E20").format = { wrapText: true, fill: "#FFF2CC" };
summary.getRange("A:A").format.columnWidth = 18;
summary.getRange("B:B").format.columnWidth = 28;
summary.getRange("C:C").format.columnWidth = 55;
summary.getRange("D:D").format.columnWidth = 65;
summary.getRange("E:E").format.columnWidth = 28;
summary.freezePanes.freezeRows(1);

const inspect = await workbook.inspect({
  kind: "table",
  range: "'Validation Summary'!A1:E20",
  include: "values,formulas",
  tableMaxRows: 20,
  tableMaxCols: 5,
  maxChars: 12000,
});
await fs.writeFile(`${outputDir}/correction_summary_inspect.ndjson`, inspect.ndjson, "utf8");
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "correction workbook formula error scan",
});
await fs.writeFile(`${outputDir}/correction_formula_errors.ndjson`, errors.ndjson, "utf8");
for (const [sheetName, range, name] of [
  ["Validation Summary", "A1:E20", "correction_summary_preview.png"],
  ["Pilot Annotation", "O1:X10", "correction_rows_upper_preview.png"],
  ["Pilot Annotation", "O11:X21", "correction_rows_lower_preview.png"],
]) {
  const preview = await workbook.render({ sheetName, range, scale: 1.3, format: "png" });
  await fs.writeFile(`${outputDir}/${name}`, new Uint8Array(await preview.arrayBuffer()));
}
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(outputPath);
