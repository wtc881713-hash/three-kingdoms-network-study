import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const projectRoot = "E:/three-kingdoms-network";
const csvPath = `${projectRoot}/outputs/cooccurrence/paragraph/edge_validation_sample_highlighted.csv`;
const outputPath = `${projectRoot}/outputs/cooccurrence/paragraph/edge_validation_workbook_highlighted.xlsx`;
const previewDir = `${projectRoot}/outputs/cooccurrence/paragraph/workbook_previews_highlighted`;

const csvText = await fs.readFile(csvPath, "utf8");
const workbook = await Workbook.fromCSV(csvText, { sheetName: "Validation" });
const validation = workbook.worksheets.getItem("Validation");
const instructions = workbook.worksheets.add("Instructions");

validation.showGridLines = false;
validation.freezePanes.freezeRows(1);
validation.freezePanes.freezeColumns(4);
validation.getRange("A1:O61").format = {
  font: { name: "Aptos", size: 10, color: "#172033" },
  verticalAlignment: "top",
};
validation.getRange("A1:O1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 10 },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  rowHeight: 38,
  borders: { preset: "outside", style: "thin", color: "#9FBAD0" },
};
validation.getRange("A2:O61").format.borders = {
  insideHorizontal: { style: "thin", color: "#D9E2F3" },
};
validation.getRange("K2:K61").format = {
  wrapText: true,
  rowHeight: 60,
  fill: "#F7F9FC",
};
validation.getRange("L2:O61").format.fill = "#FFF2CC";
validation.getRange("C2:C61").format = {
  fill: "#DDEBF7",
  font: { bold: true, color: "#1F4E78" },
};
validation.getRange("D2:D61").format = {
  fill: "#FCE4D6",
  font: { bold: true, color: "#9E480E" },
};
validation.getRange("E2:E61").format = {
  fill: "#DDEBF7",
  font: { bold: true, color: "#1F4E78" },
};
validation.getRange("F2:F61").format = {
  fill: "#FCE4D6",
  font: { bold: true, color: "#9E480E" },
};
validation.getRange("G2:H61").format.numberFormat = "#,##0";
validation.getRange("A:A").format.columnWidth = 14;
validation.getRange("B:B").format.columnWidth = 11;
validation.getRange("C:D").format.columnWidth = 11;
validation.getRange("E:F").format.columnWidth = 19;
validation.getRange("G:H").format.columnWidth = 11;
validation.getRange("I:I").format.columnWidth = 17;
validation.getRange("J:J").format.columnWidth = 23;
validation.getRange("K:K").format.columnWidth = 80;
validation.getRange("L:M").format.columnWidth = 20;
validation.getRange("N:N").format.columnWidth = 17;
validation.getRange("O:O").format.columnWidth = 32;
validation.getRange("L2:L61").dataValidation = {
  rule: { type: "list", values: ["yes", "no", "uncertain"] },
};
validation.getRange("M2:M61").dataValidation = {
  rule: { type: "list", values: ["yes", "no", "uncertain"] },
};
validation.getRange("N2:N61").dataValidation = {
  rule: { type: "list", values: ["direct", "indirect", "unclear"] },
};
const validationTable = validation.tables.add("A1:O61", true, "HighlightedEdgeValidationTable");
validationTable.style = "TableStyleMedium2";

instructions.showGridLines = false;
instructions.getRange("A1").values = [["Paragraph Co-occurrence Edge Validation"]];
instructions.getRange("A1:F1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  verticalAlignment: "center",
  rowHeight: 34,
};
instructions.getRange("A3:B8").values = [
  ["Field", "How to complete it"],
  ["Blue / SOURCE markers", "The blue columns identify character A. In the evidence, its actual name or alias is wrapped as 【SOURCE:name】."],
  ["Orange / TARGET markers", "The orange columns identify character B. In the evidence, its actual name or alias is wrapped as 【TARGET:name】."],
  ["human_same_paragraph", "Choose yes if both marked characters appear in the displayed paragraph; otherwise choose no or uncertain."],
  ["human_both_characters", "Choose yes if both names refer to the stated characters; otherwise choose no or uncertain."],
  ["interaction_type", "Choose direct for dialogue/action, indirect for co-narration, or unclear when the paragraph is insufficient."],
  ["human_notes", "Explain errors, uncertainty, or useful literary context."],
  ["Important", "Indirect interaction is not an extraction error. This network measures textual proximity, not friendship or alliance."],
].slice(0, 6);
instructions.getRange("A3:B3").format = {
  fill: "#4472C4",
  font: { bold: true, color: "#FFFFFF" },
};
instructions.getRange("A4:B8").format = {
  wrapText: true,
  verticalAlignment: "top",
  borders: { preset: "inside", style: "thin", color: "#D9E2F3" },
};
instructions.getRange("A10:B18").values = [
  ["Live review summary", "Value"],
  ["Total sampled edges", 60],
  ["Completed same-paragraph decisions", null],
  ["Completed character decisions", null],
  ["Correct edges", null],
  ["Scorable edges", null],
  ["Sample precision", null],
  ["Direct interactions", null],
  ["Indirect interactions", null],
];
instructions.getRange("B12").formulas = [["=COUNTIF('Validation'!L2:L61,\"yes\")+COUNTIF('Validation'!L2:L61,\"no\")+COUNTIF('Validation'!L2:L61,\"uncertain\")"]];
instructions.getRange("B13").formulas = [["=COUNTIF('Validation'!M2:M61,\"yes\")+COUNTIF('Validation'!M2:M61,\"no\")+COUNTIF('Validation'!M2:M61,\"uncertain\")"]];
instructions.getRange("B14").formulas = [["=COUNTIFS('Validation'!L2:L61,\"yes\",'Validation'!M2:M61,\"yes\")"]];
instructions.getRange("B15").formulas = [["=COUNTIFS('Validation'!L2:L61,\"yes\",'Validation'!M2:M61,\"yes\")+COUNTIFS('Validation'!L2:L61,\"yes\",'Validation'!M2:M61,\"no\")+COUNTIFS('Validation'!L2:L61,\"no\",'Validation'!M2:M61,\"yes\")+COUNTIFS('Validation'!L2:L61,\"no\",'Validation'!M2:M61,\"no\")"]];
instructions.getRange("B16").formulas = [["=IF(B15=0,\"Pending\",B14/B15)"]];
instructions.getRange("B17").formulas = [["=COUNTIF('Validation'!N2:N61,\"direct\")"]];
instructions.getRange("B18").formulas = [["=COUNTIF('Validation'!N2:N61,\"indirect\")"]];
instructions.getRange("A10:B10").format = {
  fill: "#4472C4",
  font: { bold: true, color: "#FFFFFF" },
};
instructions.getRange("A11:B18").format.borders = {
  insideHorizontal: { style: "thin", color: "#D9E2F3" },
};
instructions.getRange("B16").format.numberFormat = "0.0%";
instructions.getRange("A:A").format.columnWidth = 36;
instructions.getRange("B:B").format.columnWidth = 88;
instructions.getRange("A3:B18").format.wrapText = true;
instructions.freezePanes.freezeRows(1);

await fs.mkdir(previewDir, { recursive: true });
const validationPreview = await workbook.render({
  sheetName: "Validation",
  range: "A1:O10",
  scale: 1,
  format: "png",
});
await fs.writeFile(
  `${previewDir}/validation.png`,
  new Uint8Array(await validationPreview.arrayBuffer()),
);
const instructionsPreview = await workbook.render({
  sheetName: "Instructions",
  range: "A1:B18",
  scale: 1.3,
  format: "png",
});
await fs.writeFile(
  `${previewDir}/instructions.png`,
  new Uint8Array(await instructionsPreview.arrayBuffer()),
);

const inspection = await workbook.inspect({
  kind: "table",
  range: "Instructions!A10:B18",
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 4,
});
console.log(inspection.ndjson);
const titleInspection = await workbook.inspect({
  kind: "table",
  range: "Instructions!A1:F2",
  include: "values,formulas",
  tableMaxRows: 3,
  tableMaxCols: 6,
});
console.log(titleInspection.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "formula error scan",
});
console.log(errors.ndjson);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(`Saved ${outputPath}`);
