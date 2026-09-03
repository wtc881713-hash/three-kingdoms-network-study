import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const projectRoot = path.resolve(import.meta.dirname, "..", "..");
const outputDir = path.join(projectRoot, "outputs", "validation", "multi_method");
const payload = JSON.parse(await fs.readFile(
  path.join(outputDir, "validation_workbook_data.json"),
  "utf8",
));

const workbook = Workbook.create();
const instructions = workbook.worksheets.add("Instructions");
const dialogue = workbook.worksheets.add("Dialogue Validation");
const semantic = workbook.worksheets.add("Semantic Validation");
dialogue.getRange("A1:N61").values = [payload.dialogue.columns, ...payload.dialogue.rows];
semantic.getRange("A1:L61").values = [payload.semantic.columns, ...payload.semantic.rows];

instructions.showGridLines = false;
instructions.getRange("A1").values = [["Multi-Method Network Validation"]];
instructions.getRange("A1:F1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  verticalAlignment: "center",
};
instructions.getRange("A1:F1").format.rowHeight = 30;
instructions.getRange("A3:B9").values = [
  ["Dialogue review", "Check the highlighted source and target against the paragraph."],
  ["Source correct", "yes / no / uncertain: whether the named speaker is correct."],
  ["Target correct", "yes / no / uncertain: whether the other named participant is correct."],
  ["Direct exchange", "yes / no / uncertain: whether the two characters directly address or answer each other."],
  ["Semantic review", "Compare the two representative contexts and your knowledge of the novel."],
  ["Meaningful", "yes / no / uncertain: whether the semantic-context link supports a useful literary comparison."],
  ["Relation type", "faction / narrative_period / conflict / theme / role / other / unclear."],
];
instructions.getRange("D3:E8").values = [
  ["Live summary", "Value"],
  ["Dialogue rows", null],
  ["Dialogue identity precision", null],
  ["Dialogue direct-exchange rate", null],
  ["Semantic rows", null],
  ["Semantic meaningful rate", null],
];
instructions.getRange("E4:E8").formulas = [
  ["=COUNTA('Dialogue Validation'!A2:A61)"],
  ["=IFERROR(COUNTIFS('Dialogue Validation'!K2:K61,\"yes\",'Dialogue Validation'!L2:L61,\"yes\")/(COUNTIF('Dialogue Validation'!K2:K61,\"yes\")+COUNTIF('Dialogue Validation'!K2:K61,\"no\")),\"\")"],
  ["=IFERROR(COUNTIF('Dialogue Validation'!M2:M61,\"yes\")/(COUNTIF('Dialogue Validation'!M2:M61,\"yes\")+COUNTIF('Dialogue Validation'!M2:M61,\"no\")),\"\")"],
  ["=COUNTA('Semantic Validation'!A2:A61)"],
  ["=IFERROR(COUNTIF('Semantic Validation'!J2:J61,\"yes\")/(COUNTIF('Semantic Validation'!J2:J61,\"yes\")+COUNTIF('Semantic Validation'!J2:J61,\"no\")),\"\")"],
];
instructions.getRange("A3:A9").format.font = { bold: true, color: "#17365D" };
instructions.getRange("D3:E3").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D" },
};
instructions.getRange("E5:E6").format.numberFormat = "0.0%";
instructions.getRange("E8").format.numberFormat = "0.0%";
instructions.getRange("A3:E9").format.wrapText = true;
instructions.getRange("A:A").format.columnWidth = 24;
instructions.getRange("B:B").format.columnWidth = 68;
instructions.getRange("C:C").format.columnWidth = 4;
instructions.getRange("D:D").format.columnWidth = 31;
instructions.getRange("E:E").format.columnWidth = 18;

function styleValidationSheet(sheet, lastColumn, editableRange, sourceColumn, targetColumn) {
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(2);
  const header = sheet.getRange(`A1:${lastColumn}1`);
  header.format = {
    fill: "#17365D",
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
    verticalAlignment: "center",
  };
  header.format.rowHeight = 42;
  sheet.getRange(`A1:${lastColumn}61`).format.borders = {
    insideHorizontal: { style: "thin", color: "#D9E2F3" },
  };
  sheet.getRange(editableRange).format.fill = "#FFF2CC";
  sheet.getRange(`${sourceColumn}2:${sourceColumn}61`).format = {
    fill: "#DDEBF7",
    font: { bold: true, color: "#1F4E78" },
  };
  sheet.getRange(`${targetColumn}2:${targetColumn}61`).format = {
    fill: "#FCE4D6",
    font: { bold: true, color: "#9C0006" },
  };
  sheet.getRange(`A1:${lastColumn}61`).format.verticalAlignment = "top";
}

styleValidationSheet(dialogue, "N", "K2:N61", "F", "H");
dialogue.getRange("K2:M61").dataValidation = {
  rule: { type: "list", values: ["yes", "no", "uncertain"] },
};
dialogue.getRange("A:A").format.columnWidth = 13;
dialogue.getRange("B:B").format.columnWidth = 17;
dialogue.getRange("C:C").format.columnWidth = 24;
dialogue.getRange("D:E").format.columnWidth = 13;
dialogue.getRange("F:I").format.columnWidth = 14;
dialogue.getRange("J:J").format.columnWidth = 90;
dialogue.getRange("J2:J61").format.wrapText = true;
dialogue.getRange("K:M").format.columnWidth = 18;
dialogue.getRange("N:N").format.columnWidth = 35;
dialogue.getRange("N2:N61").format.wrapText = true;

styleValidationSheet(semantic, "L", "J2:L61", "C", "D");
semantic.getRange("J2:J61").dataValidation = {
  rule: { type: "list", values: ["yes", "no", "uncertain"] },
};
semantic.getRange("K2:K61").dataValidation = {
  rule: {
    type: "list",
    values: ["faction", "narrative_period", "conflict", "theme", "role", "other", "unclear"],
  },
};
semantic.getRange("A:B").format.columnWidth = 14;
semantic.getRange("C:D").format.columnWidth = 14;
semantic.getRange("E:E").format.columnWidth = 13;
semantic.getRange("E2:E61").format.numberFormat = "0.000";
semantic.getRange("F:F").format.columnWidth = 13;
semantic.getRange("G:G").format.columnWidth = 75;
semantic.getRange("H:H").format.columnWidth = 13;
semantic.getRange("I:I").format.columnWidth = 75;
semantic.getRange("G2:G61").format.wrapText = true;
semantic.getRange("I2:I61").format.wrapText = true;
semantic.getRange("J:K").format.columnWidth = 22;
semantic.getRange("L:L").format.columnWidth = 35;
semantic.getRange("L2:L61").format.wrapText = true;

const dialogueTable = dialogue.tables.add("A1:N61", true, "DialogueValidationTable");
dialogueTable.style = "TableStyleMedium2";
const semanticTable = semantic.tables.add("A1:L61", true, "SemanticValidationTable");
semanticTable.style = "TableStyleMedium2";

await fs.mkdir(outputDir, { recursive: true });
const previewDir = path.join(import.meta.dirname, "previews");
await fs.mkdir(previewDir, { recursive: true });
for (const [sheetName, fileName, range] of [
  ["Instructions", "instructions.png", "A1:E9"],
  ["Dialogue Validation", "dialogue.png", "A1:N8"],
  ["Semantic Validation", "semantic.png", "A1:L8"],
]) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(
    path.join(previewDir, fileName),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

console.log((await workbook.inspect({
  kind: "table",
  range: "Instructions!A1:E9",
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 8,
})).ndjson);
console.log((await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "formula error scan",
})).ndjson);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(outputDir, "multi_method_validation_workbook.xlsx"));
