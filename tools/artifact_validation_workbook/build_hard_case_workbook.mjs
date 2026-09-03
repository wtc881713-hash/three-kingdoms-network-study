import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const projectRoot = path.resolve(import.meta.dirname, "..", "..");
const outputDir = path.join(projectRoot, "outputs", "validation", "hard_cases");
const payload = JSON.parse(await fs.readFile(
  path.join(outputDir, "hard_case_workbook_data.json"),
  "utf8",
));

const workbook = Workbook.create();
const instructions = workbook.worksheets.add("Instructions");
const dialogue = workbook.worksheets.add("Dialogue Hard Cases");
const semantic = workbook.worksheets.add("Semantic Hard Cases");
dialogue.getRange("A1:O11").values = [payload.dialogue.columns, ...payload.dialogue.rows];
semantic.getRange("A1:M11").values = [payload.semantic.columns, ...payload.semantic.rows];

instructions.showGridLines = false;
instructions.getRange("A1").values = [["Hard-Case Researcher Validation"]];
instructions.getRange("A1:E1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  verticalAlignment: "center",
};
instructions.getRange("A1:E1").format.rowHeight = 30;
instructions.getRange("A3:B10").values = [
  ["Purpose", "Review 10 difficult cases from each method before Codex pre-fills the complete validation tables."],
  ["Dialogue source", "Choose yes, no, or uncertain for whether the named speaker is correct."],
  ["Dialogue target", "Choose yes, no, or uncertain for whether the named target/next speaker is correct."],
  ["Direct exchange", "Choose yes only when the two characters directly address or answer each other in this passage."],
  ["Semantic meaningful", "Choose yes when the pair supports a useful literary comparison, not merely similar generic wording."],
  ["Relation type", "Choose faction, narrative_period, conflict, theme, role, family, other, or unclear."],
  ["Notes", "Briefly explain no or uncertain decisions. Notes on difficult yes decisions are also useful."],
  ["Next step", "Return this reviewed workbook. Codex will learn the decision pattern and pre-fill the complete tables for your final review."],
];
instructions.getRange("D3:E8").values = [
  ["Progress", "Value"],
  ["Dialogue cases", 10],
  ["Dialogue completed", null],
  ["Semantic cases", 10],
  ["Semantic completed", null],
  ["Total completed", null],
];
instructions.getRange("E5").formulas = [["=COUNTIF('Dialogue Hard Cases'!L2:L11,\"yes\")+COUNTIF('Dialogue Hard Cases'!L2:L11,\"no\")+COUNTIF('Dialogue Hard Cases'!L2:L11,\"uncertain\")"]];
instructions.getRange("E7").formulas = [["=COUNTIF('Semantic Hard Cases'!K2:K11,\"yes\")+COUNTIF('Semantic Hard Cases'!K2:K11,\"no\")+COUNTIF('Semantic Hard Cases'!K2:K11,\"uncertain\")"]];
instructions.getRange("E8").formulas = [["=E5+E7"]];
instructions.getRange("A3:A10").format.font = { bold: true, color: "#17365D" };
instructions.getRange("D3:E3").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D" },
};
instructions.getRange("A3:E10").format.wrapText = true;
instructions.getRange("A:A").format.columnWidth = 24;
instructions.getRange("B:B").format.columnWidth = 76;
instructions.getRange("C:C").format.columnWidth = 4;
instructions.getRange("D:D").format.columnWidth = 24;
instructions.getRange("E:E").format.columnWidth = 16;

function styleSheet(sheet, lastColumn, sourceColumn, targetColumn, editableRange) {
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(2);
  sheet.getRange(`A1:${lastColumn}1`).format = {
    fill: "#17365D",
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
    verticalAlignment: "center",
  };
  sheet.getRange(`A1:${lastColumn}1`).format.rowHeight = 45;
  sheet.getRange(`A1:${lastColumn}11`).format.borders = {
    insideHorizontal: { style: "thin", color: "#D9E2F3" },
  };
  sheet.getRange(`${sourceColumn}2:${sourceColumn}11`).format = {
    fill: "#DDEBF7",
    font: { bold: true, color: "#1F4E78" },
  };
  sheet.getRange(`${targetColumn}2:${targetColumn}11`).format = {
    fill: "#FCE4D6",
    font: { bold: true, color: "#9C0006" },
  };
  sheet.getRange(editableRange).format.fill = "#FFF2CC";
  sheet.getRange(`A1:${lastColumn}11`).format.verticalAlignment = "top";
}

styleSheet(dialogue, "O", "G", "I", "L2:O11");
dialogue.getRange("L2:N11").dataValidation = {
  rule: { type: "list", values: ["yes", "no", "uncertain"] },
};
dialogue.getRange("A:A").format.columnWidth = 13;
dialogue.getRange("B:B").format.columnWidth = 52;
dialogue.getRange("B2:B11").format.wrapText = true;
dialogue.getRange("C:C").format.columnWidth = 17;
dialogue.getRange("D:D").format.columnWidth = 24;
dialogue.getRange("E:F").format.columnWidth = 13;
dialogue.getRange("G:J").format.columnWidth = 13;
dialogue.getRange("K:K").format.columnWidth = 95;
dialogue.getRange("K2:K11").format.wrapText = true;
dialogue.getRange("L:N").format.columnWidth = 18;
dialogue.getRange("O:O").format.columnWidth = 38;
dialogue.getRange("O2:O11").format.wrapText = true;

styleSheet(semantic, "M", "D", "E", "K2:M11");
semantic.getRange("K2:K11").dataValidation = {
  rule: { type: "list", values: ["yes", "no", "uncertain"] },
};
semantic.getRange("L2:L11").dataValidation = {
  rule: {
    type: "list",
    values: ["faction", "narrative_period", "conflict", "theme", "role", "family", "other", "unclear"],
  },
};
semantic.getRange("A:A").format.columnWidth = 13;
semantic.getRange("B:B").format.columnWidth = 55;
semantic.getRange("B2:B11").format.wrapText = true;
semantic.getRange("C:C").format.columnWidth = 14;
semantic.getRange("D:E").format.columnWidth = 14;
semantic.getRange("F:F").format.columnWidth = 13;
semantic.getRange("F2:F11").format.numberFormat = "0.000";
semantic.getRange("G:G").format.columnWidth = 14;
semantic.getRange("H:H").format.columnWidth = 75;
semantic.getRange("I:I").format.columnWidth = 14;
semantic.getRange("J:J").format.columnWidth = 75;
semantic.getRange("H2:H11").format.wrapText = true;
semantic.getRange("J2:J11").format.wrapText = true;
semantic.getRange("K:L").format.columnWidth = 22;
semantic.getRange("M:M").format.columnWidth = 38;
semantic.getRange("M2:M11").format.wrapText = true;

dialogue.tables.add("A1:O11", true, "DialogueHardCasesTable").style = "TableStyleMedium2";
semantic.tables.add("A1:M11", true, "SemanticHardCasesTable").style = "TableStyleMedium2";

const previewDir = path.join(import.meta.dirname, "previews");
await fs.mkdir(previewDir, { recursive: true });
for (const [sheetName, fileName, range] of [
  ["Instructions", "hard-instructions.png", "A1:E10"],
  ["Dialogue Hard Cases", "hard-dialogue.png", "A1:O6"],
  ["Semantic Hard Cases", "hard-semantic.png", "A1:M6"],
]) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(path.join(previewDir, fileName), new Uint8Array(await preview.arrayBuffer()));
}

console.log((await workbook.inspect({
  kind: "table",
  range: "Instructions!A1:E10",
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

await fs.mkdir(outputDir, { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(outputDir, "hard_case_validation_for_researcher.xlsx"));
