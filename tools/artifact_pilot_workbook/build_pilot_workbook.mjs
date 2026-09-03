import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = "E:/three-kingdoms-network";
const csvPath = process.argv[2] ?? `${root}/data/annotation/annotation_batch_01.csv`;
const outputPath = process.argv[3] ?? `${root}/outputs/few_shot_pilot/annotation_batch_01.xlsx`;
const workbookTitle = process.argv[4] ?? "Few-Shot Character Relation Pilot — Quick Guide";
const outputDir = path.dirname(outputPath);

await fs.mkdir(outputDir, { recursive: true });
const csvText = (await fs.readFile(csvPath, "utf8")).replace(/^\uFEFF/, "");
const workbook = await Workbook.fromCSV(csvText, { sheetName: "Pilot Annotation" });
const sheet = workbook.worksheets.getItem("Pilot Annotation");
const guidelines = workbook.worksheets.add("Quick Guide");
const lastRow = sheet.getUsedRange().values.length;

sheet.showGridLines = false;
sheet.freezePanes.freezeRows(1);
sheet.freezePanes.freezeColumns(6);
const used = sheet.getUsedRange();
used.format = {
  font: { name: "Aptos", size: 10, color: "#1F2937" },
  verticalAlignment: "top",
};
sheet.getRange("A1:X1").format = {
  fill: "#17365D",
  font: { name: "Aptos Display", size: 10, bold: true, color: "#FFFFFF" },
  wrapText: true,
  verticalAlignment: "center",
  rowHeight: 42,
  borders: { preset: "inside", style: "thin", color: "#5B7394" },
};
sheet.getRange(`A2:N${lastRow}`).format = {
  fill: "#F5F8FC",
  wrapText: true,
  borders: { insideHorizontal: { style: "thin", color: "#D9E2F3" } },
};
sheet.getRange(`O2:X${lastRow}`).format = {
  fill: "#FFF2CC",
  wrapText: true,
  borders: { insideHorizontal: { style: "thin", color: "#E6D690" } },
};
sheet.getRange(`A2:X${lastRow}`).format.rowHeight = 90;

const widths = {
  A: 13, B: 9, C: 13, D: 13, E: 13, F: 13, G: 11, H: 11,
  I: 58, J: 23, K: 11, L: 30, M: 22, N: 13, O: 22, P: 22,
  Q: 16, R: 16, S: 17, T: 16, U: 42, V: 15, W: 38, X: 16,
};
for (const [column, width] of Object.entries(widths)) {
  sheet.getRange(`${column}:${column}`).format.columnWidth = width;
}
sheet.getRange(`B2:B${lastRow}`).format.numberFormat = "0";
sheet.getRange(`C2:D${lastRow}`).format.numberFormat = "0";
sheet.getRange(`K2:K${lastRow}`).format.numberFormat = "0.000";
sheet.getRange(`V2:V${lastRow}`).dataValidation = {
  rule: { type: "list", values: ["1", "2", "3", "4", "5"] },
};
for (const column of ["O", "P"]) {
  sheet.getRange(`${column}2:${column}${lastRow}`).dataValidation = {
    rule: {
      type: "list",
      values: [
        "cooperation", "hierarchy_loyalty", "kinship",
        "friendship_brotherhood", "hostility_conflict",
        "deception_manipulation", "affection_romance",
        "no_clear_relation", "uncertain", "",
      ],
    },
  };
}
sheet.getRange(`Q2:Q${lastRow}`).dataValidation = {
  rule: { type: "list", values: ["A_to_B", "B_to_A", "bidirectional", "unclear"] },
};
sheet.getRange(`R2:R${lastRow}`).dataValidation = {
  rule: { type: "list", values: ["positive", "negative", "mixed", "neutral", "unclear"] },
};
sheet.getRange(`S2:S${lastRow}`).dataValidation = {
  rule: { type: "list", values: ["explicit", "implicit", "inferred", "unclear"] },
};
sheet.getRange(`T2:T${lastRow}`).dataValidation = {
  rule: { type: "list", values: ["stable", "temporary", "changing", "unclear"] },
};
sheet.getRange(`X2:X${lastRow}`).dataValidation = {
  rule: { type: "list", values: ["pending", "reviewed"] },
};

guidelines.showGridLines = false;
guidelines.getRange("A1:D1").merge();
guidelines.getRange("A1").values = [[workbookTitle]];
guidelines.getRange("A1:D1").format = {
  fill: "#17365D",
  font: { name: "Aptos Display", size: 16, bold: true, color: "#FFFFFF" },
  rowHeight: 34,
  verticalAlignment: "center",
};
guidelines.getRange("A3:B12").values = [
  ["Field", "What to enter"],
  ["primary_relation", "Choose the main relation supported by this passage."],
  ["secondary_relation", "Optional second relation; leave blank when not needed."],
  ["relation_direction", "A_to_B / B_to_A / bidirectional / unclear"],
  ["relation_polarity", "positive / negative / mixed / neutral / unclear"],
  ["relation_explicitness", "explicit / implicit / inferred / unclear"],
  ["relation_temporality", "stable / temporary / changing / unclear"],
  ["evidence_text", "Copy the shortest exact words that support your decision."],
  ["annotator_confidence", "1 = very unsure; 5 = very sure."],
  ["annotation_status", "Change pending to reviewed only after completing the row."],
];
guidelines.getRange("A3:B3").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D" },
};
guidelines.getRange("A3:B12").format.wrapText = true;
guidelines.getRange("A3:B12").format.borders = {
  insideHorizontal: { style: "thin", color: "#D9E2F3" },
};
guidelines.getRange("A14:B23").values = [
  ["Relation label", "Short meaning"],
  ["cooperation", "Work together or share an immediate goal."],
  ["hierarchy_loyalty", "Service, command, obedience, duty, or loyalty."],
  ["kinship", "Blood, marriage, adoption, or accepted family status."],
  ["friendship_brotherhood", "Personal friendship, trust, or sworn brotherhood."],
  ["hostility_conflict", "Attack, threat, punishment, rejection, or opposition."],
  ["deception_manipulation", "Trick, test, trap, persuasion, or secret control."],
  ["affection_romance", "Romance, marriage, desire, jealousy, or intimacy."],
  ["no_clear_relation", "Clear passage, but no useful relation evidence for the pair."],
  ["uncertain", "A relation may exist, but the passage is too ambiguous."],
];
guidelines.getRange("A14:B14").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D" },
};
guidelines.getRange("A14:B23").format.wrapText = true;
guidelines.getRange("A14:B23").format.borders = {
  insideHorizontal: { style: "thin", color: "#D9E2F3" },
};
guidelines.getRange("D3:D8").values = [
  ["Important rules"],
  ["Use mainly the supplied passage, not general knowledge of the novel."],
  ["The suggested relation is a rule-based hint and may be wrong."],
  ["Use uncertain when evidence is ambiguous; use no_clear_relation when the pair merely co-occurs."],
  ["Record any use of wider narrative context in annotator_notes."],
  ["Full guidance: docs/annotation_guidelines.md"],
];
guidelines.getRange("D3").format = {
  fill: "#F4B183",
  font: { bold: true, color: "#7F2704" },
};
guidelines.getRange("D4:D8").format = { fill: "#FCE4D6", wrapText: true };
guidelines.getRange("A:A").format.columnWidth = 25;
guidelines.getRange("B:B").format.columnWidth = 72;
guidelines.getRange("C:C").format.columnWidth = 4;
guidelines.getRange("D:D").format.columnWidth = 62;
guidelines.getRange("3:23").format.rowHeight = 30;
guidelines.freezePanes.freezeRows(1);

const check = await workbook.inspect({
  kind: "table",
  range: "'Pilot Annotation'!A1:X6",
  include: "values,formulas",
  tableMaxRows: 6,
  tableMaxCols: 24,
  maxChars: 5000,
});
await fs.writeFile(`${outputDir}/pilot_workbook_inspect.ndjson`, check.ndjson, "utf8");
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
await fs.writeFile(`${outputDir}/pilot_workbook_errors.ndjson`, errors.ndjson, "utf8");

for (const [sheetName, range, fileName] of [
  ["Quick Guide", "A1:D23", "quick_guide_preview.png"],
  ["Pilot Annotation", "A1:N7", "pilot_evidence_preview.png"],
  ["Pilot Annotation", "O1:X7", "pilot_review_preview.png"],
  ["Pilot Annotation", `A${Math.max(2, lastRow - 5)}:N${lastRow}`, "pilot_evidence_tail_preview.png"],
  ["Pilot Annotation", `O${Math.max(2, lastRow - 5)}:X${lastRow}`, "pilot_review_tail_preview.png"],
]) {
  const preview = await workbook.render({ sheetName, range, scale: 1.4, format: "png" });
  await fs.writeFile(`${outputDir}/${fileName}`, new Uint8Array(await preview.arrayBuffer()));
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(outputPath);
