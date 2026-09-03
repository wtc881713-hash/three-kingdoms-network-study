import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const root = "E:/three-kingdoms-network";
const inputPath = process.argv[2] ?? `${root}/outputs/few_shot_pilot/annotation_batch_01.xlsx`;
const outputDir = process.argv[3] ?? `${root}/outputs/few_shot_pilot/review_check`;
await fs.mkdir(outputDir, { recursive: true });

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));
const sheet = workbook.worksheets.getItem("Pilot Annotation");

const overview = await workbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 6000,
  tableMaxRows: 5,
  tableMaxCols: 8,
});
await fs.writeFile(`${outputDir}/overview.ndjson`, overview.ndjson, "utf8");

const review = await workbook.inspect({
  kind: "table",
  range: "'Pilot Annotation'!A1:X21",
  include: "values,formulas",
  tableMaxRows: 21,
  tableMaxCols: 24,
  tableMaxCellChars: 500,
  maxChars: 100000,
});
await fs.writeFile(`${outputDir}/completed_rows.ndjson`, review.ndjson, "utf8");

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "completed pilot formula error scan",
});
await fs.writeFile(`${outputDir}/formula_errors.ndjson`, errors.ndjson, "utf8");

const values = sheet.getRange("A1:X21").values;
await fs.writeFile(`${outputDir}/completed_values.json`, JSON.stringify(values, null, 2), "utf8");

for (const [range, name] of [
  ["A1:N8", "evidence_preview.png"],
  ["O1:X8", "review_preview.png"],
  ["O9:X21", "review_lower_preview.png"],
]) {
  const preview = await workbook.render({
    sheetName: "Pilot Annotation",
    range,
    scale: 1.3,
    format: "png",
  });
  await fs.writeFile(`${outputDir}/${name}`, new Uint8Array(await preview.arrayBuffer()));
}

console.log(`${outputDir}/completed_values.json`);
