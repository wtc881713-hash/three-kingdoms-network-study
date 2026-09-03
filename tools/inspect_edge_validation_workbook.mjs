import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";


const path = "E:/three-kingdoms-network/outputs/cooccurrence/paragraph/edge_validation_workbook_highlighted.xlsx";
const input = await FileBlob.load(path);
const workbook = await SpreadsheetFile.importXlsx(input);
const validationSheet = workbook.worksheets.getItem("Validation");
const validationValues = validationSheet.getRange("A1:O61").values;

const csvEscape = (value) => {
  const text = value === null || value === undefined ? "" : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
};
const csv = validationValues
  .map((row) => row.map(csvEscape).join(","))
  .join("\r\n");
await fs.writeFile(
  "E:/three-kingdoms-network/outputs/cooccurrence/paragraph/edge_validation_results.csv",
  `\uFEFF${csv}\r\n`,
  "utf8",
);

const summary = await workbook.inspect({
  kind: "table",
  range: "Instructions!A10:B18",
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 4,
});
console.log(summary.ndjson);

const decisions = await workbook.inspect({
  kind: "table",
  range: "Validation!A1:O61",
  include: "values",
  tableMaxRows: 65,
  tableMaxCols: 15,
  tableMaxCellChars: 160,
  maxChars: 100000,
});
console.log(decisions.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "formula error scan",
});
console.log(errors.ndjson);
