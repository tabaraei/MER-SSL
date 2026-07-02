"""
convert_iadse_labels.py — IADS-E label extractor (stdlib only)
===============================================================
The project venv has no `openpyxl`, so we parse `Sound Ratings.xlsx`
(sheet `SoundsAll`) directly as the zip/XML it really is and emit a
flat CSV the rest of the pipeline can `pd.read_csv`.

Source xlsx is READ-ONLY — only a derived CSV is written into phaseB/.

Usage:
    cd phaseB/
    python convert_iadse_labels.py \
        --xlsx "/datasets/emotions/IADS-E/Sound Ratings.xlsx" \
        --out  iadse_labels.csv

Output columns: sound_id, valence, arousal, category, description
  - sound_id : IADS-E `Sound ID` (e.g. 0001, 0015_b)
  - valence  : ValMN (SAM 1-9, NOT normalized — loader scales to [0,1])
  - arousal  : AroMN (SAM 1-9)
"""

import argparse
import csv
import os
import tempfile
import zipfile
import xml.etree.ElementTree as ET

NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _col_letters(ref: str) -> str:
    return "".join(ch for ch in ref if ch.isalpha())


def main():
    ap = argparse.ArgumentParser(description="Convert IADS-E Sound Ratings.xlsx → flat CSV")
    ap.add_argument("--xlsx", default="/datasets/emotions/IADS-E/Sound Ratings.xlsx")
    ap.add_argument("--sheet", default="SoundsAll")
    ap.add_argument("--out", default="iadse_labels.csv")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(args.xlsx) as z:
            z.extractall(tmp)

        # sheet name → sheetN.xml via workbook + rels
        wb = ET.parse(os.path.join(tmp, "xl", "workbook.xml")).getroot()
        rels = ET.parse(os.path.join(tmp, "xl", "_rels", "workbook.xml.rels")).getroot()
        rid_to_target = {
            r.get("Id"): r.get("Target")
            for r in rels
        }
        sheet_target = None
        for s in wb.find("a:sheets", NS):
            if s.get("name") == args.sheet:
                rid = s.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
                sheet_target = rid_to_target[rid]
        if sheet_target is None:
            raise SystemExit(f"Sheet '{args.sheet}' not found in {args.xlsx}")

        shared = []
        ss_path = os.path.join(tmp, "xl", "sharedStrings.xml")
        if os.path.exists(ss_path):
            for si in ET.parse(ss_path).getroot():
                shared.append("".join(t.text or "" for t in si.iter("{%s}t" % NS["a"])))

        sheet_path = os.path.join(tmp, "xl", sheet_target.replace("/", os.sep))
        root = ET.parse(sheet_path).getroot()
        rows = root.find("a:sheetData", NS).findall("a:row", NS)

        def cells(row):
            out = {}
            for c in row.findall("a:c", NS):
                v = c.find("a:v", NS)
                if v is None:
                    continue
                val = shared[int(v.text)] if c.get("t") == "s" else v.text
                out[_col_letters(c.get("r"))] = val
            return out

        header = cells(rows[0])
        lab2col = {v: k for k, v in header.items()}
        need = {"Sound ID": "sound_id", "ValMN": "valence", "AroMN": "arousal",
                "Category": "category", "Description": "description"}
        for src in ("Sound ID", "ValMN", "AroMN"):
            if src not in lab2col:
                raise SystemExit(f"Required column '{src}' missing. Found: {list(header.values())}")

        n = 0
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(list(need.values()))
            for r in rows[1:]:
                cd = cells(r)
                sid = cd.get(lab2col["Sound ID"])
                if sid in (None, ""):
                    continue
                w.writerow([
                    sid,
                    cd.get(lab2col["ValMN"], ""),
                    cd.get(lab2col["AroMN"], ""),
                    cd.get(lab2col.get("Category", ""), ""),
                    cd.get(lab2col.get("Description", ""), ""),
                ])
                n += 1

    print(f"✅ Wrote {n} IADS-E label rows → {args.out}")


if __name__ == "__main__":
    main()
