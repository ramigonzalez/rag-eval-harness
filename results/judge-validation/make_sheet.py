#!/usr/bin/env python3
"""Build a blind human-labeling sheet to validate two LLM judges.

Reproducible with seed 42. Writes labels.xlsx, labels.csv, and a separate
unblinding file key.csv (not embedded in the workbook).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

SEED = 42
N_TOTAL = 40
N_DISAGREE = 24
N_FAITH = 12
N_REL = 12
N_RANDOM = 16
DISAGREE_CONFIG_CAP = 5
TOTAL_CONFIG_TARGET = 8

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANSWERS = ROOT / "results" / "generation-100" / "answers.jsonl"
# RAGAS per-answer scores are not committed. Pass --ragas to rebuild the sheet.
DEFAULT_RAGAS = ROOT / "results" / "judge-validation" / "ragas_scores.jsonl"

ABSTENTION = "I do not know."
LABEL_COLUMNS = [
    "item",
    "question",
    "answer",
    "context_1",
    "context_2",
    "context_3",
    "context_4",
    "context_5",
    "faithful",
    "relevant",
    "notes",
]
KEY_COLUMNS = [
    "item",
    "query_id",
    "chunker",
    "retriever",
    "ours_faithfulness",
    "ragas_faithfulness",
    "ours_answer_relevancy",
    "ragas_answer_relevancy",
    "selection_reason",
]

INSTRUCTIONS_PT = """Rotulagem do juiz (prova 1)

Você está fazendo o papel do juiz humano. O objetivo é comparar as suas marcações com dois juízes de IA. Por isso esta planilha não mostra as notas deles.

Como preencher
Na aba "label", leia nesta ordem: a pergunta (no SciFact é uma afirmação científica), os 5 trechos de contexto e a resposta do modelo. Depois preencha só as colunas faithful e relevant. A coluna notes é opcional.

1) faithful (fiel)
Tudo o que a resposta afirma está apoiado nos 5 trechos?
Não julgue se o conteúdo é verdadeiro no mundo real, nem se você concorda com ele. Julgue apenas se o texto dos trechos sustenta o que a resposta diz.
- yes: tudo o que a resposta afirma está nos trechos
- partial: parte está apoiada e parte não, ou o apoio é só parcial/ambíguo
- no: a resposta inventa informação ou contradiz os trechos

2) relevant (relevante)
A resposta aborda a afirmação/pergunta feita?
- yes: responde o que foi perguntado
- partial: responde só em parte
- no: não responde a pergunta

Se um trecho estiver cortado na tela, aumente a altura da linha ou dê um duplo clique na célula. Não é preciso pesquisar o assunto; use só o que está na linha.

Há 40 itens. Cada um leva cerca de 30 a 60 segundos. Pode fazer em várias sessões; deixe em branco o que ainda não tiver rotulado.
"""


def is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def row_key(row: dict) -> tuple[str, str, str]:
    return (str(row["query_id"]), row["chunker"], row["retriever"])


def config_of(row: dict) -> tuple[str, str]:
    return (row["chunker"], row["retriever"])


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_eligible(answers_path: Path, ragas_path: Path) -> list[dict]:
    answers = load_jsonl(answers_path)
    ragas_rows = load_jsonl(ragas_path)
    ragas_index = {row_key(row): row for row in ragas_rows}

    eligible: list[dict] = []
    for answer in answers:
        if answer.get("answer") == ABSTENTION:
            continue
        if answer.get("error"):
            continue
        key = row_key(answer)
        ragas = ragas_index.get(key)
        if ragas is None:
            continue
        ours_f = ragas.get("ours_faithfulness")
        ragas_f = ragas.get("ragas_faithfulness")
        ours_r = ragas.get("ours_answer_relevancy")
        ragas_r = ragas.get("ragas_answer_relevancy")
        answer_f = answer.get("faithfulness")
        answer_r = answer.get("answer_relevancy")
        if any(is_missing(v) for v in (ours_f, ragas_f, ours_r, ragas_r, answer_f, answer_r)):
            continue
        contexts = answer.get("contexts") or []
        if len(contexts) != 5:
            continue
        eligible.append(
            {
                "query_id": str(answer["query_id"]),
                "chunker": answer["chunker"],
                "retriever": answer["retriever"],
                "question": answer["question"],
                "answer": answer["answer"],
                "contexts": list(contexts),
                "ours_faithfulness": float(ours_f),
                "ragas_faithfulness": float(ragas_f),
                "ours_answer_relevancy": float(ours_r),
                "ragas_answer_relevancy": float(ragas_r),
                "faith_gap": abs(float(ours_f) - float(ragas_f)),
                "rel_gap": abs(float(ours_r) - float(ragas_r)),
            }
        )
    return eligible


def try_pick(
    ranked: list[dict],
    used_qids: set[str],
    used_keys: set[tuple[str, str, str]],
    config_counts: Counter,
    cap: int | None,
) -> dict | None:
    for row in ranked:
        qid = row["query_id"]
        key = row_key(row)
        cfg = config_of(row)
        if qid in used_qids or key in used_keys:
            continue
        if cap is not None and config_counts[cfg] >= cap:
            continue
        return row
    return None


def mark_selected(
    row: dict,
    reason: str,
    selected: list[dict],
    used_qids: set[str],
    used_keys: set[tuple[str, str, str]],
    config_counts: Counter,
) -> None:
    item = dict(row)
    item["selection_reason"] = reason
    selected.append(item)
    used_qids.add(row["query_id"])
    used_keys.add(row_key(row))
    config_counts[config_of(row)] += 1


def select_items(eligible: list[dict], seed: int = SEED) -> list[dict]:
    by_faith = sorted(
        eligible,
        key=lambda r: (
            -r["faith_gap"],
            -r["rel_gap"],
            r["query_id"],
            r["chunker"],
            r["retriever"],
        ),
    )
    by_rel = sorted(
        eligible,
        key=lambda r: (
            -r["rel_gap"],
            -r["faith_gap"],
            r["query_id"],
            r["chunker"],
            r["retriever"],
        ),
    )
    by_max = sorted(
        eligible,
        key=lambda r: (
            -max(r["faith_gap"], r["rel_gap"]),
            -r["faith_gap"],
            -r["rel_gap"],
            r["query_id"],
            r["chunker"],
            r["retriever"],
        ),
    )

    selected: list[dict] = []
    used_qids: set[str] = set()
    used_keys: set[tuple[str, str, str]] = set()
    config_counts: Counter = Counter()

    def fill_disagreement(cap: int | None) -> None:
        n_faith = sum(1 for r in selected if r["selection_reason"] == "disagreement_faithfulness")
        n_rel = sum(1 for r in selected if r["selection_reason"] == "disagreement_relevancy")
        while n_faith < N_FAITH or n_rel < N_REL:
            picked = False
            if n_faith < N_FAITH:
                row = try_pick(by_faith, used_qids, used_keys, config_counts, cap)
                if row is not None:
                    mark_selected(
                        row,
                        "disagreement_faithfulness",
                        selected,
                        used_qids,
                        used_keys,
                        config_counts,
                    )
                    n_faith += 1
                    picked = True
            if n_rel < N_REL:
                row = try_pick(by_rel, used_qids, used_keys, config_counts, cap)
                if row is not None:
                    mark_selected(
                        row,
                        "disagreement_relevancy",
                        selected,
                        used_qids,
                        used_keys,
                        config_counts,
                    )
                    n_rel += 1
                    picked = True
            if not picked:
                break

        while len(selected) < N_DISAGREE:
            row = try_pick(by_max, used_qids, used_keys, config_counts, cap)
            if row is None:
                break
            # Attribute leftover slots to the metric with the larger gap.
            reason = (
                "disagreement_faithfulness"
                if row["faith_gap"] >= row["rel_gap"]
                else "disagreement_relevancy"
            )
            mark_selected(row, reason, selected, used_qids, used_keys, config_counts)

    fill_disagreement(DISAGREE_CONFIG_CAP)
    if len(selected) < N_DISAGREE:
        fill_disagreement(None)

    if len(selected) != N_DISAGREE:
        raise RuntimeError(
            f"Could only select {len(selected)} disagreement items (need {N_DISAGREE})"
        )

    rng = random.Random(seed)
    configs = sorted({config_of(row) for row in eligible})
    remaining_by_config: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in eligible:
        if row["query_id"] in used_qids or row_key(row) in used_keys:
            continue
        remaining_by_config[config_of(row)].append(row)
    for cfg in configs:
        rng.shuffle(remaining_by_config[cfg])

    # First pass: fill each config toward an even total of 8.
    for cfg in configs:
        need = max(0, TOTAL_CONFIG_TARGET - config_counts[cfg])
        pool = remaining_by_config[cfg]
        taken = 0
        kept: list[dict] = []
        for row in pool:
            if taken < need and row["query_id"] not in used_qids:
                mark_selected(
                    row, "random", selected, used_qids, used_keys, config_counts
                )
                taken += 1
            else:
                kept.append(row)
        remaining_by_config[cfg] = kept
        if len(selected) >= N_TOTAL:
            break

    # Second pass: if a config ran short, fill from whatever unique queries remain.
    if len(selected) < N_TOTAL:
        leftover = []
        for cfg in configs:
            leftover.extend(remaining_by_config[cfg])
        rng.shuffle(leftover)
        for row in leftover:
            if len(selected) >= N_TOTAL:
                break
            if row["query_id"] in used_qids or row_key(row) in used_keys:
                continue
            mark_selected(row, "random", selected, used_qids, used_keys, config_counts)

    if len(selected) != N_TOTAL:
        raise RuntimeError(f"Could only select {len(selected)} items (need {N_TOTAL})")

    n_random = sum(1 for r in selected if r["selection_reason"] == "random")
    if n_random != N_RANDOM:
        raise RuntimeError(f"Expected {N_RANDOM} random items, got {n_random}")

    rng_order = random.Random(seed)
    rng_order.shuffle(selected)
    for index, row in enumerate(selected, start=1):
        row["item"] = index
    return selected


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def label_records(selected: list[dict]) -> list[dict]:
    records = []
    for row in selected:
        record = {
            "item": row["item"],
            "question": row["question"],
            "answer": row["answer"],
            "faithful": "",
            "relevant": "",
            "notes": "",
        }
        for idx in range(5):
            record[f"context_{idx + 1}"] = row["contexts"][idx]
        records.append(record)
    return records


def key_records(selected: list[dict]) -> list[dict]:
    records = []
    for row in selected:
        records.append({col: row[col] for col in KEY_COLUMNS})
    return records


def style_header(cell, fill: PatternFill) -> None:
    cell.font = Font(bold=True, color="FFFFFF", name="Calibri", size=11)
    cell.fill = fill
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def write_xlsx(path: Path, labels: list[dict]) -> None:
    wb = Workbook()

    ws_inst = wb.active
    ws_inst.title = "instructions"
    ws_inst["A1"] = INSTRUCTIONS_PT.strip()
    ws_inst["A1"].alignment = Alignment(wrap_text=True, vertical="top")
    ws_inst["A1"].font = Font(name="Calibri", size=12)
    ws_inst.merge_cells("A1:F28")
    ws_inst.row_dimensions[1].height = 420
    ws_inst.column_dimensions["A"].width = 22
    for letter in "BCDEF":
        ws_inst.column_dimensions[letter].width = 18
    ws_inst.page_setup.fitToPage = True
    ws_inst.page_setup.fitToWidth = 1
    ws_inst.page_setup.fitToHeight = 1
    ws_inst.print_options.horizontalCentered = True
    ws_inst.sheet_properties.tabColor = "1F4E79"

    ws = wb.create_sheet("label")
    header_fill = PatternFill("solid", fgColor="1F4E79")
    thin = Border(
        left=Side(style="thin", color="D0D7DE"),
        right=Side(style="thin", color="D0D7DE"),
        top=Side(style="thin", color="D0D7DE"),
        bottom=Side(style="thin", color="D0D7DE"),
    )
    wrap_top = Alignment(wrap_text=True, vertical="top")
    wrap_top_center = Alignment(wrap_text=True, vertical="top", horizontal="center")
    input_fill = PatternFill("solid", fgColor="FFF4CC")
    alt_fill = PatternFill("solid", fgColor="F7F9FC")

    widths = {
        "item": 6,
        "question": 40,
        "answer": 45,
        "context_1": 46,
        "context_2": 46,
        "context_3": 46,
        "context_4": 46,
        "context_5": 46,
        "faithful": 12,
        "relevant": 12,
        "notes": 28,
    }
    for col_idx, name in enumerate(LABEL_COLUMNS, start=1):
        cell = ws.cell(row=1, column=col_idx, value=name)
        style_header(cell, header_fill)
        ws.column_dimensions[get_column_letter(col_idx)].width = widths[name]
    ws.row_dimensions[1].height = 22
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:K{len(labels) + 1}"
    ws.sheet_view.showGridLines = False

    for row_idx, record in enumerate(labels, start=2):
        ws.row_dimensions[row_idx].height = 110
        for col_idx, name in enumerate(LABEL_COLUMNS, start=1):
            cell = ws.cell(row=row_idx, column=col_idx, value=record[name])
            cell.border = thin
            if name in {"item", "faithful", "relevant"}:
                cell.alignment = wrap_top_center
            else:
                cell.alignment = wrap_top
            if name in {"faithful", "relevant", "notes"}:
                cell.fill = input_fill
            elif row_idx % 2 == 0:
                cell.fill = alt_fill
            if name == "item":
                cell.font = Font(bold=True, name="Calibri", size=11)
            else:
                cell.font = Font(name="Calibri", size=11)

    dv = DataValidation(
        type="list",
        formula1='"yes,partial,no"',
        allow_blank=True,
        showDropDown=False,
        showErrorMessage=True,
        showInputMessage=True,
    )
    dv.errorTitle = "Valor inválido"
    dv.error = "Use yes, partial ou no."
    dv.promptTitle = "Rótulo"
    dv.prompt = "yes, partial ou no"
    last_row = len(labels) + 1
    dv.add(f"I2:I{last_row}")
    dv.add(f"J2:J{last_row}")
    ws.add_data_validation(dv)

    ws.freeze_panes = "A2"
    # Keep the label sheet in front when the file opens.
    wb._sheets = [ws, ws_inst]
    wb.active = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def summarize(selected: list[dict]) -> str:
    reasons = Counter(r["selection_reason"] for r in selected)
    configs = Counter(f"{r['chunker']}/{r['retriever']}" for r in selected)
    qids = [r["query_id"] for r in selected]
    lines = [
        f"items: {len(selected)}",
        f"reasons: {dict(reasons)}",
        f"configs: {dict(sorted(configs.items()))}",
        f"unique query_id: {len(set(qids))} (duplicates: {len(qids) - len(set(qids))})",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--ragas", type=Path, default=DEFAULT_RAGAS)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    eligible = load_eligible(args.answers, args.ragas)
    selected = select_items(eligible, seed=args.seed)
    labels = label_records(selected)
    keys = key_records(selected)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = out_dir / "labels.xlsx"
    csv_path = out_dir / "labels.csv"
    key_path = out_dir / "key.csv"

    write_xlsx(xlsx_path, labels)
    write_csv(csv_path, labels, LABEL_COLUMNS)
    write_csv(key_path, keys, KEY_COLUMNS)

    print(f"eligible: {len(eligible)}")
    print(summarize(selected))
    print(f"wrote {xlsx_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {key_path}")


if __name__ == "__main__":
    main()
