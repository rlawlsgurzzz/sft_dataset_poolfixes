# Builds a single-file HTML visual report from SFT test_summary_report.json files.
# Reads one or more summary JSON files and treats the last positional argument as output HTML.
# Embeds matplotlib charts as base64 images and includes model-comparison breakdown charts.
# Does not call model endpoints and does not read per-sample CSV files.
# Uses only summary JSON content already produced by the evaluator.

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RATE_FIELDS = [
    "request_success_rate",
    "json_parse_success_rate",
    "top_level_schema_success_rate",
    "strict_schema_success_rate",
    "contract_ok_rate",
    "contract_violation_rate",
    "sac_success_rate",
    "exact_json_match_rate",
    "action_exact_match_rate",
    "semantic_or_action_valid_match_rate",
    "dialog_validity_rate",
    "empty_action_correctness_rate",
    "timeout_rate",
]

LATENCY_FIELDS = [
    "avg_latency_ms",
]

TOKEN_FIELDS = [
    "avg_output_tokens",
]

QUALITY_CHART_FIELDS = [
    "json_parse_success_rate",
    "top_level_schema_success_rate",
    "strict_schema_success_rate",
    "contract_ok_rate",
    "sac_success_rate",
    "action_exact_match_rate",
    "semantic_or_action_valid_match_rate",
    "dialog_validity_rate",
    "empty_action_correctness_rate",
]


BREAKDOWN_CHART_METRICS = [
    "json_parse_success_rate",
    "strict_schema_success_rate",
    "contract_ok_rate",
    "sac_success_rate",
    "action_exact_match_rate",
    "semantic_or_action_valid_match_rate",
]

BREAKDOWN_TABLE_FIELDS = [
    "total_samples",
    "json_parse_success_rate",
    "strict_schema_success_rate",
    "contract_ok_rate",
    "contract_violation_rate",
    "sac_success_rate",
    "action_exact_match_rate",
    "semantic_or_action_valid_match_rate",
    "dialog_validity_rate",
    "empty_action_correctness_rate",
    "avg_latency_ms",
    "p95_latency_ms",
    "avg_output_tokens",
]

FIELD_LABELS = {
    "total_samples": "samples",
    "request_success_rate": "request success",
    "json_parse_success_rate": "JSON parse",
    "top_level_schema_success_rate": "top-level schema",
    "strict_schema_success_rate": "strict schema",
    "contract_ok_rate": "contract OK",
    "contract_violation_rate": "contract violation",
    "sac_success_rate": "SAC",
    "exact_json_match_rate": "exact JSON match",
    "action_exact_match_rate": "action exact",
    "semantic_or_action_valid_match_rate": "semantic/action valid",
    "dialog_validity_rate": "dialog valid",
    "empty_action_correctness_rate": "empty action correct",
    "timeout_rate": "timeout",
    "avg_latency_ms": "avg latency ms",
    "p50_latency_ms": "p50 latency ms",
    "p95_latency_ms": "p95 latency ms",
    "avg_output_tokens": "avg output tokens",
    "p95_output_tokens": "p95 output tokens",
}

METRIC_EXPLANATIONS = [
    ("samples", "평가에 사용된 샘플 수. 현재 test 기준은 500개다."),
    ("request success", "slm에게 보낸 요청의 응답이 성공한 비율."),
    ("JSON parse", "모델 출력 문자열이 순수 JSON object로 파싱된 비율. JSON 밖 텍스트나 깨진 JSON은 실패."),
    ("top-level schema", "최상위 key가 정확히 thinking, dialog, action 세 개였던 비율."),
    ("strict schema", "top-level뿐 아니라 dialog/action/sequence/action type별 필드 구조까지 모두 맞은 비율."),
    ("contract OK", "runtime commandAnalysis와 전장 상태 기준으로 actor, target, move, skill, wait, dialog 규칙을 모두 지킨 비율."),
    ("contract violation", "contract OK의 반대"),
    ("SAC", "Strict schema AND Contract OK의 약자. 출력 구조가 엄격한 스키마를 통과했고 동시에 runtime contract도 통과한 비율이며, 사실상 바로 사용할 수 있는 응답 비율."),
    ("action exact", "모델 output.action이 gold output.action과 정확히 일치한 비율."),
    ("semantic/action valid", "action exact이거나, gold와 완전히 같지는 않아도 actor/target/action type 구조가 맞고 runtime contract를 통과한 비율."),
    ("dialog valid", "dialog가 action actor와 1:1로 대응하고 중복/불필요 dialog가 없었던 비율."),
    ("empty action correct", "gold(즉 정답)가 empty action인지 여부와 모델의 empty/non-empty action 여부가 일치한 비율. 즉 empty가 나와야할 때 나온 비율"),
    ("timeout", "응답 수집 단계에서 timeout으로 기록된 비율."),
    ("avg latency ms", "성공 응답의 평균 지연시간(ms)."),
    ("p50 latency ms", "성공 응답 지연시간의 중앙값(ms)."),
    ("p95 latency ms", "성공 응답 지연시간의 95퍼센타일(ms). tail latency 확인용."),
    ("avg output tokens", "평균 출력 토큰 수. 보통 토큰 수가 많을수록 비용과 지연시간이 증가함."),
    ("p95 output tokens", "출력 토큰 수의 95퍼센타일. 비정상적으로 긴 출력 확인용."),
]

BREAKDOWN_LABELS = {
    "by_intent_family": "intent family",
    "by_command_style": "command style",
    "by_actor_selection": "actor selection",
    "by_target_selection": "target selection",
    "by_action_pattern": "action pattern",
}


@dataclass
class ModelReport:
    model_name: str
    source_path: str
    created_at_utc: str | None
    test_path: str | None
    responses_path: str | None
    total_samples: int | None
    total_responses: int | None
    summary: dict[str, Any]
    breakdown: dict[str, Any]


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON file: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Summary report must be a JSON object: {path}")
    return value


def as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def enrich_summary(summary: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(summary)
    contract_violation_rate = as_number(enriched.get("contract_violation_rate"))
    if "contract_ok_rate" not in enriched and contract_violation_rate is not None:
        enriched["contract_ok_rate"] = 1.0 - contract_violation_rate
    return enriched


def extract_model_reports(path: Path) -> list[ModelReport]:
    data = read_json(path)
    model_summaries = data.get("model_summaries")
    if not isinstance(model_summaries, dict) or not model_summaries:
        raise ValueError(f"Missing non-empty model_summaries in {path}")

    top_breakdown = data.get("breakdown")
    top_breakdown = top_breakdown if isinstance(top_breakdown, dict) else {}

    reports: list[ModelReport] = []
    for model_name, summary in model_summaries.items():
        if not isinstance(summary, dict):
            continue
        model_breakdown = top_breakdown.get(model_name)
        if not isinstance(model_breakdown, dict):
            model_breakdown = {}

        final_name = str(model_name)
        if len(model_summaries) == 1 and isinstance(data.get("model_name"), str) and data["model_name"]:
            final_name = data["model_name"]

        reports.append(
            ModelReport(
                model_name=final_name,
                source_path=str(path),
                created_at_utc=data.get("created_at_utc") if isinstance(data.get("created_at_utc"), str) else None,
                test_path=data.get("test_path") if isinstance(data.get("test_path"), str) else None,
                responses_path=data.get("responses_path") if isinstance(data.get("responses_path"), str) else None,
                total_samples=int(data["total_samples"]) if isinstance(data.get("total_samples"), int) else None,
                total_responses=int(data["total_responses"]) if isinstance(data.get("total_responses"), int) else None,
                summary=enrich_summary(summary),
                breakdown=model_breakdown,
            )
        )

    if not reports:
        raise ValueError(f"No usable model summaries found in {path}")
    return reports


def num(value: Any, digits: int = 3) -> str:
    number = as_number(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def intish(value: Any) -> str:
    number = as_number(value)
    if number is None:
        return ""
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.3f}"


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def rate_cell(value: Any) -> str:
    number = as_number(value)
    if number is None:
        return "<td></td>"
    clamped = max(0.0, min(1.0, number))
    label = f"{number * 100:.2f}%"
    return (
        "<td class='rate-cell'>"
        f"<div class='bar-wrap'><div class='bar-fill' style='width:{clamped * 100:.3f}%'></div></div>"
        f"<span>{esc(label)}</span>"
        "</td>"
    )


def number_cell(value: Any, digits: int = 3) -> str:
    return f"<td class='num'>{esc(num(value, digits))}</td>"


def samples_cell(value: Any) -> str:
    return f"<td class='num'>{esc(intish(value))}</td>"


def metric_label(field: str) -> str:
    return FIELD_LABELS.get(field, field)


def breakdown_label(field: str) -> str:
    return BREAKDOWN_LABELS.get(field, field)


def import_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise SystemExit(
            "matplotlib is not installed. Install it with:\n"
            "  py -3.12 -m pip install matplotlib\n"
            "or:\n"
            "  python -m pip install matplotlib"
        ) from error
    return plt


def make_grouped_bar_chart(
    reports: list[ModelReport],
    fields: list[str],
    title: str,
    ylabel: str,
    as_percent: bool,
) -> str:
    plt = import_pyplot()
    labels = [report.model_name for report in reports]
    x_positions = list(range(len(labels)))
    series_count = len(fields)
    width = min(0.8 / max(series_count, 1), 0.16)

    fig_width = max(10.0, len(labels) * 1.4 + series_count * 0.8)
    fig_height = 5.6
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for index, field in enumerate(fields):
        offset = (index - (series_count - 1) / 2) * width
        values: list[float] = []
        for report in reports:
            value = as_number(report.summary.get(field))
            values.append(0.0 if value is None else (value * 100.0 if as_percent else value))
        ax.bar([x + offset for x in x_positions], values, width, label=metric_label(field))

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(loc="best", fontsize=8)
    if as_percent:
        ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def make_breakdown_model_group_chart(
    reports: list[ModelReport],
    dim: str,
    metric: str,
    max_groups: int = 30,
) -> str:
    plt = import_pyplot()

    group_names: list[str] = []
    group_seen: set[str] = set()
    for report in reports:
        groups = report.breakdown.get(dim)
        if not isinstance(groups, dict):
            continue
        for group_name in groups.keys():
            name = str(group_name)
            if name not in group_seen:
                group_seen.add(name)
                group_names.append(name)

    group_names = sorted(group_names)
    if len(group_names) > max_groups:
        group_names = group_names[:max_groups]

    model_count = len(reports)
    group_count = len(group_names)
    width = min(0.8 / max(model_count, 1), 0.18)
    x_positions = list(range(group_count))

    fig_width = max(10.0, min(28.0, group_count * 0.75 + model_count * 0.8))
    fig_height = 5.8
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for model_index, report in enumerate(reports):
        offset = (model_index - (model_count - 1) / 2) * width
        groups = report.breakdown.get(dim)
        groups = groups if isinstance(groups, dict) else {}
        values: list[float] = []
        for group_name in group_names:
            summary = groups.get(group_name)
            if not isinstance(summary, dict):
                values.append(0.0)
                continue
            enriched = enrich_summary(summary)
            value = as_number(enriched.get(metric))
            values.append(0.0 if value is None else value * 100.0)
        ax.bar([x + offset for x in x_positions], values, width, label=report.model_name)

    ax.set_title(f"{breakdown_label(dim)} - {metric_label(metric)}")
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(group_names, rotation=55, ha="right", fontsize=8)
    ax.set_ylim(0, 100)
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def html_chart(base64_png: str, alt: str) -> str:
    return f"<figure><img src='data:image/png;base64,{base64_png}' alt='{esc(alt)}'></figure>"


def build_metric_explanation_table() -> str:
    rows = [
        f"<tr><td class='metric-name'>{esc(name)}</td><td>{esc(description)}</td></tr>"
        for name, description in METRIC_EXPLANATIONS
    ]
    return (
        "<div class='table-wrap'>"
        "<table>"
        "<thead><tr><th>metric</th><th>description</th></tr></thead>"
        "<tbody>" + "\n".join(rows) + "</tbody>"
        "</table>"
        "</div>"
    )


def build_summary_table(reports: list[ModelReport]) -> str:
    fields = [
        "total_samples",
        "request_success_rate",
        "json_parse_success_rate",
        "top_level_schema_success_rate",
        "strict_schema_success_rate",
        "contract_ok_rate",
        "contract_violation_rate",
        "sac_success_rate",
        "action_exact_match_rate",
        "semantic_or_action_valid_match_rate",
        "dialog_validity_rate",
        "empty_action_correctness_rate",
        "timeout_rate",
        "avg_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "avg_output_tokens",
        "p95_output_tokens",
    ]
    header = "".join(f"<th>{esc(metric_label(field))}</th>" for field in fields)
    body_rows: list[str] = []
    for report in reports:
        cells = [f"<td class='sticky'>{esc(report.model_name)}</td>"]
        for field in fields:
            if field == "total_samples":
                cells.append(samples_cell(report.summary.get(field)))
            elif field in RATE_FIELDS:
                cells.append(rate_cell(report.summary.get(field)))
            else:
                cells.append(number_cell(report.summary.get(field)))
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<div class='table-wrap'>"
        "<table>"
        "<thead><tr><th class='sticky'>model</th>" + header + "</tr></thead>"
        "<tbody>" + "\n".join(body_rows) + "</tbody>"
        "</table>"
        "</div>"
    )


def build_delta_table(reports: list[ModelReport]) -> str:
    if len(reports) < 2:
        return "<p>Delta table requires at least two model reports.</p>"
    baseline = reports[0]
    fields = [
        "json_parse_success_rate",
        "strict_schema_success_rate",
        "contract_ok_rate",
        "contract_violation_rate",
        "sac_success_rate",
        "action_exact_match_rate",
        "semantic_or_action_valid_match_rate",
        "dialog_validity_rate",
        "empty_action_correctness_rate",
        "avg_latency_ms",
        "p95_latency_ms",
        "avg_output_tokens",
    ]

    header = "".join(f"<th>{esc(metric_label(field))}</th>" for field in fields)
    rows: list[str] = []
    for report in reports[1:]:
        cells = [f"<td class='sticky'>{esc(report.model_name)} vs {esc(baseline.model_name)}</td>"]
        for field in fields:
            current = as_number(report.summary.get(field))
            base = as_number(baseline.summary.get(field))
            if current is None or base is None:
                cells.append("<td></td>")
                continue
            delta = current - base
            if field in RATE_FIELDS:
                cells.append(f"<td class='num'>{delta * 100:+.2f} pp</td>")
            else:
                cells.append(f"<td class='num'>{delta:+.3f}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return (
        "<div class='table-wrap'>"
        "<table>"
        "<thead><tr><th class='sticky'>comparison</th>" + header + "</tr></thead>"
        "<tbody>" + "\n".join(rows) + "</tbody>"
        "</table>"
        "</div>"
    )


def available_breakdown_dimensions(reports: list[ModelReport]) -> list[str]:
    preferred = [
        "by_intent_family",
        "by_command_style",
        "by_actor_selection",
        "by_target_selection",
        "by_action_pattern",
    ]
    dims: set[str] = set()
    for report in reports:
        dims.update(str(key) for key in report.breakdown.keys())
    return [dim for dim in preferred if dim in dims]


def breakdown_rows(reports: list[ModelReport], dim: str) -> list[tuple[str, str, dict[str, Any]]]:
    rows: list[tuple[str, str, dict[str, Any]]] = []
    for report in reports:
        groups = report.breakdown.get(dim)
        if not isinstance(groups, dict):
            continue
        for group, summary in sorted(groups.items(), key=lambda item: str(item[0])):
            if isinstance(summary, dict):
                rows.append((report.model_name, str(group), enrich_summary(summary)))
    return rows


def build_breakdown_table(reports: list[ModelReport], dim: str) -> str:
    rows = breakdown_rows(reports, dim)
    if not rows:
        return "<p>No rows.</p>"

    fields = BREAKDOWN_TABLE_FIELDS
    header = "".join(f"<th>{esc(metric_label(field))}</th>" for field in fields)
    body: list[str] = []

    for model_name, group_name, summary in rows:
        cells = [f"<td class='sticky'>{esc(group_name)}</td>", f"<td>{esc(model_name)}</td>"]
        for field in fields:
            if field == "total_samples":
                cells.append(samples_cell(summary.get(field)))
            elif field in RATE_FIELDS:
                cells.append(rate_cell(summary.get(field)))
            else:
                cells.append(number_cell(summary.get(field)))
        body.append("<tr>" + "".join(cells) + "</tr>")

    return (
        "<div class='table-wrap breakdown-table'>"
        "<table>"
        "<thead><tr><th class='sticky'>group</th><th>model</th>" + header + "</tr></thead>"
        "<tbody>" + "\n".join(body) + "</tbody>"
        "</table>"
        "</div>"
    )


def build_breakdown_chart_section(reports: list[ModelReport], dim: str) -> str:
    rows = breakdown_rows(reports, dim)
    if not rows:
        return ""

    group_count = len({group for _, group, _ in rows})
    if group_count > 30:
        return "<p class='note'>Chart omitted because this breakdown has more than 30 groups. The full table is still included.</p>"

    charts: list[str] = []
    for metric in BREAKDOWN_CHART_METRICS:
        img = make_breakdown_model_group_chart(reports, dim, metric, max_groups=30)
        charts.append(html_chart(img, f"{dim} {metric}"))
    return "\n".join(charts)


def build_all_breakdowns(reports: list[ModelReport]) -> str:
    dims = available_breakdown_dimensions(reports)
    if not dims:
        return "<p>No supported breakdown data found in the input reports.</p>"

    sections: list[str] = []
    for dim in dims:
        rows = breakdown_rows(reports, dim)
        group_count = len({group_name for _, group_name, _ in rows})
        sections.append(
            "<section class='breakdown-section'>"
            f"<h3>{esc(breakdown_label(dim))} <span class='muted'>({esc(dim)}, {group_count} groups)</span></h3>"
            "<p class='note'>Each chart uses the category group as the x-axis. Bars inside each group compare models.</p>"
            + build_breakdown_chart_section(reports, dim)
            + build_breakdown_table(reports, dim)
            + "</section>"
        )
    return "\n".join(sections)


def build_html(reports: list[ModelReport], title: str) -> str:
    quality_img = make_grouped_bar_chart(reports, QUALITY_CHART_FIELDS, "Core quality rates", "Rate (%)", True)
    latency_img = make_grouped_bar_chart(reports, LATENCY_FIELDS, "Latency", "Milliseconds", False)
    token_img = make_grouped_bar_chart(reports, TOKEN_FIELDS, "Output token count", "Tokens", False)

    created = datetime.now(timezone.utc).isoformat()
    model_order = ", ".join(report.model_name for report in reports)

    css = """
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      margin: 32px;
      line-height: 1.45;
      color: #202124;
      background: #fff;
    }
    h1, h2, h3 { margin-top: 28px; }
    .muted { color: #666; font-size: 0.9em; font-weight: normal; }
    .note {
      color: #555;
      background: #f6f8fa;
      border-left: 4px solid #d0d7de;
      padding: 10px 12px;
    }
    figure {
      margin: 18px 0 28px 0;
      padding: 0;
    }
    img {
      max-width: 100%;
      height: auto;
      border: 1px solid #ddd;
      border-radius: 8px;
      background: white;
    }
    .table-wrap {
      overflow-x: auto;
      border: 1px solid #ddd;
      border-radius: 8px;
      margin: 14px 0 26px 0;
    }
    table {
      border-collapse: collapse;
      width: max-content;
      min-width: 100%;
      font-size: 13px;
    }
    th, td {
      border-bottom: 1px solid #eee;
      border-right: 1px solid #eee;
      padding: 8px 10px;
      vertical-align: middle;
      white-space: nowrap;
    }
    th {
      background: #f6f8fa;
      font-weight: 600;
      text-align: left;
      position: sticky;
      top: 0;
      z-index: 1;
    }
    .sticky {
      position: sticky;
      left: 0;
      background: #fff;
      z-index: 2;
      font-weight: 600;
    }
    th.sticky {
      background: #f6f8fa;
      z-index: 3;
    }
    .num {
      text-align: right;
      font-variant-numeric: tabular-nums;
    }
    .metric-name {
      font-weight: 600;
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    }
    .rate-cell {
      min-width: 150px;
      position: relative;
      font-variant-numeric: tabular-nums;
    }
    .bar-wrap {
      display: inline-block;
      width: 78px;
      height: 8px;
      background: #eef1f4;
      border-radius: 999px;
      margin-right: 8px;
      vertical-align: middle;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      background: #6b7280;
    }
    .breakdown-section {
      border-top: 1px solid #e5e7eb;
      padding-top: 12px;
    }
    """

    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>{esc(title)}</title>
<style>{css}</style>
</head>
<body>
<h1>{esc(title)}</h1>
<p class="muted">Created at UTC: {esc(created)}</p>
<p><b>Model order:</b> {esc(model_order)}</p>

<h2>Metric definitions</h2>
{build_metric_explanation_table()}

<h2>Model summary</h2>
{build_summary_table(reports)}

<h2>Delta from Base Model</h2>
<p class="note">Rate deltas are percentage points.</p>
{build_delta_table(reports)}

<h2>Epoch1이 Base Model보다 나은 이유:</h2>
<p class="note">베이스 모델이 강한 부분:<br>
top-level schema는 틀리면 아예 못 써먹는 수준입니다.<br><br>

strict schema는 틀릴 경우엔 일부 보정이 가능하나, 살리지 못하는 경우도 있습니다. 이건 예를 들어 move 에 이동 시간 필드를 모델이 자의적으로 넣는다든지, wait에서 대기 시간을 빼버린다든지 하는 경우입니다.<br><br>

이 두 지표는 base 모델이 강하게 나오네요. 높게 나온데는 이유가 있습니다. 이유는 뻔한데, 후술하겠습니다.<br><br><br>


1에포크가 좋은 이유:<br>
contract OK가 1에포크 쪽이 아주 강하게 나옵니다. 이건 타겟 불가한 유닛을 타겟으로 삼거나, 스킬이 아군 대상인데 적에게 사용해버리면 "내려갑니다". 즉 이건 실제 유니티 엔진에서 넘어간 정보들에 얼마나 충실했는지 보는 비율입니다. 제가 보기엔 이거 높은게 위의 strict schema보다 훨씬 중요합니다.<br><br>

SAC는 strict schema 와 contract OK를 동시에 만족하는 비율입니다. 이건 그냥 완벽한 출력, 즉 진정한 의미의 성공률이라고 보면 됩니다.<br><br>

action exact, semantic/action valid, dialog valid, empty action correct 도 모두 1에포크가 유리하게 나오네요.<br> 
action exact, semantic/action valid는 "명령에 따라 실제로 적절한 행동이 나왔는지"를 보는 지표입니다.<br>
이걸 위해서 굳이굳이 메타데이터에 gold(진짜 정답지 & 부분 점수 인정표라고 보면 됩니다)까지 꾸역꾸역 넣었던 건데요.<br><br>

베이스 모델에서 action exact와 semantic/action valid가 저렇게나 낮게 나왔다는 것은, 적절하지 않은 행동도 끼워넣거나 넣어야할 행동을 넣지 않아서 그렇습니다. 예를 들어 공격 명령이고, 상황이 적절한데도 스킬을 써버린다든지 하는 경우입니다. 이렇게 마구잡이로 액션 넣으니까 strict scema가 높게 나온겁니다...<br>avg output tokens도 약간 이 맥락입니다. 베이스 모델이 쓸데없는/넣으면 안될 액션까지 마구 넣어버린 결과죠<br><br>

action exact는 높을 당위성까지는 없지만, semantic/action valid는 반드시 높아야합니다. strict schema는 보정이 가능하지만, 이쪽은 진짜로 "명령이 어떤 의미인지"를 나타내는 지표라 틀리면 안돼요. 엉뚱한 행동을 하게 되는 비율이 바로 (100-이 비율)인데, 1에포크 정도면 딱 적절한 것 같아요. evaluator 자체를 좀 빡세게 기준을 잡아놔서 실제 게임에서 체감은 더 나을거예요.<br><br>

empty action correct는 빈 행동이 나와야될 때 올바르게 출력한 경우입니다. 이건 의외로 베이스 모델도 꽤 하네요.<br><br>

avg latency ms는 응답 시간인데, 모두 2초 이하라서 사실상 큰 영향은 없습니다.

저는 에포크 두번 돌린게 제일 결과가 좋을 거라 생각했었는데, 의외로 1에포크가 진짜 좋은 지표를 보여줬네요. 전 sft 정말 성공적이라고 봅니다.<br></p>

<h2>Core visual comparison</h2>
{html_chart(quality_img, "Core quality rates")}
{html_chart(latency_img, "Latency")}
{html_chart(token_img, "Output tokens")}

<h2>Breakdown comparisons</h2>
<p class="note">Breakdown charts use category groups on the x-axis. Each group contains one bar per model, so base/epoch models can be compared within the same category.</p>
{build_all_breakdowns(reports)}
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a single-file visual HTML report from one or more test_summary_report.json files. "
            "The last positional argument is the output .html path."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Input summary JSON files followed by output HTML path.",
    )
    parser.add_argument("--title", default="SFT Test Evaluation Visual Report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.paths) < 2:
        raise SystemExit(
            "Usage:\n"
            "  py -3.12 scripts/build_visual_report.py report_base.json report_epoch1.json report_epoch2.json report_epoch3.json final_report.html"
        )

    input_paths = [Path(item) for item in args.paths[:-1]]
    output_path = Path(args.paths[-1])

    reports: list[ModelReport] = []
    for path in input_paths:
        reports.extend(extract_model_reports(path))

    if not reports:
        raise SystemExit("No model reports loaded.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_html(reports, args.title), encoding="utf-8")
    print(f"[done] wrote {output_path}")
    print(f"[done] models: {', '.join(report.model_name for report in reports)}")
    print(f"[done] breakdown dimensions: {', '.join(available_breakdown_dimensions(reports)) or 'none'}")


if __name__ == "__main__":
    main()
