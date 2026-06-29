"""
从 nsys_logs/ 中所有 .log 文件提取 CINN JIT kernel 信息，
分析每个 kernel 融合了 Elementwise/Broadcast/Injective/Reduce 哪几类算子，
输出 Excel 表格。

用法:
    python collect_kernel_info.py

输出:
    cinn_kernel_analysis.xlsx
"""

import os
import re
import glob
import csv
import io
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

# ===== 配置 =====
LOG_DIR = "nsys_logs_dy2stcinn"
OUTPUT_FILE = "cinn_kernel_analysis_dy2stcinn.xlsx"
# ================

# ----------------------------------------------------------------
# 算子短名 -> 完整 CINN 算子名 (来自 ShortenOpName, utils.cc:658-678)
# ----------------------------------------------------------------
SHORT_TO_FULL = {
    "full": "fill_constant",
    "sum": "reduce_sum",
    "r_max": "reduce_max",
    "r_min": "reduce_min",
    "prod": "reduce_prod",
    "add": "elementwise_add",
    "mul": "elementwise_mul",
    "sub": "subtract",
    "div": "divide",
    "bc": "broadcast_to",
    "gs": "generate_shape",
    "yield": "yield_store",
}

# ----------------------------------------------------------------
# 有效运行时分类 (考虑 OpKind() 覆盖逻辑, utils.cc:855-880)
# 所有注册为 kBroadcast 的二元算子在运行时被重分类为 kElementWise,
# 仅 broadcast_to 保留为 kBroadcast
# ----------------------------------------------------------------
OP_CATEGORY = {}

_elementwise_ops = [
    # 原始 elementwise
    "exp", "erf", "sqrt", "log", "floor", "ceil", "rint", "round", "tanh",
    "log2", "log10", "trunc", "cos", "cosh", "tan", "sin", "sinh",
    "acos", "acosh", "asin", "asinh", "atan", "atanh",
    "bitwise_not", "negative", "identity", "sign", "abs", "rsqrt",
    "sigmoid", "cbrt", "clz", "popc",
    "isnan", "isfinite", "isinf",
    "scale", "const_scalar", "fill_constant", "assign_value",
    "squeeze", "expand_dims", "reshape", "cast", "yield_store",
    "arange", "logical_not", "tril", "assign_out_", "isclose",
    "select", "slice_assign", "reciprocal", "logical_right_shift",
    "generate_shape",
    # 二元算子 (注册为 kBroadcast, 运行时覆盖为 kElementWise)
    "elementwise_add", "atan2", "elementwise_mul", "subtract", "divide",
    "floor_divide", "mod", "remainder", "max", "min", "pow",
    "logical_and", "logical_or", "logical_xor",
    "greater_than", "less_than", "equal", "not_equal",
    "greater_equal", "less_equal",
    "bitwise_or", "bitwise_xor", "bitwise_and",
    "left_shift", "right_shift",
]
_broadcast_ops = ["broadcast_to"]
_injective_ops = [
    "concat", "reverse", "transpose", "slice", "gather",
    "scatter_assign", "bitcast_convert", "lookup_table",
    "one_hot", "gather_nd",
]
_reduce_ops = [
    "reduce_sum", "reduce_prod", "variance",
    "argmax", "argmin", "reduce_max", "reduce_min",
    "reduce_all", "reduce_any",
]

for op in _elementwise_ops:
    OP_CATEGORY[op] = "Elementwise"
for op in _broadcast_ops:
    OP_CATEGORY[op] = "Broadcast"
for op in _injective_ops:
    OP_CATEGORY[op] = "Injective"
for op in _reduce_ops:
    OP_CATEGORY[op] = "Reduce"

# ----------------------------------------------------------------
# 构建 token -> category 映射 (token 是 kernel 名中出现的短名)
# ----------------------------------------------------------------
TOKEN_CATEGORY = {}
# 先加不在 SHORT_TO_FULL.values() 中的算子 (它们在 kernel 名中直接用原名)
for op, cat in OP_CATEGORY.items():
    TOKEN_CATEGORY[op] = cat
# 再加短名映射 (覆盖全名)
for short, full in SHORT_TO_FULL.items():
    if full in OP_CATEGORY:
        TOKEN_CATEGORY[short] = OP_CATEGORY[full]

# 按长度降序排列, 用于贪心最长匹配
ALL_TOKENS = sorted(TOKEN_CATEGORY.keys(), key=len, reverse=True)


def parse_ops_from_kernel_name(kernel_name):
    """从 CINN kernel 函数名中提取算子 token 列表。

    kernel name 格式:
        fn_<op1>_<op2>_...__<hash>_COND__...__kernel
    """
    # 提取 ops 部分: fn_ 和 __<digits> 之间
    m = re.match(r"fn_(.*?)__\d+", kernel_name)
    if not m:
        return []

    ops_str = m.group(1)
    ops = []
    remaining = ops_str

    while remaining:
        if remaining.startswith("_"):
            remaining = remaining[1:]
            continue

        matched = False
        for token in ALL_TOKENS:
            if remaining.startswith(token):
                rest = remaining[len(token):]
                if rest == "" or rest.startswith("_"):
                    ops.append(token)
                    remaining = rest
                    matched = True
                    break

        if not matched:
            # 跳过未知 token 直到下一个 '_'
            idx = remaining.find("_")
            if idx == -1:
                ops.append(remaining)  # unknown trailing
                break
            else:
                ops.append(remaining[:idx])
                remaining = remaining[idx:]

    return ops


def classify_kernel(ops):
    """给定 op token 列表, 返回该 kernel 融合的类别集合。"""
    categories = set()
    for op in ops:
        if op in TOKEN_CATEGORY:
            categories.add(TOKEN_CATEGORY[op])
    return categories


def extract_kernels_from_log(log_path):
    """从 log 文件的 CSV 数据中提取 fn_* kernel 条目。"""
    kernels = []
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # CSV 行: 首字符是数字, 且包含 fn_
            if ",fn_" not in line:
                continue
            if not line[0].isdigit():
                continue
            try:
                reader = csv.reader(io.StringIO(line))
                fields = next(reader)
                name = fields[-1].strip()
                if not name.startswith("fn_"):
                    continue
                # 去掉末尾 _kernel 后缀 (如果有)
                clean_name = re.sub(r"_kernel$", "", name)
                time_pct = float(fields[0])
                total_ns = int(fields[1])
                num_calls = int(fields[2])
                kernels.append({
                    "name": name,
                    "clean_name": clean_name,
                    "time_pct": time_pct,
                    "total_ns": total_ns,
                    "num_calls": num_calls,
                })
            except (ValueError, IndexError, StopIteration):
                continue
    return kernels


def case_name_from_log(log_path):
    """从 log 文件名提取 case 名, 还原路径格式。"""
    base = os.path.splitext(os.path.basename(log_path))[0]
    return base.replace("__", "/")


def main():
    log_files = sorted(glob.glob(os.path.join(LOG_DIR, "*.log")))
    total_logs = len(log_files)
    print(f"[INFO] 共发现 {total_logs} 个 log 文件")
    if total_logs == 0:
        print("[WARN] 未发现 log 文件, 请检查路径")
        return

    wb = Workbook()

    # ========== Sheet 1: 详细信息 ==========
    ws_detail = wb.active
    ws_detail.title = "Kernel Detail"

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    thin_border = Border(
        left=Side(style="thin"),
        right=Side(style="thin"),
        top=Side(style="thin"),
        bottom=Side(style="thin"),
    )
    cat_yes_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")

    headers = [
        "Case", "Kernel Name", "Elementwise", "Broadcast",
        "Injective", "Reduce", "Fused Categories",
        "Parsed Ops", "Time(%)", "Total Time(ns)", "Num Calls",
    ]
    for col, h in enumerate(headers, 1):
        cell = ws_detail.cell(row=1, column=col, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")
        cell.border = thin_border

    # ========== Sheet 2: 汇总信息 ==========
    ws_summary = wb.create_sheet("Case Summary")
    summary_headers = [
        "Case", "Total Kernels",
        "Elementwise Count", "Broadcast Count",
        "Injective Count", "Reduce Count",
        "Pure Elementwise", "Pure Reduce",
        "E+R Fused", "E+B Fused", "E+I Fused",
        "Total CINN Kernel Time(ns)",
    ]
    for col, h in enumerate(summary_headers, 1):
        cell = ws_summary.cell(row=1, column=col, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")
        cell.border = thin_border

    detail_row = 2
    summary_row = 2

    for log_idx, log_path in enumerate(log_files, 1):
        case = case_name_from_log(log_path)
        kernels = extract_kernels_from_log(log_path)

        if log_idx % 100 == 0:
            print(f"  [{log_idx}/{total_logs}] 已处理...")

        # 汇总计数
        cat_counts = {"Elementwise": 0, "Broadcast": 0, "Injective": 0, "Reduce": 0}
        pure_e = 0
        pure_r = 0
        e_r_fused = 0
        e_b_fused = 0
        e_i_fused = 0
        total_kernel_ns = 0

        for k in kernels:
            ops = parse_ops_from_kernel_name(k["clean_name"])
            cats = classify_kernel(ops)

            has_e = "Elementwise" in cats
            has_b = "Broadcast" in cats
            has_i = "Injective" in cats
            has_r = "Reduce" in cats

            for c in cats:
                cat_counts[c] += 1

            if cats == {"Elementwise"}:
                pure_e += 1
            if cats == {"Reduce"}:
                pure_r += 1
            if has_e and has_r:
                e_r_fused += 1
            if has_e and has_b:
                e_b_fused += 1
            if has_e and has_i:
                e_i_fused += 1

            total_kernel_ns += k["total_ns"]

            # 写详细行
            cats_str = "+".join(sorted(cats)) if cats else "Unknown"
            # 仅保留已知算子
            known_ops = [op for op in ops if op in TOKEN_CATEGORY]
            ops_str = ", ".join(known_ops) if known_ops else ", ".join(ops)

            row_data = [
                case, k["name"],
                "Y" if has_e else "",
                "Y" if has_b else "",
                "Y" if has_i else "",
                "Y" if has_r else "",
                cats_str, ops_str,
                k["time_pct"], k["total_ns"], k["num_calls"],
            ]
            for col, val in enumerate(row_data, 1):
                cell = ws_detail.cell(row=detail_row, column=col, value=val)
                cell.border = thin_border
                # 高亮 Y
                if col in (3, 4, 5, 6) and val == "Y":
                    cell.fill = cat_yes_fill
                    cell.alignment = Alignment(horizontal="center")
                elif col in (3, 4, 5, 6):
                    cell.alignment = Alignment(horizontal="center")
            detail_row += 1

        # 写汇总行
        summary_data = [
            case, len(kernels),
            cat_counts["Elementwise"], cat_counts["Broadcast"],
            cat_counts["Injective"], cat_counts["Reduce"],
            pure_e, pure_r,
            e_r_fused, e_b_fused, e_i_fused,
            total_kernel_ns,
        ]
        for col, val in enumerate(summary_data, 1):
            cell = ws_summary.cell(row=summary_row, column=col, value=val)
            cell.border = thin_border
        summary_row += 1

    # 调整列宽
    for ws in [ws_detail, ws_summary]:
        for col in ws.columns:
            max_len = 0
            col_letter = col[0].column_letter
            for cell in col:
                if cell.value:
                    max_len = max(max_len, len(str(cell.value)))
            ws.column_dimensions[col_letter].width = min(max_len + 2, 60)

    # 冻结首行
    ws_detail.freeze_panes = "A2"
    ws_summary.freeze_panes = "A2"

    # 添加筛选
    ws_detail.auto_filter.ref = ws_detail.dimensions
    ws_summary.auto_filter.ref = ws_summary.dimensions

    wb.save(OUTPUT_FILE)
    print(f"\n[DONE] 已生成 {OUTPUT_FILE}")
    print(f"  详细表: {detail_row - 2} 条 kernel 记录")
    print(f"  汇总表: {summary_row - 2} 个 case")


if __name__ == "__main__":
    main()
