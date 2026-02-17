"""检查 FC2 data list 有效路径数量。"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _read_non_empty_lines(path: Path) -> List[str]:
    """读取文本文件并返回非空行。"""
    if not path.exists():
        return []
    lines: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            item = raw.strip()
            if item:
                lines.append(item)
    return lines


def _resolve_path(raw_path: str, base_path: Path) -> Path:
    """将原始路径解析为绝对路径。"""
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (base_path / candidate).resolve()


def _build_triplet_paths(img0_path: Path) -> Tuple[Path, Path, Path]:
    """由 img_0 路径推导 img_1 和 flow 路径。"""
    img1_path = Path(str(img0_path).replace("img_0", "img_1"))
    flow_path = Path(str(img0_path).replace("img_0.png", "flow_01.flo"))
    return img0_path, img1_path, flow_path


def _check_dirnames(dirnames: List[str], base_path: Path, keep_examples: int = 10) -> Dict[str, object]:
    """检查 FC2_dirnames.txt 的有效性。"""
    total = len(dirnames)
    valid_img0 = 0
    valid_triplet = 0
    missing_examples: List[str] = []

    for raw in dirnames:
        img0_abs = _resolve_path(raw_path=raw, base_path=base_path)
        img0_path, img1_path, flow_path = _build_triplet_paths(img0_abs)

        if img0_path.exists():
            valid_img0 += 1
        else:
            if len(missing_examples) < keep_examples:
                missing_examples.append(str(img0_path))
            continue

        if img1_path.exists() and flow_path.exists():
            valid_triplet += 1

    return {
        "total_entries": total,
        "img0_exists": valid_img0,
        "triplet_exists": valid_triplet,
        "missing_img0": total - valid_img0,
        "missing_examples": missing_examples,
    }


def _parse_indices(tokens: List[str]) -> Tuple[List[int], int]:
    """解析索引列表并统计非法 token 数量。"""
    indices: List[int] = []
    bad_tokens = 0
    for token in tokens:
        try:
            indices.append(int(token))
        except Exception:
            bad_tokens += 1
    return indices, bad_tokens


def _check_split(split_tokens: List[str], dirnames: List[str], base_path: Path, keep_examples: int = 10) -> Dict[str, object]:
    """检查 train/test 索引文件对应样本的有效性。"""
    indices, bad_tokens = _parse_indices(split_tokens)

    in_range = 0
    out_of_range = 0
    valid_img0 = 0
    valid_triplet = 0
    missing_examples: List[str] = []

    for idx in indices:
        if idx < 0 or idx >= len(dirnames):
            out_of_range += 1
            continue

        in_range += 1
        raw = dirnames[idx]
        img0_abs = _resolve_path(raw_path=raw, base_path=base_path)
        img0_path, img1_path, flow_path = _build_triplet_paths(img0_abs)

        if img0_path.exists():
            valid_img0 += 1
        else:
            if len(missing_examples) < keep_examples:
                missing_examples.append(str(img0_path))
            continue

        if img1_path.exists() and flow_path.exists():
            valid_triplet += 1

    return {
        "total_tokens": len(split_tokens),
        "bad_tokens": bad_tokens,
        "total_indices": len(indices),
        "unique_indices": len(set(indices)),
        "in_range": in_range,
        "out_of_range": out_of_range,
        "img0_exists": valid_img0,
        "triplet_exists": valid_triplet,
        "missing_img0": in_range - valid_img0,
        "missing_examples": missing_examples,
    }


def _build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="check FC2 data list validity")
    parser.add_argument("--data_list", required=True, help="目录，包含 FC2_dirnames/train/test 文件")
    parser.add_argument("--base_path", required=True, help="数据根目录，例如 /data/.../MCUFlowNet")
    parser.add_argument("--examples", type=int, default=10, help="最多输出多少条缺失样本示例")
    parser.add_argument("--json_out", default="", help="可选：保存完整 JSON 报告路径")
    return parser


def main() -> int:
    """执行检查入口。"""
    args = _build_parser().parse_args()

    data_list_dir = Path(args.data_list).resolve()
    base_path = Path(args.base_path).resolve()
    examples = max(0, int(args.examples))

    dirnames_path = data_list_dir / "FC2_dirnames.txt"
    train_path = data_list_dir / "FC2_train.txt"
    test_path = data_list_dir / "FC2_test.txt"

    dirnames = _read_non_empty_lines(dirnames_path)
    train_tokens = _read_non_empty_lines(train_path)
    test_tokens = _read_non_empty_lines(test_path)

    report = {
        "data_list_dir": str(data_list_dir),
        "base_path": str(base_path),
        "files": {
            "FC2_dirnames.txt": _check_dirnames(dirnames=dirnames, base_path=base_path, keep_examples=examples),
            "FC2_train.txt": _check_split(split_tokens=train_tokens, dirnames=dirnames, base_path=base_path, keep_examples=examples),
            "FC2_test.txt": _check_split(split_tokens=test_tokens, dirnames=dirnames, base_path=base_path, keep_examples=examples),
        },
    }

    print("=== FC2 Data List Validity Report ===")
    print(f"data_list_dir: {report['data_list_dir']}")
    print(f"base_path:     {report['base_path']}")

    d = report["files"]["FC2_dirnames.txt"]
    print("\n[FC2_dirnames.txt]")
    print(f"total_entries  : {d['total_entries']}")
    print(f"img0_exists    : {d['img0_exists']}")
    print(f"triplet_exists : {d['triplet_exists']}")
    print(f"missing_img0   : {d['missing_img0']}")

    t = report["files"]["FC2_train.txt"]
    print("\n[FC2_train.txt]")
    print(f"total_tokens   : {t['total_tokens']}")
    print(f"bad_tokens     : {t['bad_tokens']}")
    print(f"total_indices  : {t['total_indices']}")
    print(f"unique_indices : {t['unique_indices']}")
    print(f"in_range       : {t['in_range']}")
    print(f"out_of_range   : {t['out_of_range']}")
    print(f"img0_exists    : {t['img0_exists']}")
    print(f"triplet_exists : {t['triplet_exists']}")
    print(f"missing_img0   : {t['missing_img0']}")

    v = report["files"]["FC2_test.txt"]
    print("\n[FC2_test.txt]")
    print(f"total_tokens   : {v['total_tokens']}")
    print(f"bad_tokens     : {v['bad_tokens']}")
    print(f"total_indices  : {v['total_indices']}")
    print(f"unique_indices : {v['unique_indices']}")
    print(f"in_range       : {v['in_range']}")
    print(f"out_of_range   : {v['out_of_range']}")
    print(f"img0_exists    : {v['img0_exists']}")
    print(f"triplet_exists : {v['triplet_exists']}")
    print(f"missing_img0   : {v['missing_img0']}")

    if args.json_out:
        json_path = Path(args.json_out).resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\njson_report: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
