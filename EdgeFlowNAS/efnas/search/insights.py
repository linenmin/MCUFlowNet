"""Phase 1.5 (search_hybrid_v1): insights.md 的最小机器解析与 ID 管理.

设计原则: 三个不变量, 其余全自由.
  1. 每条 insight 用三级标题, 格式必须为
     ``### I-{ID} ({status}): {title}``
  2. status ∈ {active, retired, under_review}
  3. ID 必须以 ``I-`` 开头, 后续只允许字母数字和短横线;
     一旦分配后保持稳定, 修订时不要改 ID

正文 (heading 之后到下一条 heading 之前) 完全自由形式. Scientist 想写啥写啥
(自然语言、Python 代码块、ASCII 表、中英混写均可). 本模块只做最低限度的机器
解析, 不试图理解正文内容.
"""

import re
from typing import Dict, List, Optional, Set


VALID_STATUSES: Set[str] = {"active", "retired", "under_review"}

# 三级标题的强制格式. status 大小写敏感; 必须严格匹配三个枚举值之一.
# title 部分非贪婪, 用 \s*$ 吞掉行末空白与可能的 \r (CRLF 换行环境).
INSIGHT_HEADING_RE = re.compile(
    r"^###\s+(?P<id>I-[A-Za-z0-9-]+)\s*\((?P<status>active|retired|under_review)\)"
    r"\s*:\s*(?P<title>.+?)\s*$",
    re.MULTILINE,
)

# 用于 next_insight_id: 仅匹配 I-NNN (纯数字) 形式
_SEQUENTIAL_ID_RE = re.compile(r"^I-(\d+)$")

# 用于 validate_id: 完整的 ID 合法性校验
_VALID_ID_RE = re.compile(r"^I-[A-Za-z0-9-]+$")


def parse_insights(md_text: str) -> List[Dict[str, str]]:
    """解析 insights.md 全文, 返回每条 insight 的 (id, status, title, body).

    body 是从 heading 行之后到下一个 heading (或文件末尾) 之间的全部 raw
    markdown 文本, 已 strip 首尾空白. 不解析 body 内部结构.

    任何不匹配 ``INSIGHT_HEADING_RE`` 的 ``###`` 标题都视为正文的一部分,
    属于上一条 insight 的 body. 这让 Scientist 可以在正文里自由使用三级
    及以下标题做内部分段.

    Args:
        md_text: insights.md 完整文本.

    Returns:
        List[Dict[str, str]] 顺序与文件中出现顺序一致. 每个 dict 含:
            - id (str): 形如 'I-001' 或 'I-EB0-DB1'
            - status (str): 'active' / 'retired' / 'under_review'
            - title (str): 标题行 ":" 后面的一行文本 (已 strip)
            - body (str): heading 之后的 raw markdown (已 strip 首尾空白)
    """
    if not md_text:
        return []

    matches = list(INSIGHT_HEADING_RE.finditer(md_text))
    if not matches:
        return []

    results: List[Dict[str, str]] = []
    for i, m in enumerate(matches):
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        body = md_text[body_start:body_end].strip()
        results.append({
            "id": m.group("id"),
            "status": m.group("status"),
            "title": m.group("title").strip(),
            "body": body,
        })
    return results


def list_active_insights(md_text: str) -> List[Dict[str, str]]:
    """快捷过滤 status == 'active' 的 insight."""
    return [it for it in parse_insights(md_text) if it["status"] == "active"]


def list_active_ids(md_text: str) -> List[str]:
    """返回 active insight 的 ID 列表 (出现顺序)."""
    return [it["id"] for it in list_active_insights(md_text)]


def next_insight_id(
    md_text: str,
    *,
    prefix: str = "I-",
    width: int = 3,
) -> str:
    """返回下一个可用的 sequential numeric ID.

    扫文件中所有已存在的 insight ID, 提取形如 'I-NNN' (纯数字) 的部分,
    取最大值 +1 并 zero-pad 到 ``width`` 位. **非 sequential 风格的 ID** (例如
    ``I-EB0-DB1``) 被忽略, 不影响下一个数字编号.

    Args:
        md_text: insights.md 文本
        prefix: 前缀 (默认 ``"I-"``)
        width: 数字部分零填充宽度 (默认 3, 即 I-001 .. I-999)

    Returns:
        形如 'I-001' 的下一个可用 ID.
    """
    existing = parse_insights(md_text)
    max_n = 0
    for item in existing:
        m = _SEQUENTIAL_ID_RE.match(item["id"])
        if m:
            try:
                n = int(m.group(1))
                if n > max_n:
                    max_n = n
            except ValueError:
                continue
    return f"{prefix}{max_n + 1:0{width}d}"


def validate_id(insight_id: str) -> bool:
    """检查一个 ID 字符串是否符合 ``I-{字母数字+短横线}`` 格式."""
    if not insight_id:
        return False
    return bool(_VALID_ID_RE.match(insight_id))


def find_insight_by_id(md_text: str, insight_id: str) -> Optional[Dict[str, str]]:
    """按 ID 查找单条 insight; 找不到返回 None."""
    for item in parse_insights(md_text):
        if item["id"] == insight_id:
            return item
    return None


def count_by_status(md_text: str) -> Dict[str, int]:
    """统计各 status 的 insight 数量. 返回 dict 总是含三个 status 的 key."""
    counts: Dict[str, int] = {s: 0 for s in VALID_STATUSES}
    for item in parse_insights(md_text):
        if item["status"] in counts:
            counts[item["status"]] += 1
    return counts
