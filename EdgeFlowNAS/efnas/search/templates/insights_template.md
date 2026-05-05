# Search Insights

<!--
Scientist agent (Phase 3) 拥有此分隔线以下的全部内容. Coordinator 只做版本化
备份 (insights.md.gen{N}), 不修改正文.

机器解析的硬约束 (整个文件仅有的三条):

  1. 每条 insight 用三级标题, 格式必须为
        ### I-{ID} ({status}): {一行标题}

  2. {status} 必须严格是以下三选一 (大小写敏感):
        active           - 当前生效的洞察
        retired          - 已退役 (例如发现反例后)
        under_review     - 暂时挂起重新审视

  3. {ID} 必须以 'I-' 开头, 后续只允许字母数字和短横线;
     一旦分配后保持稳定, 修订时不要改 ID.
     建议 sequential 数字编号 (I-001, I-002, ...), 也允许语义化前缀
     (例如 I-EB0-DB1 表示某种维度组合的洞察).

正文 (heading 之后到下一条 heading 之前) **完全自由形式**:
  - 没有必填字段、没有字段顺序要求
  - 可以贴 Python 验证片段、画 ASCII 表、写自然语言
  - 可以中英文混写
  - 没东西可说就直接简短陈述, 不要硬凑

退役一条 insight 只需要把 (active) 改成 (retired) 即可,
**不要求**移到独立的 retired 区段; status 标签自带过滤能力.

Hardware grounding 不是必填字段. 如果 Scientist 通过 Vela 数据 corroborate
或 contradict 了某条洞察, 直接在正文里自然语言陈述; 没硬件证据时不写也没关系.
-->

---

<!--
Scientist 在此追加 / 修订 insights.

下一条新 insight 的建议 ID 由 coordinator 调用 efnas.search.insights.next_insight_id()
计算后传给 Scientist prompt; Scientist 不需要自己跟踪编号空间.
-->
