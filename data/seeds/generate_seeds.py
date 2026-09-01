from __future__ import annotations

KIND_VALUES = {"dependency", "resource", "deploy_regression", "config"}

LEAK_MECHANISMS: list[dict[str, str]] = [
    {
        "slug": "unbounded_list_append",
        "name": "无界 list append",
        "clue": (
            "怀疑是最近一次发版引入的某个 commit 里，对一个全局/请求级的 list 做了 "
            "append 但从未清理或做长度上限控制，随着请求量增加持续增长，最终把工作集内存吃满"
        ),
    },
    {
        "slug": "unbounded_dict_accumulation",
        "name": "无界 dict 累加",
        "clue": (
            "怀疑是最近一次发版引入的某个 commit 里，用一个全局 dict 按请求 key 缓存/累加"
            "计算结果，却没有配 TTL 或淘汰策略，随着不同 key 不断出现，dict 体积单调增长"
        ),
    },
    {
        "slug": "unclosed_file_handle",
        "name": "未关闭的文件句柄",
        "clue": (
            "怀疑是最近一次发版引入的某个 commit 里打开了日志/临时文件后没有用 with 语句"
            "正确关闭（或者异常路径遗漏了 close），文件句柄和内部缓冲区随请求量持续堆积"
        ),
    },
    {
        "slug": "misused_weakset",
        "name": "WeakSet 用错",
        "clue": (
            "怀疑是最近一次发版引入的某个 commit 里错误使用了 WeakSet/WeakValueDictionary"
            "（例如外部仍持有强引用，或者存进去的对象类型本身不支持被弱引用而退化成强引用），"
            "本该被回收的对象实际从未被回收"
        ),
    },
]
