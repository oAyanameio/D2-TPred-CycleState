# ENGINEERING_ISSUES

本文件是工程层问题索引，占位补齐 `PLAN.md` / `EXPERIMENT_LOG.md` / `technical_documentation.md`
中已存在的交叉引用，避免悬空文档链接。

## 作用

- 汇总代码正确性、训练协议、checkpoint、日志与可复现性相关问题。
- 作为 [PLAN.md](./PLAN.md) 中 `EI` 来源缩写的落点文档。
- 与 [COMPREHENSIVE_ANALYSIS.md](./COMPREHENSIVE_ANALYSIS.md)、[METHOD_AND_ARCHITECTURE_ANALYSIS.md](./METHOD_AND_ARCHITECTURE_ANALYSIS.md)、[../EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md) 形成可跳转链路。

## 当前状态

- 历史问题主表当前仍以内嵌形式维护在 [PLAN.md §1 主问题交叉索引](./PLAN.md#1-主问题交叉索引)。
- 详细实验证据与修复时间线见 [../EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md)。
- 代码实现与模块职责见 [technical_documentation.md](./technical_documentation.md)。

## 后续迁移建议

后续如需把 `PLAN.md` 中的 EI 条目完全拆出，可按以下结构迁移：

1. 问题编号与严重度
2. 触发文件/行号
3. 复现方式
4. 修复状态
5. 对应实验/测试证据
