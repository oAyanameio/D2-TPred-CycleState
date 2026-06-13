"""D2TPred 项目根包: CycleState / 基线 / 数据加载 / 训练与评估脚本。

为了让 ``from D2TP.data.loader import seed_worker`` 这类绝对路径
import 能工作 (例如 ``tests/test_cyclestate_protocol.py`` 的
``test_seed_worker_seeds_correctly`` 验证), 需要把仓库根目录加到
``sys.path`` 并保证本文件存在, 让 ``D2TP`` 成为合法的 regular package。
"""
