# -*- coding: utf-8 -*-
"""实验二入口：运行四个题目的官方测试用例。

    python main.py            # 运行全部官方用例
    python -m unittest discover -s tests -v   # 运行单元测试
    python experiments/run_experiments.py     # 生成实验数据与图表
"""

import sys

# 保证从任意目录调用时都能导入 src 包
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.run_all import main

if __name__ == "__main__":
    main()
