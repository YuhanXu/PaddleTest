#!/bin/env python3
# 单执行器运行封装：python single_engine_runner.py <case相对路径> <title> <yaml相对路径>
# yaml 的 testings 只含一个执行器，故 _perf_case_run 只跑该执行器
import sys

from layertest import LayerTest

if __name__ == "__main__":
    layerfile = sys.argv[1]
    title = sys.argv[2]
    testing = sys.argv[3]
    st = LayerTest(title=title, layerfile=layerfile, testing=testing)
    st._perf_case_run()
