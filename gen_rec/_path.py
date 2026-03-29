import sys, os
# 把项目根目录和 gen_rec 目录都加入 path
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_here = os.path.dirname(os.path.abspath(__file__))
for _p in [_root, _here]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
