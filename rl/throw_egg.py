'''扔鸡蛋问题
- N层楼，请你设计某种策略确定鸡蛋正好在n层楼会破，鸡蛋在每层楼破的概率都是1/2，其中最好的策略平均最少需要尝试多少次才能一定确定楼层n，假设可使用的鸡蛋为k个。
方法1:
系统的状态: N层楼用了k个鸡蛋尝试的次数为d[N][k]
状态转移方程(action-state),先试第i层，扔一个鸡蛋，消耗一次机会:
- 在i层楼鸡蛋没破，转化为d[N-i][k]
- 在i层楼鸡蛋破了，转化为d[i-1][k-1]
所以有递推公式:
d[N][k] = min_i(max(d[i-1][k-1], d[N-i][k])) + 1

方法2:
已知最大尝试次数，倒推能确定的楼层数
系统的状态: k个鸡蛋允许尝试次数m，最多可以尝试的楼层d[k][m]
状态转移方程(action-state), 假设我们从最优的某一层扔下第一个鸡蛋，必然只有两种结果：
- 鸡蛋碎了: 损失1个鸡蛋，消耗1次机会。下方能测的最大楼层数为d[k-1][m-1]。
- 鸡蛋没碎: 蛋完好，消耗 1次机会。上方能测的最大楼层数为d[k][m-1]。
d[k][m]= d[k-1][m-1] + d[k][m-1] + 1
'''

import builtins
import functools

def tag_print(func):
    """让被装饰函数内的所有 print 自动加上 [函数名] 前缀。"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        original_print = builtins.print

        @functools.wraps(original_print)
        def tagged_print(*pargs, **pkwargs):
            original_print(f"[{func.__name__}]", *pargs, **pkwargs)

        builtins.print = tagged_print
        try:
            return func(*args, **kwargs)
        finally:
            builtins.print = original_print  # 确保恢复，异常也不漏
    return wrapper

@tag_print
def solution1():
    N = 100
    K = 2
    d = [ [0] * (K+1) for _ in range(N+1)]
    rows = len(d)          # 结果为 2
    cols = len(d[0])       # 结果为 3
    print(f"Dimensions: {rows}x{cols}")
    for n in range(0, N+1):
        d[n][1] = n

    for n in range(1, N+1):
        for k in range(2, K+1):
            for i in range(1, n+1):
                d[n][k] = min(d[n][k], max(d[i-1][k-1], d[n-i][k]) + 1)
    print("1个鸡蛋最少需要尝试多少次:", d[N][1])
    print("2个鸡蛋最少需要尝试多少次:", d[N][2], "\n")

@tag_print
def solution2():
    K = 2
    m = 0
    max_try = 16 # 随机确定的数
    d = [[0] * (max_try) for _ in range(K+1)]  # 这里就不做优化了，0个鸡蛋的内存也放进去了，虽然可以直接从1个鸡蛋开始做，但是还要处理n-1比较麻烦
    while d[K][m] < 100:
        m += 1
        for k in range(1, K + 1):
            d[k][m] = d[k - 1][m - 1] + d[k][m - 1] + 1
        assert m < max_try, "尝试次数过多, 请调大max_try"
    print("1个鸡蛋最多可以尝试的楼层数:", d[1])
    print("2个鸡蛋最少需要尝试多少次:", m)


solution1()
solution2()