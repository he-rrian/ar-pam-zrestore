import numpy as np, config as C
from scipy.signal import hilbert
a = np.loadtxt(r"D:\tmp\data\4\500.txt")          # (1000 A-lines, 500 samples)
env = np.abs(hilbert(a, axis=1))                   # 沿采样轴取包络
print("txt shape", a.shape, "=> 1000 A-line x 500 采样 = 1500 um 深")
print("窗口 20:340 = %.0f-%.0f um；窗口外 340:500 = %.0f-%.0f um" % (
   20*3, 340*3, 340*3, 500*3))
tot = env.sum()
print("窗口外(340:500) 占总能量: %.1f%%" % (100*env[:,340:].sum()/tot))
print("逐列：窗口外/窗口内 能量比 中位 %.2f" % np.median(env[:,340:].sum(1)/(env[:,:340].sum(1)+1e-9)))
print("窗口外最大值 / 全局最大值: %.2f" % (env[:,340:].max()/env.max()))
