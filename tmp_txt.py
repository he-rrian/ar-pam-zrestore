import numpy as np, config as C
p = r"D:\tmp\data\4\500.txt"
a = np.loadtxt(p)
print("txt shape", a.shape, " (行=采样, 列=A-line)")
print("完整深度 %.0f um; 当前窗口 RF %d:%d = %.0f-%.0f um" % (
    a.shape[0]*C.PAM_RF_PITCH_UM, C.PAM_Z_START, C.PAM_Z_END,
    C.PAM_Z_START*C.PAM_RF_PITCH_UM, C.PAM_Z_END*C.PAM_RF_PITCH_UM))
env = np.abs(a).mean(axis=1)
print("窗口外(340:500) 占总能量: %.1f%%" % (100*env[340:].sum()/(env.sum()+1e-9)))
print("窗口外最大值/窗口内最大值: %.2f" % (env[340:].max()/(env[:340].max()+1e-9)))
