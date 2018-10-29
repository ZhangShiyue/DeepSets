t1_baseline = [0.0166609, 0.00568582, 0.000755591, 0.000471004, 0.000500957, 0.000515831,
               0.000271252, 0.000228882, 0.000413461, 0.000226764, 0.000297888]
t1_self_attn = [0.0233448, 0.0263114, 0.00331307, 0.00268856, 0.00117308, 0.00103898, 0.000361931,
                0.00030187, 0.000218742, 0.00021588, 0.000193684]
t1_self_attn_nor = [0.0121663, 0.0232391, 0.00760459, 0.00703421, 0.00175879, 0.000669206, 0.000355538,
                    0.000303406, 0.000240443, 0.000225487, 0.000207471]
t1_self_attn_con = [0.0101513, 0.0206714, 0.00600049, 0.00253864, 0.00117816, 0.000398236, 0.000473949,
                    0.000277785, 0.000228213, 0.000231039, 0.000207318]

t2_baseline = [0.00220099, 0.00186386, 0.000855192, 0.000682115, 0.000705779, 0.00035113, 0.000253712,
               8.43E-05, 6.31E-05, 8.11E-05, 5.39E-05]
t2_self_attn = [0.0735702, 0.0804073, 0.00430536, 0.357857, 0.00153898, 0.000811735, 0.00096345,
                0.000118752, 9.46E-05, 6.05E-05, 5.42E-05]
t2_self_attn_nor = [0.0246259, 0.00571341, 0.00797209, 0.00342796, 0.000824121, 0.000732635,
                    0.000693413, 0.000550545, 1.02E-04, 6.20E-05, 7.08E-05]
t2_self_attn_con = [0.0291177, 0.0385921, 0.0101796, 0.00211356, 0.0011087, 0.000718121, 0.000198684,
                    0.000130051, 6.55E-05, 5.89E-05, 5.90E-05]

t3_baseline = [0.000622578, 0.000579006, 0.000332704, 0.000273231, 0.000204162, 0.000194633, 0.000182411,
               0.000178298, 0.000183573, 0.000183874, 0.000173512]
t3_self_attn = [0.0112648, 0.0128641, 0.00165071, 0.00186716, 0.000444366, 0.000211285, 0.000184568,
                0.000204052, 0.000174239, 0.000182931, 0.000173389]
t3_self_attn_nor = [0.00873198, 0.0039597, 0.000231779, 0.00263472, 0.000370445, 0.00181685, 0.000224003,
                    0.000196575, 0.000179915, 0.000173555, 0.000173181]
t3_self_attn_con = [0.00561618, 0.00428682, 0.00168074, 0.00189217, 0.000280953, 0.00020507, 0.000182437,
                    0.000179636, 0.000187045, 0.000176872, 0.000180984]

t4_baseline = [0.134502, 0.0623199, 0.0472334, 0.0426353, 0.0379478, 0.0366223, 0.0342373, 0.0341288,
               0.0318494, 0.0305582, 0.0301914]
t4_self_attn = [0.163887, 0.134579, 0.146878, 0.0437529, 0.0401621, 0.0380939, 0.0341196, 0.0337739,
                0.0293641, 0.0249918, 0.0213156]
t4_self_attn_nor = [0.13466, 0.15528, 0.130975, 0.046175, 0.0380195, 0.0363166, 0.0342002, 0.0334218,
                    0.0270067, 0.0246149, 0.0206759]
t4_self_attn_con = [0.140108, 0.0495515, 0.0439105, 0.045983, 0.0369164, 0.0404056, 0.0335139,
                    0.0329566, 0.0265845, 0.0236268, 0.022794]

x = range(0, 11)
xlabels = ["2^7", "2^8", "2^9", "2^10", "2^11", "2^12", "2^13", "2^14", "2^15", "2^16", "2^17"]

from matplotlib import pyplot as plt

figure = plt.figure()
plt.plot(x, list(map(lambda x: x * 100, t1_baseline)), '.-', label="Baseline", markersize=10)
plt.plot(x, list(map(lambda x: x * 100, t1_self_attn)), '.-', label="Self-attn", markersize=10)
plt.plot(x, list(map(lambda x: x * 100, t1_self_attn_nor)), '.-', label="Self-attn w/o residual", markersize=10)
plt.plot(x, list(map(lambda x: x * 100, t1_self_attn_con)), '.-', label="Self-attn concat", markersize=10)
plt.xlabel("Number of set examples (N)")
plt.ylabel("Mean square error [1e-2]")
plt.xticks(x, xlabels)
plt.legend()
plt.title("Task1: Entropy estimation for rotated of 2d Gaussian.")
plt.savefig("figures/figure1.png")

figure1 = plt.figure()
plt.plot(x, list(map(lambda x: x * 100, t2_baseline)), '.-', label="Baseline", markersize=10)
plt.plot(x, list(map(lambda x: x * 100, t2_self_attn)), '.-', label="Self-attn", markersize=10)
plt.plot(x, list(map(lambda x: x * 100, t2_self_attn_nor)), '.-', label="Self-attn w/o residual", markersize=10)
plt.plot(x, list(map(lambda x: x * 100, t2_self_attn_con)), '.-', label="Self-attn concat", markersize=10)
plt.xlabel("Number of set examples (N)")
plt.ylabel("Mean square error [1e-2]")
plt.xticks(x, xlabels)
plt.legend()
plt.title("Task2: Mutual information estimation by varying correlation.")
plt.savefig("figures/figure2.png")

figure2 = plt.figure()
plt.plot(x, list(map(lambda x: x * 100, t3_baseline)), '.-', label="Baseline", markersize=10)
plt.plot(x, list(map(lambda x: x * 100, t3_self_attn)), '.-', label="Self-attn", markersize=10)
plt.plot(x, list(map(lambda x: x * 100, t3_self_attn_nor)), '.-', label="Self-attn w/o residual", markersize=10)
plt.plot(x, list(map(lambda x: x * 100, t3_self_attn_con)), '.-', label="Self-attn concat", markersize=10)
plt.xlabel("Number of set examples (N)")
plt.ylabel("Mean square error [1e-2]")
plt.xticks(x, xlabels)
plt.legend()
plt.title("Task3: Mutual information estimation by varying rank-1 strength.")
plt.savefig("figures/figure3.png")

figure3 = plt.figure()
plt.plot(x, list(map(lambda x: x * 100, t4_baseline)), '.-', label="Baseline", markersize=10)
plt.plot(x, list(map(lambda x: x * 100, t4_self_attn)), '.-', label="Self-attn", markersize=10)
plt.plot(x, list(map(lambda x: x * 100, t4_self_attn_nor)), '.-', label="Self-attn w/o residual", markersize=10)
plt.plot(x, list(map(lambda x: x * 100, t4_self_attn_con)), '.-', label="Self-attn concat", markersize=10)
plt.xlabel("Number of set examples (N)")
plt.ylabel("Mean square error [1e-2]")
plt.xticks(x, xlabels)
plt.legend()
plt.title("Task4: Mutual information on 32d random covariance matrices.")
plt.savefig("figures/figure4.png")