"""
Authors:
Date:
"""

import matplotlib.pyplot as plt

logsfiles = ["/root/paddlejob/workspace/log/run.log"]
infos = []
for logfile in logsfiles:
    infos += open(logfile).readlines()

all_losses = []
all_meanLoss = []
epoch_id_old = -1
for linet in infos:
    if "Epoch" in linet and "learning_rate" in linet and "loss:" in linet:
        line1 = linet.strip().split("loss:")
        line2 = line1[1].split("eta:")
        loss_item = float(line2[0])
        if loss_item > 10:
            continue
        all_losses.append(loss_item)

        info1 = linet.split('Epoch:')[1]
        info2 = info1.split("] [")[0]
        info3 = info2.strip()
        epoch_id_new = int(float(info3[1:]))
        if epoch_id_new != epoch_id_old:
            all_meanLoss.append([loss_item])
        else:
            all_meanLoss[-1].append(loss_item)

        epoch_id_old = epoch_id_new

for id,item in enumerate(all_meanLoss):
    print ("epoch_id:",id, "\tloss_item:", round(sum(item)/len(item), 4), "\titer_nums:",len(item) )

x = list(range(len(all_losses)))
plt.plot(x, all_losses)
plt.xlabel("iter")
plt.ylabel("loss value")
plt.title("loss iter")
plt.savefig("/root/paddlejob/workspace/log/loss_face.jpg")
plt.show()

