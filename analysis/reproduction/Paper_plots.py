####################################################
## PLOT SEEDED TOPIC-WORD POSTERIOR DISTRIBUTIONS ##
####################################################

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow_probability as tfp

tfd = tfp.distributions

pets = ["dog","cat", "litter", "cats", "dogs", "food", "box", "collar", "water", "pet"]
toys = ["toy", "game", "play", "fun", "old", "son", "year", "loves", "kids", "daughter"]
beauty = ["hair", "skin", "product", "color", "scent", "smell", "used", "dry", "using", "products"]
baby = ["baby", "seat", "diaper", "diapers", "stroller", "bottles", "son", "pump", "gate", "months"]
health = ["product", "like", "razor", "shave", "time", "day", "shaver", "better", "work", "years"]
grocery = ["tea", "taste", "flavor", "coffee", "sauce", "chocolate", "sugar", "eat", "sweet", "delicious"]


keywords = {"pets": pets, "toys": toys, "beauty": beauty, "baby": baby, "health": health, "grocery": grocery}


a_beta_tilde_S = pd.read_csv("./keyATM/paper_code/final_model_fits/VSTM/30k/30k_VSTM_a_beta_tilde_S.csv")["0"]
b_beta_tilde_S = pd.read_csv("./keyATM/paper_code/final_model_fits/VSTM/30k/30k_VSTM_b_beta_tilde_S.csv")["0"]

# ---- SINGLE PLOT
a_beta_tilde_S
b_beta_tilde_S


keywords["pets"]

pets_a = a_beta_tilde_S[0:len(keywords["pets"])]
pets_b = b_beta_tilde_S[0:len(keywords["pets"])]

pet_samples = tfd.Gamma(pets_a, pets_b).sample(1000)

df3 = pd.DataFrame(pet_samples)
df3.columns = keywords["pets"]

plt.figure(figsize = (10,7))
sns.kdeplot(data = df3, fill = True, palette = "bone", alpha = .3, linewidth = .8).set(title = "Seeded topic-word posterior distribution for keyword topic: pets")
sns.set_style("whitegrid")
sns.despine(left = True)
plt.show()

sns.kdeplot(data = df3, fill = True, palette = "RdGy", alpha = .3, linewidth = .8).set(title = "Seeded topic-word posterior distribution for keyword topic: pets")
sns.set_style("whitegrid")
sns.despine(left = True)
plt.show()


plt.figure(figsize = (12,7))
sns.kdeplot(data = df3, fill = True, palette = "binary_r", alpha = .3, linewidth = .8).set(title = "Seeded topic-word posterior distribution for keyword topic: pets")
sns.set_style("whitegrid")
sns.despine(left = True)
plt.show()


# ------- BIG PLOT:
# GET q(\tilde{\beta}) parameters
seed_posterior_params = dict()
i = 0
for topic, kw in keywords.items():
    # seed_posterior_params[topic] = [[a_beta_tilde_S.numpy()[i:i+len(kw)]],
    #                                 [b_beta_tilde_S.numpy()[i:i+len(kw)]]]
    seed_posterior_params[topic] = [a_beta_tilde_S[i:i + len(kw)],
                                    b_beta_tilde_S[i:i + len(kw)]]
    i += len(kw)

#

sp_pets = pd.DataFrame(tfd.Gamma(seed_posterior_params["pets"][0], seed_posterior_params["pets"][1]).sample(1000))
sp_pets.columns = keywords["pets"]

sns.kdeplot(sp_pets)
plt.show()

sp_toys = pd.DataFrame(tfd.Gamma(seed_posterior_params["toys"][0], seed_posterior_params["toys"][1]).sample(1000))
sp_toys.columns = keywords["toys"]

sp_beauty = pd.DataFrame(tfd.Gamma(seed_posterior_params["beauty"][0], seed_posterior_params["beauty"][1]).sample(1000))
sp_beauty.columns = keywords["beauty"]

sp_baby = pd.DataFrame(tfd.Gamma(seed_posterior_params["baby"][0], seed_posterior_params["baby"][1]).sample(1000))
sp_baby.columns = keywords["baby"]

sp_health = pd.DataFrame(tfd.Gamma(seed_posterior_params["health"][0], seed_posterior_params["health"][1]).sample(1000))
sp_health.columns = keywords["health"]

sp_grocery = pd.DataFrame(tfd.Gamma(seed_posterior_params["grocery"][0], seed_posterior_params["grocery"][1]).sample(1000))
sp_grocery.columns = keywords["grocery"]


# f = plt.figure(figsize = (8,8))
# gs = f.add_gridspec(2,2)
#
# with sns.axes_style("whitegrid"):
#     ax = f.add_subplot(gs[0,0])
#     sns.kdeplot(sp_pets, fill = True, palette = "crest", alpha = .3, linewidth = .8)
#     sns.despine(left=True)
# with sns.axes_style("whitegrid"):
#     ax = f.add_subplot(gs[0,1])
#     sns.kdeplot(sp_toys, fill = True, palette = "crest", alpha = .3, linewidth = .8)
#     sns.despine(left=True)
# with sns.axes_style("whitegrid"):
#     ax = f.add_subplot(gs[1,0])
#     sns.kdeplot(sp_beauty, fill = True, palette = "crest", alpha = .3, linewidth = .8)
#     sns.despine(left=True)
# with sns.axes_style("whitegrid"):
#     ax = f.add_subplot(gs[1,1])
#     sns.kdeplot(sp_baby, fill = True, palette = "crest", alpha = .3, linewidth = .8)
#     sns.despine(left=True)
# f.tight_layout()
# plt.show()




f = plt.figure(figsize = (10,10))
gs = f.add_gridspec(2,3)
with sns.axes_style("whitegrid"):
    ax = f.add_subplot(gs[0,0])
    sns.kdeplot(sp_pets, fill = True, palette = "bone", alpha = .3, linewidth = .8).set(title = r"$q(\tilde{\beta})_{pets}$")
    sns.despine(left=True)
with sns.axes_style("whitegrid"):
    ax = f.add_subplot(gs[0,1])
    sns.kdeplot(sp_toys, fill = True, palette = "bone", alpha = .3, linewidth = .8).set(title = r"$q(\tilde{\beta})_{toys}$")
    sns.despine(left=True)
with sns.axes_style("whitegrid"):
    ax = f.add_subplot(gs[0,2])
    sns.kdeplot(sp_beauty, fill = True, palette = "bone", alpha = .3, linewidth = .8).set(title = r"$q(\tilde{\beta})_{beauty}$")
    sns.despine(left=True)
with sns.axes_style("whitegrid"):
    ax = f.add_subplot(gs[1,0])
    sns.kdeplot(sp_baby, fill = True, palette = "bone", alpha = .3, linewidth = .8).set(title = r"$q(\tilde{\beta})_{baby}$")
    sns.despine(left=True)
with sns.axes_style("whitegrid"):
    ax = f.add_subplot(gs[1,1])
    sns.kdeplot(sp_health, fill = True, palette = "bone", alpha = .3, linewidth = .8).set(title = r"$q(\tilde{\beta})_{health}$")
    sns.despine(left=True)
with sns.axes_style("whitegrid"):
    ax = f.add_subplot(gs[1,2])
    sns.kdeplot(sp_grocery, fill = True, palette = "bone", alpha = .3, linewidth = .8).set(title = r"$q(\tilde{\beta})_{grocery}$")
    sns.despine(left=True)
f.tight_layout()
# f.suptitle("Seeded topic-word posterior distributions")
plt.show()


plt.figure(figsize = (10,7))
sns.kdeplot(sp_toys, fill=True, palette="bone", alpha=.3, linewidth=.8).set(title = "Seeded topic-word posterior distribution for keyword topic: toys")
sns.set_style("whitegrid")
sns.despine(left = True)
plt.show()





###############################
### PLOT THETA DISTRIBUTION ###
###############################

# # Heatmap thetas
# thetas_sample = document_topic_distribution.sample(500)
# thetas_sample_softmax = tf.nn.softmax(thetas_sample)
# sns.heatmap(thetas_sample_softmax, cmap="binary")
# # sns.heatmap(thetas_sample, vmin = 0.0, vmax = 1.0, cmap = "binary")
# plt.show()


##########################################
## 30k PLOT MODEL FITS - EARLY STOPPING ##
##########################################
import pandas as pd
import matplotlib.pyplot as plt

metrics_data = dict(
    sample_size =   [1,    5,    10,   15,   20,   25,   30],
    accuracy =      [0.68, 0.71, 0.73, 0.73, 0.74, 0.73, 0.73],
    precision =     [0.71, 0.73, 0.75, 0.76, 0.77, 0.76, 0.76],
    f1 =            [0.67, 0.70, 0.72, 0.72, 0.72, 0.72, 0.72],
    time =          [8,    10,   9,   16,    21,   27,   43]
)

metrics_df = pd.DataFrame(metrics_data, index = metrics_data["sample_size"])
fig, ax1 = plt.subplots(figsize = (15,7))
plt.title(f"VSTM metrics (early stopping activated)", fontsize = 15, weight = "bold", color = "0.2")
color = "tab:blue"
ax1.set_xlabel("Number of documents (in 1k)", fontsize = 13, color = "0.2")
ax1.set_ylabel("Model fit", fontsize = 15, color = "0.2")

markersizes = 15
line_weights = 2.5
lns1 = ax1.plot(metrics_df["accuracy"], color = "black", label = "Accuracy", marker = "^",
                markersize = markersizes, lw = line_weights, mec = "w", mew = "2", alpha = 0.9)
ax1.tick_params(axis = "y", labelcolor = "black")
lns2 = ax1.plot(metrics_df["precision"], color = "dimgray", label = "Precision", marker = "^",
                markersize = markersizes, lw = line_weights, mec = "w", mew = "2", alpha = 0.9)
lns3 = ax1.plot(metrics_df["f1"], color = "tan", label = "F1", marker = "^",
                markersize = markersizes, lw = line_weights, mec = "w", mew = "2", alpha = 0.9)
# ax1.xaxis.grid()
ax2 = ax1.twinx()
color = "tab:red"
ax2.set_ylabel("Computation time (in seconds)", color = "0.2", fontsize = 15)
lns4 = ax2.plot(metrics_df["time"], color = "slategrey", label = "Runtime", linestyle = "dashed",
                marker = "o", alpha = .7, markersize = markersizes, lw = line_weights, mec = "w", mew = "2")
ax2.tick_params(axis = "y", labelcolor = "black", color = "0.2")
fig.tight_layout()

# Add all line legends in one legend plot: https://stackoverflow.com/questions/5484922/secondary-axis-with-twinx-how-to-add-to-legend
lines = lns1+lns2+lns3+lns4
labs = [l.get_label() for l in lines]
ax1.legend(lines, labs, loc = 2, frameon = False, labelcolor = "0.2",
           prop = {"weight":"bold", "size":13})

plt.xticks(metrics_df.index, color = "0.2")

ax1.set_ylim(0.63,0.8)
ax2.set_ylim(5, 70)

# Macht die Axis dicker
for axis in ["bottom", "left"]:
    ax1.spines[axis].set_linewidth(2.5) # macht sie dicker
    ax1.spines[axis].set_color("0.2") # macht das schwarz weniger schwarz

# Removed die rechte und obere axis
# ax1.spines["top"].set_visible(False)
# ax1.spines["right"].set_visible(False)
# ax2.spines["top"].set_visible(False)
# ax2.spines["right"].set_visible(False)

# Macht die axis ticker dicker
ax1.tick_params(width = 2.5, labelsize = 13)
ax2.tick_params(labelsize = 13)

plt.show()






#######################################################
## 30k PLOT MODEL FITS - #### NO #### EARLY STOPPING ##
#######################################################
import pandas as pd
import matplotlib.pyplot as plt

metrics_data = dict(
    sample_size =   [1, 5, 10, 15, 20, 25, 30],
    accuracy =      [0.69, .71, .73, .74, .73, .74, .73],
    precision =     [0.65, .73, .75, .77, .76, .76, .76],
    f1 =            [0.65, .70, .72, .73, .73, .73, .72],
    time =          [6, 9, 19, 29, 41, 58, 67]
)

metrics_df = pd.DataFrame(metrics_data, index = metrics_data["sample_size"])
fig, ax1 = plt.subplots(figsize = (15,7))
plt.title(f"VSTM metrics (fixed epochs)", fontsize = 15, weight = "bold", color = "0.2")
color = "tab:blue"
ax1.set_xlabel("Number of documents (in 1k)", fontsize = 13, color = "0.2")
ax1.set_ylabel("Model fit", fontsize = 15, color = "0.2")

markersizes = 15
line_weights = 2.5
lns1 = ax1.plot(metrics_df["accuracy"], color = "black", label = "Accuracy", marker = "^",
                markersize = markersizes, lw = line_weights, mec = "w", mew = "2", alpha = 0.9)
ax1.tick_params(axis = "y", labelcolor = "black")
lns2 = ax1.plot(metrics_df["precision"], color = "dimgray", label = "Precision", marker = "^",
                markersize = markersizes, lw = line_weights, mec = "w", mew = "2", alpha = 0.9)
lns3 = ax1.plot(metrics_df["f1"], color = "tan", label = "F1", marker = "^",
                markersize = markersizes, lw = line_weights, mec = "w", mew = "2", alpha = 0.9)
# ax1.xaxis.grid()
ax2 = ax1.twinx()
color = "tab:red"
ax2.set_ylabel("Computation time (in seconds)", color = "0.2", fontsize = 15)
lns4 = ax2.plot(metrics_df["time"], color = "slategrey", label = "Runtime", linestyle = "dashed",
                marker = "o", alpha = .7, markersize = markersizes, lw = line_weights, mec = "w", mew = "2")
ax2.tick_params(axis = "y", labelcolor = "black", color = "0.2")
fig.tight_layout()

# Add all line legends in one legend plot: https://stackoverflow.com/questions/5484922/secondary-axis-with-twinx-how-to-add-to-legend
lines = lns1+lns2+lns3+lns4
labs = [l.get_label() for l in lines]
ax1.legend(lines, labs, loc = 2, frameon = False, labelcolor = "0.2",
           prop = {"weight":"bold", "size":13})

plt.xticks(metrics_df.index, color = "0.2")

ax1.set_ylim(0.63,0.8)
ax2.set_ylim(5, 70)

# Macht die Axis dicker
for axis in ["bottom", "left"]:
    ax1.spines[axis].set_linewidth(2.5) # macht sie dicker
    ax1.spines[axis].set_color("0.2") # macht das schwarz weniger schwarz

# Removed die rechte und obere axis
# ax1.spines["top"].set_visible(False)
# ax1.spines["right"].set_visible(False)
# ax2.spines["top"].set_visible(False)
# ax2.spines["right"].set_visible(False)

# Macht die axis ticker dicker
ax1.tick_params(width = 2.5, labelsize = 13)
ax2.tick_params(labelsize = 13)

plt.show()






#############################################
## 1m PLOT BOOTSTRAP FITS - EARLY STOPPING ##
#############################################

import pandas as pd
import matplotlib.pyplot as plt

metrics_data = dict(
    sample_size =   [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    accuracy =      [0.72, 0.70 , 0.7, 0.71, 0.71, 0.71, 0.71, 0.71, 0.7, 0.7],
    precision =     [0.75, 0.73, 0.74, 0.74, 0.74, 0.74, 0.74, 0.74, 0.74, 0.74],
    f1 =            [0.71, 0.69, 0.69, 0.69, 0.70, 0.69, 0.69, 0.69, 0.69, 0.69],
    time =          [177, 373, 631, 1321, 1277, 1721, 2078, 2590, 3188, 3713]
)

metrics_df = pd.DataFrame(metrics_data, index = metrics_data["sample_size"])
fig, ax1 = plt.subplots(figsize = (12,7))
plt.title(f"VSTM bootstrap metrics (early stopping activated)", fontsize = 15, weight = "bold", color = "0.2")
color = "tab:blue"
ax1.set_xlabel("Number of documents (in 1k)", fontsize = 13, color = "0.2")
ax1.set_ylabel("Model fit", fontsize = 15, color = "0.2")

markersizes = 15
line_weights = 2.5
lns1 = ax1.plot(metrics_df["accuracy"], color = "black", label = "Accuracy", marker = "^",
                markersize = markersizes, lw = line_weights, mec = "w", mew = "2", alpha = 0.9)
ax1.tick_params(axis = "y", labelcolor = "black")
lns2 = ax1.plot(metrics_df["precision"], color = "darkslateblue", label = "Precision", marker = "^",
                markersize = markersizes, lw = line_weights, mec = "w", mew = "2", alpha = 0.9)
lns3 = ax1.plot(metrics_df["f1"], color = "maroon", label = "F1", marker = "^",
                markersize = markersizes, lw = line_weights, mec = "w", mew = "2", alpha = 0.9)
# ax1.xaxis.grid()
ax2 = ax1.twinx()
color = "tab:red"
ax2.set_ylabel("Computation time (in seconds)", color = "0.2", fontsize = 15)
lns4 = ax2.plot(metrics_df["time"], color = "slategrey", label = "Runtime", linestyle = "dashed",
                marker = "o", alpha = .7, markersize = markersizes, lw = line_weights, mec = "w", mew = "2")
ax2.tick_params(axis = "y", labelcolor = "black", color = "0.2")
fig.tight_layout()

# Add all line legends in one legend plot: https://stackoverflow.com/questions/5484922/secondary-axis-with-twinx-how-to-add-to-legend
lines = lns1+lns2+lns3+lns4
labs = [l.get_label() for l in lines]
ax1.legend(lines, labs, loc = 2, frameon = False, labelcolor = "0.2",
           prop = {"weight":"bold", "size":13})

plt.xticks(metrics_df.index, color = "0.2")

ax1.set_ylim(0.65,0.8)
ax2.set_ylim(0, 7500)

# Macht die Axis dicker
for axis in ["bottom", "left"]:
    ax1.spines[axis].set_linewidth(2.5) # macht sie dicker
    ax1.spines[axis].set_color("0.2") # macht das schwarz weniger schwarz

# Removed die rechte und obere axis
# ax1.spines["top"].set_visible(False)
# ax1.spines["right"].set_visible(False)
# ax2.spines["top"].set_visible(False)
# ax2.spines["right"].set_visible(False)

# Macht die axis ticker dicker
ax1.tick_params(width = 2.5, labelsize = 13)
ax2.tick_params(labelsize = 13)

plt.show()




########################################################
## 1m PLOT BOOTSTRAP FITS - ### NO ### EARLY STOPPING ##
########################################################

import pandas as pd
import matplotlib.pyplot as plt

metrics_data = dict(
    sample_size =   [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    accuracy =      [0.72, .73, .72, .71, .71, .71, .71, .71, .7, 0.69],
    precision =     [.75, .77, .74, .74, .75, .75, .75, .75, .73, 0.73],
    f1 =            [.7, .72, .71, .69, .70, .7, .70, .7, .69, 0.68],
    time =          [311, 743, 1217, 1811, 2470, 3331, 4087, 4987, 6329, 7121]
)

metrics_df = pd.DataFrame(metrics_data, index = metrics_data["sample_size"])
fig, ax1 = plt.subplots(figsize = (12,7))
plt.title(f"VSTM bootstrap metrics (150 epochs)", fontsize = 15, weight = "bold", color = "0.2")
color = "tab:blue"
ax1.set_xlabel("Number of documents (in 1k)", fontsize = 13, color = "0.2")
ax1.set_ylabel("Model fit", fontsize = 15, color = "0.2")

markersizes = 15
line_weights = 2.5
lns1 = ax1.plot(metrics_df["accuracy"], color = "black", label = "Accuracy", marker = "^",
                markersize = markersizes, lw = line_weights, mec = "w", mew = "2", alpha = 0.9)
ax1.tick_params(axis = "y", labelcolor = "black")
lns2 = ax1.plot(metrics_df["precision"], color = "darkslateblue", label = "Precision", marker = "^",
                markersize = markersizes, lw = line_weights, mec = "w", mew = "2", alpha = 0.9)
lns3 = ax1.plot(metrics_df["f1"], color = "maroon", label = "F1", marker = "^",
                markersize = markersizes, lw = line_weights, mec = "w", mew = "2", alpha = 0.9)
# ax1.xaxis.grid()
ax2 = ax1.twinx()
color = "tab:red"
ax2.set_ylabel("Computation time (in seconds)", color = "0.2", fontsize = 15)
lns4 = ax2.plot(metrics_df["time"], color = "slategrey", label = "Runtime", linestyle = "dashed",
                marker = "o", alpha = .7, markersize = markersizes, lw = line_weights, mec = "w", mew = "2")
ax2.tick_params(axis = "y", labelcolor = "black", color = "0.2")
fig.tight_layout()

# Add all line legends in one legend plot: https://stackoverflow.com/questions/5484922/secondary-axis-with-twinx-how-to-add-to-legend
lines = lns1+lns2+lns3+lns4
labs = [l.get_label() for l in lines]
ax1.legend(lines, labs, loc = 2, frameon = False, labelcolor = "0.2",
           prop = {"weight":"bold", "size":13})

plt.xticks(metrics_df.index, color = "0.2")

ax1.set_ylim(0.65,0.8)
ax2.set_ylim(0, 7500)

# Macht die Axis dicker
for axis in ["bottom", "left"]:
    ax1.spines[axis].set_linewidth(2.5) # macht sie dicker
    ax1.spines[axis].set_color("0.2") # macht das schwarz weniger schwarz

# Removed die rechte und obere axis
# ax1.spines["top"].set_visible(False)
# ax1.spines["right"].set_visible(False)
# ax2.spines["top"].set_visible(False)
# ax2.spines["right"].set_visible(False)

# Macht die axis ticker dicker
ax1.tick_params(width = 2.5, labelsize = 13)
ax2.tick_params(labelsize = 13)

plt.show()




###############
## LOSS PLOT ##
###############

metrics = pd.read_csv("./keyATM/paper_code/final_model_fits/VSTM/30k/30k_VSTM_loss_metrics.csv")
import matplotlib.pyplot as plt


fig, ax1 = plt.subplots(figsize = (12,7))
plt.title(f"VSTM model fit on 30k Amazon customer feedbacks", fontsize = 15, weight = "bold", color = "0.2")
color = "tab:blue"
ax1.set_xlabel("Epoch", fontsize = 13, color = "0.2")
ax1.set_ylabel("Negative ELBO loss", fontsize = 15, color = "0.2")

markersizes = 15
line_weights = 2.5
lns1 = ax1.plot(metrics["neg_elbo"], color = "black", label = "Negative ELBO",
                markersize = markersizes, lw = line_weights, mec = "w", mew = "2", alpha = 0.9)
ax1.tick_params(axis = "y", labelcolor = "black")

ax2 = ax1.twinx()
color = "tab:red"
ax2.set_ylabel("Accuracy", color = "0.2", fontsize = 15)
lns4 = ax2.plot(metrics["accuracy"], color = "slategrey", label = "Accuracy", linestyle = "dashed", alpha = .7, markersize = markersizes, lw = line_weights, mec = "w", mew = "2")
ax2.tick_params(axis = "y", labelcolor = "black", color = "0.2")
fig.tight_layout()

# Add all line legends in one legend plot: https://stackoverflow.com/questions/5484922/secondary-axis-with-twinx-how-to-add-to-legend
lines = lns1+lns4
labs = [l.get_label() for l in lines]
ax1.legend(lines, labs, loc = 2, frameon = False, labelcolor = "0.2",
           prop = {"weight":"bold", "size":13})

# plt.xticks(metrics.index, color = "0.2")

ax1.set_ylim(0.63*1e7,1.35*1e7)
# ax2.set_ylim(5, 70)

# Macht die Axis dicker
for axis in ["bottom", "left"]:
    ax1.spines[axis].set_linewidth(2.5) # macht sie dicker
    ax1.spines[axis].set_color("0.2") # macht das schwarz weniger schwarz

# Removed die rechte und obere axis
# ax1.spines["top"].set_visible(False)
# ax1.spines["right"].set_visible(False)
# ax2.spines["top"].set_visible(False)
# ax2.spines["right"].set_visible(False)

# Macht die axis ticker dicker
ax1.tick_params(width = 2.5, labelsize = 13)
ax2.tick_params(labelsize = 13)

plt.show()



#######################
## RUNTIME BOOTSTRAP ##
#######################

import matplotlib.pyplot as plt

metrics_data_no_early_stopping = dict(
    sample_size =   [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    accuracy =      [0.72, .73, .72, .71, .71, .71, .71, .71, .7, 0.69],
    precision =     [.75, .77, .74, .74, .75, .75, .75, .75, .73, 0.73],
    f1 =            [.7, .72, .71, .69, .70, .7, .70, .7, .69, 0.68],
    time =          [311, 743, 1217, 1811, 2470, 3331, 4087, 4987, 6329, 7121],
    epochs=         [150, 150, 150, 150, 150, 150, 150, 150, 150, 150]
)


metrics_data_early_stopping = dict(
    sample_size =   [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    accuracy =      [0.72, 0.70 , 0.7, 0.71, 0.71, 0.71, 0.71, 0.71, 0.7, 0.7],
    precision =     [0.75, 0.73, 0.74, 0.74, 0.74, 0.74, 0.74, 0.74, 0.74, 0.74],
    f1 =            [0.71, 0.69, 0.69, 0.69, 0.70, 0.69, 0.69, 0.69, 0.69, 0.69],
    time =          [177, 373, 631, 1321, 1277, 1721, 2078, 2590, 3188, 3713],
    epochs=         [75, 75, 75, 90, 75, 75, 75, 75, 75, 75]
)



markersizes = 15
line_weights = 2.5
fig, ax = plt.subplots(figsize = (12,7))
plt.title("VSTM bootstrap experiment computation time", fontsize = 15, weight = "bold", color = "0.2")
plt.plot(metrics_data_no_early_stopping["sample_size"], metrics_data_no_early_stopping["time"], color = "black", label = "Fixed epochs", markersize = markersizes,
         lw = line_weights, mec = "w", mew = "2", alpha = 0.9, marker = "o")
plt.plot(metrics_data_early_stopping["sample_size"], metrics_data_early_stopping["time"], color = "gray", label = "Early stopping", markersize = markersizes,
         lw = line_weights, mec = "w", mew = "2", alpha = 0.9, marker = "o")

for idx, i in enumerate(metrics_data_no_early_stopping["epochs"]):
    plt.annotate(f"{i}",
                 xy = (metrics_data_no_early_stopping["sample_size"][idx]-10, metrics_data_no_early_stopping["time"][idx]+200),
                 alpha = 0.9, color = "black")
for idx, i in enumerate(metrics_data_early_stopping["epochs"]):
    plt.annotate(f"{i}",
                 xy = (metrics_data_early_stopping["sample_size"][idx]-10, metrics_data_early_stopping["time"][idx]+200),
                 alpha = 0.9, color = "gray")

ax.set_xticks(metrics_data_no_early_stopping["sample_size"])
plt.xlabel("Number of documents (in 1k)", fontsize = 13, color = "0.2")
plt.ylabel("Computation time (in seconds)", fontsize = 15, color = "0.2")
plt.legend(loc = 2, frameon = False, labelcolor = "0.2",
           prop = {"weight":"bold", "size":13})
ax.spines["bottom"].set_linewidth(2.5)
ax.spines["bottom"].set_color("0.2")
ax.spines["left"].set_linewidth(2.5)
ax.spines["left"].set_color("0.2")
ax.tick_params(width = 2.5, labelsize = 13)
ax.tick_params(labelsize = 13)
ax.tick_params(axis = "y", labelcolor = "black", color = "0.2")
plt.tight_layout()
plt.show()



