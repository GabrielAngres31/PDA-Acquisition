import matplotlib.pyplot as plt 
import pandas as pd

df = pd.read_csv("posterstats.csv")

print(df)

dpg_mapping = {1:"#fde725", 
            2:"#90d743", 
            3:"#35b779", 
            4:"#21918c", 
            5:"#31688e", 
            6:"#443983", 
            7:"#440154"}

bar_labels = df.dpg
bar_colors = [dpg_mapping[i] for i in bar_labels]
print(bar_colors)

plt.style.use("seaborn-v0_8")
fig = plt.figure(figsize = (8,6))
ax1 = fig.add_subplot(111)
ax1.bar(df.File, df.Stomata, color=bar_colors)

colors = {"1 dpg":"#fde725", 
            "2 dpg":"#90d743", 
            "3 dpg":"#35b779", 
            "4 dpg":"#21918c", 
            "5 dpg":"#31688e", 
            "6 dpg":"#443983", 
            "7 dpg":"#440154"}        
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels)

ax1.set_ylabel("# of Stomata")
#ax1.set_xticklabels(df.File, rotation = 45)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right" )
ax1.set_xlabel("Sample ID")
plt.title("Cotyledon stomatal count with dpg")
plt.tight_layout
plt.show()