#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


onehot = pd.read_csv("../onehot.csv",index_col=[0])


# In[11]:


train_val = pd.read_csv("train_val.csv", index_col=[0])
train_pat, val_pat = train_test_split(train_val.sample(frac=1, random_state=42).drop_duplicates(subset="Patient ID", keep="first"), test_size=0.1, random_state=42)


# In[12]:


train = train_val.merge(train_pat[["Patient ID"]])


# In[13]:


val = train_val.merge(val_pat[["Patient ID"]])


# In[14]:


train.to_csv("train.csv")
val.to_csv("val.csv")


# In[15]:


targets = train.columns[-14:].tolist()


# In[16]:


train_pt = train.sample(frac=1, random_state=42).drop_duplicates(subset="Patient ID", keep="first").reset_index(drop=True) # ensure no patient overlapping


# In[17]:


import random
y_np = train_pt[targets].to_numpy()
X_np = train_pt["Patient ID"].to_numpy()


# In[18]:


import numpy as np
from sklearn.model_selection import train_test_split

def partition_data(X, y, num_clients, alpha=0.5):
    """
    Partition data into non-IID subsets using Dirichlet distribution.
    Args:
        X (numpy array): Feature data.
        y (numpy array): Multi-label targets.
        num_clients (int): Number of clients.
        alpha (float): Dirichlet distribution parameter; smaller means more skew.
    Returns:
        List of tuples [(X_client1, y_client1), (X_client2, y_client2), ...]
    """
    # Calculate label distributions for each class
    label_distribution = np.random.dirichlet([alpha] * num_clients, size=y.shape[1])
    # return label_distribution

    # Assign samples to clients
    client_data = {i: ([], []) for i in range(num_clients)}
    for i, label_vector in enumerate(y):
        # Assign the sample to one of the clients based on label distribution
        # return label_vector
        client_idx = np.argmax(np.dot(label_vector, label_distribution))
        client_data[client_idx][0].append(X[i])
        client_data[client_idx][1].append(label_vector)

    # Convert lists to numpy arrays
    client_data = [(np.array(X_list), np.array(y_list)) for X_list, y_list in client_data.values()]

    return client_data

# # Example usage
# X = np.random.rand(1000, 20)  # Example feature data
# y = np.random.randint(0, 2, (1000, 5))  # Example multi-label targets
# clients = partition_data(X, y, num_clients=5, alpha=0.5)


# In[47]:


minn = 1e6
best = 0
while True:
    state = random.randint(0, 2**32-1)
    np.random.seed(state)
    partd = partition_data(X_np, y_np, 6, alpha=0.3)
    m = min([len(partd[i][0]) for i in range(6)])
    # if std < minn:
    #     minn = std
    #     best = i
    if m > 1000:
        print("best m", m)
        print("random_state", state)
        break
    


# In[48]:


np.random.seed(state)
partd = partition_data(X_np, y_np, 6, alpha=0.3)
print([len(partd[i][0]) for i in range(6)])


# In[49]:


pd.options.display.float_format = '{:.2f}'.format


# In[50]:


km_df = pd.DataFrame({"Patient ID":X_np, "cluster":np.zeros(X_np.shape)})
km_df = km_df.set_index("Patient ID")
for i in range(6):
    km_df.loc[partd[i][0],"cluster"] = i


# In[51]:


km_df = km_df.map(lambda x: int(x))


# In[52]:


km_df = km_df.reset_index()


# In[53]:


clustered = train_pt.merge(km_df)


# In[54]:


for s in range(6):
    fig, ax = plt.subplots()
    ax = clustered[clustered["cluster"]==s][targets].sum().plot.barh()
    ax.set_title(f"Data Distribution of Site {s+1}")


# In[55]:


fig, ax = plt.subplots(figsize=(5,5))
ax = (clustered["cluster"]+1).value_counts().sort_index().plot.bar()
fig.tight_layout()


# In[56]:


# nf_pt = nf_pt.reset_index(drop=True)
# for s in range(6):
#     avg = len(nf_pt)//6
#     if s == 5:
#         nf_pt.loc[(s+1)*avg:, "cluster"] = s+1
#     else:
#         nf_pt.loc[s*avg:(s+1)*avg, "cluster"] = s+1


# In[57]:


# clustered = pd.concat([clustered, nf_pt])


# In[58]:


train_splits = {}
val_splits = {}
sites = [i+1 for i in range(6)]
for s in sites:
    train_splits[s] = train.merge(clustered[clustered["cluster"]==s-1][["Patient ID"]], on="Patient ID").sample(frac=0.8, random_state=42)
    val_splits[s] = train.merge(clustered[clustered["cluster"]==s-1][["Patient ID"]], on="Patient ID").sample(frac=0.2, random_state=42)


# In[59]:


for s in sites:
    train_splits[s].to_csv(f"train_{s}.csv", index=False)
    val_splits[s].to_csv(f"val_{s}.csv", index=False)


# In[60]:


new_ds = pd.concat([pd.read_csv("test.csv"), pd.read_csv("val.csv")]\
                  + [pd.read_csv(f"train_{i}.csv") for i in range(1,7)]+ [pd.read_csv(f"val_{i}.csv") for i in range(1,7)])


# In[61]:


colors = [(0.235, 0.702, 0.443), (0.576, 0.439, 0.859), (0.255, 0.412, 0.882), (0.0, 0.502, 0.502),  (0.85, 0.85, 0.95), (1.0, 0.855, 0.725),  ]


# In[62]:


fig, ax = plt.subplots()
pd.DataFrame({"original (NIH)":onehot[onehot.columns[-15:]].sum(), "ours":new_ds[new_ds.columns[-15:]].sum()}).plot.bar(ax=ax, zorder=3, color=colors)
ax.grid(zorder=0)
ax.text(4.5, 50000, f"original: {len(onehot):,} images, ours: {len(new_ds):,} images", backgroundcolor='floralwhite')


# In[63]:


dist = pd.DataFrame({"original (NIH)":onehot[onehot.columns[-15:]].sum(), "ours":new_ds[new_ds.columns[-15:]].sum()})


# In[64]:


dist.loc[:, "NIH_percent"] = (dist["original (NIH)"] / sum(dist["original (NIH)"])).apply(lambda x: round(x,4))
dist.loc[:, "ours_percent"] = (dist["ours"] / sum(dist["ours"])).apply(lambda x: round(x,4))


# In[65]:


pd.options.display.float_format = '{:.2%}'.format
dist[["NIH_percent", "ours_percent"]]


# In[66]:


test = pd.read_csv("test.csv")
val = pd.read_csv("val.csv")
trains_vals = pd.concat([pd.read_csv(f"train_{i}.csv") for i in range(1,7)]+ [pd.read_csv(f"val_{i}.csv") for i in range(1,7)])


# In[67]:


fig, ax = plt.subplots(figsize=(12,5))
pd.DataFrame({"trains":trains_vals[trains_vals.columns[-15:]].sum(), "val":val[val.columns[-15:]].sum(),  "test":test[test.columns[-15:]].sum()}).plot.bar(ax=ax, zorder=3, color=colors)
ax.grid(zorder=0)
ax.text(5.5, 7500, f"train {len(trains_vals):,} / validate {len(val):,} / test {len(test):,}", backgroundcolor='floralwhite')


# In[68]:


sites = {}
sums = {}
for i in range(1,7):
    sites[i] = pd.concat([pd.read_csv(f"train_{i}.csv"), pd.read_csv(f"val_{i}.csv")])
    sums[f"site {i}"] = sites[i][sites[i].columns[-15:]].sum()


# In[69]:


colors = [(0.235, 0.702, 0.443), (0.576, 0.439, 0.859), (0.255, 0.412, 0.882), (0.0, 0.502, 0.502),  (0.65, 0.65, 0.75), (1.0, 0.7, 0.5),  ]


# In[70]:


fig, ax = plt.subplots(figsize=(12,5))
pd.DataFrame(sums).plot.bar(ax=ax, zorder=3, color=colors)
ax.grid(zorder=0)
ax.text(7, 2000, "\n".join([f"site_{i}: {len(sites[i]):,} images" for i in range(1,7)]), backgroundcolor='floralwhite')


# ### Full/Half/Scarce

# In[71]:


val.to_csv("full/val.csv", index=False)
val.to_csv("half/val.csv", index=False)
val.to_csv("scarce/val.csv", index=False)
for i in range(1,7):
    train = pd.read_csv(f"train_{i}.csv")
    val = pd.read_csv(f"val_{i}.csv")
    train.to_csv(f"full/train_{i}.csv")
    val.to_csv(f"full/val_{i}.csv")
    train.sample(frac=0.5, random_state=42).to_csv(f"half/train_{i}.csv")
    val.sample(frac=0.5, random_state=42).to_csv(f"half/val_{i}.csv")
    train.sample(frac=0.1, random_state=42).to_csv(f"scarce/train_{i}.csv")
    val.sample(frac=0.1, random_state=42).to_csv(f"scarce/val_{i}.csv")

