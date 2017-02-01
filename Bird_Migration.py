import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

bird_data = pd.read_csv("./bird_tracking.csv")
bird_names = pd.unique(bird_data.bird_name)

plt.figure(figsize=(7,7))
for name in bird_names:
    ix = bird_data.bird_name == name
    x, y = bird_data.longitude[ix], bird_data.latitude[ix]
    plt.plot(x, y, ".", label=name)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(loc="lower right")


ix = bird_data.bird_name == "Eric"
speed = bird_data.speed_2d[ix]
ind = np.isnan(speed)
plt.hist(speed[~ind], bins=np.linspace(0,30,20), normed=True)
plt.xlabel("2D speed m/s")
plt.ylabel("Frequency")


