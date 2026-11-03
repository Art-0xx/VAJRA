import matplotlib.pyplot as plt
import numpy as np

# The data we observed
labels = ['Baseline Model', 'Defended Model']
asr_before = [98.75, 16.31]
f1_before = [0.01, 0.68]

x = np.arange(len(labels))
width = 0.35

fig, ax1 = plt.subplots(figsize=(8, 6))

# ASR Bars
rects1 = ax1.bar(x - width/2, asr_before, width, label='Attack Success Rate (%)', color='crimson')
# Robust F1 Bars
rects2 = ax1.bar(x + width/2, f1_before, width, label='Robust F1 (Under Attack)', color='seagreen')

ax1.set_ylabel('Score')
ax1.set_title('Impact of PGD Adversarial Training (PGD eps=0.05)')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend(loc='upper center')
ax1.set_ylim(0, 110)

# Add text labels
for rect in rects1:
    height = rect.get_height()
    ax1.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
for rect in rects2:
    height = rect.get_height()
    ax1.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("artifacts/figures/defense_comparison.png", dpi=200)
print("Saved defense_comparison.png")