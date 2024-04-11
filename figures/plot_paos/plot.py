from upf_tools.projectors import Projectors
from upf_tools import UPFDict
import matplotlib.pyplot as plt

def plot_projs(projs, labels):
    fig, ax = plt.subplots(figsize=(2, 2), dpi=600)
    ax = projs.plot(ax)
    leg = ax.get_legend()
    for text, label in zip(leg.texts, labels):
        text.set_text(label)
    ax.set_xlabel('$r$ (Bohr)')
    ax.set_yticks([])#label('PAO')
    ax.set_xlim([0, 15])
    plt.tight_layout()
    return ax

old_upf = UPFDict.from_upf('Li.upf')
old_projs = Projectors.from_str(old_upf.to_dat())
new_projs = Projectors.from_file('Li_with_2p.dat')

old_projs[1].y *= -1

plot_projs(old_projs, labels=['1s', '2s'])
plt.savefig('old_projs.png')

plot_projs(new_projs, labels=['1s', '2s', '2p'])
plt.savefig('new_projs.png')

plt.close('all')
