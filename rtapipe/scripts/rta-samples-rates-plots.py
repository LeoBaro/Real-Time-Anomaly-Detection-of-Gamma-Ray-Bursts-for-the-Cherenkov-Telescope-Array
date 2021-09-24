import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 18, 'lines.markersize': 6,'legend.markerscale': 5, 'lines.linewidth':1, 'lines.linestyle':'-'})

def tbts(ws, t):
    _tbts = ws*t
    print(f"tbts={_tbts}")
    return _tbts

def tsr(t, nroff):
    _tsr = t / nroff
    print(f"tsr={_tsr}")    
    return _tsr

def tbb(ws, t, nroff, bs):
    return tbts(ws, t) + (bs-1) * tsr(t, nroff)

def bt(t, nroff, bs):
    return bs * tsr(t, nroff)

def br(t, nroff, bs):
    return 1  / (bs * tsr(t, nroff))

def scatterplot(x, y, slabel, xlabel, ylabel, title, xlim=None, ylim=None, filename=None, show=True):
    fig, ax = plt.subplots(1,1,figsize=(10,15))
    ax.scatter(x,y,label=slabel)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if filename is not None:
        plt.savefig(filename, dpi=600)
    if show:    
        plt.show()        
    return fig

if __name__=='__main__':
    
    t = 1
    ws = 5
    nroff = np.arange(1, 16, step=1)
    bs = 5

    # batch bootstrap time variando il numero di regioni OFF 
    # scatterplot(nroff, tbb(ws,t,nroff,bs), "tbb (seconds)", "number of OFF regions", "tbb", "Bootstrap time to obtain a first batch", filename="tbb.png", show=True)

    # batch rate variando il numero di regioni OFF
    scatterplot(nroff, bt(t,nroff,bs), "bach time (seconds)", "number of OFF regions", "bt", "Batch time", filename="bt.png", show=False)
    scatterplot(nroff, br(t,nroff,bs), "bach rate (batch/second)", "number of OFF regions", "br", "Batch rate", filename="br.png", show=False)
    plt.show()