import matplotlib.pyplot as plt
import pyregion
from astropy.io import fits
from matplotlib.colors import SymLogNorm
from matplotlib import rc
from astropy.wcs import WCS

# plot sky map ---!
def showSkymap(file, reg='none', col='black', suffix='none', title='skymap', xlabel='R.A. (deg)', ylabel='Dec (deg)', fontsize=12, show=True, tex=True):
    with fits.open(file) as hdul:
        wcs = WCS(hdul[0].header)
        data = hdul[0].data
        hdr = hdul[0].header
    
    plt.rc('text', usetex=False) if tex else None
    ax = plt.subplot(111)
    # load region ---!
    if reg != 'none' :
        r = pyregion.open(reg).as_imagecoord(hdr)
        for i in range(len(r)):
            r[i].attr[1]['color'] = col
            patch_list, text_list = r.get_mpl_patches_texts()
            for p in patch_list:
                ax.add_patch(p)
            for t in text_list:
                ax.add_artist(t)
    # plot with projection ---!
    plt.subplot(projection=wcs)
    plt.imshow(data, cmap='jet', norm=SymLogNorm(1), interpolation='gaussian')
    plt.grid(color='white', ls='solid')
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.colorbar().set_label('cts')
    # save fig ---!
    if suffix != 'none' :
        outputname = file.replace('.fits', '_%s.png' % suffix)
    else :
        outputname = file.replace('.fits', '.png')

    plt.savefig(outputname)

    # show fig ---!
    plt.show() if show else None
    plt.close()
    return