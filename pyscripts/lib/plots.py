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


def showLightCurve(file, figsize=(15,15), axisLim ='auto', title='lightcurve', yscale=('lin','log'), xscale=('lin','log'),
                   show = True, tex=True):

  fig = plt.figure(figsize=figsize)
  plt.rc('text', usetex=True) if tex else None
  ax1 = plt.subplot(211, yscale=yscale[0], xscale=xscale[0])
  ax2 = plt.subplot(212, yscale=yscale[1], xscale=xscale[1])

  for i in range(len(file)):
    with fits.open(file) as hdul:
     data = hdul[1].data
     t_mjd = data.field(0)  # days
     et_mjd = data.field(1)  # days
     prefact = data.field(6)  # ph/cm^2/s/MeV
     e_prefact = data.field(7)  # ph/cm^2/s/MeV
     index = data.field(8)
     e_index = data.field(9)
     TS = data.field(10)
     diff_uplim = data.field(11) # ph/cm^2/s/MeV
     flux_uplim = data.field(12) # ph/cm^2/s
     Eflux_uplim = data.field(13) # erg/cm^2/s

    pnts = []
    e_pnts = []
    t_pnts = []
    et_pnts = []
    ul_pnts = []
    eul_pnts = []
    tul_pnts = []
    etul_pnts = []
    # list flux point or upper limit ---!
    for el in range(len(data)):
      if TS[el] > 9 and 2.0*e_prefact[el] < prefact[el] :
        pnts.append(prefact[el])
        e_pnts.append(e_prefact[el])
        t_pnts.append(t_mjd[el])
        et_pnts.append(et_mjd[el])
      else :
        ul_pnts.append(diff_uplim[el])
        eul_pnts.append(0.5*diff_uplim[el])
        tul_pnts.append(t_mjd[el])
        etul_pnts.append(et_mjd[el])

    # linear ---!
    ax1.errorbar(t_pnts, pnts, xerr=et_pnts, yerr=e_pnts, fmt='o', mec='k', label='data')
    ax1.errorbar(tul_pnts, ul_pnts, xerr=[etul_pnts, etul_pnts], yerr=eul_pnts, uplims=True, fmt='bo', mec='k')
    ax1.axis(axisLim) if axisLim != 'auto' else None
    ax1.grid()
    ax1.set_xlabel('t (MJD)')
    ax1.set_ylabel('dN/dE (ph/$cm^2$/s/MeV)')
    ax1.set_title('lightcurve') if title == 'none' else plt.title(title)
    # log ---!
    ax2.errorbar(t_pnts, pnts, xerr=et_pnts, yerr=e_pnts, fmt='o', mec='k', label='data')
    ax2.errorbar(tul_pnts, ul_pnts, xerr=[etul_pnts, etul_pnts], yerr=eul_pnts, uplims=True, fmt='bo', mec='k')
    ax2.axis(axisLim) if axisLim != 'auto' else None
    ax2.grid()
    ax2.set_xlabel('t (MJD)')
    ax2.set_ylabel('dN/dE (ph/$cm^2$/s/MeV)')
    ax2.set_title('lightcurve') if title == 'none' else plt.title(title)
  ax1.legend()
  ax2.legend()
  # adjust ---!
  plt.subplots_adjust(hspace=0.5)
  # save fig ---!
  extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
  fig.savefig(file.replace('.fits', '.png'), bbox_inches=extent.expanded(1.3, 1.3))
  extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
  fig.savefig(file.replace('.fits', '_log.png'), bbox_inches=extent.expanded(1.3, 1.3))
  # show fig ---!
  plt.show() if show else None
  plt.close()
  return    