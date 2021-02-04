  #!/bin/bash
  
  # src only
  python pyscripts/phlists_to_photometry_plot.py -obs ~/phd/DATA/obs/2021-02-04_10-41-42_st_grb_tr_1_os_0 -md windowed -wsize 50 -wstep 5 -rad 1 -pl 1
  python pyscripts/phlists_to_photometry_plot.py -obs ~/phd/DATA/obs/2021-02-04_10-41-42_st_grb_tr_1_os_0 -md cumulative -wsize 50 -rad 1 -pl 1

  # bkg+src
  python pyscripts/phlists_to_photometry_plot.py -obs ~/phd/DATA/obs/2021-02-04_10-58-36_st_grb_tr_1_os_600/ -md windowed -wsize 50 -wstep 5 -rad 1 -pl 1
  python pyscripts/phlists_to_photometry_plot.py -obs ~/phd/DATA/obs/2021-02-04_10-58-36_st_grb_tr_1_os_600/ -md cumulative -wsize 50 -rad 1 -pl 1

  # bkg only
  python pyscripts/phlists_to_photometry_plot.py -obs ~/phd/DATA/obs/2021-02-04_10-41-38_st_bkg_tr_1_os_0/ -md windowed -wsize 50 -wstep 5 -rad 1 -pl 1
  python pyscripts/phlists_to_photometry_plot.py -obs ~/phd/DATA/obs/2021-02-04_10-41-38_st_bkg_tr_1_os_0/ -md cumulative -wsize 50 -rad 1 -pl 1
 