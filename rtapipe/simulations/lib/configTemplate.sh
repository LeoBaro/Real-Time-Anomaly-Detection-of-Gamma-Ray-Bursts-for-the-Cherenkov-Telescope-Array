#!/bin/bash

if [[ -z "${1}" ]]; then
    printf "Error: no arguments have been passed to the function. Expected: directory to store the configuration file\n"
    echo ""
else
    configFilePath="$1/config.yaml"

    echo "
setup:
    simtype: XXX                      # grb -> src+bkg; bkg -> empty fields; skip -> skip sim;
                                    # wobble -> LST-like runs (str)
    runid: run0406_ID000126           # can be all or any template (list or str) in catalog
    trials: XXX                       # realisations per runid
    start_count: XXX                  # starting count for seed (int)
    scalefluxfactor: XXX                # scale src nominal flux by factor (float)

simulation:
    caldb: prod3b                     # calibration database
    irf: South_z40_average_LST_30m    # istrument response function
    tobs: XXX                         # total obs time (s)
    onset: XXX                        # time of bkg only a.k.a. delayed onset of burst (s)
    delay: 0                         # delayed start of observation (s) (float)
    emin: XXX                        # simulation minimum energy (TeV)
    emax: XXX                        # simulation maximum energy (TeV)
    roi: XXX                          # region of interest radius (deg)
    offset: 0                      # 'gw' -> from alert; value -> otherwise (deg) (str/float)
    nruns:                        # numer of runs (of lenght=tobs) for wobble simtype (int)


analysis:
    skypix:                       # pixel size in skymap (deg) (float)
    skyroifrac:                   # ratio between skymap axis and roi (float)
    smooth:                       # Gaussian corr. kernel rad. (deg) (float)
    maxsrc:                       # number of hotspot to search for (float)
    sgmthresh: 3                  # blind-search acc. thresh. in Gaussian sigma (float)
    usepnt: yes                   # use pointing for RA/DEC (bool)
    exposure:                     # exposure times for the analysis (s) (float)
    binned: no                    # perform binned or unbinned analysis (bool)
    blind: yes                    # requires blind-search (bool)
    tool: ctools                  # which science tool (str)
    type: 3d                      # 1d on/off or 3d full-fov (str)
    cumulative: no
    lightcurve: no
    index: -2.1

options:
    set_ebl: True                     # uses the EBL absorbed template
    extract_data: True                # if True extracts lightcurves and spectra
    plotsky: False                    # if True generates skymap plot (bool)

path:
    data: $DATA                       # all data should be under this folder
    ebl: $DATA/ebl_tables/gilmore_tau_fiducial.csv
    model: $DATA/models
    merger: $DATA/mergers                     # folder of alerts (str)
    bkg: $DATA/models/CTAIrfBackground.xml    # file of background model (str)
    catalog: $DATA/templates/grb_afterglow/GammaCatalogV1.0
    " > $configFilePath
    echo $configFilePath
fi
