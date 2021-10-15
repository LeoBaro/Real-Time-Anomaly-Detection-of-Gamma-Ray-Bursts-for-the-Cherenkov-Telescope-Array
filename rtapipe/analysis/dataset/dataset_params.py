
def get_dataset_params(integration_time, integration_type):

    if integration_type not in ["t","te"]:
        raise ValueError("The integration value must be 't' or 'te'")

    dataset_params = {
        "1" : {
            "tobs" : 1800, "onset" : 900,
            "bkg" : f"/data/datasets/ap_data/t_1/ap_data_for_training_and_testing_NORMALIZED/simtype_bkg_os_0_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_0.15_roi_2.5/integration_{integration_type}_type_bkg_window_size_10_region_radius_0.2",
            "grb" : f"/data/datasets/ap_data/t_1/ap_data_for_training_and_testing_NORMALIZED/simtype_grb_os_900_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_0.15_roi_2.5/integration_{integration_type}_type_grb_window_size_10_region_radius_0.2"
        },
        "5" : {
            "tobs" : 360, "onset" : 180,
            "bkg" : f"/data/datasets/ap_data/t_5/ap_data_for_training_and_testing_NORMALIZED/simtype_bkg_os_0_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_0.15_roi_2.5/integration_{integration_type}_type_bkg_window_size_10_region_radius_0.2",
            "grb" : f"/data/datasets/ap_data/t_5/ap_data_for_training_and_testing_NORMALIZED/simtype_grb_os_900_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_0.15_roi_2.5/integration_{integration_type}_type_grb_window_size_10_region_radius_0.2"        
        },
        "10" : {
            "tobs" : 180, "onset" : 90,
            "bkg" : f"/data/datasets/ap_data/t_10/ap_data_for_training_and_testing_NORMALIZED/simtype_bkg_os_0_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_0.15_roi_2.5/integration_{integration_type}_type_bkg_window_size_10_region_radius_0.2",
            "grb" : f"/data/datasets/ap_data/t_10/ap_data_for_training_and_testing_NORMALIZED/simtype_grb_os_900_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_0.15_roi_2.5/integration_{integration_type}_type_grb_window_size_10_region_radius_0.2"
        }
    }

    dataset_params = dataset_params[integration_time]

    dataset_params["integration_time"] = integration_time
    dataset_params["integration_type"] = integration_type

    return dataset_params

