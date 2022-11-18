import os 
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler

from rtapipe.lib.plotting.APPlot import APPlot
from rtapipe.lib.rtapipeutils.FileSystemUtils import parse_params
from rtapipe.lib.datasource.Photometry3 import SimulationParams, OnlinePhotometry
from rtapipe.lib.rtapipeutils.SequenceUtils import extract_sub_windows, extract_sub_windows_pivot


class DataManager:
    """
    Use two saparate DataManager objects to handle train and test data!!!
    If you want to scale test data, use the same scaler that was used for train data.
    """
    REGION_RADIUS = 0.2
    MAX_OFFSET = 2

    def __init__(self, output_dir):
        """
        data: will have the following shape (TEMPLATE, N_SAMPLES, N_POINTS, N_FEATURES)
        """
        self.output_dir = Path(output_dir)
        self.fits_files = []
        self.data = {}
        self.multiple_templates = None
        self.scaler = None

    @staticmethod
    def extract_runid_from_name(name):
        return name.split("runid_")[1].split("_trial")[0]

    @staticmethod
    def split_array_with_percentage(arr, percentage):
        split_point = int(len(arr)*(percentage/100))
        return np.split(arr, [split_point])

    @staticmethod
    def scale(scaler, data):
        if scaler is None:
            raise ValueError("Scaler is None")
        return scaler.transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)

    @staticmethod
    def load_fits_data(dataset_folder: str, limit = None):
        fits_files = [os.path.join(dataset_folder, file) for file in os.listdir(dataset_folder) if file.endswith(".fits")]
        if limit:
            fits_files = fits_files[:limit]
        print("Loaded {} files".format(len(fits_files)))
        return fits_files    

    @staticmethod
    def get_fits_from_template(fits_files, template):
        return [file for file in fits_files if template in file].pop()

    def load_saved_data(self, integration_time, tsl):
        cache_dir = self.output_dir.joinpath("data_cache")
        for file in os.listdir(cache_dir):
            if file.endswith(".npy"):
                if "it_"+str(integration_time) in file and "tsl_"+str(tsl) in file:
                    print(f"Loading cached data from {file}")
                    template = file.split("_it_")[0]
                    self.data[template] = np.load(cache_dir.joinpath(file))
        print(f"[{datetime.now()}] Loaded data from {cache_dir}")

    def store_scaler(self, dest_path):
        if self.scaler is None:
            raise ValueError("Scaler is None. Call fit_scaler() first.")
        Path(dest_path).mkdir(parents=True, exist_ok=True)
        with open(Path(dest_path).joinpath('fitted_scaler.pickle'), 'wb') as handle:
            pickle.dump(self.scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_scaler(self, scaler_path):
        with open(scaler_path, 'rb') as handle:
            self.scaler = pickle.load(handle)

    def transform_to_timeseries(self, fits_files: list, sim_params: SimulationParams, add_target_region: bool, integration_time: int, tsl: int, number_of_energy_bins: int, normalize: bool, threads: int, multiple_templates: bool):
        """
        For each photon list, it creates a time series of the integrated light curve.
        The data is then saved in a numpy array.
        The phometry is performed using the region that correspond to the position of the 
        source that is moved by an offset from the pointing.
        If the photons list contains no source, the region still correspond to the position
        of the source (took from the template).
        multiple_templates: if True the dataset is composed by multiple templates, otherwise it is composed by a single template. 
                            The single template case will allow photometry optimizations.
        """
        normalize = True
        with_metadata = False

        if add_target_region:
            integrate_from_regions = "src"
        else:
            integrate_from_regions = "bkg"

        if not multiple_templates:
            online_photometry = OnlinePhotometry(sim_params, integration_time=integration_time, tsl=tsl, number_of_energy_bins=number_of_energy_bins)
            runid = DataManager.extract_runid_from_name(fits_files[0])
            print(f"[{datetime.now()}] Preconfiguring regions. Normalization: {normalize} - Template: {runid}")
            online_photometry.set_template(runid)
            online_photometry.preconfigure_regions(regions_radius=DataManager.REGION_RADIUS, max_offset=DataManager.MAX_OFFSET, example_fits=fits_files[0], add_target_region=add_target_region, remove_overlapping_regions_with_target=False, compute_effective_area_for_normalization=normalize)
            print(f"[{datetime.now()}] Found {online_photometry.get_number_of_regions('bkg')} regions and {online_photometry.get_number_of_regions('src')} target regions")


        for photon_list in tqdm(fits_files):
            online_photometry = OnlinePhotometry(sim_params, integration_time=integration_time, tsl=tsl, number_of_energy_bins=number_of_energy_bins)
            runid = DataManager.extract_runid_from_name(photon_list)
            online_photometry.set_template(runid)

            cd, cd_err, fd, fd_err, metadata = online_photometry.integrate(
                                                                    photon_list, 
                                                                    normalize, 
                                                                    threads, 
                                                                    with_metadata, 
                                                                    regions_radius=DataManager.REGION_RADIUS, 
                                                                    max_offset=DataManager.MAX_OFFSET, 
                                                                    example_fits=photon_list, 
                                                                    add_target_region=add_target_region, 
                                                                    remove_overlapping_regions_with_target=False, 
                                                                    integrate_from_regions=integrate_from_regions
                                                        )

            if runid not in self.data:
                self.data[runid] = fd
            else:
                self.data[runid] = np.append(self.data[runid], fd, axis=0)

        # save data as binary array
        cache_dir = self.output_dir.joinpath("data_cache")
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True)
        for runid, data in self.data.items():
            np.save(cache_dir.joinpath(f"{runid}_it_{integration_time}_tsl_{tsl}.npy"), data)
        print(f"[{datetime.now()}] Saved data to {cache_dir}")





    def get_train_set(self, template, sub_window_size, stride, validation_split=70, make_plots=False):
        """
        This method:
            - extract subsequences from the timeseries
            - splits those in train and validation set
            - fit a scaler on the train set
            - normalize the train and validation set
        """
        if len(self.data) == 0:
            raise ValueError("No data loaded") 

        if template not in self.data:
            raise ValueError(f"Template {template} not found")       

        print(f"[{datetime.now()}] Extracting subsequences of {sub_window_size} points with stride {stride} from {len(self.data[template])} time series")

        sequences = None
        for ts in self.data[template]:
            s = extract_sub_windows(ts, start=0, stop=len(ts), sub_window_size=sub_window_size, stride_size=stride)
            if sequences is None:
                sequences = s
            else:
                sequences = np.append(sequences, s, axis=0)
        
        print(f"[{datetime.now()}] Extracted {len(sequences)} subsequences")

        labels = np.array([False for i in range(len(sequences))])

        train_x, val_x = DataManager.split_array_with_percentage(sequences, validation_split)
        train_y, val_y = DataManager.split_array_with_percentage(labels, validation_split)

        print(f"[{datetime.now()}] Train set shape: {train_x.shape} - Validation set shape: {val_x.shape}")

        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.scaler.fit(train_x.reshape(-1, train_x.shape[-1]))

        print(f"[{datetime.now()}] Data is scaled to 0-1")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir.joinpath('fitted_scaler.pickle'), 'wb') as handle:
            pickle.dump(self.scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return DataManager.scale(self.scaler, train_x), train_y, DataManager.scale(self.scaler, val_x), val_y

    def get_test_set(self, template, onset, integration_time, sub_window_size, stride, verbose=False):
        """
        This method:
        """
        if template not in self.data:
            raise ValueError(f"Template {template} not found. Available templates: {list(self.data.keys())}")

        if self.scaler is None:
            raise ValueError("Scaler not found")

        pivot_idx = onset//integration_time + 1 
        if verbose:
            print("Pivot index: ", pivot_idx)

        test_x_tot = None
        labels_tot = None
        for ts in self.data[template]:
            windows_before_pivot, windows_after_pivot = extract_sub_windows_pivot(ts, sub_window_size=sub_window_size, stride_size=stride, pivot_idx=pivot_idx)
            windows_before_pivot = DataManager.scale(self.scaler, windows_before_pivot)
            windows_after_pivot = DataManager.scale(self.scaler, windows_after_pivot)
            if verbose:
                print("windows_before_pivot: ", windows_before_pivot.shape)
                print("windows_after_pivot: ", windows_after_pivot.shape)
            test_x = np.concatenate((windows_before_pivot, windows_after_pivot), axis=0)
            #print("test_x: ", test_x.shape)
            labels = np.array([False for i in range(len(windows_before_pivot))]+[True for i in range(len(windows_after_pivot))])
            #print("labels: ", labels.shape)
            if test_x_tot is None:
                test_x_tot = test_x
                labels_tot = labels
            else:
                test_x_tot = np.concatenate((test_x_tot, test_x), axis=0)
                labels_tot = np.concatenate((labels_tot, labels), axis=0)

        if verbose:
            print(f"[{datetime.now()}] Loaded {len(self.data[template])} timeseries from template {template}.")
            print(f"Single file shape before sub-windowing: {self.data[template][0].shape}. Single file shape after sub-windowing: {test_x.shape}")
            print(f"[{datetime.now()}] test_x shape:", test_x_tot.shape)
            print(f"[{datetime.now()}] test_y shape:", labels_tot.shape)

        #test_x_tot = DataManager.scale(self.scaler, test_x_tot)
            
        return test_x_tot, labels_tot



    @staticmethod
    def plot_timeseries(template_name, data, trials, sim_params, output_dir, max_flux=None, labels=[]):
        """
        data has the shape (trials, timepoints, channels)
        """
        params = {
            "runid":   template_name,
            "trial":   trials,
            "simtype": sim_params.simtype,
            "onset":   sim_params.onset,
            "delay":   0,
            "offset":  0.5,
            "itype":   "te",
            "itime":   5,
            "normalized": True,
            "maxflux" : max_flux
        }        
        applot = APPlot()
        for i in range(trials):
            applot.plot_from_numpy(data[i], params, labels)
            name = f"template_{template_name}_trial_{i}_{datetime.now()}.png"
            applot.save(output_dir, name)
            print(f"Saved: {output_dir}/{name}")
 