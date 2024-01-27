import yaml
import pickle
import numpy as np 
import pandas as pd 
from pathlib import Path
from rtapipe.lib.utils.misc import dotdict
from rtapipe.lib.models.anomaly_detector_builder import AnomalyDetectorBuilder
from rtapipe.lib.evaluation.pval import get_pval_table, get_threshold_for_sigma
from rtapipe.lib.plotting.PlotConfig2 import PlotConfig2

def load_model(model_id):
    with open("./trained_models.yaml", "r") as f:
        try:
            configs = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    model_config = [c for c in configs["models"] if c["id"] == model_id]
    model_config = dotdict(model_config.pop())
    model_config.ad = AnomalyDetectorBuilder.getAnomalyDetector(name=model_config.name, timesteps=model_config.timesteps, nfeatures=model_config.nfeatures, load_model="True", training_epoch_dir=model_config.path, training=False)
    model_config.pvalue_table = get_pval_table(model_config.pval_path) 
    return model_config

def filter_templates(templates_detections, model, other_model, which, sigma, detection_index_max):

    templates = []
    
    for template_name, template_result in templates_detections.items():
    
        model_detected       = template_result[model][f"{sigma}s_detection"]
        
        if which == "common":
            other_model_detected = template_result[other_model][f"{sigma}s_detection"]
            if model_detected and other_model_detected:
                model_first_detection_index       = template_result[model][f"{sigma}s_detections_indexes"][0]
                other_model_first_detection_index = template_result[other_model][f"{sigma}s_detections_indexes"][0]
                
                if (model_first_detection_index <= detection_index_max) and (other_model_first_detection_index <= detection_index_max): 
                    templates.append(template_name)

        elif which == "first":
            if model_detected:

                model_first_detection_index       = template_result[model][f"{sigma}s_detections_indexes"][0]
                
                if (model_first_detection_index <= detection_index_max): 
                    templates.append(template_name)                

                    
        elif which == "first-only":

            other_model_detected = template_result[other_model][f"{sigma}s_detection"]

            if model_detected and (not other_model_detected):

                model_first_detection_index       = template_result[model][f"{sigma}s_detections_indexes"][0]
                
                if (model_first_detection_index <= detection_index_max): 
                    templates.append(template_name)                
        

        elif which == "second":
            other_model_detected = template_result[other_model][f"{sigma}s_detection"]
            if other_model_detected:

                other_model_first_detection_index       = template_result[other_model][f"{sigma}s_detections_indexes"][0]
                
                if (other_model_first_detection_index <= detection_index_max): 
                    templates.append(template_name)                

                    
        elif which == "second-only":
            other_model_detected = template_result[other_model][f"{sigma}s_detection"]
            if (not model_detected) and other_model_detected:

                other_model_first_detection_index       = template_result[other_model][f"{sigma}s_detections_indexes"][0]
                
                if (other_model_first_detection_index <= detection_index_max): 
                    templates.append(template_name)                
                
        
        
        elif which == "none":
            other_model_detected = template_result[other_model][f"{sigma}s_detection"]
            if (not model_detected) and (not other_model_detected):
                templates.append(template_name)

    
        #model_first_detection_index      = template_result[model][f"{sigma}s_detections_indexes"][0]
        #othermodel_first_detection_index = template_result[other_model][f"{sigma}s_detections_indexes"][0]
    return templates

class APPlot:

    def __init__(self):
        self.pc = PlotConfig2()
        self.fig = None
        self.ax = None

    def read_data(self, csv_file_path):
        df = pd.read_csv(str(csv_file_path))
        if "TMIN" in df.columns:
            df["TCENTER"] = (df["TMAX"] + df['TMIN'])/2
        if "EMIN" in df.columns:
            df["ECENTER"] = (df["EMAX"] + df['EMIN'])/2
        return df

    def set_layout(self, params):
        self.ax.set_xlabel(r'Time $\left[s\right]$')

        if params["normalized"]:
            self.fig.suptitle(r"$Light curve$")
            self.ax.set_ylabel(r'Flux $\left[\times 10^{9}\,\mathrm{ph}/\mathrm{s} / \mathrm{cm}^{2}\right]$')
        else:
            self.fig.suptitle(r"$Counts in region$")
            self.ax.set_ylabel(r"$Counts$")

        self.ax.set_title(f"Template: {params['runid']}, Integration time: {params['itime']}")
        self.ax.legend(title="Energy bins [TeV]:", prop=self.pc.legend_kw, loc='upper left')
        
    def plot_from_numpy(self, data, params, labels=[]):
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=self.pc.fig_size)
        for i in range(data.shape[1]):
            if len(labels) > 0:
                label = labels[i]
            else:
                label = f"Feature {i}"   
            error_bar_kwargs = {
                "ls": "none",
                "marker": 'o',
                "color": self.pc.colors[i], 
                "ecolor": self.pc.colors[i]
            }     
            #self.ax.errorbar(range(data.shape[0]), data[:,i], label=label, **self.pc.error_kw, **error_bar_kwargs)
            number_of_points = data.shape[0]

            self.ax.plot(np.array(range(0,number_of_points))*5, data[:,i]*1e9, color=self.pc.colors[i], marker='o', markersize=6, linestyle='dashed', label=label)
        
        number_of_points = data.shape[0]
            
        if data.max()*1e9 > 4:
            #print("Exceed max: ", data.max()*1e9)
            self.ax.set_ylim(0, data.max()*1e9+1)
        else:
            self.ax.set_ylim(0, 4)

            
            
        if params["onset"] > 0:
            pivot_idx = params["onset"]
            self.ax.axvline(x = pivot_idx, color = "grey", linestyle = '-', alpha=0.7)
            props = dict(boxstyle='round', facecolor='none', linestyle='-',edgecolor='grey', alpha=0.7)
            self.ax.text(0.58,0.9,'GRB start',bbox=props, horizontalalignment='center', transform=self.ax.transAxes)
        
        self.set_layout(params)
    

    def save(self, outputDir, outputFilename):
        if self.fig:
            outputDir = Path(outputDir)
            outputDir.mkdir(parents=True, exist_ok=True)
            outputFilePath = outputDir.joinpath(outputFilename).with_suffix(".png")
            self.fig.savefig(str(outputFilePath), dpi=100)
            outputFilePath = outputDir.joinpath(outputFilename).with_suffix(".svg")
            self.fig.savefig(str(outputFilePath))
            #print(f"Produced: {outputFilePath}")
            plt.close()
            return str(outputFilePath)
        print("No plot has been generated yet!")

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
        "itime":   1,
        "normalized": True,
        "maxflux" : max_flux
    }        
    applot = APPlot()
    for i in range(trials):
        applot.plot_from_numpy(data[i], params, ["0.04 - 0.117", "0.117 - 0.342", "0.342 - 1"])
        name = f"template_{template_name}_itime_1"
        applot.save(output_dir, name)


def plot_nn_metrics2(metrics, model_config, output_dir, fig_name, y_lim=(0.4, 1.05), annotate_after=3, title="", axtitle="", showFig=False, saveFig=True):
    pc = PlotConfig2()
    fig, ax = plt.subplots(1,1,figsize=(12,8))
    ax.plot(metrics["threshold"], metrics["accuracy"], label="Accuracy", marker='o', markersize=3, linestyle='--')
    for i,j in zip(metrics["threshold"][annotate_after:],metrics["accuracy"][annotate_after:]):
        ax.annotate(str(round(j,3)),xy=(i,j+0.01), xytext=(0,1),  textcoords="offset points", fontsize=15)    
    
    ax.plot(metrics["threshold"], metrics["precision"], label="Precision", marker='o', markersize=3, linestyle='--')
    for i,j in zip(metrics["threshold"][annotate_after:],metrics["precision"][annotate_after:]):
        ax.annotate(str(round(j,3)),xy=(i,j+0.01), xytext=(0,1),  textcoords="offset points", fontsize=15)

    ax.plot(metrics["threshold"], metrics["recall"], label="Recall", marker='o', markersize=3, linestyle='--')
    for i,j in zip(metrics["threshold"][annotate_after:],metrics["recall"][annotate_after:]):
        ax.annotate(str(round(j,3)),xy=(i,j+0.01), xytext=(0,1),  textcoords="offset points", fontsize=15)

    fig.suptitle(title)
    ax.set_title(axtitle, fontsize=18)
    ax.set_ylabel("Metrics")
    ax.set_xlabel("Threshold")
    for sigma_thresh in range(1,6):
        sigma_threshold = get_threshold_for_sigma(model_config.pvalue_table, sigma_thresh)
        ax.axvline(x = sigma_threshold, color='grey', linestyle="dotted", alpha=0.8) 
        ax.text(sigma_threshold+0.00001, y_lim[0]+0.05, f"{sigma_thresh}$\sigma$", fontsize = 18, color="grey")

    xticks = [round(m, 5) for m in metrics["threshold"]]    
    ax.set_xticks(xticks, xticks)
    for tick in ax.get_xticklabels():
        tick.set_rotation(-45)    
    
    
    ax.set_ylim(y_lim)
    ax.legend(loc='upper right', prop={'size': 20}, bbox_to_anchor=(0.9,0.98))
    
    #plt.tight_layout()
    
    if showFig:
        plt.show()

    if saveFig:
        fig.savefig(Path(output_dir).joinpath(f"{fig_name}.png"), dpi=pc.dpi)
        print("Saved figure to: ", Path(output_dir).joinpath(f"{fig_name}.png"))
        fig.savefig(Path(output_dir).joinpath(f"{fig_name}.svg"))
        print("Saved figure to: ", Path(output_dir).joinpath(f"{fig_name}.svg"))        
        
    plt.close()


def get_first_detections(templates_detections, model, sigma):
    """
    template_result[model][f"{sigma}s_detections_indexes"] contains a list of indexes, from 0 to n. 
    Each index correspond to a time bin in which a sigma-detection has been performed. 
    """
    first_detection_indexes = []
    for template_result in templates_detections.values():
            if len(template_result[model][f"{sigma}s_detections_indexes"]) > 0:
                first_detection_indexes.append(template_result[model][f"{sigma}s_detections_indexes"][0])
    return np.array(sorted(first_detection_indexes))

def crop_to_5(s):
    if s > 5: 
        s = 5.00001
    return s

def fix_independence(df, module_val):
    for time_bin in df.index:
        if int(time_bin.split("-")[1])%module_val != 0:
            df.loc[time_bin] = 0
    return df




def get_templates_detections(rnn_st, cnn_st, lima_st):
    
    templates_detections = dotdict({})
    
    for templ_det in lima_st:
        templates_detections[templ_det] = dotdict({}) 
        templates_detections[templ_det]["li_ma"] = dotdict({})
        templates_detections[templ_det]["cnn"] = dotdict({})
        templates_detections[templ_det]["rnn"] = dotdict({})

    for templ_det in lima_st:
        detections_3s = lima_st[templ_det] >= 3
        detections_5s = lima_st[templ_det] >= 5
        templates_detections[templ_det]["li_ma"]["3s_detection"] = detections_3s.any()
        templates_detections[templ_det]["li_ma"]["5s_detection"] = detections_5s.any()
        templates_detections[templ_det]["li_ma"]["3s_detections_indexes"] = np.where(detections_3s == True)[0]
        templates_detections[templ_det]["li_ma"]["5s_detections_indexes"] = np.where(detections_5s == True)[0]

    for templ_det in rnn_st:
        detections_3s = rnn_st[templ_det] >= 3
        detections_5s = rnn_st[templ_det] >= 5
        templates_detections[templ_det]["rnn"]["3s_detection"] = detections_3s.any()
        templates_detections[templ_det]["rnn"]["5s_detection"] = detections_5s.any()
        templates_detections[templ_det]["rnn"]["3s_detections_indexes"] = np.where(detections_3s == True)[0]
        templates_detections[templ_det]["rnn"]["5s_detections_indexes"] = np.where(detections_5s == True)[0]
        
    for templ_det in cnn_st:
        detections_3s = cnn_st[templ_det] >= 3
        detections_5s = cnn_st[templ_det] >= 5
        templates_detections[templ_det]["cnn"]["3s_detection"] = detections_3s.any()
        templates_detections[templ_det]["cnn"]["5s_detection"] = detections_5s.any()
        templates_detections[templ_det]["cnn"]["3s_detections_indexes"] = np.where(detections_3s == True)[0]
        templates_detections[templ_det]["cnn"]["5s_detections_indexes"] = np.where(detections_5s == True)[0] 

    return templates_detections

def filter_templates(templates_detections, model, other_model, which, sigma, detection_index_min, detection_index_max):

    templates = []
    #print("detection_index_min:",detection_index_min)
    #print("detection_index_max:",detection_index_max)

    for template_name, template_result in templates_detections.items():
    
        model_detected       = template_result[model][f"{sigma}s_detection"]
        
        if which == "common":
            other_model_detected = template_result[other_model][f"{sigma}s_detection"]
            if model_detected and other_model_detected:
                model_first_detection_index       = template_result[model][f"{sigma}s_detections_indexes"][0]
                other_model_first_detection_index = template_result[other_model][f"{sigma}s_detections_indexes"][0]
                
                if (model_first_detection_index >= detection_index_min and model_first_detection_index <= detection_index_max) and (other_model_first_detection_index >= detection_index_min and other_model_first_detection_index <= detection_index_max): 
                    templates.append(template_name)

        elif which == "first":
            if model_detected:

                model_first_detection_index = template_result[model][f"{sigma}s_detections_indexes"][0]
                
                if (model_first_detection_index >= detection_index_min and model_first_detection_index <= detection_index_max): 
                    templates.append(template_name)                

                    
        elif which == "first-only":

            other_model_detected = template_result[other_model][f"{sigma}s_detection"]

            if model_detected and (not other_model_detected):

                model_first_detection_index       = template_result[model][f"{sigma}s_detections_indexes"][0]
                
                if (model_first_detection_index >= detection_index_min and model_first_detection_index <= detection_index_max): 
                    templates.append(template_name)                
        

        elif which == "second":
            other_model_detected = template_result[other_model][f"{sigma}s_detection"]
            if other_model_detected:

                other_model_first_detection_index       = template_result[other_model][f"{sigma}s_detections_indexes"][0]
                
                if (other_model_first_detection_index >= detection_index_min and other_model_first_detection_index <= detection_index_max): 
                    templates.append(template_name)                

                    
        elif which == "second-only":
            other_model_detected = template_result[other_model][f"{sigma}s_detection"]
            if (not model_detected) and other_model_detected:

                other_model_first_detection_index       = template_result[other_model][f"{sigma}s_detections_indexes"][0]
                
                if (other_model_first_detection_index >= detection_index_min and other_model_first_detection_index <= detection_index_max): 
                    templates.append(template_name)                
                
        
        
        elif which == "none":
            other_model_detected = template_result[other_model][f"{sigma}s_detection"]
            if (not model_detected) and (not other_model_detected):
                templates.append(template_name)

    
        #model_first_detection_index      = template_result[model][f"{sigma}s_detections_indexes"][0]
        #othermodel_first_detection_index = template_result[other_model][f"{sigma}s_detections_indexes"][0]
    return templates


def evaluate_metrics(model_config, test_all_x, test_all_y, output_dir, filename, additional_tresholds=[]):
    output_file = Path(output_dir).joinpath(filename)
    if output_file.exists():
        print(f"Loading {filename} from {output_file}...")
        with open(output_file, "rb") as f:
            results = pickle.load(f)
        return results

    results = {
        "threshold" : [],
        "accuracy" : [],
        "precision" : [],
        "recall" : [],
        "false_positive_rate" : []
    }
    for sigma_threshold in additional_tresholds:
        model_config.ad.threshold = sigma_threshold
        metrics = model_config.ad.evaluate(test_all_x, test_all_y)
        results["threshold"] += [sigma_threshold]
        for key, value in metrics.items():
            if key in ["accuracy", "precision", "recall", "false_positive_rate"]:
                results[key] += [value]

    for sigma_thresh in range(1,6):
        sigma_threshold = get_threshold_for_sigma(model_config.pvalue_table, sigma_thresh)
        model_config.ad.threshold = sigma_threshold
        #print(f"Threshold: {model_config_rnn.ad.threshold} corresponding to {get_sigma_from_ts(model_config_rnn.pvalue_table, model_config_rnn.ad.threshold)} sigma")
        metrics = model_config.ad.evaluate(test_all_x, test_all_y)
        results["threshold"] += [sigma_threshold]
        for key, value in metrics.items():
            if key in ["accuracy", "precision", "recall", "false_positive_rate"]:
                results[key] += [value]
 
    with open(output_file, "wb") as f:
        print("Generating output file: ", output_file)
        pickle.dump(results, f)

    return results








import math
class IncrementalMean:
    
    def __init__(self, model):
        self.mean = 0.0
        self.var = 0.0
        self.n = 0
        self.means = []
        self.stds = []
        self.model = model
        self.tbins = []
        
    def update(self, tbin_max, samples):
        for s in samples:
            self.n += 1            
            delta = s - self.mean
            self.mean += delta / self.n
            self.var += delta * (s - self.mean)
        mean = round(self.mean,2)
        std = math.sqrt(
                        self.var/(self.n - 1)
             )
        std = round(std, 2)
        self.means.append(mean)
        self.stds.append(std)
        self.tbins.append(f"{tbin_max}")
        #print(self.means)

    def __str__(self):
        mean = round(self.mean,2)
        std = math.sqrt(
                        self.var/(self.n - 1)
             )
        std = round(std, 2)
        #self.means.append(mean)
        #self.stds.append(std)
        return f"{mean} +- {std}"

    def to_latex(self):
        print(f"\n{self.model} Converting in Latex {self.tbins} ")
        df = pd.DataFrame(
            {
                'tbins': self.tbins,
                'mean': self.means,
                'stds': self.stds
            }
        )  
        #print(df)
        name = f"./{self.model}_dd.tex"
        df.to_latex(name, header=False, index=False)
        # print file contenst
        with open(name, 'r') as f:
            print(f.read())

    
def get_new_samples(templates_detections, templates, model, sigma, onset_index, integration_time):
    first_detection_indexes = []
    for template_name in templates:
        first_detection_index = templates_detections[template_name][model][f"{sigma}s_detections_indexes"][0]
        first_detection_indexes.append(first_detection_index)
    first_detection_time_relative_to_tt = (np.array(first_detection_indexes)-onset_index)*integration_time 
    return first_detection_time_relative_to_tt

def get_dd(model, templates_detections, sigma, onset_index, integration_time, tmin, tmaxs, show_common_only=True):
    
    im_model = IncrementalMean(model)
    im_model_common_lima = IncrementalMean(model+"_common_li_ma")
    
    for tmax in tmaxs:
        
        tbin_min=f"{tmin*integration_time}-{tmin*integration_time+integration_time*5}"
        tbin_max=f"{tmax*integration_time}-{tmax*integration_time+integration_time*5}"
        #print(f"{tbin_min}=>{tbin_max}")
        
        templates = filter_templates(templates_detections, model, None, "first", sigma, tmin, tmax)
        common_with_lima = filter_templates(templates_detections, model, "li_ma", "common", sigma, tmin, tmax)
        
        new_samples = get_new_samples(templates_detections, templates, model, sigma, onset_index, integration_time)
        im_model.update(tbin_max, new_samples)
        #print(f"{model} {im_model}")

        new_samples = get_new_samples(templates_detections, common_with_lima, model, sigma, onset_index, integration_time)
        im_model_common_lima.update(tbin_max, new_samples)
        #print(f"{model} {im_model_common_lima} [common LiMa]\n")
    
    if show_common_only:
        im_model_common_lima.to_latex()
    else:
        im_model.to_latex()
        im_model_common_lima.to_latex()


def detection_tables(templates_detections, integration_time, SIGMA, ONSET_INDEX, TMIN, TMAXs):
    

    for TMAX in TMAXs:
        
        print(f"\nFrom {TMIN*integration_time}-{TMIN*integration_time+integration_time*5} TO From {TMAX*integration_time}-{TMAX*integration_time+integration_time*5}")

        rnn = filter_templates(templates_detections, "rnn", None, "first", SIGMA, TMIN, TMAX)
        print("  RNN:",len(rnn))

        cnn = filter_templates(templates_detections, "cnn", None, "first", SIGMA, TMIN, TMAX)
        print("  CNN:",len(cnn))

        lima = filter_templates(templates_detections, "li_ma", None, "first", SIGMA, TMIN, TMAX)
        print("  LI_MA:",len(lima))

        common_rnn_lima = filter_templates(templates_detections, "rnn", "li_ma", "common", SIGMA, TMIN, TMAX)
        print(f"  Common RNN-LI_MA detections: {len(common_rnn_lima)}")
        common_cnn_lima = filter_templates(templates_detections, "cnn", "li_ma", "common", SIGMA, TMIN, TMAX)
        print(f"  Common CNN-LI_MA detections: {len(common_cnn_lima)}")

        rnn_no_cnn = filter_templates(templates_detections, "rnn", "cnn", "first-only", SIGMA, TMIN, TMAX)
        print("  RNN but no CNN",len(rnn_no_cnn))
        rnn_no_lima = filter_templates(templates_detections, "rnn", "li_ma", "first-only", SIGMA, TMIN, TMAX)
        print("  RNN but no LI_MA",len(rnn_no_lima))    

        cnn_no_rnn = filter_templates(templates_detections, "cnn", "rnn", "first-only", SIGMA, TMIN, TMAX)
        print("  CNN but no RNN",len(cnn_no_rnn))
        cnn_no_lima = filter_templates(templates_detections, "cnn", "li_ma", "first-only", SIGMA, TMIN, TMAX)
        print("  CNN but no LI_MA",len(cnn_no_lima))       

        lima_no_rnn = filter_templates(templates_detections, "li_ma", "rnn", "first-only", SIGMA, TMIN, TMAX)
        print("  LI_MA but no RNN",len(lima_no_rnn))
        lima_no_cnn = filter_templates(templates_detections, "li_ma", "cnn", "first-only", SIGMA, TMIN, TMAX)
        print("  LI_MA but no CNN",len(lima_no_cnn))           