
this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

output_dir_plot="$this_dir/plot_output"
output_dir_ap="$this_dir/ap_output"

rtapipe_path=/data01/homes/baroncelli/phd/rtapipe

mkdir -p ${output_dir_plot}
python $rtapipe_path/scripts/plot_ap_timeseries.py \
    --dir $output_dir_ap \
    --output-dir ${output_dir_plot}


