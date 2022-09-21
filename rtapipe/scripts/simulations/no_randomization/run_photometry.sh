

# get the path to the current script
this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

rtapipe_path=/data01/homes/baroncelli/phd/rtapipe

output_dir_ap="$this_dir/ap_output"
mkdir -p "$output_dir_ap"

python $rtapipe_path/lib/dataset/generate_ap_data.py \
        -dd $this_dir/sim_output/run0406_ID000126 \
        -itype t \
        -itime 1 \
        -rr 0.2 \
        -norm no \
        -lim 5 \
        -out "$output_dir_ap" \
        -proc 20

python $rtapipe_path/lib/dataset/generate_ap_data.py \
        -dd $this_dir/sim_output/run0406_ID000126 \
        -itype te \
        -itime 1 \
        -rr 0.2 \
        -norm no \
        -lim 5 \
        -out "$output_dir_ap" \
        -proc 20

python $rtapipe_path/lib/dataset/generate_ap_data.py \
        -dd $this_dir/sim_output/run0406_ID000126 \
        -itype t \
        -itime 1 \
        -rr 0.2 \
        -norm yes \
        -lim 5 \
        -out "$output_dir_ap" \
        -proc 20

python $rtapipe_path/lib/dataset/generate_ap_data.py \
        -dd $this_dir/sim_output/run0406_ID000126 \
        -itype te \
        -itime 1 \
        -rr 0.2 \
        -norm yes \
        -lim 5 \
        -out "$output_dir_ap" \
        -proc 20

python $rtapipe_path/lib/dataset/generate_ap_data.py \
        -dd $this_dir/sim_output/run0406_ID000126 \
        -itype t \
        -itime 5 \
        -rr 0.2 \
        -norm no \
        -lim 5 \
        -out "$output_dir_ap" \
        -proc 20

python $rtapipe_path/lib/dataset/generate_ap_data.py \
        -dd $this_dir/sim_output/run0406_ID000126 \
        -itype te \
        -itime 5 \
        -rr 0.2 \
        -norm no \
        -lim 5 \
        -out "$output_dir_ap" \
        -proc 20

python $rtapipe_path/lib/dataset/generate_ap_data.py \
        -dd $this_dir/sim_output/run0406_ID000126 \
        -itype t \
        -itime 5 \
        -rr 0.2 \
        -norm yes \
        -lim 5 \
        -out "$output_dir_ap" \
        -proc 20

python $rtapipe_path/lib/dataset/generate_ap_data.py \
        -dd $this_dir/sim_output/run0406_ID000126 \
        -itype te \
        -itime 5 \
        -rr 0.2 \
        -norm yes \
        -lim 5 \
        -out "$output_dir_ap" \
        -proc 20

