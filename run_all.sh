
top -n 1 -b > results/top_0_before.txt

# EXPERIMENT 1
# Inference over different batch sizes:
# python3.7 run_inference.py

python3.7 ravaen_payload/run_inference.py --batch_size 32
python3.7 ravaen_payload/run_inference.py --batch_size 64
python3.7 ravaen_payload/run_inference.py --batch_size 128

python3.7 ravaen_payload/run_inference.py --batch_size 32 --override_channels 10 --log_name "highres10band" --nosave True
python3.7 ravaen_payload/run_inference.py --batch_size 64 --override_channels 10 --log_name "highres10band" --nosave True
python3.7 ravaen_payload/run_inference.py --batch_size 128 --override_channels 10 --log_name "highres10band" --nosave True

top -n 1 -b > results/top_1_after_inference.txt

# EXPERIMENT 2
# python3.7 run_train.py

python3.7 tile_classifier/run_train.py --batch_size 32
python3.7 tile_classifier/run_train.py --batch_size 64
python3.7 tile_classifier/run_train.py --batch_size 128
python3.7 tile_classifier/run_train.py --batch_size 256

python3.7 tile_classifier/run_train.py --batch_size 32 --multiclass True --numclasses 4
python3.7 tile_classifier/run_train.py --batch_size 64 --multiclass True --numclasses 4
python3.7 tile_classifier/run_train.py --batch_size 128 --multiclass True --numclasses 4
python3.7 tile_classifier/run_train.py --batch_size 256 --multiclass True --numclasses 4

top -n 1 -b > results/top_2_after_train.txt


