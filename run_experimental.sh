
# EXPERIMENT 1
# Inference over different batch sizes:
# python3.7 run_inference.py

python3.7 ravaen_payload/run_inference.py --batch_size 2
python3.7 ravaen_payload/run_inference.py --batch_size 4
python3.7 ravaen_payload/run_inference.py --batch_size 8
python3.7 ravaen_payload/run_inference.py --batch_size 16
#python3.7 ravaen_payload/run_inference.py --batch_size 32
#python3.7 ravaen_payload/run_inference.py --batch_size 64
#python3.7 ravaen_payload/run_inference.py --batch_size 128

python3.7 ravaen_payload/run_inference.py --batch_size 2 --override_channels 10 --log_name "highres10band" --nosave True
python3.7 ravaen_payload/run_inference.py --batch_size 4 --override_channels 10 --log_name "highres10band" --nosave True
python3.7 ravaen_payload/run_inference.py --batch_size 8 --override_channels 10 --log_name "highres10band" --nosave True
python3.7 ravaen_payload/run_inference.py --batch_size 16 --override_channels 10 --log_name "highres10band" --nosave True
#python3.7 ravaen_payload/run_inference.py --batch_size 32 --override_channels 10 --log_name "highres10band" --nosave True
#python3.7 ravaen_payload/run_inference.py --batch_size 64 --override_channels 10 --log_name "highres10band" --nosave True
#python3.7 ravaen_payload/run_inference.py --batch_size 128 --override_channels 10 --log_name "highres10band" --nosave True

# rgb
python3.7 ravaen_payload/run_inference.py --batch_size 64 --override_channels 3 --log_name "exp3band" --nosave True
python3.7 ravaen_payload/run_inference.py --batch_size 64 --override_channels 6 --log_name "exp6band" --nosave True
python3.7 ravaen_payload/run_inference.py --batch_size 64 --override_channels 8 --log_name "exp8band" --nosave True

