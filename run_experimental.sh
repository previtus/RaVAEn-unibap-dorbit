
# EXPERIMENT 1
# Inference over different batch sizes:
# python3.7 run_inference.py

python3.7 ravaen_payload/run_inference.py --batch_size 4 --save_only_k_latents 1
python3.7 ravaen_payload/run_inference.py --batch_size 8 --save_only_k_latents 1
python3.7 ravaen_payload/run_inference.py --batch_size 16 --save_only_k_latents 1

python3.7 ravaen_payload/run_inference.py --batch_size 4 --override_channels 10 --log_name "highres10band" --nosave True
python3.7 ravaen_payload/run_inference.py --batch_size 8 --override_channels 10 --log_name "highres10band" --nosave True
python3.7 ravaen_payload/run_inference.py --batch_size 16 --override_channels 10 --log_name "highres10band" --nosave True

