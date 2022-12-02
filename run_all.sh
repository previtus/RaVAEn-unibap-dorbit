# EXPERIMENT 1
# Inference over different batch sizes:
# python3.7 run_inference.py

python3.7 ravaen_payload/run_inference.py --batch_size 2 --selected_images first_30
python3.7 ravaen_payload/run_inference.py --batch_size 4 --selected_images first_30
python3.7 ravaen_payload/run_inference.py --batch_size 8 --selected_images first_30
python3.7 ravaen_payload/run_inference.py --batch_size 16 --selected_images first_30
python3.7 ravaen_payload/run_inference.py --batch_size 32 --selected_images first_30
python3.7 ravaen_payload/run_inference.py --batch_size 64 --selected_images first_30
python3.7 ravaen_payload/run_inference.py --batch_size 128 --selected_images first_30

# EXPERIMENT 2
# python3.7 run_train.py

python3.7 tile_classifier/run_train.py --batch_size 2
python3.7 tile_classifier/run_train.py --batch_size 4
python3.7 tile_classifier/run_train.py --batch_size 8
python3.7 tile_classifier/run_train.py --batch_size 16
python3.7 tile_classifier/run_train.py --batch_size 32
python3.7 tile_classifier/run_train.py --batch_size 64
python3.7 tile_classifier/run_train.py --batch_size 128
python3.7 tile_classifier/run_train.py --batch_size 256
