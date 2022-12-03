
# EXPERIMENT 1
# Inference over different batch sizes:
# python3.7 run_inference.py

python3.7 ravaen_payload/run_inference.py --batch_size 2 --force_dummy_model True --log_name "noweights"
python3.7 ravaen_payload/run_inference.py --batch_size 4 --force_dummy_model True --log_name "noweights"
python3.7 ravaen_payload/run_inference.py --batch_size 8 --force_dummy_model True --log_name "noweights"
python3.7 ravaen_payload/run_inference.py --batch_size 16 --force_dummy_model True --log_name "noweights"
python3.7 ravaen_payload/run_inference.py --batch_size 32 --force_dummy_model True --log_name "noweights"
python3.7 ravaen_payload/run_inference.py --batch_size 64 --force_dummy_model True --log_name "noweights"
python3.7 ravaen_payload/run_inference.py --batch_size 128 --force_dummy_model True --log_name "noweights"

python3.7 ravaen_payload/run_inference.py --batch_size 2 --force_dummy_data True --log_name "nodata"
python3.7 ravaen_payload/run_inference.py --batch_size 4 --force_dummy_data True --log_name "nodata"
python3.7 ravaen_payload/run_inference.py --batch_size 8 --force_dummy_data True --log_name "nodata"
python3.7 ravaen_payload/run_inference.py --batch_size 16 --force_dummy_data True --log_name "nodata"
python3.7 ravaen_payload/run_inference.py --batch_size 32 --force_dummy_data True --log_name "nodata"
python3.7 ravaen_payload/run_inference.py --batch_size 64 --force_dummy_data True --log_name "nodata"
python3.7 ravaen_payload/run_inference.py --batch_size 128 --force_dummy_data True --log_name "nodata"

python3.7 ravaen_payload/run_inference.py --batch_size 2 --force_dummy_data True --force_dummy_model True --log_name "nodatanoweights"
python3.7 ravaen_payload/run_inference.py --batch_size 4 --force_dummy_data True --force_dummy_model True --log_name "nodatanoweights"
python3.7 ravaen_payload/run_inference.py --batch_size 8 --force_dummy_data True --force_dummy_model True --log_name "nodatanoweights"
python3.7 ravaen_payload/run_inference.py --batch_size 16 --force_dummy_data True --force_dummy_model True --log_name "nodatanoweights"
python3.7 ravaen_payload/run_inference.py --batch_size 32 --force_dummy_data True --force_dummy_model True --log_name "nodatanoweights"
python3.7 ravaen_payload/run_inference.py --batch_size 64 --force_dummy_data True --force_dummy_model True --log_name "nodatanoweights"
python3.7 ravaen_payload/run_inference.py --batch_size 128 --force_dummy_data True --force_dummy_model True --log_name "nodatanoweights"

python3.7 ravaen_payload/run_inference.py --batch_size 2 --override_channels 10 --log_name "highres10band"
python3.7 ravaen_payload/run_inference.py --batch_size 4 --override_channels 10 --log_name "highres10band"
python3.7 ravaen_payload/run_inference.py --batch_size 8 --override_channels 10 --log_name "highres10band"
python3.7 ravaen_payload/run_inference.py --batch_size 16 --override_channels 10 --log_name "highres10band"
python3.7 ravaen_payload/run_inference.py --batch_size 32 --override_channels 10 --log_name "highres10band"
python3.7 ravaen_payload/run_inference.py --batch_size 64 --override_channels 10 --log_name "highres10band"
python3.7 ravaen_payload/run_inference.py --batch_size 128 --override_channels 10 --log_name "highres10band"


