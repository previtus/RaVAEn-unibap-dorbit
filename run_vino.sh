
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.7

# by default using encoder_model_mu

python3.7 ravaen_payload/run_inference_openvino.py --batch_size 32
python3.7 ravaen_payload/run_inference_openvino.py --batch_size 64
python3.7 ravaen_payload/run_inference_openvino.py --batch_size 128

python3.7 ravaen_payload/run_inference_openvino.py --batch_size 32 --openvino_device 'CPU' --log_name "log_openvinooncpu"
python3.7 ravaen_payload/run_inference_openvino.py --batch_size 64 --openvino_device 'CPU' --log_name "log_openvinooncpu"
python3.7 ravaen_payload/run_inference_openvino.py --batch_size 128 --openvino_device 'CPU' --log_name "log_openvinooncpu"

