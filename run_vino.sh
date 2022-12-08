
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.7

#cd openvino_conversions
#python3.7 model_to_vino_03_minimal.py

python3.7 ravaen_payload/run_inference_openvino.py --batch_size 4
python3.7 ravaen_payload/run_inference_openvino.py --batch_size 8
python3.7 ravaen_payload/run_inference_openvino.py --batch_size 16
python3.7 ravaen_payload/run_inference_openvino.py --batch_size 32
python3.7 ravaen_payload/run_inference_openvino.py --batch_size 64
python3.7 ravaen_payload/run_inference_openvino.py --batch_size 128

python3.7 ravaen_payload/run_inference_openvino.py --batch_size 4 --openvino_device 'CPU' --log_name "log_openvinooncpu"
python3.7 ravaen_payload/run_inference_openvino.py --batch_size 8 --openvino_device 'CPU' --log_name "log_openvinooncpu"
python3.7 ravaen_payload/run_inference_openvino.py --batch_size 16 --openvino_device 'CPU' --log_name "log_openvinooncpu"
python3.7 ravaen_payload/run_inference_openvino.py --batch_size 32 --openvino_device 'CPU' --log_name "log_openvinooncpu"
python3.7 ravaen_payload/run_inference_openvino.py --batch_size 64 --openvino_device 'CPU' --log_name "log_openvinooncpu"
python3.7 ravaen_payload/run_inference_openvino.py --batch_size 128 --openvino_device 'CPU' --log_name "log_openvinooncpu"


# for comparison, the cpu runs with torch:
python3.7 ravaen_payload/run_inference.py --batch_size 4
python3.7 ravaen_payload/run_inference.py --batch_size 8
python3.7 ravaen_payload/run_inference.py --batch_size 16
python3.7 ravaen_payload/run_inference.py --batch_size 32
python3.7 ravaen_payload/run_inference.py --batch_size 64
python3.7 ravaen_payload/run_inference.py --batch_size 128

