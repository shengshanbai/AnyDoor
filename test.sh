export PYTHONPATH=.:$PYTHONPATH
export XFORMERS_DISABLED=true
python -m debugpy --connect 127.0.0.1:5678 --wait-for-client infer.py
#python run_inference.py
