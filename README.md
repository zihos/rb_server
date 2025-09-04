# rb_server

sam vit-h pt [download](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

conda 
```
conda create -n fastapi python=3.10 -y
conda activate fastapi
pip install fastapi uvicorn[standard]

# server activate
uvicorn main:app --reload

pip install fastapi uvicorn[standard] pillow python-multipart

#sam
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
pip install opencv-python pycocotools matplotlib onnxruntime onnx
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python scripts/export_onnx_model.py --checkpoint /home/zio/segment-anything/sam_vit_h_4b8939.pth --model-type vit_h --output /home/zio/segment-anything/sam_onnx_example.onnx
pip install ultralytics
```
