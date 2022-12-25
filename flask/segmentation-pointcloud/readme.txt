python>=3.8
pip install -r requirements.txt
To test the installation use
#可以只測試這個(pytorch)
python -c "import open3d.ml.torch as ml3d"  
#可不用測試(tensorflow)   
python -c "import open3d.ml.tf as ml3d"

指令
python test.py --data_path /home/awinlab/Documents/howting/data/f2.ply --ckpt_path ./logs/SparseEncDec_Semantic3D_torch/checkpoint

(--data_path放點雲檔  --ckpt_path 模型位置)

model.py  --> 神經網路架構
torch_dataloader.py  dataset.py  pipline.py-->數據處理數據讀取 執行模型
(上面4個程式不好改勁量不要動)
config是參數檔

test.py 有加入簡單的註解 然後讀ply檔轉成npy  生成label那邊是讀上一個資料匣data的資料匣做處理 所以data/test是必要得(如果不行這樣我再改)
 

