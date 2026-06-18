
# conda activate AirVLN

# cd ./AirVLN
# echo $PWD


# nohup python -u ./airsim_plugin/AirVLNSimulatorServerTool.py --gpus 0,1,2,3,4,5,6,7 &
# nohup python -u ./airsim_plugin/AirVLNSimulatorServerTool.py --gpus 0,1 &

python -u ./src/vlnce_src/train.py \
--run_type collect \
--policy_type seq2seq \
--collect_type TF \
--name AerialVLN \
--batchSize 8


