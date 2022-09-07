# path
data_dic=self_training
electra_model=./ckpt/electra_en_base
teacher_model=./ckpt/teacher
judge_model=./ckpt/judge
model_name=pytorch_model.bin
log_dir=./log

# setting
teacher_msl=128
judge_msl=256
lr=2e-5
num_train_epochs=20
gradient_accumulation_steps=1
teacher_batch_size=64
judge_batch_size=32
temp=0.5

# self-training
data_scale_list=(500 1000 1500 2000 3000 4000 5000 6000 7000 8000 10000 15000 20000 25000 30000 35000 40000 45000 50000 55000 60000 65000 70000 75000 80000 85000 90000 100000 100000 100000)

# tag data
nohup python ./src/teacher.py\
                    --task_name disfluency\
                    --do_unlabel\
                    --do_eval\
                    --do_test\
                    --do_lower_case\
                    --max_seq_length ${teacher_msl}\
                    --batch_size ${teacher_batch_size}\
                    --learning_rate ${lr}\
                    --pretrain_model_dir ${teacher_model}\
                    --pretrain_model_name ${model_name}\
                    --model_name_or_path ${electra_model}\
                    --use_new_model\
                    --unlabel_size 500\
                    --data_dir ./${data_dic}/run_data/500\
                    --output_dir ./${data_dic}/run_data/500

wait
# judge data
nohup python ./src/total_judge.py\
                    --task_name disfluency\
                    --do_eval\
                    --do_test\
                    --do_lower_case\
                    --max_seq_length ${judge_msl}\
                    --batch_size ${judge_batch_size}\
                    --learning_rate ${lr}\
                    --model_name_or_path ${electra_model}\
                    --use_new_model\
                    --pretrain_model_dir ${judge_model}\
                    --pretrain_model_name ${model_name}\
                    --thre 0\
                    --do_tagging\
                    --data_dir ./${data_dic}/run_data/500\
                    --output_dir ./${data_dic}/run_data/500\
                    --temp ${temp}
wait
for((i=0;i<28;i+=1)); 
do
data_scale=${data_scale_list[i]}
data_scale_next=${data_scale_list[((${i}+1))]}
# train student
nohup python ./src/teacher.py\
                    --task_name disfluency\
                    --do_train\
                    --do_eval\
                    --do_test\
                    --do_lower_case\
                    --max_seq_length ${teacher_msl}\
                    --batch_size ${teacher_batch_size}\
                    --gradient_accumulation_steps ${gradient_accumulation_steps}\
                    --learning_rate ${lr}\
                    --num_train_epochs ${num_train_epochs}\
                    --pretrain_model_dir ${teacher_model}\
                    --pretrain_model_name ${model_name}\
                    --model_name_or_path ${electra_model}\
                    --use_new_model\
                    --seed ${data_scale}\
                    --unlabel_size ${data_scale}\
                    --data_dir ./${data_dic}/run_data/${data_scale}\
                    --output_dir ./${data_dic}/run_model/${data_scale}\
                    --log_dir ${log_dir}\
                    --judge_score

wait
# tag data
nohup python ./src/teacher.py\
                    --task_name disfluency\
                    --do_unlabel\
                    --do_eval\
                    --do_test\
                    --do_lower_case\
                    --max_seq_length ${teacher_msl}\
                    --batch_size ${teacher_batch_size}\
                    --learning_rate ${lr}\
                    --pretrain_model_dir ./${data_dic}/run_model/${data_scale}\
                    --pretrain_model_name ${model_name}\
                    --model_name_or_path ${electra_model}\
                    --use_new_model\
                    --seed ${data_scale_next}\
                    --unlabel_size ${data_scale_next}\
                    --data_dir ./${data_dic}/run_data/${data_scale}\
                    --output_dir ./${data_dic}/run_data/${data_scale_next}

wait
cp ./${data_dic}/run_data/${data_scale}/dev.tsv ./${data_dic}/run_data/${data_scale_next}/dev.tsv
cp ./${data_dic}/run_data/${data_scale}/test.tsv ./${data_dic}/run_data/${data_scale_next}/test.tsv
cp ./${data_dic}/run_data/${data_scale}/unlabel.tsv ./${data_dic}/run_data/${data_scale_next}/unlabel.tsv

# judge data
nohup python ./src/total_judge.py\
                    --task_name disfluency\
                    --do_eval\
                    --do_test\
                    --do_lower_case\
                    --max_seq_length ${judge_msl}\
                    --batch_size ${judge_batch_size}\
                    --learning_rate ${lr}\
                    --model_name_or_path ${electra_model}\
                    --use_new_model\
                    --pretrain_model_dir ${judge_model}\
                    --pretrain_model_name ${model_name}\
                    --thre 0\
                    --do_tagging\
                    --data_dir ./${data_dic}/run_data/${data_scale_next}\
                    --output_dir ./${data_dic}/run_data/${data_scale_next}\
                    --temp ${temp}
wait

done