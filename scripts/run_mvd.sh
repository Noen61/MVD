python -m torch.distributed.launch --nproc_per_node=4 main.py \
--do_train \
--kb zeshel \
--task_name mvd \
--data_dir data/zeshel \
--entity_data_dir data/zeshel/entity \
--output_dir output/mvd \
--bert_model bert-base-uncased \
--pretrain_retriever output/retriever/retriever.bin \
--pretrain_teacher output/teacher/teacher.bin \
--max_seq_length 128 \
--cand_num 16 \
--infer_view_type global-local \
--train_view_type local \
--max_ent_length 128 \
--max_view_length 40 \
--max_view_num 10 \
--num_train_epochs 5 \
--train_batch_size 16 \
--gradient_accumulation_steps 16 \
--learning_rate 2e-5 \
--eval_per_epoch 1 \
--eval_batch_size 1024 \
--top_k 100 \
--faiss \
--best_checkpoint_metric top64_hits \
--seed 10000 \
--do_lower_case
