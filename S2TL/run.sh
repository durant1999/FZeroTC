python3 -u S2TL.py  \
--method pet  \
--pattern_ids  1 \
--data_dir ./data/ag_news_csv/ag_news_csv \
--model_type roberta  \
--model_name_or_path  roberta-base \
--task_name  agnews \
--output_dir ./outputs/agnews \
--do_train \
--pet_repetitions 1 \
--pet_per_gpu_train_batch_size 16 \
--pet_per_gpu_unlabeled_batch_size 32 \
--pet_num_train_epochs 10 \
--lm_training \
--no_distillation \
--known_classes 2 \
--limited_clusters 4 \
--total_classes 4 \
--per_known_class_samples_on_labeled_semantics_pool 5 \
--pic_name agnews_b \
> ./agnews_log/agnews_b.txt 1>&1

