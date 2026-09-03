 ./run_V_FP.sh 2>&1 | tee run_V_FP.sh.log

 ./run_c1.sh 2>&1 | tee run_c1.sh.log

uv run python -m src.prepare_tokenizers --data_path data --src_vocab_size 256 --tgt_vocab_size 1000 --output_dir outputs

uv run python -m src.prepare_tokenizers --data_path data --src_vocab_size 256 --tgt_vocab_size 1000 --output_dir outputs

uv run python -m src.prepare_tokenizers --data_path data --src_vocab_size 256 --tgt_vocab_size 1000 --output_dir outputs 2>&1 | tee outputs/tokenizer_training.log

22:43


uv run python -m src.train \
  --config C1 \
  --data_path data \
  --output_dir outputs \
  --epochs 20 \
  --batch_size 16 \
  --lr 0.0005 \
  --d_model 128 \
  --num_heads 4 \
  --num_layers 3 \
  --d_ff 512 \
  --max_len 1024 \
  --use_wandb

chmod +x run_c1.sh
./run_c1.sh 2>&1 | tee outputs/run_c1.log


 ./run_c1.sh 2>&1 | tee outputs/run_c1.log


./run_c1.sh 2>&1 | tee outputs/run_c1.log

3:40

# Upload C1 Base Model
uv run python upload_to_hf.py --config C1 --repo_id likhitbhogadi/anlp-a1-C1

# Upload C2 RoPE Model
uv run python upload_to_hf.py --config C2 --repo_id likhitbhogadi/anlp-a1-C2

# Upload C3 GQA Model
uv run python upload_to_hf.py --config C3 --repo_id likhitbhogadi/anlp-a1-C3

# Upload C4 RMSNorm Model
uv run python upload_to_hf.py --config C4 --repo_id likhitbhogadi/anlp-a1-C4

# Upload C5 BLT Model
uv run python upload_to_hf.py --config C5 --repo_id likhitbhogadi/anlp-a1-C5