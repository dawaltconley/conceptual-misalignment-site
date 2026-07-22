for f in mengzi/*.txt; do
  chap=$(basename "$f" .txt)
  python -m xunzi.run seg \
    --input "$f" --output "../segpos/chapters-autofix/$chap.jsonl" \
    --unit line --chapter "$chap" \
    --arch api --api-base http://127.0.0.1:8080/v1 \
    --max-new-tokens 2048
done
