#!/bin/bash

# batch_inference_pretrain.sh
# 指定した期間の各時刻に対してinference_pretrain.pyを実行し、中間特徴量を抽出するスクリプト
# 使用方法: ./batch_inference_pretrain.sh START_DATE END_DATE
# 例: ./batch_inference_pretrain.sh "2024-01-15 00:00" "2024-01-16 23:00"

set -e

# 引数チェック
if [ $# -ne 2 ]; then
    echo "使用方法: $0 START_DATE END_DATE"
    echo "例: $0 \"2024-01-15 00:00\" \"2024-01-16 23:00\""
    echo "日付形式: YYYY-MM-DD HH:MM"
    exit 1
fi

START_DATE="$1"
END_DATE="$2"

# 日付の妥当性チェック
if ! date -d "$START_DATE" >/dev/null 2>&1; then
    echo "エラー: 開始日時が無効です: $START_DATE"
    exit 1
fi

if ! date -d "$END_DATE" >/dev/null 2>&1; then
    echo "エラー: 終了日時が無効です: $END_DATE"
    exit 1
fi

# 開始日時が終了日時より前かチェック
if [[ $(date -d "$START_DATE" +%s) -gt $(date -d "$END_DATE" +%s) ]]; then
    echo "エラー: 開始日時が終了日時より後になっています"
    exit 1
fi

echo "特徴量抽出開始"
echo "開始日時: $START_DATE"
echo "終了日時: $END_DATE"
echo "===================="

# 現在の日時を開始日時に設定
current_date="$START_DATE"

# 1時間毎にループ
while [[ $(date -d "$current_date" +%s) -le $(date -d "$END_DATE" +%s) ]]; do
    # YYYYMMDD_HHMMSS形式に変換（inference_pretrain.pyの引数形式に合わせる）
    datetime_str=$(date -d "$current_date" +"%Y%m%d_%H0000")
    
    echo "処理中: $current_date (引数: $datetime_str)"
    
    # mlディレクトリに移動してinference_pretrain.pyを実行
    if (cd ./ml && python inference_pretrain.py \
        --params params/main/params.yaml \
        --datetime "$datetime_str" \
        --fold 1 \
        --pretrain_checkpoint checkpoints/pretrain/ours.pth); then
        echo "✅ 特徴量抽出完了: $current_date"
    else
        echo "❌ 特徴量抽出エラー: $current_date"
    fi
    
    echo "--------------------"
    sleep 1 # 1秒待機
    
    # 1時間進める
    current_date=$(date -d "$current_date + 1 hour" "+%Y-%m-%d %H:%M")
done

echo "===================="
echo "すべての特徴量抽出が完了しました" 