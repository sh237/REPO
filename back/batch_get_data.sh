#!/bin/bash

# batch_get_data.sh
# 指定した期間のデータを1時間毎に取得するスクリプト
# 使用方法: ./batch_get_data.sh START_DATE END_DATE
# 例: ./batch_get_data.sh "2024-01-15 00:00" "2024-01-16 23:00"

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

echo "データ取得開始"
echo "開始日時: $START_DATE"
echo "終了日時: $END_DATE"
echo "===================="

# 現在の日時を開始日時に設定
current_date="$START_DATE"

# 1時間毎にループ
while [[ $(date -d "$current_date" +%s) -le $(date -d "$END_DATE" +%s) ]]; do
    # MMDDHH形式に変換（get_data.pyの引数形式に合わせる）
    mmddhh=$(date -d "$current_date" +"%m%d%H")
    
    echo "処理中: $current_date (引数: $mmddhh)"
    
    # get_data.pyを実行
    if python ./back/get_data.py "$mmddhh"; then
        echo "✅ 完了: $current_date"
    else
        echo "❌ エラー: $current_date"
    fi
    
    echo "--------------------"
    sleep 3 # 1秒待機
    
    # 1時間進める
    current_date=$(date -d "$current_date + 1 hour" "+%Y-%m-%d %H:%M")
done

echo "===================="
echo "すべてのデータ取得が完了しました" 