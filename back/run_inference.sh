#!/bin/bash

# 手動実行用スクリプト: update.ymlのworkflowを拡張版
# 指定した時間数分のデータを取得してから、inference_pretrain, inferenceを実行し、結果をコミット・プッシュします
#
# 使用例:
#   ./run_inference.sh 4                    # 直近4時間分を処理
#   ./run_inference.sh 4 20250617_15        # 2025年6月17日15時から過去4時間分を処理
#   ./run_inference.sh                      # デフォルト2時間分を処理

# エラーが発生しても処理を続ける
set +e

# デフォルトは2時間
HOURS=${1:-2}

# 開始時間の指定（YYYYMMDD_HH形式）
if [ -n "$2" ]; then
    START_TIME="$2"
    # 入力形式の検証
    if [[ ! "$START_TIME" =~ ^[0-9]{8}_[0-9]{2}$ ]]; then
        echo "❌ Error: Invalid start time format. Use YYYYMMDD_HH (e.g., 20250617_15)"
        exit 1
    fi
    echo "🚀 Starting manual inference workflow for $HOURS hours from $START_TIME..."
else
    START_TIME=""
    echo "🚀 Starting manual inference workflow for latest $HOURS hours..."
fi

# スクリプトの場所を基準にルートディレクトリを設定
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

echo "📍 Working directory: $(pwd)"
echo ""

# エラーカウンタ
DATA_ERRORS=0
FEATURE_ERRORS=0
INFERENCE_ERRORS=0

# Step 1: 全時間のデータ取得（2秒間隔でsleep）
echo "----------------------------------------------------------------------"
echo "📥 Step 1: Fetching data for $HOURS hours..."
echo "----------------------------------------------------------------------"

for ((i=0; i<HOURS; i++)); do
    if [ -n "$START_TIME" ]; then
        # 指定された開始時間から i 時間前を計算
        START_YEAR=${START_TIME:0:4}
        START_MONTH=${START_TIME:4:2}
        START_DAY=${START_TIME:6:2}
        START_HOUR=${START_TIME:9:2}
        START_DATETIME="${START_YEAR}-${START_MONTH}-${START_DAY} ${START_HOUR}:00:00"
        TARGET_DATETIME_MDH=$(date -u -d "$START_DATETIME $i hours ago" +"%m%d%H")
        TARGET_DATETIME_DISPLAY=$(date -u -d "$START_DATETIME $i hours ago" +"%Y-%m-%d %H:00 UTC")
    else
        # 現在時刻から i 時間前を計算
        TARGET_DATETIME_MDH=$(date -u -d "$i hours ago" +"%m%d%H")
        TARGET_DATETIME_DISPLAY=$(date -u -d "$i hours ago" +"%Y-%m-%d %H:00 UTC")
    fi
    
    echo "  Fetching data for $TARGET_DATETIME_DISPLAY ($TARGET_DATETIME_MDH)..."
    
    if python back/get_data.py $TARGET_DATETIME_MDH; then
        echo "  ✅ Data fetching complete for $TARGET_DATETIME_DISPLAY"
    else
        echo "  ❌ Data fetching failed for $TARGET_DATETIME_DISPLAY"
        ((DATA_ERRORS++))
    fi
    
    # 最後の画像取得以外はsleep
    if [ $i -lt $((HOURS-1)) ]; then
        echo "  💤 Sleeping for 2 seconds..."
        sleep 2
    fi
done

echo ""

# Step 2: 全時間の特徴量抽出
echo "----------------------------------------------------------------------"
echo "🔧 Step 2: Extracting pretrain features for all hours..."
echo "----------------------------------------------------------------------"

for ((i=0; i<HOURS; i++)); do
    if [ -n "$START_TIME" ]; then
        # 指定された開始時間から i 時間前を計算
        START_YEAR=${START_TIME:0:4}
        START_MONTH=${START_TIME:4:2}
        START_DAY=${START_TIME:6:2}
        START_HOUR=${START_TIME:9:2}
        START_DATETIME="${START_YEAR}-${START_MONTH}-${START_DAY} ${START_HOUR}:00:00"
        TARGET_DATETIME_YMD_HMS=$(date -u -d "$START_DATETIME $i hours ago" +"%Y%m%d_%H0000")
        TARGET_DATETIME_DISPLAY=$(date -u -d "$START_DATETIME $i hours ago" +"%Y-%m-%d %H:00 UTC")
    else
        # 現在時刻から i 時間前を計算
        TARGET_DATETIME_YMD_HMS=$(date -u -d "$i hours ago" +"%Y%m%d_%H0000")
        TARGET_DATETIME_DISPLAY=$(date -u -d "$i hours ago" +"%Y-%m-%d %H:00 UTC")
    fi
    
    echo "  Extracting features for $TARGET_DATETIME_DISPLAY ($TARGET_DATETIME_YMD_HMS)..."
    
    cd ml
    if python inference_pretrain.py --params params/main/params.yaml --datetime $TARGET_DATETIME_YMD_HMS --fold 1 --pretrain_checkpoint checkpoints/pretrain/ours.pth; then
        echo "  ✅ Feature extraction complete for $TARGET_DATETIME_DISPLAY"
    else
        echo "  ❌ Feature extraction failed for $TARGET_DATETIME_DISPLAY"
        ((FEATURE_ERRORS++))
    fi
    cd ..
done

echo ""

# Step 3: 全時間の推論実行
echo "----------------------------------------------------------------------"
echo "🧠 Step 3: Running inference for all hours..."
echo "----------------------------------------------------------------------"

for ((i=0; i<HOURS; i++)); do
    if [ -n "$START_TIME" ]; then
        # 指定された開始時間から i 時間前を計算
        START_YEAR=${START_TIME:0:4}
        START_MONTH=${START_TIME:4:2}
        START_DAY=${START_TIME:6:2}
        START_HOUR=${START_TIME:9:2}
        START_DATETIME="${START_YEAR}-${START_MONTH}-${START_DAY} ${START_HOUR}:00:00"
        TARGET_DATETIME_YMD_HMS=$(date -u -d "$START_DATETIME $i hours ago" +"%Y%m%d_%H0000")
        TARGET_DATETIME_DISPLAY=$(date -u -d "$START_DATETIME $i hours ago" +"%Y-%m-%d %H:00 UTC")
    else
        # 現在時刻から i 時間前を計算
        TARGET_DATETIME_YMD_HMS=$(date -u -d "$i hours ago" +"%Y%m%d_%H0000")
        TARGET_DATETIME_DISPLAY=$(date -u -d "$i hours ago" +"%Y-%m-%d %H:00 UTC")
    fi
    
    echo "  Running inference for $TARGET_DATETIME_DISPLAY ($TARGET_DATETIME_YMD_HMS)..."
    
    if python ml/inference.py --params params/main/params.yaml --fold 1 --data_root ./ml/datasets --cuda_device -1 --history 4 --trial_name 090 --mode test --resume_from_checkpoint ./ml/checkpoints/main/ours.pth --datetime $TARGET_DATETIME_YMD_HMS; then
        echo "  ✅ Inference complete for $TARGET_DATETIME_DISPLAY"
    else
        echo "  ❌ Inference failed for $TARGET_DATETIME_DISPLAY"
        ((INFERENCE_ERRORS++))
    fi
done

echo ""

# Step 4: コミット・プッシュ
echo "----------------------------------------------------------------------"
echo "📤 Step 4: Committing and pushing changes..."
echo "----------------------------------------------------------------------"
git add data/
git add ml/datasets/

# 変更があるかチェック
if git diff --cached --quiet; then
    echo "No changes to commit."
else
    TIMESTAMP=$(date +"%H:%M")
    if [ -n "$START_TIME" ]; then
        COMMIT_MESSAGE="🤖Update data [manual-batch-${HOURS}h-from-${START_TIME}] at $TIMESTAMP"
    else
        COMMIT_MESSAGE="🤖Update data [manual-batch-${HOURS}h] at $TIMESTAMP"
    fi
    echo "Committing with message: $COMMIT_MESSAGE"
    
    if git commit -m "$COMMIT_MESSAGE"; then
        echo "✅ Commit successful."
        
        # upstreamブランチが設定されていない場合の対処
        echo "Pushing changes..."
        if git push 2>/dev/null; then
            echo "✅ Push successful."
        else
            echo "Setting upstream branch 'origin main' and pushing..."
            if git push --set-upstream origin main; then
                echo "✅ Push with upstream successful."
            else
                echo "❌ Push failed."
            fi
        fi
    else
        echo "❌ Commit failed."
    fi
fi

echo ""
echo "----------------------------------------------------------------------"
echo "📊 SUMMARY"
echo "----------------------------------------------------------------------"
echo "✅ Manual batch inference workflow completed!"
if [ -n "$START_TIME" ]; then
    echo "📊 Processed $HOURS hours of data from $START_TIME in batch mode."
else
    echo "📊 Processed $HOURS hours of data in batch mode."
fi

# エラーサマリー
TOTAL_ERRORS=$((DATA_ERRORS + FEATURE_ERRORS + INFERENCE_ERRORS))
if [ $TOTAL_ERRORS -eq 0 ]; then
    echo "🎉 No errors occurred during processing!"
else
    echo "⚠️  Error Summary:"
    [ $DATA_ERRORS -gt 0 ] && echo "   📥 Data fetching errors: $DATA_ERRORS"
    [ $FEATURE_ERRORS -gt 0 ] && echo "   🔧 Feature extraction errors: $FEATURE_ERRORS"
    [ $INFERENCE_ERRORS -gt 0 ] && echo "   🧠 Inference errors: $INFERENCE_ERRORS"
    echo "   Total errors: $TOTAL_ERRORS"
fi

echo "📊 Check data/ and ml/datasets/ directories for updated files." 