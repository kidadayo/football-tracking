from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import timedelta, datetime
from PIL import Image, ImageDraw, ImageFont

def calculate_iou(box1, box2):
    """バウンディングボックス間のIoUを計算する"""
    # box format: [x1, y1, x2, y2]
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]
    
    # 交差領域の座標を計算
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # 交差領域の面積を計算
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # 各ボックスの面積を計算
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # IoUを計算
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0
    
    return iou

def filter_overlapping_boxes(boxes, scores, iou_threshold=0.3):
    """IoUに基づいて重複するバウンディングボックスをフィルタリング"""
    if len(boxes) == 0:
        return [], []
    
    # スコアでソート
    indices = np.argsort(scores)[::-1]
    boxes = boxes[indices]
    scores = scores[indices]
    
    keep = []
    
    while len(indices) > 0:
        keep.append(indices[0])
        if len(indices) == 1:
            break
            
        ious = np.array([calculate_iou(boxes[indices[0]], boxes[i]) for i in indices[1:]])
        indices = indices[1:][ious < iou_threshold]
    
    return boxes[keep], scores[keep]

def detect_and_filter_players(frame, model):
    """プレイヤーを検出し、IoUフィルタリングを適用"""
    # YOLOで検出を実行
    results = model.predict(frame, classes=[0], conf=0.3)  # 信頼度閾値を上げて確実な検出のみを採用
    detections = results[0].boxes.data
    
    if len(detections) == 0:
        return results, 0
    
    # バウンディングボックスとスコアを取得
    boxes = detections[:, :4].cpu().numpy()
    scores = detections[:, 4].cpu().numpy()
    
    # IoUフィルタリングを適用
    filtered_boxes, filtered_scores = filter_overlapping_boxes(boxes, scores)
    
    # 結果を更新
    results[0].boxes.data = results[0].boxes.data[np.isin(detections[:, :4].cpu().numpy(), filtered_boxes).all(axis=1)]
    
    return results, len(filtered_boxes)

def draw_japanese_text(image, text, position, font_size=32):
    # OpenCV画像をPIL画像に変換
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    
    try:
        # システムにインストールされているフォントを使用
        font = ImageFont.truetype('C:\\Windows\\Fonts\\meiryo.ttc', font_size)
    except:
        # フォントが見つからない場合はデフォルトフォントを使用
        font = ImageFont.load_default()
    
    # 黒い縁取り（8方向）
    for offset_x, offset_y in [(x, y) for x in [-3,3] for y in [-3,3]]:
        draw.text((position[0] + offset_x, position[1] + offset_y), text, font=font, fill=(0, 0, 0))
    
    # 白い文字
    draw.text(position, text, font=font, fill=(255, 255, 255))
    
    # PIL画像をOpenCV画像に戻す
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def count_players(video_path):
    # YOLOv8モデルの初期化
    model = YOLO('yolov8n.pt')
    
    # 動画ファイルを開く
    video = cv2.VideoCapture(str(video_path))
    
    # 動画の情報を取得
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps  # 動画の長さ（秒）
    
    # 予想処理時間を計算（1フレームあたり約0.1秒と仮定）
    estimated_time = total_frames * 0.1  # 秒単位
    estimated_duration = timedelta(seconds=int(estimated_time))
    
    # 分析情報を表示
    print("\n===== 分析情報 =====")
    print(f"動画の長さ: {timedelta(seconds=int(video_duration))}")
    print(f"総フレーム数: {total_frames}")
    print(f"フレームレート: {fps}fps")
    print(f"解像度: {frame_width}x{frame_height}")
    print(f"予想処理時間: {estimated_duration}")
    
    # ユーザーに確認
    while True:
        response = input("\n分析を実行しますか？ (y/n): ").lower()
        if response in ['y', 'n']:
            break
        print("yまたはnで入力してください。")
    
    if response == 'n':
        print("分析を中止しました。")
        video.release()
        return
    
    # フレーム間隔の設定
    frames_per_second = int(fps)  # スクリーンショット用（1秒間隔）
    excel_interval = int(fps * 0.5)  # Excelデータ用（0.5秒間隔）
    
    # リサイズ比率の計算（幅1280pxを基準）
    resize_ratio = 1280 / frame_width
    process_width = 1280
    process_height = int(frame_height * resize_ratio)
    
    # 出力ディレクトリの設定
    current_datetime = datetime.now().strftime("%Y%m%d%H%M")
    output_dir = video_path.parent / current_datetime
    output_dir.mkdir(exist_ok=True)
    
    # 結果を保存するリスト
    results_data = []
    
    print("\n動画を分析中...")
    
    # 分析開始時刻を記録
    start_time = datetime.now()
    
    # フレームごとの処理
    frame_idx = 0
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
            
        # 処理用に画像をリサイズ
        process_frame = cv2.resize(frame, (process_width, process_height))
            
        # YOLOで検出を実行し、IoUフィルタリングを適用
        results, person_count = detect_and_filter_players(process_frame, model)
        
        # 時間情報の計算
        current_time = timedelta(seconds=frame_idx/fps)

        # 0.25秒ごとにExcelデータを記録
        if frame_idx % excel_interval == 0:
            # データを記録
            results_data.append({
                '経過時間': str(current_time),
                '検出人数': person_count
            })

        # 1秒ごとに画像を保存
        if frame_idx % frames_per_second == 0:
            # バウンディングボックスを描画
            annotated_frame = results[0].plot()

            # 検出人数を画像に追記
            text = f"検出人数：{person_count}人"
            annotated_frame = draw_japanese_text(
                annotated_frame,
                text,
                (10, 30),
                font_size=50  # フォントサイズを調整（リサイズに合わせて）
            )

            # 画像を保存（圧縮率50%）
            video_time = str(current_time).replace(':', '').replace('.', '')
            image_path = output_dir / f"{video_time}_{person_count}人.jpg"
            cv2.imwrite(str(image_path), annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])

            # シンプルなプログレス表示
            progress = (frame_idx / total_frames) * 100
            elapsed_time = datetime.now() - start_time
            estimated_total_time = elapsed_time * (total_frames / (frame_idx + 1))
            remaining_time = estimated_total_time - elapsed_time
            
            print(f"\r処理進捗: {progress:.0f}% (残り時間: {str(remaining_time).split('.')[0]})", end="")
        
        frame_idx += 1
    
    # リソースの解放
    video.release()
    print("\n分析完了!")
    
    # 実際の処理時間を表示
    total_time = datetime.now() - start_time
    print(f"実際の処理時間: {str(total_time).split('.')[0]}")
    
    # 結果をDataFrameに変換
    df = pd.DataFrame(results_data)
    
    # 基本統計の計算
    avg_count = df['検出人数'].mean()
    max_count = df['検出人数'].max()
    min_count = df['検出人数'].min()
    
    # 統計情報の表示
    print(f"\n===== 分析結果 =====")
    print(f"平均検出人数: {avg_count:.1f}人")
    print(f"最大検出人数: {max_count}人")
    print(f"最小検出人数: {min_count}人")
    
    # 結果をExcelファイルに出力
    excel_path = output_dir / f"player_counts_{current_datetime}.xlsx"
    
    # 統計情報のDataFrame
    stats_df = pd.DataFrame({
        '統計項目': ['平均検出人数', '最大検出人数', '最小検出人数'],
        '値': [f"{avg_count:.1f}人", f"{max_count}人", f"{min_count}人"]
    })
    
    # ExcelWriterでファイルを作成
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 詳細データをシート1に出力
        df.to_excel(writer, sheet_name='詳細データ', index=False)
        
        # 統計情報をシート2に出力
        stats_df.to_excel(writer, sheet_name='統計情報', index=False)
        
    print(f"\nExcelファイルを保存しました: {excel_path}")

if __name__ == "__main__":
    video_path = Path(r"C:\Users\rfgra\Desktop\douga\6079618-uhd_3840_2160_25fps.mp4")
    if not video_path.exists():
        print(f"エラー: 動画ファイルが見つかりません: {video_path}")
    else:
        count_players(video_path)
