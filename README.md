# manazashi

## 0.データ
動画  
https://teams.microsoft.com/_#/school/files/%E4%B8%80%E8%88%AC?threadId=19%3A53f497dfc6b240b9b0a4434cb00db9d2%40thread.skype&ctx=channel&context=DCON2020%252Fmanazashi

重み  
https://drive.google.com/open?id=12ehli-RGLoQ7b7ExD0buAsaVAPtKe97i
→ manazashi/weights以下に配置すること

#### 1.clone_repository.py
今は使わなくていい


## 1.前処理 (preprocessingフォルダ)
ffmpegを使用して動画から学習データ用の静止画を切り出すプログラム (ffmpegのインストールが必要)
#### 1.preprocessing/split_video.py
指定fpsで静止画を切り出す。
```
--input -i 入力ファイル(.mp4 or .m4v)
--out -o   出力先フォルダ(画像の保存先)
--rate -r　 フレームレート(1sに何枚切り出すか)    
--size -s  画像の出力サイズ
```

#### 2.preprocessing/make_cropped_image.py
フレーム内に検出された人物矩形を切り出す。

## 2.デモ
#### 1.demo.py
m2detとmtcnnによる人物検出と顔検出を組み合わせたデモ。
静止画、動画、カメラ入力に対応。
デフォルトでは1fpsで推論。
gui時、動画クリックで一時停止。
```
// 入力
--config -c m2detの設定ファイルを指定
--weight -w m2detのweightパス(mtcnnはパス固定)
--directory -f 画像ディレクトリ or 画像ファイル
--video 動画入力モードのフラグ（プログラム実行後にパス指定)
--camera カメラ入力オプション。カメラidを指定（内臓は0）。
// 表示
--show スコア推移グラフと検出結果を別ウィンドウで表示。
--gui 上記の情報等を1つのウィンドウにまとめて表示。一時停止等も可能。
//その他
--fixed 初めの5フレームを使用し、人物bboxの位置をキャリブレーション。cpu時は使用推奨。
--crop 検出した人物画像の抽出
```

## 3.その他
#### 1.research/embedding.py
画像特徴量をResNetで抜き出す。
tensorboardのログを残すので可視化可能。
`tensoborad logdir=./logs`
