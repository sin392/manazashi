import cv2

if __name__ == '__main__':

    # 定数定義
    ESC_KEY = 27     # Escキー
    INTERVAL= 33     # 待ち時間
    FRAME_RATE = 30  # fps

    # ウィンドウ名
    WINDOW_NAME = "extraction"

    # フォント
    font = cv2.FONT_HERSHEY_DUPLEX
    font_size = 1.0

    # 描画
    color = (0, 0, 225)
    pen_w = 2

    # 顔検出
    cascade_file = "haarcascade_frontalface_default.xml"
    minSize = (100, 100)

    # 動画ファイル
    mov = "movie.mp4"

    # プレフィックス
    prfx = "smpl"

    # 動画取得
    cap = cv2.VideoCapture(mov)

    # 読込開始
    end_flag, c_frame = cap.read()

    # ウィンドウの準備
    cv2.namedWindow(WINDOW_NAME)

    # 分類器の生成
    cascade = cv2.CascadeClassifier(cascade_file)

    # カウンター
    cnt_frame = 0
    cnt_face = 0

    text = ""

    # 変換処理ループ
    while end_flag == True:

        img = c_frame
        img_out = img.copy()

        # 白黒画像に変換
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 画像内の顔を検出
        face_list = cascade.detectMultiScale(img_gray, minNeighbors=3, minSize=minSize)

        for (pos_x, pos_y, w, h) in face_list:

            # 検出した顔を切り出す
            img_face = img[pos_y:pos_y+h, pos_x:pos_x+w]
            img_face = cv2.resize(img_face, minSize)

            # 検出した顔に四角を描画
            cv2.rectangle(img_out, (pos_x, pos_y), (pos_x+w, pos_y+h), color, thickness = pen_w)

            # 検出状況
            text = "{}_{:09}_{:05}.jpg".format(prfx, cnt_frame, cnt_face)

            # 検出した画像を保存
            cv2.imwrite("faces/" + text, img_face)

            cnt_face += 1

        # 進捗表示
        cv2.putText(img_out, text, (50, 50), font, font_size, color)

        # フレーム表示
        cv2.imshow(WINDOW_NAME, img_out)

        # Escキーで終了
        key = cv2.waitKey(INTERVAL)
        if key == ESC_KEY:
            break

        # 次のフレーム読み込み
        end_flag, c_frame = cap.read()
        cnt_frame += 1

    # 終了処理
    cv2.destroyAllWindows()
    cap.release()
