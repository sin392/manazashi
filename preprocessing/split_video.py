import os
import glob
import argparse
import subprocess  # ターミナルで実行するコマンドを実行できる

def split_video(INPUT='', OUTPUT='', frame_rate=3, size=224):
    # video -> image
    command = f'ffmpeg -i {INPUT} -vcodec png -vf scale={size}:{size} -r {frame_rate} {OUTPUT}'
    print(command)
    subprocess.call(command, shell=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='specify Image folder directory')
    parser.add_argument('-o', '--output', help='Directory to output the result')
    parser.add_argument('-r', '--rate', type=int, default=3)
    parser.add_argument('-s', '--size', type=int, default=224)
    args = parser.parse_args()

    rate = args.rate

    if os.path.splitext(args.input)[1] in (".mp4", ".MP4"):
        video_list = [args.input]
    else:
        video_list = glob.glob(os.path.join(args.input, "*"))

    for video in video_list:
        folder = os.path.splitext(os.path.basename(video))[0]
        os.makedirs(os.path.join(args.output, folder), exist_ok=True)

        sub_folder = os.path.join(args.output, folder, f'Image_r{rate}')
        os.makedirs(sub_folder, exist_ok=True)

        output = os.path.join(sub_folder, 'image_%05d.png')
        print(output)

        split_video(INPUT=video, OUTPUT=output, frame_rate=args.rate, size=args.size)

if __name__ == '__main__':
    main()