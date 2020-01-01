import subprocess
import os
import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":
    # prepare face-net
    # if not os.path.exists("facenet_pytorch"):
    #     command = ["git", "clone", "https://github.com/timesler/facenet-pytorch.git", "facenet_pytorch"]
    #     subprocess.call(command)

    # prepare M2Det
    if not os.path.exists("M2Det"):
        command = ["git", "clone", "https://github.com/qijiezhao/M2Det.git", "M2Det"]
        subprocess.call(command)
        os.chdir("M2Det")
        command = ["sh", "make.sh"]
        subprocess.call(command)
        os.chdir("../")

    if not os.path.exists("weights"): os.mkdir("weights")

    if not os.path.exists("weights/m2det512_vgg.pth"):
        # 正常にDLできないのでひとまずは手動でDLすること

        # file_id = 'https://drive.google.com/file/d/1NM1UDdZnwHwiNDxhcP-nndaWj24m-90L/view'
        # destination = 'weights/m2det512_vgg.pth'
        # download_file_from_google_drive(file_id, destination)