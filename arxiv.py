import socket
import os
import re
import random
import time
import glob

from urllib.request import urlopen, urlretrieve, Request


arxiv_url = "https://arxiv.org"
format_url = arxiv_url + "/format/{}"
e_print_url = arxiv_url + "/e-print/{}"

# use arxiv sanity to fetch papers
arxiv_sanity_pdf_path = "arxiv-sanity-preserver/data/pdf"
# source downloaded to here
tmp_folder = "tmp"
# final processed data
processed_data_path = "data"

socket.setdefaulttimeout(5)
user_agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36"
wait_time = 5


def fetch_format(url, dest):
    try:
        q = Request(url)
        q.add_header("User-Agent", user_agent)
        with urlopen(q) as uo:
            response = uo.read()

        regex = re.compile(b'<a href="/e-print/(.*?)">Download source</a>', re.S)
        link = re.findall(regex, response)[0]
        filename, _ = urlretrieve(e_print_url.format(link.decode("utf-8")), dest)
    except Exception as e:
        return str(e)

    return True


def detex_file(f, tmp_out_file, writer):
    os.system("detex -n {} > {}/{}".format(
        f, tmp_folder, tmp_out_file))

    with open(os.path.join(tmp_folder, tmp_out_file), "rb") as r:
        for line in r:
            line = line.strip()
            if len(line) > 0:
                writer.write(line+b"\n")
        writer.write(b"\n")


def recursive_listdir(folder, tmp_out_file, writer):
    has_written = False
    find = folder + "{}.tex"
    slash = "/*"
    while True:
        tex_list = glob.glob(find.format(slash), recursive=True)
        if len(tex_list) == 0 and len(slash) > 4:
            break

        slash += "/*"
        for tex in tex_list:
            if "(" not in tex and ")" not in tex:
                tex_name = tex.replace(" ", "\ ") # handle space within path name
                detex_file(tex_name, tmp_out_file, writer)
                has_written = True

    return has_written


def detex(id):
    dest = os.path.join(tmp_folder, id+".tar")
    tmp_out_file = "out.txt"
    has_written = False
    write_file = os.path.join(processed_data_path, id+".txt")
    with open(write_file, "wb") as w:
        has_written = recursive_listdir(tmp_folder, tmp_out_file, w)

    if not has_written:
        print("WARNING: {} cannot find tex file.".format(id))
        return -1

    return 0


def get_processed_data():
    processed_data = {}
    for f in os.listdir(processed_data_path):
        processed_data[f[:-4]] = 0

    return processed_data


def main():
    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)

    if not os.path.exists(processed_data_path):
        os.mkdir(processed_data_path)

    processed_data = get_processed_data()

    filelist = os.listdir(arxiv_sanity_pdf_path)
    total_num = len(filelist)
    success_num = 0
    process_num = 0
    for f in filelist:
        process_num += 1
        id = f[:-4]
        if id not in processed_data and "." in id:
            dest = os.path.join(tmp_folder, id+".tar")
            res = fetch_format(format_url.format(id), dest)
            if res == True:
                # uncompress
                os.system("tar -zxf {} --directory {}".format(dest, tmp_folder))
                # detex
                ret = detex(id)
                # delete downloaded tmp files
                os.system("rm -rf {}/*".format(tmp_folder))

                success_num += 1
                if ret == 0:
                    print("[{:5d}/{:5d}|{:5d}]\t{} has been processed successfully.".format(
                        success_num, process_num, total_num, id))
            else:
                print("[{:5d}/{:5d}|{:5d}]\tfetch {} failed, error is {}".format(
                    success_num, process_num, total_num, id, res))

            time.sleep(wait_time + random.uniform(0, 3))
        else:
            if "." in id:
                success_num += 1
                print("[{:5d}/{:5d}|{:5d}]\t{} has already existed, skipping.".format(
                    success_num, process_num, total_num, id))


def manual_process():
    for root, subfolders, files in os.walk(tmp_folder):
        for subfolder in subfolders:
            if "." in subfolder:
                tmp_out_file = "out.txt"
                write_file = os.path.join(processed_data_path, subfolder+".txt")
                with open(write_file, "wb") as w:
                    res = recursive_listdir(os.path.join(tmp_folder, subfolder), tmp_out_file, w)
                    if res:
                        print("{} has been processed.".format(subfolder))
                    else:
                        print("{} cannot find tex files".format(subfolder))


if __name__ == "__main__":
    main()
    #manual_process()
