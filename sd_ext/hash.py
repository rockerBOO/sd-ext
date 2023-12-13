import hashlib

# BUF_SIZE is totally arbitrary, change for your app!
BUF_SIZE = 65536  # lets read stuff in 64kb chunks!


def hash_file(file, buf_size=BUF_SIZE):
    sha1 = hashlib.sha1()

    with open(file, "rb") as f:
        while True:
            data = f.read(buf_size)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest()
