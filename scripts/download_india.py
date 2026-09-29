"""Fetch only RDD2022/India.zip (~500 MB) out of the 13 GB RDD2022 Figshare archive via HTTP range requests."""
import io, sys, zipfile, requests, pathlib

URL = "https://ndownloader.figshare.com/files/38030910"
OUT = pathlib.Path(__file__).resolve().parent.parent / "data"
BLOCK = 16 * 1024 * 1024


class RangeFile(io.RawIOBase):
    def __init__(self):
        self.s = requests.Session()
        r = self.s.get(URL, headers={"Range": "bytes=0-0"}, stream=True)
        self.size = int(r.headers["Content-Range"].split("/")[1]); r.close()
        self.pos = 0; self.cache_start = -1; self.cache = b""

    def _fetch(self, start, end):
        for _ in range(5):
            try:
                r = self.s.get(URL, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
                if r.status_code in (200, 206):
                    return r.content
            except requests.RequestException:
                pass
        raise IOError("range fetch failed")

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        n = min(n, self.size - self.pos)
        out = b""
        while n > 0:
            if not (self.cache_start <= self.pos < self.cache_start + len(self.cache)):
                self.cache_start = self.pos
                self.cache = self._fetch(self.pos, min(self.pos + BLOCK, self.size) - 1)
            o = self.pos - self.cache_start
            chunk = self.cache[o:o + n]
            out += chunk; self.pos += len(chunk); n -= len(chunk)
        return out

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)


rf = RangeFile()
z = zipfile.ZipFile(rf)
info = z.getinfo("RDD2022/India.zip")
OUT.mkdir(exist_ok=True)
with z.open(info) as f, open(OUT / "India.zip", "wb") as o:
    done = 0
    while chunk := f.read(8 << 20):
        o.write(chunk); done += len(chunk)
        print(f"{done >> 20} / {info.file_size >> 20} MiB", flush=True)
print("done -> data/India.zip")
